#!/usr/bin/env julia
"""
Automated GILDAS-CLASS Pipeline
-------------------------------
Line search mode
Version 1.3

Copyright (C) 2024 - Andrés Megías Toledano

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Libraries and functions.
import PyPlot as plt
import RollingFunctions: rolling
import Statistics as stats
import Random as random
import StatsBase: mad
import Interpolations: LinearInterpolation, Line
import SciPy.interpolate as scipy_interpolate
import Distributions: Normal
import Formatting: printfmtln
import ArgParse as argparse
import YAML as yaml
import FITSIO as fitsio
import DelimitedFiles as dlmfiles

# Custom functions.

function safe_realpath(path)
    """
    Return the full path corresponding to the input text.

    Parameters
    ----------
    path : String
        Text of the path to be converted.

    Returns
    -------
    full_path : String
        Text of the full path corresponding to the input text.
    """
    if ispath(path)
        return realpath(path)
    end
    if path != ""
        a, b = splitdir(path)
        return joinpath(safe_realpath(a), b)
    else
        return ""
    end
 end

function madn(x)
    """
    Return the normalized median absolute deviation (MAD) of the input data.

    Parameters
    ----------
    x : Array
        Input data.

    Returns
    -------
    y : Float
        Normalized median absolute deviation.
    """
    y = mad(x, normalize=true)
    return y
end

function rolling_function(func::Function, y::Vector{Float64}, roll_size::Int)
    """
    Apply a function in a rolling way, in windows of the specified size.

    Parameters
    ----------
    y : Vector
        Input data.
    func : Function
        Function to be applied.
    size : Int
        Size of the windows to group the data. It must be odd.

    Returns
    -------
    y_f : Vector
        Resultant array.
    """
    min_size = 1
    roll_size = Int(roll_size) + (Int(roll_size) + 1) % 2
    roll_size = max(min_size, roll_size)
    N = length(y)
    y_c = rolling(func, y, roll_size)
    M = min(N, roll_size) ÷ 2
    y_1 = zeros(M)
    y_2 = zeros(M)
    for i in 0:M-1
        j1 = 1
        j2 = max(min_size, 2*i)
        y_1[1+i] = func(y[j1:j2])
        j1 = N + 1 - max(min_size, 2*i)
        j2 = N
        y_2[end-i] = func(y[j1:j2])
    end
    y_f = vcat(y_1, y_c, y_2)
    return y_f
end

function get_windows(x::Vector{Float64}, cond::Vector{Bool};
                        margin::Float64=0.0, width::Float64=10.0)
    """
    Return the windows of the empty regions of the input array.

    Parameters
    ----------
    x : Vector
        Input data.
    cond : Vector
        Indices of the empty regions of data.
    margin : Float, optional
        Relative margin added to the windows found initially.
        The default is 0.0.
    width: Float, optional
        Minimum separation in points between two consecutive windows.

    Returns
    -------
    windows : Vector
        List of the inferior and superior limits of each window.
    inds : Vector
        List of indices that define the filled regions if the data.
    """

    N = length(x)
    separation = abs.(diff(x))
    reference = stats.median(separation)
    all_inds = 1:N

    var_inds = diff(vcat([0], Int.(cond), [0]))
    cond1 = (var_inds .== 1)[1:end-1]
    cond2 = (var_inds .== -1)[2:end]
    inds = vcat(all_inds[cond1], all_inds[cond2])
    inds = reshape(sort(inds), (2,length(inds)÷2))'

    windows = x[inds]
    for i in 1:size(windows)[1]
        window = windows[i,:]
        center = stats.mean(window)
        semiwidth = (window[2] - window[1]) / 2
        semiwidth = max(3*reference, semiwidth, 0.1*width*reference)
        semiwidth = (1 + margin) * semiwidth - 1E-9
        windows[i,:] = [center - semiwidth, center + semiwidth]
        windows[i,1] = max(x[1], windows[i,1])
        windows[i,2] = min(windows[i,2], x[end])
    end
    i = 1
    while i < size(windows)[1]
        difference = windows[i+1,1] - windows[i,2]
        if difference < width*reference
            windows[i,1] = min(windows[i,1], windows[i+1,1])
            windows[i,2] = max(windows[i,2], windows[i+1,2])
            windows = windows[1:end .!= i+1, :]
        else
            i += 1
        end
    end
    return windows
end

function regions_args(x::Vector{Float64}, windows::Matrix{Float64};
                      margin::Float64=0.0)
    """
    Select the regions of the input array specified by the given windows.

    Parameters
    ----------
    x : Vector
        Input data.
    wins : Matrix
        Windows that specify the regions of the data.

    Returns
    -------
    cond : Vector
        Resultant condition array.
    """
    cond = ones(Bool, length(x))
    dx = stats.median(diff(x))
    for i in 1:size(windows)[1]
        x1, x2 = windows[i,1], windows[i,2]
        cond .*= ((x .<= x1 - dx*margin) .+ (x .>= x2 + dx*margin))
    end
    return cond
end

function sigma_clip_args(y::Vector{Float64}; sigmas::Float64=6.0, iters::Int=2)
    """
    Apply a sigma clip and return a mask of the remaining data.

    Parameters
    ----------
    y : Vector
        Input data.
    sigmas : Float, optional
        Number of standard deviations used as threshold. The default is 6.0.
    iters : Int, optional
        Number of iterations performed. The default is 3.

    Returns
    -------
    cond : Vector
        Mask of the remaining data after applying the sigma clip.
    """
    cond = ones(Bool, length(y))
    abs_y = abs.(y)
    for i in 1:iters
        cond .*= abs_y .< sigmas*madn(abs_y[cond])
    end
    return cond
end

function rolling_sigma_clip_args(x::Vector{Float64}, y::Vector{Float64};
                                 smooth::Int, sigmas::Float64=6.0, iters::Int=2)
    """
    Apply a rolling sigma clip and return a mask of the remaining data.

    Parameters
    ----------
    x : Vector
        Dependent variable.
    y : Vector
        Independent variable.
    size : Int
        Size of the windows to group the data. It must be odd.
    sigmas : Float, optional
        Number of standard deviations used as threshold. The default is 4.0.
    iters : Int, optional
        Number of iterations performed. The default is 3.

    Returns
    -------
    cond : Vector
        Mask of the remaining data after applying the sigma clip.
    """
    cond = ones(Bool, length(y))
    abs_y = abs.(y)
    for i in 1:iters
        rolling_mad = rolling_function(madn, abs_y[cond], 2*smooth)
        itp = LinearInterpolation(x[cond], rolling_mad, extrapolation_bc=Line())
        rolling_mad = itp(x)
        cond .*= abs_y .< sigmas.*rolling_mad
    end
    return cond
end

function fit_baseline(x::Vector{Float64}, y::Vector{Float64};
                      windows::Matrix{Float64}, smooth_size::Int)
    """
    Fit the baseline of the curve ignoring the specified windows.

    Parameters
    ----------
    x : Vector
        Independent variable.
    y : Vector
        Dependent variable.
    windows : Matrix
        Windows that specify the regions of the data.
    smooth_size : Int
        Size of the filter applied for the fitting of the baseline.

    Returns
    -------
    yf : Vector
        Baseline of the curve.
    """
    cond = regions_args(x, windows)
    x_ = x[cond]
    y_ = y[cond]
    y_s = rolling_function(stats.median, y_, smooth_size)
    s = sum((y_s - y_).^2)
    spl = scipy_interpolate.UnivariateSpline(x_, y_, s=s)
    yf = spl(x)
    return yf
end

function identify_lines(x::Vector{Float64}, y::Vector{Float64}; smooth_size::Int,
    line_width::Float64, sigmas::Float64, iters::Int=2,
    rolling_sigma_clip::Bool=false)
    """
    Identify the lines of the spectrum and fits the baseline.

    Parameters
    ----------
    x : Vector
        Frequency.
    y : Vector
        Intensity.
    smooth_size : Int
        Size of the filter applied for the fitting of the baseline.
    line_width : Float
        Reference line width for merging close windows.
    sigmas : Float
        Threshold for identifying the outliers.
    iters : Int, optional
        Number of iterations of the process. The default is 2.
    rolling_sigma_clip: Bool, optional
        Use a rolling sigma clip for finding the outliers.

    Returns
    -------
    y3 : Vector
        Estimated baseline.
    windows: Matrix
        Values of the windows of the identified lines.
    """

    local y2 = rolling_function(stats.median, y, smooth_size)

    local windows

    for i in 1:iters

        cond = []
        if rolling_sigma_clip
            cond = rolling_sigma_clip_args(x, y.-y2, smooth=smooth_size,
                                            sigmas=sigmas)
        else
            cond = sigma_clip_args(y.-y2, sigmas=sigmas)
        end
        _cond = Vector(.!cond)

        windows = get_windows(x, _cond, margin=1.5, width=line_width)

        if i < iters
            y2 = fit_baseline(x, y, windows=windows, smooth_size=smooth_size)
        end

    end

    return windows
end

function load_spectrum(file::String; load_fits::Bool=false)
    """
    Load the spectrum from the given input file.

    Parameters
    ----------
    file : String
        Path of the plain text file (.dat) to load, without the extension.
    load_fits : Bool
        If true, load also a .fits file and return the HDU list.

    Returns
    -------
    x : Vector
        Frequency.
    y : Vector
        Intensity.
    hdul : HDU list (FITSIO)
        List of the HDUs (Header Data Unit).
    """
    data = dlmfiles.readdlm("$file.dat")
    x = data[:,1]
    y = data[:,2]
    if sum(isnan.(data)) != 0
        println("Data of file $file is corrupted.")
        throw(Exception)
    end
    if load_fits
        hdul = fitsio.FITS("$file.fits")
    else
        hdul = nothing
    end
    return x, y, hdul
end

function save_yaml_dict(dictionary::Dict, file_path::String; replace::Bool=false)
    """
    Save the input YAML dictionary into a file.

    Parameters
    ----------
    dictionary : Dict
        Dictionary that wants to be saved.
    file_path : String
        Path of the output file.
    default_flow_style : Bool, optional
        The flow style of the output YAML file. The default is False.
    replace : Bool, optional
        If true, replace the output file in case it existed. If false, load the
        existing output file and merge it with the input dictionary.
        The default is false.

    Returns
    -------
    nothing.
    """
    file_path = safe_realpath(file_path)
    if ! replace & isfile(file_path)
        old_dict = yaml.load_file(file_path)
        new_dict = merge(old_dict, dictionary)
    else
        new_dict = dictionary
    end
    yaml.write_file(file_path, new_dict)
end

# Arguments.
aps = argparse.ArgParseSettings()

argparse.@add_arg_table! aps begin
    "folder"
    arg_type = String
    required = true
    "file"
    arg_type = String
    required = true
    "--smooth"
    arg_type = Int
    default = 20
    "--width"
    arg_type = Float64
    default = 6.0
    "--threshold"
    arg_type = Float64
    default = 8.0
    "--plots_folder"
    arg_type = String
    default = "plots"
    "--rolling_sigma"
    action = :store_true
    "--save_plots"
    action = :store_true
end

#%%

args = argparse.parse_args(aps)
original_folder = safe_realpath(pwd())
cd(safe_realpath(args["folder"]))

#%% Calculations.

windows_dict = Dict()

# Processing of each spectrum.
for file in split(args["file"], ",")

    # Loading of the data file.
    frequency, intensity = load_spectrum(String(file))

    # Identification of the lines and reduction of the spectrum.
    windows =
        identify_lines(frequency, intensity, smooth_size=args["smooth"],
                       line_width=args["width"], sigmas=args["threshold"],
                       iters=2, rolling_sigma_clip=args["rolling_sigma"])
    intensity_cont = fit_baseline(frequency, intensity, windows=windows,
                                  smooth_size=args["smooth"])
    intensity_red = intensity .- intensity_cont

    # Windows.

    num_windows = size(windows)[1]

    if num_windows != 0
        println("$num_windows windows identified for $file.")
    else
        windows = [[frequency[1], frequency[2]/2]]
        println("No lines identified for $file.")
    end

    windows_dict[file] = [[windows[i,1], windows[i,2]] for i in 1:num_windows]

    #%% Plots.

    if args["save_plots"]

        plt.figure(1, figsize=(10,7))
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)

        sp1 = plt.subplot(2,1,1)
        plt.step(frequency, intensity, where="mid", color="black", ms=6)
        for i in 1:num_windows
            x1, x2 = windows[i,:]
            plt.axvspan(x1, x2, color="gray", alpha=0.3)
        end
        plt.plot(frequency, intensity_cont, "tab:green", label="fitted baseline")
        plt.ticklabel_format(style="sci", useOffset=false)
        plt.margins(x=0)
        plt.xlabel("frequency (MHz)")
        plt.ylabel("original intensity (K)")
        plt.legend(loc="upper right")

        plt.subplot(2,1,2, sharex=sp1)
        for i in 1:num_windows
            x1, x2 = windows[i,:]
            plt.axvspan(x1, x2, color="gray", alpha=0.3)
        end
        plt.step(frequency, intensity_red, where="mid", color="black")
        plt.ticklabel_format(style="sci", useOffset=false)
        plt.margins(x=0)
        plt.xlabel("frequency (MHz)")
        plt.ylabel("reduced intensity (K)")

        plt.suptitle("Full spectrum - $file", fontweight="semibold")
        plt.tight_layout(pad=0.7, h_pad=1.0)

        dx = stats.median(diff(frequency))
        plt.matplotlib.rc("font", size=8)
        num_lines = size(windows)[1]
        num_plots = 1 + (num_lines - 1)//15
        for i in 0:num_plots-1
            fig = plt.figure(2+i, figsize=(12, 6))
            plt.clf()
            for j in 0:Int(round(min(num_lines - 15*i, 15))-1)
                plt.subplot(3, 5, j+1)
                j += Int(15*i)
                x1, x2 = windows[j+1,:]
                margin = max(args["width"]*dx, 0.4*(x2-x1))
                cond = (frequency .> x1 - margin) .* (frequency .< x2 + margin)
                xj = frequency[cond]
                yrj = intensity_red[cond]
                plt.step(xj, yrj, where="mid", color="black")
                plt.axvspan(x1, x2, color="gray", alpha=0.2)
                plt.margins(x=0, y=0.1)
                if j+1 > min(15*(i+1), num_lines) - 5
                    plt.xlabel("frequency (MHz)")
                end
                if j%5 == 0
                    plt.ylabel("reduced intensity (K)")
                end
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
                plt.locator_params(axis="x", nbins=1)
                plt.locator_params(axis="y", nbins=3)
                plt.ticklabel_format(style="sci", useOffset=false)

            end
            window_num = ""
            if num_plots > 1
                window_num = " ($(Int(i)+1))"
            end
            plt.suptitle("Identified lines$window_num - $file",
                             fontweight="semibold")
            fig.align_ylabels()
            plt.tight_layout(pad=1.2, h_pad=0.6, w_pad=0.1)

        end

        if args["save_plots"]
            cd(original_folder)
            cd(safe_realpath(args["plots_folder"]))
            plt.figure(1)
            plt.savefig("spectrum-$file.png", dpi=200)
            print("    ")
            println("Saved plot in $(args["plots_folder"])spectrum-$file.png.")
            for i in 1:num_plots
                i = Int(i)
                plt.figure(1+i)
                plt.savefig("lines-$(file)_$i.png", dpi=200)
                print("    ")
                println("Saved plot in $(args["plots_folder"])lines-$(file)_$i.png.")
            end
            println()
            cd(original_folder)
            cd(safe_realpath(args["folder"]))
        end

    end

end

# Export of the frequency windows of each spectrum.
save_yaml_dict(windows_dict, "frequency_windows.yaml")
println("Saved windows in $(args["folder"])frequency_windows.yaml.")

println()
