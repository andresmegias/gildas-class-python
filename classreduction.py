#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Reduction Pipeline for CLASS
-----------------------------------
Reduction mode
Version 1.0

Copyright (C) 2022 - Andrés Megías Toledano

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
import os
import gc
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import median_abs_deviation
from scipy.interpolate import UnivariateSpline

# Custom functions.

def rolling_window(y, window):
    """
    Group the input data according to the specified window size.
    
    Function by Erik Rigtorp.

    Parameters
    ----------
    y : array
        Input data.
    window : int
        Size of the windows to group the data.

    Returns
    -------
    y2
        Output array.
    """
    shape = y.shape[:-1] + (y.shape[-1] - window + 1, window)
    strides = y.strides + (y.strides[-1],)
    y_w = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    return y_w

def rolling_function(func, y, size, **kwargs):
    """
    Apply a function in a rolling way, in windows of the specified size.

    Parameters
    ----------
    y : array
        Input data.
    func : function
        Function to be applied.
    size : int
        Size of the windows to group the data. It must be odd.
    **kwargs : (various)
        Keyword arguments of the function to be applied.

    Returns
    -------
    y_f : array
        Resultant array.

    """
    size = int(size) + (int(size) + 1) % 2
    size = max(7, size)
    min_size = size//3
    N = len(y)
    y_c = func(rolling_window(y, size), -1, **kwargs)
    M = min(N, size) // 2
    y_1, y_2 = np.zeros(M), np.zeros(M)
    for i in range(M):
        j1 = 0
        j2 = max(min_size, 2*i)
        y_1[i] = func(y[j1:j2], **kwargs)
        j1 = N - max(min_size, 2*i)
        j2 = N
        y_2[-i-1] = func(y[j1:j2], **kwargs)
    y_f = np.concatenate((y_1, y_c, y_2))
    return y_f

def regions_args(x, windows, margin=0):
    """
    Select the regions of the input array specified by the given windows.

    Parameters
    ----------
    x : array
        Input data.
    windows : array
        Windows that specify the regions of the data.
    margin : float
        Margin for the windows.

    Returns
    -------
    cond : array (bool)
        Resultant condition array.
    """
    cond = np.ones(len(x), dtype=bool)
    dx = np.median(np.diff(x))
    for x1, x2 in windows:
        cond *= (x <= x1 - dx*margin) + (x >= x2 + dx*margin)
    return cond

def reduce_curve(x, y, windows, smooth_size):
    """
    Reduce the curve ignoring the specified windows.

    Parameters
    ----------
    x : array
        Independent variable.
    y : array
        Dependent variable.
    windows : array
        Windows that specify the regions of the data.
    smooth_size : int
        Size of the filter applied for the fitting of the baseline.

    Returns
    -------
    y3 : array
        Reduced array.
    """
    
    cond = regions_args(x, windows)
    x_ = x[cond]
    y_ = y[cond]
    
    y_2 = rolling_function(np.median, y_, smooth_size)
  
    s = len(x_)*(1.0*np.std(y_2-y_))**2
    spl = UnivariateSpline(x_, y_, s=s)
    y3 = spl(x)
    
    y2 = y.copy()
    y2[~cond] = y3[~cond]
    
    y3 = rolling_function(np.median, y2, smooth_size)
    y3 = rolling_function(np.mean, y3, smooth_size//3)
        
    return y3

def sigma_clip_args(y, sigmas=6.0, iters=2):
    """
    Apply a sigma clip and return a mask of the remaining data.

    Parameters
    ----------
    y : array
        Input data.
    sigmas : float, optional
        Number of deviations used as threshold. The default is 6.0.
    iters : int, optional
        Number of iterations performed. The default is 3.

    Returns
    -------
    cond : array (bool)
        Mask of the remaining data after applying the sigma clip.
    """
    cond = np.ones(len(y), dtype=bool)
    abs_y = abs(y)
    for i in range(iters):
        cond *= abs_y < sigmas*median_abs_deviation(abs_y[cond], scale='normal')
    return cond

def load_spectrum(file, load_fits=False):
    """
    Load the spectrum from the given input file.

    Parameters
    ----------
    file : str
        Path of the plain text file (.dat) to load, without the extenxion.
    load_fits : bool
        If True, load also a .fits file and return the HDU list. 

    Returns
    -------
    x : array
        Frequency.
    y : array
        Intensity.
    hdul : HDU list (astropy)
        List of the HDUs (Header Data Unit).
    """
    data = np.loadtxt(file + '.dat')
    x = data[:,0]
    y = data[:,1]
    if np.sum(np.isnan(data)) != 0:
        raise Exception('Data of file {} is corrupted.'.format(file))
    if load_fits:
        hdul = fits.open(file + '.fits')
        if 'BLANK' in hdul[0].header:
            del hdul[0].header['BLANK']
    else:
        hdul = None
    return x, y, hdul

def save_yaml_dict(dictionary, file_path, default_flow_style=False, replace=False):
    """
    Save the input YAML dictionary into a file.

    Parameters
    ----------
    dictionary : dict
        Dictionary that wants to be saved.
    file_path : str
        Path of the output file.
    default_flow_style : bool, optional
        The flow style of the output YAML file. The default is False.
    replace : bool, optional
        If True, replace the output file in case it existed. If False, load the
        existing output file and merge it with the input dictionary.
        The default is False.

    Returns
    -------
    None.
    """
    file_path = os.path.realpath(file_path)
    if not replace and os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            old_dict = yaml.safe_load(file)
        new_dict = {**old_dict, **dictionary}
    else:
        new_dict = dictionary
    with open(file_path, 'w') as file:
        yaml.dump(new_dict, file, default_flow_style=default_flow_style)

def get_rms_noise(x, y, windows=[], sigmas=3.5, margin=0., iters=3):
    """
    Obtain the RMS noise of the input data, ignoring the given windows.

    Parameters
    ----------
    x : array
        Independent variable.
    y : array
        Dependent variable.
    windows : array, optional
        Windows of the independent variable that will be avoided in the
        calculation of the RMS noise. The default is [].
    sigmas : float, optional
        Number of deviations used as threshold for the sigma clip applied to
        the data before the calculation of the RMS noise. The default is 6.0.
    margin : float, optional
        Relative frequency margin that will be ignored for calculating the RMS
        noise. The default is 0.
    iters : int, optional
        Number of iterations performed for the sigma clip applied to the data
        before the calculation of the RMS noise. The default is 3.

    Returns
    -------
    rms_noise : float
        Value of the RMS noise of the data.
    """
    N = len(x)
    i1, i2 = int(margin*N), int((1-margin)*N)
    x = x[i1:i2]
    y = y[i1:i2]
    cond = regions_args(x, windows)
    y = y[cond]
    cond = sigma_clip_args(y, sigmas=sigmas, iters=iters)
    y = y[cond]
    rms_noise = np.sqrt(np.mean(y**2)) 
    return rms_noise

def find_rms_region(x, y, rms_noise, windows=[], rms_threshold=0.1,
                    offset_threshold=0.05, reference_width=200, min_width=120,
                    max_iters=1000):
    """
    Find a region of the input data that has a similar noise than the one given.

    Parameters
    ----------
    x : array
        Independent variable.
    y : array
        Dependent variable.
    rms_noise : float
        The value of the RMS used as a reference.
    windows : array, optional
        The regions of the independent variable that should be ignored.
        The default is [].
    rms_threshold : float, optional
        Maximum relative difference that can exist between the RMS noise of the
        searched region and the reference RMS noise. The default is 0.1.
    offset_threshold : float, optional
        Maximum value, in units of the reference RMS noise, that the mean value
        of the dependent variable can have in the searched region.
        The default is 0.05.
    reference_width : int, optional
        Size of the desired region, in number of channels. The default is 200.
    min_width : int, optional
        Minimum size of the desired region, in number of channels.
        The default is 120.
    max_iters : int, optional
        Maximum number of iterations that will be done to find the desired
        region. The default is 1000.

    Returns
    -------
    rms_region : list
        Frequency regions of the desired region.
    """
    i = 0
    local_rms = 0
    offset = 1*rms_noise
    while not (abs(local_rms - rms_noise) / rms_noise < rms_threshold
               and abs(offset) / rms_noise < offset_threshold):
        width = max(min_width, reference_width)
        resolution = np.median(np.diff(x))
        central_freq = np.random.uniform(x[0] + width/2*resolution,
                                         x[-1] - width/2*resolution)
        region_inf = central_freq - width/2*resolution
        region_sup = central_freq + width/2*resolution
        cond = (x > region_inf) & (x < region_sup)
        y_ = y[cond]
        valid_range = True
        for x1, x2 in windows:
            if (region_inf < x1 < region_sup) or (region_inf < x2 < region_sup):
                valid_range = False
        if valid_range:
            local_rms = float(np.sqrt(np.mean(y_**2)))
            offset = np.mean(y_)
        i += 1
        if i > max_iters:
            return []
        
    rms_region = [float(central_freq - width/2*resolution),
                  float(central_freq + width/2*resolution)]
    return rms_region

# Arguments.
parser = argparse.ArgumentParser()
parser.add_argument('folder')
parser.add_argument('file')
parser.add_argument('-smooth', default=20, type=int)
parser.add_argument('-sigmas', default=4, type=float)
parser.add_argument('-rms_margin', default=0.1, type=float)
parser.add_argument('-plots_folder', default='plots')
parser.add_argument('--no_plots', action='store_true')
parser.add_argument('--save_plots', action='store_true')
parser.add_argument('--rms_check', action='store_true')
args = parser.parse_args()
original_folder = os.path.realpath(os.getcwd())
os.chdir(args.folder)

#%%

if not args.rms_check:
    if os.path.isfile('frequency_windows.yaml'):
        with open('frequency_windows.yaml') as file:
            all_windows = yaml.safe_load(file)
    else:
        print('\nWarning: The file frequency_windows.yaml is missing. '
              + 'No frequency windows will be used.\n')
        all_windows = []
        
rms_noises = {}
frequency_ranges = {}
reference_frequencies = {}
rms_regions = {}
resolutions = {}

for file in args.file.split(','):
    
    # Loading of the data file.
    frequency, intensity, hdul = load_spectrum(file, load_fits=True)
    fits_data = hdul[0]
    frequency_ranges[file] = [float(frequency[0]), float(frequency[-1])]
    resolutions[file] = hdul[0].header['cdelt1'] / 1e6
    reference_frequencies[file] = hdul[0].header['restfreq'] / 1e6
    # Reduction.
    if not args.rms_check:
        windows = all_windows[file]
    else:
        windows = [[frequency[0], frequency[1]/10]]
    if args.smooth > len(frequency):
        args.smooth = len(frequency)
    intensity_cont = reduce_curve(frequency, intensity, windows, args.smooth)
    intensity_red = intensity - intensity_cont
    
    # Noise.
    rms_noise = get_rms_noise(frequency, intensity_red, windows,
                              sigmas=args.sigmas, iters=3, margin=args.rms_margin)

    rms_noises[file] = float(1e3*rms_noise)
    # Noise regions.
    if not args.rms_check:
        rms_region = \
            find_rms_region(frequency, intensity_red, rms_noise=rms_noise,
                            windows=windows, rms_threshold=0.1,
                            offset_threshold=0.05, reference_width=2*args.smooth)
        if len(rms_region) == 0:
            print('Warning: No RMS region was found for spectrum {}.'.format(file))
            rms_region = [float(frequency[0]), float(frequency[-1])]
        rms_regions[file] = rms_region
    
    # Output.
    output_file = file + '-r'
    fits_data = np.float32(np.zeros((1,1,1,len(intensity))))
    fits_data[0,0,0,:] = np.float32(intensity_red)
    hdul[0].data = fits_data
    hdul[0].scale('float32')
    hdul.writeto(output_file + '.fits', overwrite=True)
    hdul.close()
    output_data = np.array([frequency, intensity_red]).transpose()
    np.savetxt(output_file + '.dat', output_data)
    print('Saved reduced spectrum in {}{}.fits.'.format(args.folder, output_file))
    print('Saved reduced spectrum in {}{}.dat.'.format(args.folder, output_file))
    
    gc.collect()
    
    if not args.no_plots or args.save_plots:   
        
        plt.figure(1, figsize=(10,7))
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)
        
        sp1 = plt.subplot(2,1,1)
        plt.step(frequency, intensity, where='mid', color='black', ms=6)
        if not args.rms_check:
            for x1, x2 in windows:
                plt.axvspan(x1, x2, color='gray', alpha=0.3)
        plt.plot(frequency, intensity_cont, 'tab:green', label='fitted baseline')
        plt.ticklabel_format(style='sci', useOffset=False)
        plt.margins(x=0)
        plt.xlabel('frequency (MHz)')
        plt.ylabel('intensity (K)')
        plt.legend(loc='upper right')
        plt.tight_layout()
    
        plt.subplot(2,1,2, sharex=sp1)
        plt.step(frequency, intensity_red, where='mid', color='black')
        if not args.rms_check:
            for x1, x2 in windows:
                plt.axvspan(x1, x2, color='gray', alpha=0.3)
        plt.ticklabel_format(style='sci', useOffset=False)
        plt.margins(x=0)
        plt.xlabel('frequency (MHz)')
        plt.ylabel('reduced intensity (K)')
        plt.tight_layout()

        title = 'Full spectrum - {}'.format(file)
        fontsize = max(7, 12 - 0.1*max(0, len(title) - 85))
        plt.suptitle(title, fontsize=fontsize, fontweight='semibold')
        plt.tight_layout(pad=0.7, h_pad=0.6, w_pad=0.1)

        if args.save_plots:
            if args.rms_check:
                quality = 100
            else:
                quality = 200
            os.chdir(original_folder)
            os.chdir(os.path.realpath(args.plots_folder))
            file_name = 'spectrum-{}.png'.format(file)
            if args.rms_check:
                file_name = file_name.replace('spectrum-rms', 'rms-spectrum')
            plt.savefig(file_name, dpi=quality)
            os.chdir(original_folder)
            os.chdir(os.path.realpath(args.folder))       
            print("    Saved plot in {}{}.".format(args.plots_folder, file_name))
        
        if not args.no_plots:
            plt.show()
        else:
            plt.close('all')
            
    print()
    gc.collect()
            
        
# Export of the rms noise of each spectrum.
save_yaml_dict(rms_noises, 'rms_noises.yaml', default_flow_style=False)
print('Saved RMS noises in {}rms_noises.yaml.'.format(args.folder))        

# Export of the frequency ranges of each spectrum.
save_yaml_dict(frequency_ranges, 'frequency_ranges.yaml',
               default_flow_style=None)
print('Saved frequency ranges in {}frequency_ranges.yaml.'.format(args.folder))

# Export of the reference frequencies ranges of each spectrum.
save_yaml_dict(reference_frequencies, 'reference_frequencies.yaml',
               default_flow_style=None)
print('Saved frequency ranges in {}reference_frequencies.yaml.'.format(args.folder))

# Export of the RMS regions of each spectrum.
save_yaml_dict(rms_regions, 'rms_regions.yaml',
               default_flow_style=None)
print('Saved RMS regions in {}rms_regions.yaml.'.format(args.folder))

# Export of the frequency resolution of each spectrum.
save_yaml_dict(resolutions, 'frequency_resolutions.yaml',
               default_flow_style=False)
print('Saved frequency resolutions in {}frequency_resolutions.yaml.'
      .format(args.folder))

print()