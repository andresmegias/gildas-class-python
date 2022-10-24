#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Reduction Pipeline for CLASS
-----------------------------------
Line search mode
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

def full_path(text):
    """
    Obtain the full path described in a text string.
    
    Corrects the format of the path, allowing the use of operating system
    abbreviations (for example, './', '../' and '~' for Unix).
    
    Parameters
    ----------
    text : str
        Text of the path, which can include operating system abbreviations.
        
    Returns
    -------
    path : str
        Text of the full path, so that Python can use it.
    """
    path = text
    if path.startswith('~'):
        path = os.path.expanduser('~') + path[1:]  
    path = str(os.path.realpath(path))
    return path

 

def get_windows(x, cond, margin=0.0, width=10.0):
    """
    Return the windows of the empty regions of the input array.

    Parameters
    ----------
    x : array
        Input data.
    cond : array (bool)
        Indices of the empty regions of data.
    margin : float, optional
        Relative margin added to the windows found initially.
        The default is 0.0.
    width: float, optional
        Minimum separation in points between two consecutive windows.

    Returns
    -------
    windows : array (float)
        List of the inferior and superior limits of each window.
    inds : array (int)
        List of indices that define the filled regions if the data.
    """
    N = len(x)
    separation = abs(np.diff(x))
    reference = np.median(separation)
    all_inds = np.arange(N)
    var_inds = np.diff(np.concatenate(([0], np.array(cond, dtype=int), [0])))
    cond1 = (var_inds == 1)[:-1]
    cond2 = (var_inds == -1)[1:]
    inds = np.append(all_inds[cond1], all_inds[cond2])
    inds = np.sort(inds).reshape(-1,2)
    windows = x[inds]
    for i, window in enumerate(windows):
        center = np.mean(window)
        semiwidth = (window[1] - window[0]) / 2
        semiwidth = max(3*reference, semiwidth, 0.1*width*reference)
        semiwidth = (1 + margin) * semiwidth - 1E-9
        windows[i,:] = [center - semiwidth, center + semiwidth]
    i = 0
    while i < len(windows) - 1:
        difference = windows[i+1,0] - windows[i,1]
        if difference < width*reference:
            windows[i,0] = min(windows[i,0], windows[i+1,0])
            windows[i,1] = max(windows[i,1], windows[i+1,1])
            windows = np.delete(windows, i+1, axis=0)
        else:
            i += 1
    windows = np.maximum(x[0], windows)
    windows = np.minimum(windows, x[-1])
    return windows

def regions_args(x, wins):
    """
    Select the regions of the input array specified by the given windows.

    Parameters
    ----------
    x : array
        Input data.
    wins : array
        Windows that specify the regions of the data.

    Returns
    -------
    cond : array (bool)
        Resultant condition array.
    """
    cond = np.ones(len(x), dtype=bool)
    for x1, x2 in wins:
        cond *= (x < x1) + (x > x2)
    return cond

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
    min_size = 3
    size = int(size) + (int(size) + 1) % 2
    size = max(min_size, size)
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

def rolling_sigma_clip_args(x, y, smooth_size, sigmas=6.0, iters=2):
    """
    Apply a rolling sigma clip and return a mask of the remaining data.

    Parameters
    ----------
    x : array
        Dependent variable.
    y : array
        Independent variable.
    size : int
        Size of the windows to group the data. It must be odd.
    sigmas : float, optional
        Number of standard deviations used as threshold. The default is 6.0.
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
        rolling_mad = rolling_function(median_abs_deviation, abs_y[cond],
                                       size=2*smooth_size, scale='normal')
        rolling_mad = np.interp(x, x[cond], rolling_mad)

        cond *= np.less(abs_y, sigmas*rolling_mad)
    return cond

def sigma_clip_args(y, sigmas=6.0, iters=2):
    """
    Apply a sigma clip and return a mask of the remaining data.

    Parameters
    ----------
    y : array
        Input data.
    sigmas : float, optional
        Number of standard deviations used as threshold. The default is 4.0.
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

def identify_lines(x, y, smooth_size, line_width, sigmas, iters=2,
                   rolling_sigma_clip=False):
    """
    Identify the lines of the spectrum and fits the baseline.

    Parameters
    ----------
    x : array
        Frequency.
    y : array
        Intensity.
    smooth_size : int
        Size of the filter applied for the fitting of the baseline.
    line_width : float
        Reference line width for merging close windows.
    sigmas : float
        Threshold for identifying the outliers.
    iters : int, optional
        Number of iterations of the process. The default is 2.
    rolling_sigma_clip: bool, optional
        Use a rolling sigma clip for finding the outliers.

    Returns
    -------
    y3 : array
        Estimated baseline.
    windows: array
        Values of the windows of the identified lines.
    """

    y2 = rolling_function(np.median, y, smooth_size)  
    
    for i in range(iters):
 
        if rolling_sigma_clip:
            cond = rolling_sigma_clip_args(x, y-y2, smooth_size, sigmas, iters=2)
        else:
            cond = sigma_clip_args(y-y2, sigmas=sigmas, iters=2)
        windows = get_windows(x, ~cond, margin=1.5, width=line_width)
        
        if i+1 < iters:
            y3 = reduce_curve(x, y, windows, smooth_size)  
            y2 = y3
        
    return windows

def load_spectrum(file, load_fits=False):
    """
    Load the spectrum from the given input file.

    Parameters
    ----------
    file : str
        Path of the plain text file (.dat) to load, without the extension.
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

# Arguments.
parser = argparse.ArgumentParser()
parser.add_argument('folder', default='./')
parser.add_argument('file', default='spectrum.dat')
parser.add_argument('-smooth', default=20, type=int)
parser.add_argument('-width', default=6, type=int)
parser.add_argument('-threshold', default=6., type=float)
parser.add_argument('-plots_folder', default='plots', type=str)
parser.add_argument('--rolling_sigma', action='store_true')
parser.add_argument('--no_plots', action='store_true')
parser.add_argument('--save_plots', action='store_true')
args = parser.parse_args()
original_folder = full_path(os.getcwd())
os.chdir(full_path(args.folder))
np.seterr('raise')
    
#%% Calculations.

windows_dict = {}

# Processing of each spectrum.
for file in args.file.split(','):

    # Loading of the data file.
    frequency, intensity, _ = load_spectrum(file)

    # Identification of the lines and reduction of the spectrum.
    windows = \
        identify_lines(frequency, intensity, smooth_size=args.smooth,
                       line_width=args.width, sigmas=args.threshold, iters=2,
                       rolling_sigma_clip=args.rolling_sigma)
    intensity_cont = reduce_curve(frequency, intensity, windows, args.smooth)
    intensity_red = intensity - intensity_cont
    
    # Windows.
    if len(windows) != 0:
        print('{} windows identified for {}.'
              .format(str(len(windows)), file))
    else:
        windows = np.array([[frequency[0], frequency[1]/2]])
        print('No lines identified for {}.'.format(file))
    windows_dict[file] = [[x1, x2] for x1, x2 in windows.tolist()]
    
    gc.collect()
        
    #%% Plots.
    
    if not args.no_plots or args.save_plots:   
        
        fig = plt.figure(1, figsize=(10,7))
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)
        
        sp1 = plt.subplot(2,1,1)
        plt.step(frequency, intensity, where='mid', color='black', ms=6)
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
        for x1, x2 in windows:
            plt.axvspan(x1, x2, color='gray', alpha=0.3)
        plt.step(frequency, intensity_red, where='mid', color='black')
        plt.ticklabel_format(style='sci', useOffset=False)
        plt.margins(x=0)
        plt.xlabel('frequency (MHz)')
        plt.ylabel('reduced intensity (K)')
        plt.tight_layout()

        plt.suptitle('Full spectrum - {}'.format(file),
                     fontweight='semibold')
        fig.align_ylabels()
        plt.tight_layout(pad=0.7, h_pad=0.6, w_pad=0.1)

        dx = np.median(np.diff(frequency)) 
        plt.rcParams['font.size'] = 8
        num_lines = len(windows)
        num_plots = 1 + (num_lines - 1)//15
        for i in range(num_plots):
            fig = plt.figure(2+i, figsize=(12, 6))
            plt.clf()
            for j in range(min(num_lines - 15*i, 15)):
                plt.subplot(3, 5, j+1)
                j += 15*i
                x1, x2 = windows[j]
                margin = max(args.width*dx, 0.4*(x2-x1))
                cond = (frequency > x1 - margin) * (frequency < x2 + margin)
                xj = frequency[cond]
                yrj = intensity_red[cond]
                plt.step(xj, yrj, where='mid', color='black')
                plt.axvspan(x1, x2, color='gray', alpha=0.2)
                plt.margins(x=0, y=0.1)
                if j+1 > min(15*(i+1), num_lines) - 5:
                    plt.xlabel('frequency (MHz)')
                if j%5 == 0:
                    plt.ylabel('reduced intensity (K)')
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
                plt.locator_params(axis='x', nbins=1)
                plt.locator_params(axis='y', nbins=3)
                plt.ticklabel_format(style='sci', useOffset=False)
            window_num = ''
            if num_plots > 1:
                window_num = ' ({})'.format(i+1)
            plt.suptitle('Identified lines{} - {}'.format(window_num, file),
                             fontweight='semibold')
            fig.align_ylabels()
            plt.tight_layout(pad=1.2, h_pad=0.6, w_pad=0.1)
        
        if args.save_plots:
            os.chdir(original_folder)
            os.chdir(full_path(args.plots_folder))
            plt.figure(1)
            plt.savefig('spectrum-{}.png'.format(file), dpi=200)
            print("    Saved plot in {}spectrum-{}.png."
                  .format(args.plots_folder, file))
            for i in range(num_plots):
                plt.figure(2+i)
                plt.savefig('lines-{}_{}.png'.format(file, i+1), dpi=200)
                print("    Saved plot in {}lines-{}_{}.png."
                      .format(args.plots_folder, file, i+1))
            print()
            os.chdir(original_folder)
            os.chdir(full_path(args.folder))
        
        # Show plots.
        if not args.no_plots:
            plt.show()
        else:
            plt.close('all')
            
    gc.collect()
    
# Export of the frequency windows of each spectrum.
save_yaml_dict(windows_dict, 'frequency_windows.yaml',
                  default_flow_style=None)
print('Saved windows in {}frequency_windows.yaml.'.format(args.folder))

print()
