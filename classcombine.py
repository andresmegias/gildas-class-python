#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Reduction Pipeline for CLASS
-----------------------------------
Combine mode
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
import re
import copy
import time
import platform
import argparse
import subprocess
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.stats import median_abs_deviation
from matplotlib.colors import hsv_to_rgb

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
    min_size = 5
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
        

def enlarge_mask(input_cond, iters=1):
    """
    Enlarge the input mask according to the number iterations.

    Parameters
    ----------
    input_cond : array (bool)
        Logic array which defines the input mask.
    iters : int, optional
        Number of units thath the mask will grow in each direction. The
        default is 1.

    Returns
    -------
    cond : array (bool)
        Resultant logic array.

    """
    cond = copy.copy(input_cond)
    N = len(cond)
    for j in range(iters):
        for i in range(0,N-1):
            if input_cond[i] == False and input_cond[i+1] == True:
                cond[i] = True
        for i in range(1,N):
            if input_cond[i-1] == True and input_cond[i] == False:
                cond[i] = True
        input_cond = copy.copy(cond)
    return cond

def combine_equivalent_spectra(x, yy, weights=[1.,1.], clean=False, noises=0,
                               abs_threshold=5., rel_threshold=0.1, size=40,
                               margin=3):
    """
    Combine the given list of spectra, with different possible options.

    Parameters
    ----------
    x : array
        Independent variable.
    yy : list (array)
        List of the dependant variables.
    weights : list, optional
        List of weights for each spectrum. The default is [1,1].
    clean : bool, optional
        If True, perform a cleaning and a interpolation in the regions where
        the values of the spectra differ too much, using the given reference
        noises.
    noises : list, optional
        List of the reference noises for each spectrum.
    abs_threshold : float, optional
        Value for identifying channels with differences greater than this number
        of times the reference noise. The default is 5.
    rel_threshold : float, optional
        Value for identifying channels with relative differences greater than
        this number. The default is 0.1
    size : int, optional
        Size of the median filter applied when doing the interpolation for the
        cleaning.

    Raises
    ------
    Exception
        It raises if clean == True and you don't specify the noises.

    Returns
    -------
    ym : array
        Averaged dependent variable.
    windows : list
        If clean == True, windows is the list of windows that define the
        regions cleaned.
    """
    if clean and len(noises) == 0 and noises == 0:
        raise Exception('You have to specify the reference noises.')
    weights = np.array(weights)
    weights = weights / weights.sum()
    weights = weights.reshape(-1,1)
    n = len(yy)
    N = len(yy[0])
    x, yy = np.array(x), np.array(yy)
    ym = np.sum(yy*weights, axis=0)
    for i in range(n):
        inds_nan = np.arange(N)[np.isnan(yy[i])]
        for j in inds_nan:
            inds_els, weights_els = [], []
            for k in range(n):
                if not np.isnan(yy[k,j]):
                    inds_els += [k]
                    weights_els += [weights[k]]
            if len(inds_els) > 0:
                inds_els = np.array(inds_els)
                weights_els = np.array(weights_els).reshape(-1,1)
                weights_els = weights_els / weights_els.sum()
                ym[j] = np.sum(yy[inds_els,j]*weights_els)
            else:
                ym[j] = np.nan
            for l in range(n):
                yy[l,j] = ym[j]
    if clean:
        cond = np.zeros(N, bool)
        for j in range(n):
            for k in list(np.arange(n)[np.arange(n) > j]):
                diff = abs(yy[j] - yy[k])
                cond1 = diff > abs_threshold * max(noises[j], noises[k])
                cond2 = diff > (rel_threshold
                                * np.median([abs(yy[j]), abs(yy[k])], axis=0))
                cond += (cond1 * cond2)
        cond = enlarge_mask(cond, iters=margin)
        if cond.sum() > 0 and (~cond).sum() > 0:
            ym[cond] = np.interp(x[cond], x[~cond], ym[~cond])
        if len(ym) > size:
            ys = rolling_function(np.median, ym, size=size)
            ys = rolling_function(np.mean, ys, size=max(3,size//6))
            ym[cond] = ys[cond]
        windows = get_windows(x, cond, margin=0, width=0)
    else:
        windows = []
    return ym, windows

def combine_two_spectra(x1, y1, x2, y2, w1=1, w2=1, clean=False, r1=0,
                        r2=0, abs_threshold=5., rel_threshold=0.1, size=40,
                        margin=3):
    """
    Combine two spectra which overlap at some region, with different options.

    Parameters
    ----------
    x1 : array
        Independent variable of the first spectrum.
    y1 : array
        Dependent variable of the first spectrum.
    x2 : array
        Independent variable of the second spectrum.
    y2 : array
        Dependent variable of the second spectrum.
    w1 : float, optional
        Weight for the first spectrum. The default is 1.
    w2 : float, optional
        Weight for the second spectrum. The default is 1.
    clean : bool, optional
        If True, perform a cleaning and a interpolation in the regions where
        the values of the spectra differ too much, using the given reference
        noises.
    r1 : float, optional
        Reference noise for the first spectrum.
    r2 : float, optional
        Reference noise for the second spectrum
    abs_threshold : float, optional
        Value for identifying channels with differences greater than this number
        of times the reference noise. The default is 5.
    rel_threshold : float, optional
        Value for identifying channels with relative differences greater than
        this number. The default is 0.1
    size : int, optional
        Size of the median filter applied when doing the interpolation for the
        cleaning.

    Raises
    ------
    Exception
        It raises if clean == True and you don't specify the noises.

    Returns
    -------
    xm : array
        Averaged independent variable.
    ym : array
        Averaged dependent variable.
    windows : list
        If clean == True, windows is the list of windows that define the
        regions cleaned.
    """
    noises = [r1, r2]
    w = w1 + w2
    weights = [w1/w, w2/w]
    xc_i = max(x1[0], x2[0])
    xc_f = min(x1[-1], x2[-1])
    if xc_i >= xc_f:
        raise Exception("The two spectra must overlap.")     
    xx = np.concatenate((x1, x2))
    yy = np.concatenate((y1, y2))
    ids = np.concatenate((1*np.ones(len(x1), int), 2*np.ones(len(x2), int)))
    inds = np.argsort(xx)
    xx = xx[inds]
    yy = yy[inds]
    ids = ids[inds]
    cond = (xx >= xc_i) * (xx <= xc_f)
    all_inds = np.arange(len(xx))
    i1 = all_inds[cond][0]
    i2 = all_inds[cond][-1]
    if i1 != 0:
        if abs(xx[i1-1] - xx[i1]) < abs(xx[i1+1] - xx[i1]):
            i1 = i1 - 1
    if i2 != len(xx)-1:
        if abs(xx[i2+1] - xx[i2]) < abs(xx[i2-1] - xx[i2]):
            i2 = i2 + 1
    x_l = xx[:i1]
    y_l = yy[:i1]
    x_r = xx[i2:]
    y_r = yy[i2:]
    x1_c = xx[i1:i2][ids[i1:i2] == 1]
    x2_c = xx[i1:i2][ids[i1:i2] == 2]
    y1_c = yy[i1:i2][ids[i1:i2] == 1]
    y2_c = yy[i1:i2][ids[i1:i2] == 2]
    if len(x1_c) <= len(x2_c):
        xm_c = x1_c
        y_c1 = y1_c
        y_c2 = np.interp(xm_c, x2_c, y2_c)
    else:
        xm_c = x2_c
        y_c2 = y2_c
        y_c1 = np.interp(xm_c, x1_c, y1_c)
    yy_c = [y_c1, y_c2]
    ym_c, windows = \
        combine_equivalent_spectra(xm_c, yy_c, weights=weights, clean=clean,
                                   noises=noises, abs_threshold=abs_threshold,
                                   rel_threshold=rel_threshold, size=size,
                                   margin=margin)
    xm = np.concatenate((x_l, xm_c, x_r))
    ym = np.concatenate((y_l, ym_c, y_r))
    return xm, ym, windows

def combine_spectra(xx, yy, weights=None, clean=False, noises=None,
                    abs_threshold=5., rel_threshold=0.1, size=40, margin=3):
    """
    Combine the given input spectra, which must overlap at some region.

    Parameters
    ----------
    xx : list (array)
        List of the independent variables of the input spectra. They must
        overlap at least in pairs, with the given order.
    yy : list (array)
        List of the dependent variables of the input spectra.
    weights : list (float), optional
        Weights for averaging the given spectra. By default, the weights are
        equal.
    clean : bool, optional
        If True, perform a cleaning and a interpolation in the regions where
        the values of the spectra differ too much, using the given reference
        noises.
    noises : list, optional
        List of the reference noises for each spectrum.
    abs_threshold : float, optional
        Value for identifying channels with differences greater than this number
        of times the reference noise. The default is 5.
    rel_threshold : float, optional
        Value for identifying channels with relative differences greater than
        this number. The default is 0.1
    size : int, optional
        Size of the median filter applied when doing the interpolation for the
        cleaning.

    Raises
    ------
    Exception
        It raises if clean == True and you don't specify the noises.

    Returns
    -------
    xm : array
        Averaged independent variable.
    ym : array
        Averaged dependent variable.
    windows : list
        If clean == True, windows is the list of windows that define the
        regions cleaned.
    """
    if type(noises) == type(None):
        noises =  [np.nan for x in xx]
    if type(weights) == type(None):
        weights = np.ones(len(xx))
    weights = np.array(weights)
    xm = copy.copy(xx[0])
    ym = copy.copy(yy[0])
    noises_m = copy.copy(noises[0])
    weights_m = copy.copy(weights[0])
    windows = np.zeros((0,2))
    for i in range(1, len(xx)):
        xm, ym, wins_i = \
            combine_two_spectra(xm, ym, xx[i], yy[i], w1=weights_m, w2=weights[i],
                                clean=clean, r1=noises_m, r2=noises[i],
                                abs_threshold=abs_threshold,
                                rel_threshold=rel_threshold, size=size,
                                margin=margin)
        noises_m = 1 / np.sqrt(1/noises_m**2 + 1/noises[i]**2)
        weights_m = 1 / noises_m**2
        if len(wins_i) > 0:
            windows = np.concatenate((windows, np.array(wins_i)))
    return xm, ym, windows

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
    path = os.path.realpath(path)
    return path

def get_formatted_dictionary(dictionary, previous_name='', previous_index=0,
                             internal_use=False):
    """
    Convert the input compact spectra dictionary into the correct format.

    Parameters
    ----------
    dictionary : dict
        Dictionary of file names for making the average.
    previous_name : str, optional
        Variable used internally to create the sections name for the new
        dictionary.
    previous_index : int, optional
        Variable used internally to determine the level of nesting for the new
        dictionary.
    internal use : bool, optional
        Variable used for recursively calling the function internaly.

    Returns
    -------
    new_dict_list : list(dict)
        List that contains one dictionary with the desired format for each
        level of indention in the original dictionary.
    new_name : str
        Variable used internally to create the sections name for the new
        dictionary.
    """
    new_dict = {}
    final_names = []
    for name,item in zip(dictionary.keys(), dictionary.values()):
        new_name = previous_name + '-'*(previous_name != '') + name
        i = previous_index + 1
        if type(item) == dict:
            new_dict[str(i)+'-'+name] = []
            for item_name in dictionary[name]:
                new_dict[str(i)+'-'+name] += [item_name]
            result, new_name = \
                get_formatted_dictionary(item, new_name, i, internal_use=True)
            new_dict = {**result, **new_dict}
        elif type(item) == list:
            new_dict[str(i)+'-'+new_name] = item
        final_names += [name]
    if internal_use:
        return new_dict, new_name
    else:
        new_dict
        numbers = []
        for name in new_dict:
            i = int(name.split('-')[0]) - 1
            numbers += [i]
        numbers = set(numbers)
        new_dict_list = [{} for i in numbers]
        for name in new_dict:
            splitted_name = name.split('-')
            i = int(splitted_name[0]) - 1
            new_name = '-'.join(splitted_name[1:])
            new_dict_list[i][new_name] = new_dict[name]
        new_dict_list = list(reversed(new_dict_list))
        return new_dict_list, final_names

def remove_extra_spaces(input_text):
    """
    Remove extra spaces from a text string.

    Parameters
    ----------
    input_text : str
        Input text string.

    Returns
    -------
    text : str
        Resulting text.
    """
    text = input_text
    for i in range(15):
        text = text.replace('  ', ' ')
    if text.startswith(' '):
        text = text[1:]
    return text

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

def get_rms_noise(x, y, windows=[], sigmas=4., iters=3):
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
    iters : int, optional
        Number of iterations performed for the sigma clip applied to the data
        before the calculation of the RMS noise. The default is 3.

    Returns
    -------
    rms_noise : float
        Value of the RMS noise of the data.
    """
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
        The frequency windows of the regions that should be ignored.
        The default is [].
    rms_threshold : float, optional
        Maximum relative difference that can exist between the RMS noise of the
        searched region and the reference RMS noise. The default is 0.1.
    offset_threshold : float, optional
        Maximum value, in units of the reference RMS noise, that the mean value
        of the independent variable can have in the searched region.
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
               and abs(offset) < offset_threshold * rms_noise):
        width = max(min_width, args.smooth)
        resolution = np.median(np.diff(x))
        central_freq = np.random.uniform(x[0] + width/2*resolution,
                                         x[-1] - width/2*resolution)
        region_inf = central_freq - width/2*resolution
        region_sup = central_freq + width/2*resolution
        cond = (x > region_inf) & (x < region_sup)
        y = y[cond]
        valid_range = True
        for x1, x2 in windows:
            if (region_inf < x1 < region_sup) or (region_inf < x2 < region_sup):
                valid_range = False
        if valid_range:
            local_rms = 1e3 * float(np.sqrt(np.mean(y**2)))
            offset = 1e3 * np.mean(y)
        i += 1
        if i > max_iters:
            rms_region = []
            break
        rms_region = [float(central_freq - width/2*resolution),
                      float(central_freq + width/2*resolution)]
    return rms_region

#%%

# Start.
time1 = time.time()

# Folder separator.
separator = '/'
operating_system = platform.system()
if operating_system == 'Windows':
    separator = '\\'

# Arguments.
parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--plots', action='store_true')
args = parser.parse_args()

# Default options.
default_options = {
    'spectra folder': 'exported',
    'output folder': 'output',
    'plots folder': 'plots',
    'class extension': '',
    'rms noise units': 'mK',
    'intensity limits (K)': [],
    'ghost lines': {
        'clean lines': False,
        'absolute intensity threshold (rms)': 10.,
        'relative intensity threshold': 0.1,
        'smoothing factor': 40,
        'margin': 1,
        },
    'extra note': '',
    'files': {},
    'frequency units': 'MHz',
    }

# Reading of the configuration file.
config_path = full_path(args.config)
if os.path.isfile(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')
options = {**default_options, **config}
config_folder = full_path(separator.join(args.config.split(separator)[:-1]))
os.chdir(config_folder)

# Reading of the variables.
spectra_folder = options['spectra folder']
output_folder = options['output folder']
plots_folder = options['plots folder']
extra_note = options['extra note']
ext = options['class extension']
intensity_limits = options['intensity limits (K)']
if 'ghost lines' in config:
    options['ghost lines'] = {**default_options['ghost lines'],
                              **config['ghost lines']}
clean = options['ghost lines']['clean lines']
abs_threshold = options['ghost lines']['absolute intensity threshold (rms)']
rel_threshold = options['ghost lines']['relative intensity threshold']
lines_margin = options['ghost lines']['margin']
size = options['ghost lines']['smoothing factor']
files = options['files']
rms_units = options['rms noise units']
frequency_units = options['frequency units']
if frequency_units == 'MHz':
      to_MHz = 1.
elif frequency_units == 'GHz':
      to_MHz = 1000.
if rms_units == 'mK':
    to_K = 0.001
elif rms_units == 'K':
    to_K = 1
all_subfolder = 'all' + separator

# Checks.
if not spectra_folder.endswith(separator):
    spectra_folder += separator
if not output_folder.endswith(separator):
    output_folder += separator
if not plots_folder.endswith(separator):
    plots_folder += separator
if spectra_folder.startswith('.'):
    spectra_folder.replace('.','')
if output_folder.startswith('.'):
    output_folder.replace('.','')
if plots_folder.startswith('.'):
    plots_folder.replace('.','')
if not os.path.isdir(full_path(plots_folder)):
    os.mkdir(full_path(plots_folder))
if not os.path.isdir(full_path(output_folder + all_subfolder)):
    os.mkdir(full_path(output_folder + all_subfolder))

skip_plots = False

with open(spectra_folder + 'rms_noises.yaml') as file:
    rms_noises = yaml.safe_load(file)
with open(spectra_folder + 'doppler_corrections.yaml') as file:
    doppler_corr = yaml.safe_load(file)
with open(spectra_folder + 'frequency_windows.yaml') as file:
    frequency_windows = yaml.safe_load(file)

for ext_ in [ext, '.dat', '.fits']:
    prev_files = Path(output_folder + all_subfolder).glob('**/*-c*'+ext_)
    prev_files = [str(pp) for pp in prev_files]
    for file in prev_files:
        os.remove(file)

#%% Calculations.

files_list, final_names = get_formatted_dictionary(files)

# Class file for doing the coresponding averages.
all_spectra = []
doppler_text = []

for files in files_list:
    
    script = []
    
    for title in files:
        script += ['file out {}{}{}-c-temp{} m /overwrite'
                  .format(output_folder, all_subfolder, title, ext)]
        title_input = re.sub(r'\([^()]*\)', '', title).replace('--','-')
        if title_input.endswith('-'):
            title_input = title_input[:-1]
        for element in files[title]:
            input_spectrum = '{}-{}-c{}'.format(title_input, element, ext)
            if len(extra_note) > 0 and input_spectrum.count('-' + extra_note) > 1:
                input_spectrum = input_spectrum.replace(extra_note + '-', '')
            if not os.path.isfile(output_folder + all_subfolder + input_spectrum):
                input_spectrum = input_spectrum.replace('-c'+ext, ext)
            script += ['file in {}{}{}'
                       .format(output_folder, all_subfolder, input_spectrum)]
            script += ['find /all', 'get first', 'write']
            file_path = '{}/{}-{}-c.fits'.format(spectra_folder, title_input,
                                                 element)
            if len(extra_note) > 0 and file_path.count('-' + extra_note) > 1:
                file_path = file_path.replace(extra_note + '-', '')
            if os.path.isfile(file_path):
                os.remove(file_path)
            script += ['fits write {} /mode spectrum'.format(file_path)]
        script += ['file out {}{}{}-c{} m /overwrite'
                   .format(output_folder, all_subfolder, title, ext)]
        script += ['file in {}{}{}-c-temp{}'
                   .format(output_folder, all_subfolder, title, ext)]
        script += ['find /all', 'stitch', 'write']
        script += ['modify doppler', 'modify doppler *']
        all_spectra += [title]
        file_path = '{}/{}-c.fits'.format(spectra_folder, title)
        if os.path.isfile(file_path):
            os.remove(file_path)
        script += ['fits write {} /mode spectrum'.format(file_path)]
        # script += ['greg {}/all/{}-c.dat /formatted'.format(output_folder, title)]
        
    script += ['exit']
    for i in range(len(script) - 1):
        script[i] += '\n'
    with open('combine.class', 'w') as file:
        file.writelines(script)
        
    p = subprocess.Popen(['class', '@combine.class'],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    prev_line = ''
    for text_line in p.stdout:
        text_line = text_line.decode('utf-8').replace('\n','')
        print(text_line)
        if 'Doppler factor' in text_line and '***' not in text_line:
            if (not 'Doppler factor' in prev_line
                    or 'Doppler factor' in prev_line and '***' in prev_line): 
                doppler_text += [remove_extra_spaces(text_line).split(' ')[-1]]
        if 'I-OBSERVATORY' not in text_line:
            prev_line = text_line

doppler_corr = {}
if len(doppler_text) == len(all_spectra):
    for i in range(len(doppler_text)):
        spectrum = str(all_spectra[i])
        doppler_corr[spectrum] = doppler_text[i]
else:
    print(all_spectra)
    raise Exception('Error reading Doppler corrections.')

save_yaml_dict(doppler_corr, './{}doppler_corrections.yaml'.format(spectra_folder),
               default_flow_style=False)
print('\nSaved Doppler corrections in {}doppler_corrections.yaml.'
      .format(spectra_folder))
print()

temp_files = Path(output_folder + all_subfolder).glob('**/*temp'+ext)
temp_files = [str(pp) for pp in temp_files]
for file in temp_files:
    os.remove(file)           

prev_files = Path(output_folder).glob('**/*{}-r-c{}'.format(extra_note, ext))
prev_files = [str(pp) for pp in prev_files]
for file in prev_files:
    os.remove(file)
            
for files in files_list: 
    
    script = []
    
    for title in files:
        
        discard_cleaning = False
        title_input = re.sub(r'\([^()]*\)', '', title).replace('--','-')
        if title_input.endswith('-'):
            title_input = title_input[:-1]
        output_spectrum = '{}-r-c'.format(title)
        xx, yy, noises, weights, windows_groups, obs_times, beam_effs = \
            [], [], [], [], [], [], []
            
        num_spectra = len(files[title])    
        plt.figure(1, figsize=(14,8))
        plt.clf()
        
        for i,element in enumerate(files[title]):
            
            input_spectrum = '{}-{}-r-c'.format(title_input, element)
            if len(extra_note) > 0 and input_spectrum.count('-' + extra_note) > 1:
                input_spectrum = input_spectrum.replace(extra_note + '-', '')
            # Loading of the data files.
            file_path = spectra_folder + input_spectrum
            if not os.path.isfile(file_path + '.dat'):
                file_path = file_path.replace('-r-c', '-r')
            data = np.loadtxt(file_path + '.dat')
            frequency = data[:,0] * to_MHz
            intensity = data[:,1]
            hdul = fits.open(file_path + '.fits')
            fits_data = hdul[0].data
            obs_time = hdul[0].header['obstime']
            beam_eff = hdul[0].header['beameff']
            input_spectrum = input_spectrum.replace('-r-c','').replace('-r','')
            noise = rms_noises[input_spectrum] * to_K
            windows = frequency_windows[input_spectrum]
            # Storing of the data.
            xx += [frequency]
            yy += [intensity]
            noises += [noise]
            weights += [1/noise**2]
            windows_groups += [windows]
            obs_times += [obs_time]
            beam_effs += [beam_eff]
            # Plot.
            color = hsv_to_rgb((i/len(files[title]), 1, 0.8))
            alpha = 0.1 + 0.9/len(files[title])
            plt.step(frequency, intensity, where='mid', color=color,
                     label=input_spectrum, alpha=alpha)
    
        if len(files[title]) > 0:
            
            final_frequency, final_intensity, windows = \
                combine_spectra(xx, yy, weights=weights,
                                clean=clean, noises=noises,
                                rel_threshold=rel_threshold,
                                abs_threshold=abs_threshold,
                                size=size, margin=lines_margin)
    
            # Plot.
            plt.figure(1)
            plt.step(final_frequency, final_intensity, where='mid', color='black',
                     alpha=1, label=output_spectrum.replace('-r-c',''))
            for w1, w2 in windows:
                plt.axvspan(w1, w2, color='gray', alpha=0.5)
            plt.margins(x=0)
            plt.xlabel('frequency (MHz)')
            plt.ylabel('reduced intensity (K)')
            if len(intensity_limits) > 0:
                plt.ylim(intensity_limits)
            plt.title(output_spectrum.replace('-r-c',''), fontweight='bold')
            fontsize = 12 - 4*min(1, num_spectra/20)
            ncol = max(1, num_spectra//8)
            plt.legend(fontsize=fontsize, ncol=ncol)
            plt.tight_layout()
            plt.savefig(plots_folder + 'combined-'
                        + output_spectrum.replace('.dat', '.png'), dpi=200)
            if args.plots and not skip_plots:
                plt.pause(0.1)
                plt.show(block=False)
                print(output_spectrum)
                input_text = \
                    input('Options: Next (Enter), Discard (x), Skip (s). ')
                if input_text == 's':
                    skip_plots = True
                elif input_text == 'x':
                    discard_cleaning = True
            if discard_cleaning:
                final_frequency, final_intensity, _ = \
                    combine_spectra(xx, yy, xm=final_frequency, weights=weights,
                                    clean=False)
            # New noise.
            new_windows = []
            for windows in windows_groups:
                for x1,x2 in windows:
                    new_windows += [[float(x1), float(x2)]]
            reference_width = np.median(np.diff(new_windows, axis=1))
            new_noise = get_rms_noise(final_frequency, final_intensity,
                                      windows=new_windows)
            rms_noises[output_spectrum.replace('-r-c','')] = float(1e3*new_noise)
            frequency_windows[output_spectrum.replace('-r-c','')] = new_windows
     
            # Output.
            file_path = spectra_folder + \
                (output_spectrum + '.fits').replace('-r-c.fits', '-c.fits')
            hdul = fits.open(file_path)
            fits_data = np.float32(np.zeros((1,1,1,len(final_intensity))))
            fits_data[0,0,0,:] = np.float32(final_intensity)
            hdul[0].data = fits_data
            hdul[0].scale('float32')
            hdul[0].header['obstime'] = float(np.sum(obs_times))
            rest_freq = hdul[0].header['restfreq']
            initial_freq = final_frequency[0] * to_MHz * 1e6
            resolution = hdul[0].header['cdelt1']
            ind_ref = 1 + (rest_freq - initial_freq) / resolution
            show_diagnostic_plot = False
            if show_diagnostic_plot:
                plt.figure(7)
                plt.plot(final_frequency*to_MHz*1e6,
                         1 + np.arange(len(final_frequency)))
                plt.axvline(rest_freq)
                plt.axhline(hdul[0].header['crpix1'])
                plt.axhline(ind_ref, color='red')
                plt.xlabel('frequency (Hz)')
                plt.ylabel('channel number')
                plt.show()
                input()
            hdul[0].header['crpix1'] = ind_ref
            hdul[0].header['beameff'] = float(np.mean(beam_effs))
            if 'blank' in hdul[0].header:
                del hdul[0].header['blank']
            if '(' in title:
                lines_tels = title.split('(')[1].split(')')[0]
                lines = []
                for line in lines_tels.split('+'):
                    lines += [line.split('-')[0]]
                lines = '+'.join(lines)
                
                max_chars = 10
                if len(lines) > 0:
                    hdul[0].header['line'] = lines
            hdul.writeto(spectra_folder + output_spectrum + '.fits',
                         overwrite=True)
            hdul.close()
            print('Saved reduced spectrum in {}.fits.'.format(output_spectrum))
            script += ['fits read {}.fits'.format(spectra_folder
                                                  + output_spectrum)]
            script += ['modify doppler {}'
                    .format(doppler_corr[output_spectrum.replace('-r-c','')])]
            script += ['greg {}{}.dat /formatted'.format(spectra_folder,
                                                         output_spectrum)]
    
    script += ['exit']
    for i in range(len(script) - 1):
        script[i] += '\n'
    with open('combine.class', 'w') as file:
        file.writelines(script)
    subprocess.run(['class', '@combine.class'])
    print()

# Export of the rms noise of each spectrum.
save_yaml_dict(rms_noises, spectra_folder + 'rms_noises.yaml',
                  default_flow_style=None)
print('Saved RMS noises in {}rms_noises.yaml.'.format(spectra_folder))

# Export of the frequency windows of each spectrum.
save_yaml_dict(frequency_windows, spectra_folder + 'frequency_windows.yaml',
                  default_flow_style=None)
print('Saved frequency windows in {}frequency_windows.yaml.'.format(spectra_folder))

# Creation of a class script for generating the final class files.
if len(ext) > 0 and len(final_names) > 0:
    print()
    script = []
    for name in final_names:
        class_file = name + '-r-c' + ext
        script += ['file out {}{} m /overwrite'.format(output_folder, class_file)]
        input_spectrum = '{}-r-c.fits'.format(name)
        script += ['fits read {}{}'.format(spectra_folder, input_spectrum)]
        script += ['modify doppler {}'.format(doppler_corr[name])]
        script += ['write']
    script += ['exit']
    for i in range(len(script) - 1):
        script[i] += '\n'
    with open('combine.class', 'w') as file:
        file.writelines(script)
    # Running of the class file.
    subprocess.run(['class', '@combine.class'])
    print()
    for name in final_names:
        print('Saved final spectrum in {}{}.'.format(output_folder, class_file))
    print()

# Removing backuo files.
backup_files_1 = Path(spectra_folder).glob('**/*.dat~')
backup_files_1 = [str(pp) for pp in backup_files_1]
backup_files_2 = Path(output_folder + all_subfolder).glob('**/*.dat~')
backup_files_2 = [str(pp) for pp in backup_files_2]
backup_files = backup_files_1 + backup_files_2
for file in backup_files:
    os.remove(file)

# Elapsed time.
if not config_path.endswith('config-combine-auto.yaml'):
    time2 = time.time()
    total_time = int(time2 - time1)
    minutes, seconds = total_time//60, total_time%60
    text = ('Elapsed time: {} min + {} s.\n'
            .format(minutes, seconds))
    if minutes == 0:
        text = text.replace('0 min + ', '')
        if seconds == 0:
            text = text.replace('0 s', '< 1 s')
    print(text)
