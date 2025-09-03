#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated GILDAS-CLASS Pipeline
-------------------------------
Averaging mode
Version 1.4

Copyright (C) 2025 - Andrés Megías Toledano

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
from matplotlib.ticker import ScalarFormatter

# Custom functions.

def rolling_function(func, x, size, **kwargs):
    """
    Apply a function in a rolling way, in windows of the specified size.

    Parameters
    ----------
    x : array
        Input data.
    func : function
        Function to be applied.
    size : int
        Size of the windows to group the data. It must be odd.
    **kwargs : (various)
        Keyword arguments of the function to be applied.

    Returns
    -------
    y : array
        Resultant array.
    """
    
    def rolling_window(x, window):
        """
        Group the input data according to the specified window size.
        
        Function by Erik Rigtorp.

        Parameters
        ----------
        x : array
            Input data.
        window : int
            Size of the windows to group the data.

        Returns
        -------
        y : array
            Output array.
        """
        shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
        strides = x.strides + (x.strides[-1],)
        y = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return y
    
    N = len(x)
    if size <= 0:
        raise Exception('Window size must be positive.')
    size = int(size)
    if size == 1 or N == 0:
        return x
    size = min(N, size)
    N = len(x)
    y_c = func(rolling_window(x, size), -1, **kwargs)
    M = min(N, size) // 2
    min_size = 1
    y_1, y_2 = np.zeros(M), np.zeros(M)
    for i in range(M):
        j1 = 0
        j2 = max(min_size, 2*i)
        y_1[i] = func(x[j1:j2], **kwargs)
        j1 = N - max(min_size, 2*i)
        j2 = N
        y_2[-i-1] = func(x[j1:j2], **kwargs)
    y = np.concatenate((y_1, y_c, y_2))
    if size % 2 == 0:
        y = y[1:]/2 + y[:-1]/2
    
    return y

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

def average_equivalent_spectra(x, yy, weights=[1.,1.], clean=False, noises=0,
                       abs_threshold=5., rel_threshold=0.1, size=40, margin=3):
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

def average_two_spectra(x1, y1, x2, y2, w1=1, w2=1, clean=False, r1=0, r2=0,
                       abs_threshold=5., rel_threshold=0.1, size=40, margin=3):
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
    ym_c, windows = average_equivalent_spectra(xm_c, yy_c, weights=weights,
                       clean=clean, noises=noises, abs_threshold=abs_threshold,
                       rel_threshold=rel_threshold, size=size, margin=margin)
    xm = np.concatenate((x_l, xm_c, x_r))
    ym = np.concatenate((y_l, ym_c, y_r))
    return xm, ym, windows

def average_spectra(xx, yy, weights=None, clean=False, noises=None,
                    abs_threshold=5., rel_threshold=0.1, size=40, margin=3):
    """
    Average the given input spectra, which must overlap at some region.

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
            average_two_spectra(xm, ym, xx[i], yy[i], w1=weights_m, w2=weights[i],
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
    internal_use : bool, optional
        Variable used for recursively calling the function internaly.

    Returns
    -------
    new_dict_list : list(dict)
        List that contains one dictionary with the desired format for each
        level of indentation in the original dictionary.
    final_names : list
        List of correct final file names.
    """
    new_dict = {}
    final_names = []
    prev_first_alt = ''
    for (name,item) in zip(dictionary.keys(), dictionary.values()):
        previous_name = previous_name
        new_name = (previous_name + '-'*(previous_name != '') + name)
        first = new_name.split('-')[0]
        first_alt = '-'.join(name.split('-')[:-1])
        new_name = (new_name.replace(first+'-'+first, first)
                    .replace(first_alt+'-'+first_alt, first_alt))
        if prev_first_alt != '':
            new_name = new_name.replace(prev_first_alt+'-'+prev_first_alt,
                                        prev_first_alt)
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
        prev_first_alt = first_alt
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
    'use compact file names': True,
    'spectra folder': 'exported',
    'output folder': 'output',
    'plots folder': 'plots',
    'class extension': '',
    'rms noise units': 'mK',
    'intensity limits (K)': [],
    'sources-lines-telescopes': {},
    'default telescopes': [],
    'ghost lines': {
        'clean lines': False,
        'absolute intensity threshold (rms)': 10.,
        'relative intensity threshold': 0.1,
        'smoothing factor': 40,
        'margin': 1,
        },
    'extra note': '',
    'averaged spectra': {},
    'combined spectra': {},
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
abs_threshold = float(options['ghost lines']['absolute intensity threshold (rms)'])
rel_threshold = float(options['ghost lines']['relative intensity threshold'])
lines_margin = int(options['ghost lines']['margin'])
size = int(options['ghost lines']['smoothing factor'])
sources_lines_telescopes = options['sources-lines-telescopes']
default_telescopes = options['default telescopes']
averaged_files = options['averaged spectra']
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
for source in sources_lines_telescopes:
    for line in sources_lines_telescopes[source]:
        if sources_lines_telescopes[source][line] == 'default':
            sources_lines_telescopes[source][line] = default_telescopes
linewidth = 0.1
yticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
          1., 2., 3., 4., 5., 6., 7., 8., 9.,
          0, 10., 100., 1000., 10000.]
yticklabels = [0.01, '', '', '', 0.05, '', '', '', '',
               0.1, '', '', '', '', '', '', '', '',
               1, '', '', '', '', '', '', '', '',
              '0', 10, 100, 1000, 10000]
yticks += [-y for y in yticks]
for y in copy.copy(yticklabels):
    if y not in ('', '0'):
        y = '$-$'+str(y)
    yticklabels += [y]

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
with open(spectra_folder + 'rms_regions.yaml') as file:
    rms_regions = yaml.safe_load(file)
with open(spectra_folder + 'doppler_corrections.yaml') as file:
    doppler_corr = yaml.safe_load(file)
with open(spectra_folder + 'frequency_windows.yaml') as file:
    frequency_windows = yaml.safe_load(file)
with open(spectra_folder + 'frequency_ranges.yaml') as file:
    frequency_ranges = yaml.safe_load(file)
with open(spectra_folder + 'reference_frequencies.yaml') as file:
    reference_frequencies = yaml.safe_load(file)

# for ext_ in [ext, '.dat', '.fits']:
#     prev_files = Path(output_folder + all_subfolder).glob('**/*-a*'+ext_)
#     prev_files = [str(pp) for pp in prev_files]
#     for file in prev_files:
#         os.remove(file)

#%% Calculations. Averaging and grouping into same CLASS files.

files_list, final_names = get_formatted_dictionary(averaged_files)
for (i, name) in enumerate(final_names):
    if extra_note != '' and  '-'+extra_note not in name:
        final_names[i] = name + '-'+extra_note

# CLASS file for doing the averages without reducing, to get the file info.
all_spectra = []
doppler_text = []

for files in files_list:
    
    script = []
    
    for title in files:
            
        telescopes = ['']
        for source in sources_lines_telescopes:
            for line in sources_lines_telescopes[source]:
                if (source in title and (line in title or
                        any([line in element for element in files[title]]))):
                    telescopes = sources_lines_telescopes[source][line]
                    break
        if telescopes == 'default':
            telescopes = default_telescopes
            
        for telescope in telescopes:
            telescope = telescope.replace('*', '')
            if telescope != '':
                telescope = '-' + telescope
            output_spectrum = title + telescope
            if extra_note != '' and '-'+extra_note not in output_spectrum:
                output_spectrum += '-'+extra_note
            script += ['file out {}{}{}-a-temp{} m /overwrite'
                      .format(output_folder, all_subfolder, output_spectrum, ext)]
            for element in files[title]:
                input_spectrum = ('{}-{}{}'.format(title, element, telescope)
                                  .replace(title+'-'+title, title))
                if len(extra_note) > 0 and input_spectrum.count('-'+extra_note) > 1:
                    input_spectrum = input_spectrum.replace(extra_note+'-', '')
                input_path = output_folder + all_subfolder + input_spectrum + ext
                if not os.path.exists(input_path):
                    input_path = input_path.replace(ext, '-a'+ext)
                script += ['file in ' + input_path]
                script += ['find /all', 'get first', 'write']
            script += ['file out {}{}{}-a{} m /overwrite'
                       .format(output_folder, all_subfolder, output_spectrum, ext)]
            script += ['file in {}{}{}-a-temp{}'
                       .format(output_folder, all_subfolder, output_spectrum, ext)]
            script += ['find /all', 'stitch', 'write']
            script += ['modify doppler', 'modify doppler *']
            all_spectra += [output_spectrum]
            output_path = spectra_folder + output_spectrum + '-a.fits'
            if os.path.isfile(output_path):
                os.remove(output_path)
            script += ['fits write {} /mode spectrum'.format(output_path)]
            # script += ['greg {}/all/{}.dat /formatted'.format(output_folder, title)]
        
    script += ['exit']
    script = '\n'.join(script) + '\n'
    with open('averaging.class', 'w') as file:
        file.writelines(script)
        
    p = subprocess.Popen(['class', '@averaging.class'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

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
plt.figure(1, figsize=(14.,8.))


# Averaging of the spectra and grouping into CLASS files.
            
for files in files_list: 
    
    script = []
    
    for (j,title) in enumerate(files):
        
        telescopes = ['']
        for source in sources_lines_telescopes:
            for line in sources_lines_telescopes[source]:
                if (source in title and (line in title or
                        any([line in element for element in files[title]]))):
                    telescopes = sources_lines_telescopes[source][line]
                    break
        if telescopes == 'default':
            telescopes = default_telescopes
        
        discard_cleaning = False
        if title.endswith('-'):
            title = title[:-1]
            
        for telescope in telescopes:
            
            telescope = telescope.replace('*', '')
            if telescope != '':
                telescope = '-' + telescope
                
            output_spectrum = title + telescope
            if extra_note != '' and '-'+extra_note not in output_spectrum:
                output_spectrum += '-'+extra_note
            xx, yy, noises, weights, windows_groups, obs_times, beam_effs = \
                [], [], [], [], [], [], []
            mads = []
                
            num_spectra = len(files[title])    
            plt.figure(1)
            plt.clf()
            
            for (i,element) in enumerate(files[title]):
                
                input_spectrum = ('{}-{}{}-r'.format(title, element, telescope)
                                  .replace(title+'-'+title, title))
                if (len(extra_note) > 0
                        and input_spectrum.count('-'+extra_note) > 1):
                    input_spectrum = input_spectrum.replace(extra_note+'-', '')
                if not os.path.exists(spectra_folder + input_spectrum + '.dat'):
                    input_spectrum += '-a'
                # Loading of the data files.
                file_path = spectra_folder + input_spectrum
                data = np.loadtxt(file_path + '.dat')
                frequency = data[:,0] * to_MHz
                intensity = data[:,1]
                intensity[intensity < -100.] = 0.
                hdul = fits.open(file_path + '.fits')
                fits_data = hdul[0].data
                obs_time = hdul[0].header['obstime']
                beam_eff = hdul[0].header['beameff']
                input_spectrum = input_spectrum.replace('-r','').replace('-a', '')
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
                alpha = 0.3 + 0.7/len(files[title])
                plt.step(frequency, intensity, where='mid', color=color,
                         alpha=alpha, lw=linewidth)
                plt.step([], [], where='mid', color=color, alpha=alpha,
                         label=input_spectrum)
                # if len(intensity_limits) == 0:
                    # mad = median_abs_deviation(intensity, scale='normal')
                    # mads += [mad]
        
            if len(files[title]) > 0:
                
                final_frequency, final_intensity, windows = average_spectra(xx, yy,
                        weights=weights, clean=clean, noises=noises,
                        rel_threshold=rel_threshold, abs_threshold=abs_threshold,
                        size=size, margin=lines_margin)
        
                # Plot.
                plt.figure(1)
                plt.step(final_frequency, final_intensity, where='mid',
                         color='black', alpha=1, lw=linewidth)
                plt.step([], [], where='mid', color='black', label=output_spectrum)
                for (w1,w2) in windows:
                    plt.axvspan(w1, w2, color='gray', alpha=0.5)
                plt.margins(x=0)
                plt.locator_params(axis='both', nbins=12)
                plt.xlabel('frequency (MHz)')
                plt.ylabel('reduced intensity (K)')
                if len(intensity_limits) > 0:
                    plt.ylim(intensity_limits)
                else:
                    plt.yscale('symlog', linthresh=0.1)
                    # plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
                    # plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                y1, y2 = plt.ylim()
                yticks_ = copy.copy(yticks)
                yticklabels_ = copy.copy(yticklabels)
                i = 0
                for y in yticks:
                    if y < y1 or y > y2 or (y2-y1) > 20. and abs(y) == 0.01:
                        del yticks_[i]
                        del yticklabels_[i]
                    else:
                        i += 1
                plt.yticks(yticks_, yticklabels_)
                plt.title(output_spectrum, fontweight='bold', pad=10.)
                fontsize = 12. - 4.*min(1, num_spectra/20)
                ncol = max(1, num_spectra//8)
                plt.legend(fontsize=fontsize, ncol=ncol)
                plt.tight_layout()
                filename = plots_folder + 'averaged-' + output_spectrum + '.pdf'
                plt.savefig(filename)
                plt.plot(f'Saved plot in {filename}.')
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
                    final_frequency, final_intensity, _ = average_spectra(xx, yy,
                                  xm=final_frequency, weights=weights, clean=False)
                # New noise.
                new_windows = []
                for windows in windows_groups:
                    for (x1,x2) in windows:
                        new_windows += [[float(x1), float(x2)]]
                reference_width = np.median(np.diff(new_windows, axis=1))
                new_noise = get_rms_noise(final_frequency, final_intensity,
                                          windows=new_windows)
                new_freq_range = [float(final_frequency[0]), float(final_frequency[-1])]
                rms_noises[output_spectrum] = float(1e3*new_noise)
                rms_regions[output_spectrum] = rms_regions[input_spectrum]
                frequency_windows[output_spectrum] = new_windows
                frequency_ranges[output_spectrum] = new_freq_range
                reference_frequencies[output_spectrum] = \
                    reference_frequencies[input_spectrum]
         
                # Output.
                file_path = spectra_folder + output_spectrum + '-a.fits'
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
                    plt.figure(2)
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
                hdul.writeto(spectra_folder + output_spectrum + '-r-a.fits',
                             overwrite=True)
                hdul.close()
                print('Saved reduced spectrum in {}-r-a.fits.'.format(output_spectrum))
                script += ['fits read {}-r-a.fits'
                           .format(spectra_folder + output_spectrum)]
                script += ['modify doppler {}'.format(doppler_corr[output_spectrum])]
                script += ['greg {}{}-r-a.dat /formatted'
                           .format(spectra_folder, output_spectrum)]
                
    script += ['exit']
    script = '\n'.join(script) + '\n'
    with open('averaging.class', 'w') as file:
        file.writelines(script)
    subprocess.run(['class', '@averaging.class'])
    print()

# Export of the rms noise of each spectrum.
save_yaml_dict(rms_noises, spectra_folder + 'rms_noises.yaml',
               default_flow_style=None)
print('Saved RMS noises in '+spectra_folder+'rms_noises.yaml.')

# Export of the rms regions of each spectrum.
save_yaml_dict(rms_regions, spectra_folder + 'rms_regions.yaml',
               default_flow_style=None)
print('Saved RMS regions in '+spectra_folder+'rms_regions.yaml')

# Export of the frequency windows of each spectrum.
save_yaml_dict(frequency_windows, spectra_folder + 'frequency_windows.yaml',
               default_flow_style=None)
print('Saved frequency windows in '+spectra_folder+'frequency_windows.yaml.')

# Export of the frequency ranges of each spectrum.
save_yaml_dict(frequency_ranges, spectra_folder + 'frequency_ranges.yaml',
               default_flow_style=None)
print('Saved frequency windows in '+spectra_folder+'frequency_ranges.yaml.')

# Export of the reference frequencies of each spectrum.
save_yaml_dict(reference_frequencies, spectra_folder + 'reference_frequencies.yaml',
               default_flow_style=None)
print('Saved frequency windows in '+spectra_folder+'reference_frequencies.yaml.')

# Combining of the spectra into CLASS files including telescopes.

# Creation of a CLASS script to group the individual files.
final_spectra = {}
script = ['']
for name in final_names:
    class_file = name + '-r-a' + ext
    script += ['file out {}{} m /overwrite'.format(output_folder, class_file)]
    telescopes = ['']
    for source in sources_lines_telescopes:
        for line in sources_lines_telescopes[source]:
            if source in name and line in name:
                telescopes = sources_lines_telescopes[source][line]
                break
    if telescopes == ['']:
        for source in sources_lines_telescopes:
            if source in name:
                line = list(sources_lines_telescopes[source])[0]
                telescopes = sources_lines_telescopes[source][line]
                break
    if telescopes == 'default':
        telescopes = default_telescopes
    final_spectra[name] = []
    for telescope in telescopes:
        telescope = telescope.replace('*', '')
        input_spectrum = name
        if telescope != '':
            input_spectrum += '-'+telescope
        input_path = '{}{}-r.fits'.format(spectra_folder, input_spectrum)
        if not os.path.exists(input_path):
            input_path = input_path.replace('-r.fits', '-r-a.fits')
        script += ['fits read ' + input_path]
        script += ['modify doppler {}'.format(doppler_corr[input_spectrum])]
        script += ['write']
        final_spectra[name] += [input_spectrum]
script += ['exit']
script = '\n'.join(script) + '\n'
with open('averaging.class', 'w') as file:
    file.write(script)
# Running of the CLASS file.
subprocess.run(['class', '@averaging.class'])
print()

# Creation of a CLASS script to group the files by source.
script = ['']
for source in sources_lines_telescopes:
    output_path = '{}{}-all-r-a{}'.format(output_folder, source, ext)
    script += ['file out {} m /overwrite'.format(output_path)]
    files_source = [name+'-r-a'+ext for name in final_names if name.startswith(source)]
    for file in files_source:
        script += ['file in {}{}'.format(output_folder, file)]
        script += ['find /all']
        for spectrum in final_spectra[name]:
            script += ['get next', 'write']
script += ['exit']
script = '\n'.join(script) + '\n'
with open('averaging-grouping.class', 'w') as file:
    file.write(script)
# Running of the CLASS file.
subprocess.run(['class', '@averaging-grouping.class'])
print()

# Final files.
print('Created CLASS files:    (folder {})'.format(output_folder))
for name in final_names:
    class_file = name + '-r-a' + ext
    print('- ' + class_file)
for source in sources_lines_telescopes:
    files_source = [name+'-r-a'+ext
                    for name in final_names if name.startswith(source)]
    if len(files_source) > 1:
        for file in files_source:
            print('- ' + file)
print()

#%% Ending.

# Removing backup files.
backup_files_1 = Path(spectra_folder).glob('**/*.dat~')
backup_files_1 = [str(pp) for pp in backup_files_1]
backup_files_2 = Path(output_folder + all_subfolder).glob('**/*.dat~')
backup_files_2 = [str(pp) for pp in backup_files_2]
backup_files_3 = Path(output_folder + all_subfolder).glob('**/*-temp'+ext)
backup_files_3 = [str(pp) for pp in backup_files_3]
backup_files_4 = Path(output_folder + all_subfolder).glob('**/*-a'+ext)
backup_files_4 = [str(pp) for pp in backup_files_4]
backup_files = backup_files_1 + backup_files_2 + backup_files_3 + backup_files_4
for file in backup_files:
    os.remove(file)

# Elapsed time.
if not config_path.endswith('config-averaging-auto.yaml'):
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