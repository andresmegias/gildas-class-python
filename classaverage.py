#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated GILDAS-CLASS Pipeline
-------------------------------
Averaging mode
Version 1.2

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
import os
import yaml
import time
import platform
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import hsv_to_rgb

# Function for obtaining paths.
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
parser.add_argument('--plot', action='store_true')
parser.add_argument('--original', action='store_true')
args = parser.parse_args()

# Default options.
default_options = {
    'spectra folder': 'exported',
    'output folder': 'output',
    'plots folder': 'plots',
    'input files': {},
    'extra note': ''
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
input_files = options['input files']
extra_note = options['extra note']
output_files = []

# Checks.
if not spectra_folder.endswith(separator):
    spectra_folder += separator
if not output_folder.endswith(separator):
    output_folder += separator
if spectra_folder.startswith('.'):
    spectra_folder.replace('.','')
if output_folder.startswith('.'):
    output_folder.replace('.','')

#%% Calculations.

for input_file in input_files:

    if 'all spectra' in input_files[input_file]:
        all_spectra = input_files[input_file]['all spectra']
        source = input_file.split('-')[0]
        all_spectra = [source + '-' + spectrum for spectrum in all_spectra]
    else:
        raise Exception('No spectra selected for input file {}.'
                .format(input_file))
        
    if 'averaged spectra' in input_files[input_file]:
        averaged_spectra = input_files[input_file]['averaged spectra']
        for i, spectra_group in enumerate(averaged_spectra):
            for j, spectrum in enumerate(spectra_group):
                averaged_spectra[i][j] = source + '-' + spectrum
    else:
        if not args.plot:
            raise Exception('No average specified.')
        
    plt.figure(1, figsize=(12,6))
    plt.clf()
    plt.axhline(y=0, color='gray', linestyle='--')    
    
    for i, file in enumerate(all_spectra):
    
        if args.original:
            ext = '.dat'
        else:
            ext = '-r.dat'
        # Loading of the data files.
        data = np.loadtxt(full_path(spectra_folder + file + ext))
        frequency = data[:,0] / 1E3
        intensity = data[:,1]
        if np.sum(np.isnan(data)) != 0:
            raise Exception('Data of file {} is corrupted.'.format(file))
        color = hsv_to_rgb((i/len(all_spectra), 1, 0.8))
        plt.step(frequency, intensity, where='mid', label=file, color=color)
        x = frequency[0] + 0.2 * ((np.random.random()-0.5) * 
                                  (frequency[-1]-frequency[0]))
        y = 0.8*intensity.mean() + 0.2*intensity.max()
        plt.text(x, y, file, rotation=90, color=color)
    
    plt.legend(ncol=4, fontsize='small', loc='lower center')
    plt.margins(x=0)
    plt.ticklabel_format(style='sci', useOffset=False)
    plt.xlabel('frequency (GHz)')
    if args.original:
        plt.ylabel('intensity (K)')
    else:
        plt.ylabel('reduced intensity (K)')
    source = input_file.split('-')[0]
    plt.suptitle('All spectra - {}'.format(source),
                 fontweight='semibold')
    plt.tight_layout()
    
    os.chdir(full_path(plots_folder))
    filename = input_file.split('.')[0] + '.png'
    plt.savefig(filename, dpi=200)
    os.chdir(config_folder)
        
    if args.plot:
        
        plt.show()
        
    else:
        
        plt.close('all')
        
        # Creation of the class file.
        
        ext = '.' + input_file.split('.')[-1]
        output_file = input_file.replace(ext, '-a' + ext)
        output_files += [output_file]
        script = []
        
        # Non modified spectra.
        
        script += ['file in ' + output_folder + input_file]
        script += ['file out ' + output_folder + output_file + ' m /overwrite']
        script += ['find /all', 'list']
        averaged_spectra_flat = [item for sublist in averaged_spectra
                                 for item in sublist]
        for spectrum in all_spectra:
            if spectrum not in averaged_spectra_flat:
                if len(extra_note) > 0:
                    params = spectrum.replace(extra_note+'-', '').split('-')
                else:
                    params = spectrum.split('-')
                script += ['set source ' + params[0]]
                script += ['set line ' + params[1]]
                script += ['set telescope *{}*'.format(params[2])]
                script += ['find /all', 'list', 'get first', 'write']
        script += ['set source *', 'set line *', 'set telescope *']
        
        # Open yaml files.
        with open(spectra_folder + 'rms_regions.yaml') as file:
            rms_regions = yaml.safe_load(file)
        with open(spectra_folder + 'frequency_ranges.yaml') as file:
            frequency_ranges = yaml.safe_load(file)
        with open(spectra_folder + 'reference_frequencies.yaml') as file:
            reference_frequencies = yaml.safe_load(file)
            
        # Averaged spectra.
        
        windows = []
        script += ['file in ' + output_folder + input_file]
        script += ['find /all', 'list']
        lines = []
        
        # Selection of spectra to be averaged.
        for spectra_group in averaged_spectra:
            group_name = spectra_group[0]
            script += ['file out ' + output_folder + group_name
                     + '-temp' + ext + ' m /overwrite']
            lines = []
            for spectrum in spectra_group:               
                lines += [spectrum.split('-')[1]]
            lines = list(np.unique(lines))
            lines = '+'.join(lines)
            for spectrum in spectra_group:
                if len(extra_note) > 0:
                    params = spectrum.replace(extra_note+'-', '').split('-')
                else:
                    params = spectrum.split('-')
                script += ['set source ' + params[0]]
                script += ['set line ' + params[1]]
                script += ['set telescope *{}*'.format(params[2])]
                script += ['find /all', 'list', 'get first']
                script += ['modify line ' + lines]
                rms_region = rms_regions[spectrum]
                freq_range = frequency_ranges[spectrum]
                freq0 = reference_frequencies[spectrum]
                rms_windows = [freq_range[0], rms_region[0],
                               rms_region[1], freq_range[1]]
                for i,freq in enumerate(rms_windows):
                    rms_windows[i] = freq - freq0
                script += ['set window {} {} {} {}'.format(*rms_windows)]
                script += ['base 0', 'write']
            
        # Average of the selected groups of spectra.
        script += ['set source *', 'set line *', 'set telescope *']
        for spectra_group in averaged_spectra:
            group_name = spectra_group[0]
            script += ['file in ' + output_folder + group_name + '-temp' + ext]
            script += ['file out ' + output_folder + output_file]
            script += ['find /all', 'list']
            script += ['set weight sigma', 'stitch /resample /nocheck']
            script += ['write']
        
        # End of the script.
        script += ['exit']
        for i in range(len(script) - 1):
            script[i] += '\n'
            
        # Writing of the class file.
        with open('average.class', 'w') as file:
            file.writelines(script)
        
        # Running of the class file.
        subprocess.run(['class', '@average.class'])

# Ending.
if len(output_files) > 0:  
    temp_files = Path(output_folder).glob('**/*-temp.*')
    temp_files = [str(pp) for pp in temp_files]
    if len(temp_files) != 0:
        for file in temp_files:
            os.remove(file)
    print('\nCreated CLASS files:    (folder {})'.format(output_folder))    
    for file in output_files:      
        print('- ' + file)
    print()

# Elapsed time.
if not config_path.endswith('config-average-auto.yaml'):
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
    