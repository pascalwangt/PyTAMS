# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:14:51 2019

@author: pasca
"""
import numpy as np
import h5py

import matplotlib as mpl
import matplotlib.pyplot as plt

import histogram_to_trajectory
import trajectory_to_score_function
import warp_score_function_ell
import interpolate

import sys
sys.path.append('4D_system')
import qg_4D as qg
import plot4D

plt.close('all')

test_histogram_to_trajectory = 1
plotit = 1

test_trajectory_to_score_function = 0

test_warp_score_function_ell = 0
test_interpolation = 0
test_write_hdf5 = 0



#%%
if test_histogram_to_trajectory:

    """
    input:
    histogram
    start and end states
    binslist
    binlengths
    """
    
    #import file
    output_path = '../outputs/refined_histo_4D.hdf5'
    #output_path  = '4D_system/histo_material.hdf5'
    output_path = '../outputs/more_refined_4D.hdf5'
    #output_path  = '4D_system/histo6.hdf5'
    #output_path = '../outputs/histo.hdf5'
    #output_path = '../plots/compare_mcnormal_4D/normal.hdf5'
    output_path = '../outputs/histo_005.hdf5' #the one used to create the trajectories
    
    #output_path = '../outputs/qg_4D_retour.hdf5' #very good looking, factor = 2
    output_path = '../outputs/report_4D_refined.hdf5'
    
    index = 0
    
    TAMS_output_file = h5py.File(output_path, 'r')
    print(f'Opening HDF5 file {output_path} ...\n')
    print('Available entries:')
    for entry in list(TAMS_output_file.keys()):
        print(entry)
    sigma_grid = TAMS_output_file[f'transition_probability_grid_TAMS'].attrs.get('sigma_grid')
    
    histogram_collection = TAMS_output_file['probability_density'][:]
    binslist= TAMS_output_file['listbins'][:]
    
    
    TAMS_output_file.close()
    print('File closed.')

    #%%
    #arguments
    histogram = histogram_collection[index]
    initial_state = qg.initial_state
    target_state = qg.target_state
    
    
    binlengths = None
    factor= 1
    factors = [factor, factor, factor, factor]
    res = 100
    print ([len(bins) for bins in binslist]) 

    
    #result
#    path_unfilled, positions, histogram, binslist = histogram_to_trajectory.get_positions(histogram,
#                                                                                          initial_state,
#                                                                                          target_state,
#                                                                                          binslist,
#                                                                                          binlengths,
#                                                                                          factors = factors,
#                                                                                          cleaned = 1,
#                                                                                          res=res,
#                                                                                          filled = 1,
#                                                                                          smoothed=1,
#                                                                                          smooth = 10,
#                                                                                          version = 1)
#    
    
    if plotit:
        trajectory=None
        
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [0,1],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = trajectory)
        
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [0,2],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = trajectory,
                              save_path = '../../Report/overleaf/histo4D')
        
        
        
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [0,3],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = trajectory)
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [1,2],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = trajectory)
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [1,3],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = trajectory)
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [2,3],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = trajectory)
        
    
    
    
#%%
if test_trajectory_to_score_function:


    """
    input:
    trajectory
    transverse decay
    """
    
    
    #arguments
    trajectory = positions
    decay = 3
    
    #result
    score_function = trajectory_to_score_function.score_function_maker(trajectory.T, decay,)
    

#%%
if test_write_hdf5:
    
    """
    input:
    score_function
    listbins
    path
    """
    
    #arguments
    write_score_function = score_function
    listbins= [np.linspace(-2, 2, 20), 
                np.linspace(-7, 7, 20),
                np.linspace(-7, 7, 20),
                np.linspace(-7, 7, 20)]
    path = 'score_values_4D_3'
    
    #result
    interpolate.function_to_hdf5(write_score_function, listbins, filename = path)

#%%
write_path=1
if write_path:
    with h5py.File('4D_system/trajectory2.hdf5', 'a') as file:
        file.create_dataset('original_path', data = path_unfilled)
        file.create_dataset('filled_path', data = positions)
        file.close()

read_path = 0
if read_path:
    with h5py.File('4D_system/trajectory_8pt.hdf5', 'r') as file:
        p1 = file['original_path'][:]
        p2 = file['filled_path'][:]
        file.close()
    
    if plotit:
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [0,1],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = p2,
                              save_path = '01')
        
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [0,2],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = p2,
                              save_path = '02')
        
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [0,3],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = p2,
                              save_path = '03')
        
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [1,2],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = p2,
                              save_path = '12')
        
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [1,3],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = p2,
                              save_path = '13')
        
        plot4D.histogram_proj(histogram,
                              binslist,
                              indices = [2,3],
                              target_state = qg.target_state,
                              initial_state = qg.initial_state,
                              trajectory = p2,
                              save_path = '23')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    