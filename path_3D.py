# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:14:51 2019

@author: pasca
"""
import numpy as np
import h5py


import histogram_to_trajectory
import trajectory_to_score_function

import interpolate

import sys
sys.path.append('lorenz/')
import lorenz.lorenz as lrz
import plot3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

test_histogram_to_trajectory = 1
plotit = 1

test_trajectory_to_score_function = 0

test_warp_score_function_ell = 0
test_interpolation = 0




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
    output_path = 'lorenz/histo.hdf5'
    output_path = '../outputs/histolorenz.hdf5'
    #output_path = '../outputs/histolorenzbis.hdf5'
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
    initial_state = lrz.initial_state
    target_state = lrz.target_state
    
    
    binlengths = None
    factor= 4
    factors = [factor,factor, factor]
    res = 100

    
    #result
    path_unfilled, positions, histogram, binslist = histogram_to_trajectory.get_positions(histogram,
                                                                                          initial_state,
                                                                                          target_state,
                                                                                          binslist,
                                                                                          binlengths,
                                                                                          factors = factors,
                                                                                          cleaned = 1,
                                                                                          res=res,
                                                                                          filled = 1,
                                                                                          smoothed=1,
                                                                                          smooth = 10,
                                                                                          version = 1)
    
    
    if plotit:
        trajectory=positions
        instanton = phi_i
        
        plot3D.histogram_proj(histogram,
                              binslist,
                              indices = [0,1],
                              target_state = lrz.target_state,
                              initial_state = lrz.initial_state,
                              trajectory = trajectory,
                              instanton = instanton)
        
        plot3D.histogram_proj(histogram,
                              binslist,
                              indices = [0,2],
                              target_state = lrz.target_state,
                              initial_state = lrz.initial_state,
                              trajectory = trajectory,
                              instanton = instanton)

        plot3D.histogram_proj(histogram,
                              binslist,
                              indices = [1,2],
                              target_state = lrz.target_state,
                              initial_state = lrz.initial_state,
                              trajectory = trajectory,
                              instanton = instanton)
        
        
        
        
        """
        fig = plt.figure()
        

        ax = fig.add_subplot(111, projection='3d')
        ax.patch.set_facecolor('white')
        ax.tick_params(axis='both', which='major', pad=-2)
        
        xx, yy, zz = np.meshgrid(*map(lambda v: v[:-1]-v[1:], binslist))
        
        xx = np.ravel(xx)
        yy = np.ravel(yy)
        zz = np.ravel(zz)
        
        histogramf = np.ravel(histogram)
        
        
        print(histogram.shape)
        print(xx.shape)
        
        thr = 100
        ax.plot_trisurf(yy[histogramf>thr],xx[histogramf>thr], zz[histogramf>thr], color= 'blue', alpha=0.1,)
        
        
        xmin = -10
        xmax = 10
        ymin = -10
        ymax = 10
        zmin = 0
        zmax = 15
        ax = fig.add_subplot(111, projection='3d')
        ax.patch.set_facecolor('white')
        ax.tick_params(axis='both', which='major', pad=-2)
        
        vs = positions.T
        
        
        #ax.plot(vs[:,1], vs[:,0], zs=vs[:,2], zdir = 'z', linewidth = 0.2, color = 'C1')
        
        #plt.title(r'$\sigma$'+' = {}'.format(sigma1))
        ax.set_ylabel('x', labelpad = -7)
        ax.set_xlabel('y', labelpad = -7)
        ax.set_zlabel('z', labelpad = -7)
        
        ax.set_xlim(xmin, xmax)
        ax.set_zlim(zmin, zmax)
        ax.set_ylim(ymin, ymax)
        
        
        plt.scatter(initial_state[1], initial_state[0], zs=initial_state[2], marker = 'o', label = '$X_A$', s = 40, color = 'black', zorder = 1000)
        plt.scatter(target_state[1], target_state[0], zs=target_state[2] ,marker = 'x', label = '$X_B$', s= 40, color = 'black', zorder = 1000)
        
    
        plt.scatter(initial_state[1], initial_state[0], zs=initial_state[2], marker = 'o', label = '$X_A$', s = 40, color = 'black', zorder = 1000)
        plt.scatter(target_state[1], target_state[0], zs=target_state[2] ,marker = 'x', label = '$X_B$', s= 40, color = 'black', zorder = 1000)
        #plt.legend()
        
        plt.xticks([-5, 0,5])
        plt.yticks([-5, 0,5])
        ax.set_zticks([ 0,5,10,15])
        
        ax.text2D(0.1, 0.9, '(b)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
        """
    
    
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
write_path=0
if write_path:
    with h5py.File('lorenz/trajectory2.hdf5', 'a') as file:
        file.create_dataset('original_path', data = path_unfilled)
        file.create_dataset('filled_path', data = positions)
        file.close()

read_path = 0
if read_path:
    with h5py.File('lorenz/trajectoryspiral.hdf5', 'r') as file:
        p1 = file['original_path'][:]
        p2 = file['filled_path'][:]
        file.close()
    
    if plotit:
        plot3D.histogram_proj(histogram,
                              binslist,
                              indices = [0,1],
                              target_state = lrz.target_state,
                              initial_state = lrz.initial_state,
                              trajectory = p2,
                              save_path = '01')
        
        plot3D.histogram_proj(histogram,
                              binslist,
                              indices = [0,2],
                              target_state = lrz.target_state,
                              initial_state = lrz.initial_state,
                              trajectory = p2,
                              save_path = '02')
        
        plot3D.histogram_proj(histogram,
                              binslist,
                              indices = [1,2],
                              target_state = lrz.target_state,
                              initial_state = lrz.initial_state,
                              trajectory = p2,
                              save_path = '12')
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    