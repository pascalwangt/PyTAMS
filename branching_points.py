# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:29:35 2019

@author: pasca
"""

import functools
import ellipsoid_fun
import numpy as np
import matplotlib.pyplot as plt
import h5py
import triple_well_2D.triple_well as tw

#
#output_path = '../outputs/testee_high.hdf5'
#
#output_path = '../outputs/branch_2_normal.hdf5'
#output_path = '../outputs/mean.hdf5'
output_path = '../outputs/report_branch_normal.hdf5'

 
index = 0
end = -1

with h5py.File(output_path, 'r') as TAMS_output_file:
    print(f'Opening HDF5 file {output_path} ...\n')
    print('Available entries:')
    for entry in list(TAMS_output_file.keys()):
        print(entry)
    sigma_grid = TAMS_output_file[f'transition_probability_grid_TAMS'].attrs.get('sigma_grid')
    sigma = sigma_grid[index]
    
    print(sigma)
    branching_points = TAMS_output_file['branching_points'][:]
    branching_scores = TAMS_output_file['branching_scores'][:]
    
    pre_branching_scores = TAMS_output_file['pre_branching_scores'][:]
    
    pre_branching_points = TAMS_output_file['pre_branching_points'][:]
    pre_branching_points = np.squeeze(pre_branching_points)
    
    print(pre_branching_points.shape)
    print(branching_points.shape)
    
    plt.scatter(branching_points[:end,0], branching_points[:end,1], label = 'branching points, $\phi_{ell}$', s = 0.1)
    #plt.scatter(pre_branching_points[:end,0], pre_branching_points[:end,1], label = 'pre branching points', s= 0.1)
    
    plt.scatter(tw.initial_state[0],tw.initial_state[1], marker = 'o',  color = 'black', s=40, zorder = 10)
    plt.scatter(tw.target_state[0], tw.target_state[1], marker = 'x', color = 'black', s=40, zorder = 10)
    
    #plt.scatter(pre_branching_points[:,0], pre_branching_points[:,1], label = 'pre-branching points', marker = 'D', c = np.arange(len(pre_branching_points)), cmap = plt.cm.plasma)
    #plt.scatter(branching_points[:,0], branching_points[:,1], label = 'branching points',c = np.linspace(0,1, len(pre_branching_points)), cmap = plt.cm.plasma)
 
    plt.legend(loc = 'upper left')
    CS = ellipsoid_fun.draw_ellipsoid_2D(tw.force_matrix, tw.initial_state, noise=sigma, confidence=0.95)
    CS.collections[0].set_label('confidence ellipsoid')
    TAMS_output_file.close()
    plt.ylim(-5, 25)
    plt.xlim(-10,10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.text(-0.2, 1, '(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    plt.savefig('../../Report/overleaf/branch_normal', bbox_inches = 'tight')

#plt.scatter(tw.initial_state[0], tw.initial_state[1], marker = 'o', s=20)

#

#plt.scatter(pre_branching_points[:,0], pre_branching_points[:,1], label = 'pre-branching points', marker = 'D', c = np.arange(len(pre_branching_points)), cmap = plt.cm.plasma)
#plt.scatter(branching_points[:,0], branching_points[:,1], label = 'branching points',c = np.linspace(0,1, len(pre_branching_points)), cmap = plt.cm.plasma)


#%%

output_path = '../outputs/test_pre_branch_custom.hdf5' 
index = -1


with h5py.File(output_path, 'r') as TAMS_output_file:
    print(f'Opening HDF5 file {output_path} ...\n')
    print('Available entries:')
    for entry in list(TAMS_output_file.keys()):
        print(entry)
    sigma_grid = TAMS_output_file[f'transition_probability_grid_TAMS'].attrs.get('sigma_grid')
    sigma = sigma_grid[index]
    
    branching_points = TAMS_output_file['branching_points'][:]
    pre_branching_points = TAMS_output_file['pre_branching_points'][:]
    
    TAMS_output_file.close()
    


"""
plt.scatter(pre_branching_points[:,0], pre_branching_points[:,1], label = 'pre-branching points custom')
plt.scatter(branching_points[:,0], branching_points[:,1], label = 'branching points custom')


plt.legend(loc = 'lower right')
"""
#%%

output_path = '../outputs/test_pre_branch_composite.hdf5' 
index = -1


with h5py.File(output_path, 'r') as TAMS_output_file:
    print(f'Opening HDF5 file {output_path} ...\n')
    print('Available entries:')
    for entry in list(TAMS_output_file.keys()):
        print(entry)
    sigma_grid = TAMS_output_file[f'transition_probability_grid_TAMS'].attrs.get('sigma_grid')
    sigma = sigma_grid[index]
    branching_points = TAMS_output_file['branching_points'][:]
    pre_branching_points = TAMS_output_file['pre_branching_points'][:]
    TAMS_output_file.close()
    


"""
plt.scatter(pre_branching_points[:,0], pre_branching_points[:,1], label = 'pre-branching points custom')
plt.scatter(branching_points[:,0], branching_points[:,1], label = 'branching points custom')
"""

#plt.legend(loc = 'lower right')