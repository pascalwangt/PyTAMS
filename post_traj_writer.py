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

import ellipsoid_fun

from triple_well_2D import triple_well as tw
from triple_well_2D import tools_2D as too

#%%
check = 1
write = 0
check_score_function = 0

read_path = 1
if read_path:
    with h5py.File('triple_well_2D/traj_report.hdf5', 'r') as file:
        p1 = file['original_path'][:]
        filled_path2 = file['filled_path'][:]
        file.close()
        
#%%

    
"""
input:
histogram
start and end states
binslist
binlengths
res (filling of the positions)
"""

#import file
output_path = '../outputs/histogram_material.hdf5'
output_path = '../outputs/2D_histogram_new.hdf5'
output_path = '../outputs/new.hdf5'
output_path = 'triple_well_2D/histo_3.hdf5'
output_path = '../outputs/custom_3.hdf5'
output_path = '../outputs/normal.hdf5'
output_path = '../outputs/mean.hdf5'
output_path = '../outputs/report_meanhist.hdf5'
output_path = '../outputs/report_meanhist_ell.hdf5'
output_path = '../outputs/report_meanhist_ell2.hdf5' #used for the report with factor 14
output_path = '../plots/report_wall/normal_low.hdf5'
#output_path = '../plots/report_wall/normal_high.hdf5'

index = -1

TAMS_output_file = h5py.File(output_path, 'r')
histogram_collection = TAMS_output_file['probability_density'][:]
sigma_grid = TAMS_output_file[f'transition_probability_grid_TAMS'].attrs.get('sigma_grid')
t_traj = TAMS_output_file[f'transition_probability_grid_TAMS'].attrs.get('trajectory_length')
N = TAMS_output_file[f'transition_probability_grid_TAMS'].attrs.get('N_particles')
noise = sigma_grid[index]
binslist= TAMS_output_file['listbins'][:]
mean_trajectory= TAMS_output_file['mean_trajectory'][:]
TAMS_output_file.close()



#arguments
histogram = histogram_collection[index]
initial_state = tw.initial_state
target_state = tw.target_state
binslist = binslist
binlengths = [1,1]
factor = 4
factors = [factor,factor]
res = 100
filename = 'triple_well_2D/traj_report.hdf5'


#result
path, filled_path, histogram, binslist = histogram_to_trajectory.get_positions(histogram,
                                                                               initial_state,
                                                                               target_state,
                                                                               binslist,
                                                                               binlengths,
                                                                               factors = factors,
                                                                               cleaned = 1,
                                                                               res=res,
                                                                               filled = 1,
                                                                               smoothed=1,
                                                                               smooth = 3,
                                                                               version = 1,
                                                                               undersample = 100)

    
print(len(filled_path[0]))
    
if check:
    #figure
    fig = plt.figure()
    ax = plt.gca()
    
    
    
    #plot histogram
    im = ax.pcolormesh(binslist[0],binslist[1],histogram.T+1, norm=mpl.colors.LogNorm(vmin=histogram.min()+1, vmax=histogram.max()))
    locator = mpl.ticker.LogLocator(base=10) 
    cbar = fig.colorbar(im, ticks = locator)
    
    #plot trajectory
    plt.plot(filled_path2[0], filled_path2[1], label = 'estimation', color = 'red')
    plt.plot(phi_i[0], phi_i[1], label = f'instanton')
    #plt.plot(mean_trajectory[0,:,0]*2, mean_trajectory[0,:,1]*2, label = 'mean', color = 'orange')
    
    #plot initial and target state
    plt.scatter(-target_state[0], target_state[1], marker = 'o', s = 40, color = 'black')
    plt.scatter(target_state[0], target_state[1], marker = 'x',  s= 40, color = 'black')
    
#    ellipsoid_fun.draw_ellipsoid_2D(tw.force_matrix, tw.initial_state,noise, confidence=0.95)
#    ellipsoid_fun.draw_ellipsoid_2D(tw.force_matrix, tw.initial_state,noise, confidence=0.99)
#    ellipsoid_fun.draw_ellipsoid_2D(tw.force_matrix, tw.initial_state,noise, confidence=0.999)
    
    #centers = list(map(lambda v: (v[:-1]-v[1:])/2, binslist))
    #plt.contour(centers[0],centers[1], histogram.T+1, levels = [5*10**4], linewidth = 5)
    
    plt.tick_params(which = 'both', direction = 'out')
    plt.xlabel('x', labelpad = -1)
    plt.ylabel('y', labelpad= -2)
    plt.ylim(-20,30)
    plt.xlim(-20, 20)
    plt.gca().set_facecolor(plt.cm.viridis(0))
    plt.text(-0.2, 1, '',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    
    plt.legend(loc = 'lower right')
    plt.savefig('../../Report/overleaf/low_noise_histo', bbox_inches = 'tight')
    #plt.show()
    
if write:
     with h5py.File(filename, 'a') as file:
        file.create_dataset('original_path', data = path)
        file.create_dataset('filled_path', data = filled_path)
        file.close()
        
#%%

if check_score_function:
    decay = 2
    score_function = trajectory_to_score_function.score_function_maker(filled_path.T, decay)
    
    
    #visualise
    plt.figure()
    print(len(filled_path[0]))
    #filled_path = histogram_to_trajectory.positions_filler(filled_path, 100)
    print(len(filled_path[0]))
    plt.scatter(filled_path[0], filled_path[1], label = 'path', color = 'red', zorder = 10, s=3)
    #plt.text(-0.2, 1,'(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    too.visualise_scorefunction(score_function =score_function, 
                                initial_state = tw.initial_state,
                                target_state = tw.target_state,
                                score_threshold=None,
                                force_matrix=tw.force_matrix,
                                xmin = -20, xmax = 20, ymin = -15, ymax = 30,
                                new_figure=False,
                                nb_levels = 9,
                                save_path = '../../Report/overleaf/custom_2a')
    

    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    