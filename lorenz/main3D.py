# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:53:44 2019

@author: pasca
"""

import sys
sys.path.append('../')

import numpy as np

import TAMS_class_nd


import lorenz
import plot3D





#%%

output_path = 'histo.hdf5'
save_path = 'figures/'

#%%

#sys.stdout = open('outputs/twr_custom_run1_out', "a", buffering = 1)

 #%%
run = TAMS_class_nd.TAMS_object(system = lorenz,
                             
                                score_function  = lorenz.score_function_ellipsoid_maker(param = 0.01),  #function, hdf5 
                                score_function_name = 'linear',

                                t_trajectory =50,
                                dt = 0.01,
                                
                                interpolate_score = False,
                                interpolating_grids = [np.linspace(-5,5,1000), np.linspace(-5,5,1000)],
                                
                                warp = True,
                                score_threshold = 'auto')


# %%
def runit():
    transition_probability_grid_TAMS, probability_density, quadrant_samples = run.TAMS_run(sigma_grid = np.array([0.8]),
                                                                                           N_particles = 100,
                                                                                           N_samples = 1,
                                                                                           Nitermax = 10000,
                                                                                           output_path = output_path,
                                                                                           verbose = 1,
                                                                                           listbins = [np.linspace(-12, 12, 100), 
                                                                                                       np.linspace(-12, 12, 100),
                                                                                                       np.linspace(0, 20, 100)])


#%%
run_MC = 0
if run_MC:
    traj = run.monte_carlo_run(sigma_grid = np.array([1.1]),
                                                                                           N_particles = 10000,)
#%%
check_traj = 1

sigma = 1.1
if check_traj:
    
    plot3D.plot_trajectory(run.time_array, run.force, run.initial_state, run.target_state, sigma, noise_matrix = run.noise_matrix, force_matrix = run.force_matrix, 
                           xmin = -10, xmax = 10, ymin = 10, ymax = -10, zmin = 0, zmax = 15, save_path = None, vs = traj)
    
#    plot3D.check(run.time_array, run.force, run.initial_state, run.target_state, sigma, noise_matrix = run.noise_matrix, force_matrix = run.force_matrix, 
#                           save_path=None, xmin = None, xmax = None, ymin = None, ymax =None)
                                                                                     

#%%
check_proj = 0
if check_proj:
    plot3D.draw_projection(run.initial_state, run.target_state,
                           indices = [0,1], proj = run.target_state, lims = [15,10],
                           sigma = sigma, noise_matrix = run.noise_matrix,
                           function = run.score_function,
                           force_matrix = run.force_matrix)





