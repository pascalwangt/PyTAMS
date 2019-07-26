# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:31:15 2019

@author: pasca
"""

import sys
sys.path.append('../')

import numpy as np

import TAMS_class_nd


import double_well_1D.doublewell1D as double_well
import tools1D as too




#%%

output_path = '../../outputs/testee_high.hdf5'
save_path = 'transition2'

#%%

#sys.stdout = open('outputs/twr_custom_run1_out', "a", buffering = 1)

 #%%
run = TAMS_class_nd.TAMS_object(system = double_well,
                             
                                score_function  = double_well.score_function_linear,  #function, hdf5 
                                score_function_name = 'custom',

                                t_trajectory = 500,
                                dt = 0.01,
                                
                                interpolate_score = False,
                                interpolating_grids = [np.linspace(-5,5,1000), np.linspace(-5,5,1000)],
                                
                                warp = True,
                                score_threshold = 'auto')

#%%
check = 1
if check:
    sigma1=0.3
    #too.trajectory_plot(run.time_array, run.dt, run.force, run.initial_state, run.target_state, sigma1, save_path, ymin = -2, ymax = 2)
    
    #too.visualise_scorefunction(run.score_function, run.target_state, sigma=sigma1, ymax= 30)
# %%
run_algorithm = 0
if run_algorithm:
    transition_probability_grid_TAMS,histogram, quadrant_samples = run.TAMS_run(sigma_grid = np.array([3]),
                                                                                N_particles = 100,
                                                                                N_samples = 1,
                                                                                Nitermax = 10000,
                                                                                output_path = output_path,
                                                                                verbose = 1,
                                                                                listbins = [np.linspace(-13, 13, 100)],
                                                                                quadrant = True,
                                                                                branching_points=True)

#%%
run_MC = 0
if run_MC:
    transition_probability_grid_MC, computing_time, number_timesteps = run.monte_carlo_run(sigma_grid = np.array([1.7]),
                                                                                           N_particles = 10000,
                                                                                           )


