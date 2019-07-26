# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:31:15 2019

@author: pasca
"""

import sys
sys.path.append('/home/staff/wang0104/TAMS_python_nd/')
sys.path.append('../')

import numpy as np

import TAMS_class_nd
from triple_well_2D import triple_well





#%%

output_path = 'output.hdf5'
sys.stdout = open('out.txt', "a", buffering = 1)

#%%

#sys.stdout = open('outputs/twr_custom_run1_out', "a", buffering = 1)

 #%%
run = TAMS_class_nd.TAMS_object(system = triple_well,
                             
                                score_function  = triple_well.score_function_custom_maker(decay = 4),  #function, hdf5 
                                score_function_name = 'custom',

                                t_trajectory = 200,
                                dt = 0.01,
                                
                                interpolate_score = False,
                                interpolating_grids = [np.linspace(-5,5,1000), np.linspace(-5,5,1000)],
                                
                                warp = True,
                                score_threshold = 'auto')
# %%
run_algorithm = 1
if run_algorithm:
    transition_probability_grid_TAMS, mean_hitting_time, computing_time, histogram, mean_traj, number_timesteps = run.TAMS_run(
                                                                                                            sigma_grid = np.flip(np.linspace(1.1,1.7,6)),
                                                                                                            N_particles = 300,
                                                                                                            N_samples = 10,
                                                                                                            Nitermax = 50000,
                                                                                                            output_path = output_path,
                                                                                                            verbose = 0,
                                                                                                            listbins = [np.linspace(-3, 3, 200), 
                                                                                                                        np.linspace(-3, 7, 200)])

#%%
run_MC = 0
if run_MC:
    transition_probability_grid_MC, computing_time, number_timesteps = run.monte_carlo_run(sigma_grid = np.flip(np.linspace(1.1,1.7,6)),
                                                                                           N_particles = 100000,
                                                                                           output_path=output_path)


