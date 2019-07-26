# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:31:15 2019

@author: pasca
"""

import sys
sys.path.append('/home/staff/wang0104/TAMS_python_nd/')
sys.path.append('/home/staff/wang0104/TAMS_python_nd/4D_system/')

import numpy as np

import TAMS_class_nd
import qg_4D






#%%

output_path = 'output.hdf5'
sys.stdout = open('out.txt', "a", buffering = 1)

 #%%
run = TAMS_class_nd.TAMS_object(system = qg_4D,
                             
                                score_function  = qg_4D.score_function_ellipsoid_maker(param = 50),  #function, hdf5 
                                score_function_name = 'normal',

                                t_trajectory = 80,
                                dt = 0.01,
                                
                                interpolate_score = False,
                                interpolating_grids = [np.linspace(-5,5,1000), np.linspace(-5,5,1000)],
                                
                                warp = True,
                                score_threshold = 'auto')

# %%
run_algorithm = 1
if run_algorithm:
    transition_probability_grid_TAMS, mean_hitting_time, computing_time, histogram, mean_traj, number_timesteps = run.TAMS_run(
                                                                                                            sigma_grid = np.array([0.03, 0.02, 0.015, 0.01]),
                                                                                                            N_particles = 200,
                                                                                                            N_samples = 5,
                                                                                                            Nitermax = 10000,
                                                                                                            output_path = output_path,
                                                                                                            verbose = 0,
                                                                                                            listbins = [np.linspace(-1.2, 1.2, 30), 
                                                                                                                        np.linspace(-4, 4, 30),
                                                                                                                        np.linspace(-4, 4, 30),
                                                                                                                        np.linspace(-4, 4, 30)])

#%%
run_MC = 0
if run_MC:
    transition_probability_grid_MC, computing_time, number_timesteps = run.monte_carlo_run(sigma_grid = np.array([0.03, 0.02, 0.015, 0.01]),
                                                                                           N_particles = 10000,
                                                                                           )


