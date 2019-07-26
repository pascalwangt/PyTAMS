# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:31:15 2019

@author: pasca
"""

import sys
sys.path.append('../')

import numpy as np

import TAMS_class_nd
import read_output_hdf5


import simple_dw 
import triple_well_2D.tools_2D as too




#%%

output_path = '../../outputs/testee_high.hdf5'
save_path = '../../../Report/overleaf/simple_notfit'

#%%

#sys.stdout = open('outputs/twr_custom_run1_out', "a", buffering = 1)

 #%%
run = TAMS_class_nd.TAMS_object(system = simple_dw,
                             
                                score_function  = simple_dw.score_function_fred,  #function, hdf5 
                                score_function_name = 'custom',

                                t_trajectory = 200,
                                dt = 0.01,
                                
                                interpolate_score = False,
                                interpolating_grids = [np.linspace(-5,5,1000), np.linspace(-5,5,1000)],
                                
                                warp = True,
                                score_threshold = 'auto')

#%%
check = 1
if check:
    sigma1=0.25
#    too.trajectory_plot(time_array=run.time_array, dt=run.dt, 
#                                         force=run.force, initial_state=run.initial_state, target_state=run.target_state, 
#                                         sigma1=sigma1,
#                                         save_path=save_path,
#                                         force_matrix = run.force_matrix,
#                                         xmin = -1.5, xmax =1.5, ymin = -1, ymax = 1)
    
    too.visualise_scorefunction(run.score_function,run.initial_state,  run.target_state, score_thresholds = [0.12, 0.1, 0.07], colors = ['C1', 'red', 'orange'], save_path = save_path, sigma=sigma1, force_matrix=  run.force_matrix, xmin = -2, xmax =2, ymin = -1, ymax = 1)
# %%
run_algorithm = 0
if run_algorithm:
    transition_probability_grid_TAMS,histogram, quadrant_samples = run.TAMS_run(sigma_grid = np.array([3]),
                                                                                N_particles = 100,
                                                                                N_samples = 1,
                                                                                Nitermax = 10000,
                                                                                output_path = output_path,
                                                                                verbose = 1,
                                                                                listbins = [np.linspace(-13, 13, 100), 
                                                                                            np.linspace(-8, 25, 100)],
                                                                                histbis = False,
                                                                                quadrant = True,
                                                                                branching_points=True)

#%%
run_MC = 0
if run_MC:
    transition_probability_grid_MC, computing_time, number_timesteps = run.monte_carlo_run(sigma_grid = np.array([1.7]),
                                                                                           N_particles = 10000,
                                                                                           )
#%%
density_plot = 0
if density_plot:
    read_output_hdf5.probability_density_plot(output_path = output_path,
                                              histogram = histogram,
                                              index = 0,
                                              save_path = 'histo.png',
                                              score_function = run.score_function,
                                              threshold = 7.30e-02,
                                              xbins = np.linspace(-13, 13, 1000),
                                              ybins = np.linspace(-13, 13, 1000))
    
#%%
mean_traj_plot = 0

if mean_traj_plot:
    read_output_hdf5.mean_trajectory_plot(index = 4,
                                          output_path=output_path,
                                          save_path=save_path+'mean_traj.png',
                                          mean_traj=None,
                                          sigmax = None,
                                          sigmay = None)

#%%
write_theory = 0
if write_theory:
    eyring_kramers.write_theory_2D(run.t_trajectory,
                                   output_path = output_path)


# %%
read_prob = 0
if read_prob:
    read_output_hdf5.probability_sigma_plot(output_path = output_path,
                                            save_path= save_path+'kramers_comp.png', 
                                            score_function_names = ['custom'],
                                            theory_plot = 0)



