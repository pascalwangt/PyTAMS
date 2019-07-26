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


import qg_4D
import plot4D





#%%

output_path = 'histo_material.hdf5'
save_path = 'figures/'

#%%

#sys.stdout = open('outputs/twr_custom_run1_out', "a", buffering = 1)

 #%%
run = TAMS_class_nd.TAMS_object(system = qg_4D,
                             
                                score_function  = qg_4D.score_function_ellipsoid_maker(param = 0.005),  #function, hdf5 
                                score_function_name = 'linear', 

                                t_trajectory =1000,
                                dt = 0.01,
                                
                                interpolate_score = False,
                                interpolating_grids = [np.linspace(-5,5,1000), np.linspace(-5,5,1000)],
                                
                                warp = True,
                                score_threshold = 'auto')

# %%
def runit():
    #transition_probability_grid_TAMS, histogram, quadrant_samples 
    return run.TAMS_run(
                                                                                                            sigma_grid = np.array([0.03]),
                                                                                                            N_particles = 1000,
                                                                                                            N_samples = 1,
                                                                                                            Nitermax = 10000,
                                                                                                            output_path = None,
                                                                                                            verbose = 1,
                                                                                                            listbins = [np.linspace(-1.2, 1.2, 50), 
                                                                                                                        np.linspace(-4, 4, 50),
                                                                                                                        np.linspace(-4, 4, 50),
                                                                                                                        np.linspace(-4, 4, 50)],
                                                                                                            geom_stopping = 0.2)

#%%
run_MC = 0
if run_MC:
    transition_probability_grid_MC, computing_time, number_timesteps = run.monte_carlo_run(sigma_grid = np.array([1]),
                                                                                           N_particles = 10000,
                                                                                           geom_stopping = 0.5)


#%%
check_traj = 0
sigma1 = 0.5
sigma2 = 0.06
if check_traj:
#    a = plot4D.trajectory(run.time_array, run.force, run.initial_state, run.target_state, sigma1, noise_matrix = run.noise_matrix, force_matrix = run.force_matrix, save_path=None,
#                                     index_time = [0,2], index_traj = [0,2], label_time = ['x', 'y'], label_traj = ['x', 'y'], geom_thresh = 0.1,
#                                     score_function = None, xmin = -2, xmax = 2, ymin = -4, ymax = 4)
    
    plot4D.check(run.time_array, run.force, run.initial_state, run.target_state, sigma2, noise_matrix = run.noise_matrix, force_matrix = run.force_matrix, save_path='../../../Report/overleaf/basin' ,
                 index_time = [0,2], index_traj = [0,2], label_time = ['x', 'y'], label_traj = ['$A_1$', '$A_3$'],
                 score_function = None, xmin = -1, xmax = 1, ymin = -3, ymax = 3)
#%%
plot4D.plot_psi(run.saddle_state, res = 20, savepath = '../../../Report/overleaf/psia')
#%%
check_func = 0
sigma = 0.07
if check_func:
    plot4D.draw_projection(run.initial_state, run.target_state,
                           indices = [0,2], proj = run.target_state, lims = [6,6],
                           sigma = sigma, noise_matrix = qg_4D.noise_matrix,
                           function = run.score_function,
                           force_matrix = run.force_matrix)
# %%
read_prob = 0
if read_prob:
    read_output_hdf5.probability_sigma_plot(output_path = output_path,
                                            save_path= save_path+'kramers_comp.png', 
                                            score_function_names = ['custom'],
                                            theory_plot = 0)



