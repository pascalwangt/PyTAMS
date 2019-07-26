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


from triple_well_2D import triple_well





#%%

output_path = '../../outputs/histogram_material.hdf5'
save_path = 'figures/'

#%%

#sys.stdout = open('outputs/twr_custom_run1_out', "a", buffering = 1)

 #%%
run = TAMS_class_nd.TAMS_object(system = triple_well,
                             
                                score_function  = triple_well.score_function_linear,  #function, hdf5 
                                score_function_name = 'custom',

                                t_trajectory = 25,
                                dt = 0.01,
                                
                                interpolate_score = False,
                                interpolating_grids = [np.linspace(-5,5,1000), np.linspace(-5,5,1000)],
                                
                                warp = True,
                                score_threshold = 'auto')

#%%
check = 0
if check:
    read_output_hdf5.trajectory_plot_2D_compare_sigma(sigma1 = 0, sigma2 = 3.5, 
                                                      time_array = run.time_array, 
                                                      draw_threshold = 'no',              #'score' or 'geom'
                                                      geom_threshold = run.geom_threshold,
                                                      score_threshold= run.score_threshold, 
                                                      score_function = run.score_function, 
                                                      initial_state = run.initial_state,
                                                      target_state = run.target_state,
                                                      solver_scheme = run.solver_scheme_no_stop,
                                                      xmin = -3.5, xmax = 3.5, ymin = -4, ymax = 6,
                                                      save_path = save_path+'compare_sigma.png')

#%%
def runit():
    transition_probability_grid_TAMS, mean_hitting_time, computing_time, histogram, mean_traj, number_timesteps = run.TAMS_run(
                                                                                                            sigma_grid = np.array([1.7]),
                                                                                                            N_particles = 1000,
                                                                                                            N_samples = 5,
                                                                                                            Nitermax = 10000,
                                                                                                            output_path = output_path,
                                                                                                            verbose = 0,
                                                                                                            listbins = [np.linspace(-3, 3, 400), 
                                                                                                                        np.linspace(-3, 7, 400)])



#%%
run_MC = 0
if run_MC:
    transition_probability_grid_MC, computing_time, number_timesteps = run.monte_carlo_run(sigma_grid = np.array([2.1, 2, 1.9]),
                                                                                           N_particles = 10000,
                                                                                           )
#%%
density_plot = 1
if density_plot:
    read_output_hdf5.probability_density_plot(output_path = output_path,
                                              histogram = histogram,
                                              index = 0,
                                              save_path = 'histo.png',
                                              score_function = run.score_function,
                                              threshold = None,
                                              xbins = np.linspace(-3, 3, 400),
                                              ybins = np.linspace(-3, 7, 400))
    
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



