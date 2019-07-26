# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:31:15 2019

@author: pasca
"""

import sys
sys.path.append('../')

import numpy as np

import TAMS_class_nd
import triple_well_2D.triple_well as triple_well
import tools_2D as too

 #%%
run = TAMS_class_nd.TAMS_object(system = triple_well, #module where the system information is stored (force matrix, initial/target state ...)
                                
                                #score function 
                                score_function  = triple_well.score_function_custom_maker('traj_report.hdf5', decay = 20),
                                
                                #time
                                t_trajectory = 20, #trajectory duration
                                dt = 0.01,         #time step
                                
                                #optional, interpolation
                                interpolate_score = False, #True if you want to use interpolated version of the score function, or path to the hdf5 file where the grid is available on 
                                interpolating_grids = [np.linspace(-5,5,1000), np.linspace(-5,5,1000)], #if interpolate_score is True, interpolation grid bins
                                )

#%%
#diagnostics
check = 0
if check:
    sigma1= 3
    
    #plot trajectory
    too.trajectory_plot(time_array=run.time_array, dt=run.dt, 
                                         force=run.force, initial_state=run.initial_state, target_state=run.target_state, 
                                         sigma1=sigma1,
                                         save_path=None,
                                         force_matrix=run.force_matrix,
                                         xmin = -13, xmax =13, ymin = -8, ymax = 25)
    
    #visualise score function and ellipsoid
    too.visualise_scorefunction(run.score_function, run.initial_state, run.target_state, sigma=sigma1, force_matrix = run.force_matrix, ymax= 30, nb_levels = 15) 


#%% TAMS algorithm
run_algorithm = 1
if run_algorithm:
    transition_probability_grid_TAMS, histogram, quadrant_samples = run.TAMS_run(  #TAMS parameters
                                                                                   sigma_grid = np.array([2.5]),            #noise values
                                                                                   N_particles = 100,                       #number of particles
                                                                                   N_samples = 1,                           #number of samples
                                                                                   Nitermax = 10000,                        #maximum number of iterations
                                                                                   warp = True,                             #set to True if you want the target set to be the ellipsoid (automatic target score, only works if the score function is normalized to 1)
                                                                                   
                                                                                   #output path and verbose
                                                                                   output_path = '../../outputs/demo.hdf5', #path to save the hdf5 output file, set to None if you do not want to save
                                                                                   verbose = 1,                             #for printing scores at each iteration
                                                                                   
                                                                                   #histogram bins
                                                                                   listbins = [np.linspace(-20, 20, 100),   #bins for the histogram
                                                                                               np.linspace(-20, 20, 100)],
                                                                                   
                                                                                   #more options
                                                                                   quadrant = True,                         #if you want to get samples for the initial direction (works if sigma_grid is of length 1 and with 1 sample)
                                                                                   branching_points=True,                   #if you want to compute and write branching points into the output file
                                                                                   quadrant_factor = 1000,                  #save len(T)/quadrant factor points after the last ellipsoid exit
                                                                                   geom_stopping = False,                   #set to float to set the target set geometrically to a ball of radius geom_stopping around the target state
                                                                                   score_stopping = False,                  #set to float to set the target score level manually
                                                                                   histbis = False                          #weighted histogram
                                                                                   )
#%% Monte Carlo
run_MC = 0
if run_MC:
    transition_probability_grid_MC, computing_time, number_timesteps = run.monte_carlo_run(sigma_grid = np.array([1.7]),    #noise values
                                                                                           N_particles = 10000,             #number of particles
                                                                                           output_path = '../../outputs/demo.hdf5')
#%% Plot histogram
histogram_plot = 0

if histogram_plot:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots
    im = ax.pcolormesh(np.linspace(-20, 20, 100),np.linspace(-20, 20, 100),histogram.T+1, norm=mpl.colors.LogNorm(vmin=histogram.min()+1, vmax=histogram.max()))
    locator = mpl.ticker.LogLocator(base=10) 
    cbar = fig.colorbar(im, ticks = locator)
    
    plt.scatter(-run.initial_state[0], run.initial_state[1], marker = 'o', s = 40, color = 'black')
    plt.scatter(run.target_state[0], run.target_state[1], marker = 'x',  s= 40, color = 'black')
    plt.tick_params(which = 'both', direction = 'out')
    plt.xlabel('x', labelpad = -1)
    plt.ylabel('y', labelpad= -2)
#    plt.ylim(-20,30)
#    plt.xlim(-20, 20)
    plt.gca().set_facecolor(plt.cm.viridis(0))
    plt.text(-0.2, 1, '',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    
    plt.legend(loc = 'lower right')


