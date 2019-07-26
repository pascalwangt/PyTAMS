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
import warp_score_function_ell
import interpolate


test_histogram_to_trajectory = 0
plotit = 0

test_trajectory_to_score_function = 0

test_warp_score_function_ell = 0
test_interpolation = 0
test_write_hdf5 = 0
test_read_hdf5 = 0

optimize_parameter = 0

#%%
noise = 1.3
decay = 3
path = 'triple_well_2D/score_values_custom_02.hdf5'

#%%
if optimize_parameter:
    import triple_well_2D.triple_well as tw
    import ellipsoid_fun
    import functools
    """
    input:
    score function maker
    initial guess
    force matrix
    noise
    """

    #arguments
    score_function_maker = tw.score_function_ellipsoid_maker
    force_matrix = tw.force_matrix
    noise = noise
    initial_guess = 3
    initial_state = tw.initial_state
    target_state = tw.target_state
    threshold_param = tw.threshold_simexp_param
    
    #result 
    covariance_matrix_start, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, noise)
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, noise)
    
    ell = ellipsoid_fun.get_ellipsoid_array(target_state, quad_form_target, level, bound)
    score_function, score_threshold = ellipsoid_fun.optimize_param(score_function_maker, initial_guess, ell, noise, functools.partial(threshold_param, level = level))
#%%
if test_histogram_to_trajectory:
    
    """
    input:
    histogram
    start and end states
    binslist
    binlengths
    """
    
    #import file
    output_path = '../outputs/linear.hdf5'
    index = 7
    
    TAMS_output_file = h5py.File(output_path, 'r')
    histogram_collection = TAMS_output_file['probability_density'][:]
    TAMS_output_file.close()
    

    
    #arguments
    histogram = histogram_collection[index]
    initial_state = np.array([-1.6734231 , 0.04399182])
    target_state = np.array([1.6734231 , 0.04399182])
    binslist = [np.linspace(-3, 3, 200), np.linspace(-4, 5, 200)]
    binlengths = [0.2, 0.4]

    
    #result
    positions_unfilled, positions, histogram, binslist = histogram_to_trajectory.get_positions(histogram,
                                                                           initial_state,
                                                                           target_state,
                                                                           binslist,
                                                                           binlengths,
                                                                           cleaned = 1,
                                                                           res = 20)
    
    
    if plotit:
        #figure
        fig, ax = plt.subplots()
        
        #plot histogram
        im = ax.pcolormesh(binslist[0],binslist[1],histogram.T+1, norm=mpl.colors.LogNorm(vmin=histogram.min()+1, vmax=histogram.max()))
        locator = mpl.ticker.LogLocator(base=10) 
        cbar = fig.colorbar(im, ticks = locator)
        
        #plot trajectory
        plt.plot(positions[0], positions[1], label = 'path', color = 'red')
        
        #plot initial and target state
        plt.scatter(target_state[0], target_state[1], marker = 'x', label = 'target', s = 40, color = 'black')
        plt.scatter(-target_state[0], target_state[1], marker = 'x', label = 'start', s= 40)
        
        plt.tick_params(which = 'both', direction = 'out')
            
        plt.legend(loc = 'lower right')
        plt.show()
    
    
#%%
if test_trajectory_to_score_function:
    import triple_well_2D.triple_well as tw
    import triple_well_2D.tools_2D as too

    """
    input:
    trajectory
    transverse decay
    """
    
    
    #arguments
    trajectory = positions
    decay = decay
    sigma = noise
    
    #result
    score_function = trajectory_to_score_function.score_function_maker(trajectory.T, decay)
    
    
    #visualise
    too.visualise_scorefunction(score_function =score_function,
                                sigma = sigma, 
                                target_state = tw.target_state,
                                score_threshold=None,
                                force_matrix=tw.force_matrix,
                                xmin = -4, xmax = 4, ymin = -3, ymax = 6)

#%%7
if test_warp_score_function_ell:
    import triple_well_2D.triple_well as tw
    import triple_well_2D.tools_2D as too
    """
    input:
    score_function
    force_matrix
    equilibrium_point
    noise
    """
    
    
    #arguments
    #score_function = score_function
    force_matrix = tw.force_matrix
    equilibrium_point = tw.target_state
    noise = noise
    
    score_function = tw.score_function_custom_maker(filename = 'triple_well_2D/trajectory.hdf5', decay=2)
    
    #result
    warped_score_function, threshold = warp_score_function_ell.remap_score_function_ell(score_function, force_matrix, equilibrium_point, noise)
    
    
    #visualise
    too.visualise_scorefunction_level(score_function = score_function,
                                      score_threshold= 2*threshold,
                                      xmin = -4, xmax = 4, ymin = -3, ymax = 6)
    
    too.visualise_scorefunction(score_function = warped_score_function,
                                sigma = noise, 
                                target_state = tw.target_state,
                                score_threshold= threshold,
                                force_matrix=tw.force_matrix,
                                xmin = -4, xmax = 4, ymin = -3, ymax = 6,
                                new_figure = False)
    


#%%
if test_interpolation:
    
    """
    input:
    score_function
    listbins
    """
    
    #arguments
    warped_score_function = warped_score_function
    listbins = [np.linspace(-3.5,3.5,400), np.linspace(-3,8,400)]
    
    #result
    interp_score_function = interpolate.function_to_intfunction(warped_score_function, listbins)
    
    #plot
    too.visualise_scorefunction(score_function = interp_score_function,
                                sigma = sigma, 
                                target_state = tw.target_state,
                                score_threshold= threshold,
                                force_matrix=tw.force_matrix,
                                xmin = -4, xmax = 4, ymin = -3, ymax = 6)
#%%
if test_write_hdf5:
    
    """
    input:
    score_function
    listbins
    path
    """
    
    #arguments
    write_score_function = score_function
    listbins = [np.linspace(-3.5,3.5,400), np.linspace(-3,8,400)]
    path = path
    
    #result
    interpolate.function_to_hdf5(write_score_function, listbins, filename = path)

    
#%%
if test_read_hdf5:
    
    """
    input:
    path
    """
    
    #arguments
    path = path
    
    #result
    interp_score_function = interpolate.hdf5_to_intfunction()
    
    #plot
    too.visualise_scorefunction(score_function = interp_score_function,
                                sigma = noise, 
                                target_state = tw.target_state,
                                score_threshold=None,
                                force_matrix= None,
                                xmin = -4, xmax = 4, ymin = -3, ymax = 6)
    

    

    
    
    
    
    
#%%
plot_report = 1
if plot_report:
    import simple_dw.simple_dw as sdw 
    import triple_well_2D.tools_2D as too
    """
    input:
    score_function
    force_matrix
    equilibrium_point
    noise
    """
    
    
    #arguments
    #score_function = score_function
    force_matrix = sdw.force_matrix
    equilibrium_point = sdw.target_state
    noise = 0.25
    
    score_function = sdw.score_function_fred
    
    #result
    warped_score_function, threshold = warp_score_function_ell.remap_score_function_ell(score_function, force_matrix, equilibrium_point, noise)
    
    
    #visualise
    too.visualise_scorefunction_level(score_function = score_function,
                                      score_threshold= 2*threshold,
                                      xmin = -2, xmax = 2, ymin = -1, ymax = 1)
    
    too.visualise_scorefunction(score_function = warped_score_function,
                                sigma = noise, 
                                target_state = sdw.target_state,
                                initial_state = sdw.initial_state,
                                score_threshold= threshold,
                                force_matrix=sdw.force_matrix,
                                xmin = -2, xmax = 2, ymin = -1, ymax = 1,
                                new_figure = False,
                                save_path = '../../Report/overleaf/simple_warp')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    