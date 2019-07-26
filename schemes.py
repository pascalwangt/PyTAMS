# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:23:03 2019

@author: pasca
"""
import numpy as np

def dW(dt, dims): 
    """Noise term in Euler-Maruyama scheme"""
    return np.random.normal(loc = 0.0, scale = np.sqrt(dt), size = dims)


def Euler_Maruyama_no_stop(start_index, x0, sigma, force, time_array_length, dt, dims, noise_matrix=None): 
    """returns solution of the SDE using Euler Maruyama scheme """
    n_timestep_max = time_array_length - start_index
    vs = np.empty((n_timestep_max,dims))
    vs[0] = x0
    if noise_matrix is None:
        for n_timestep in range(1, n_timestep_max):
            vs[n_timestep] = vs[n_timestep-1] + force(vs[n_timestep-1]) * dt + sigma * dW(dt, dims)
    else:
        for n_timestep in range(1, n_timestep_max):
            vs[n_timestep] = vs[n_timestep-1] + force(vs[n_timestep-1]) * dt + sigma * np.dot(noise_matrix, dW(dt, dims))
    return vs


def Euler_Maruyama_score_stop(start_index, x0, sigma, score_thresh, dims, score_function, force, time_array_length, dt): 
    """ returns number of calculated timesteps and solution, stops when score is greater than 1-score_thresh, the rest of the trajecotry is nan"""
    n_timestep_max = time_array_length - start_index
    vs = np.zeros((n_timestep_max,dims))
    vs[0] = x0
    for n_timestep in range(1, n_timestep_max):
        vs[n_timestep] = vs[n_timestep-1] + force(vs[n_timestep-1]) * dt + sigma * dW(dt, dims)
        if score_function(vs[n_timestep])>1-score_thresh:
            vs[n_timestep+1:] = np.nan 
            return n_timestep, vs
    return n_timestep_max, vs


def Euler_Maruyama_geom_stop(start_index, x0, sigma, geom_thresh, target_state, dims, score_function, force, time_array_length, dt): 
    """ returns number of calculated timesteps and solution, stops when score is greater than 1-score_thresh, the rest of the trajecotry is nan """
    n_timestep_max = time_array_length - start_index
    vs = np.zeros((n_timestep_max,dims))
    vs[0] = x0
    for n_timestep in range(1, n_timestep_max):
        vs[n_timestep] = vs[n_timestep-1] + force(vs[n_timestep-1]) * dt + sigma * dW(dt, dims)
        if np.linalg.norm(vs[n_timestep]-target_state)<geom_thresh:
            vs[n_timestep+1:] = np.nan 
            return n_timestep, vs
    return n_timestep_max, vs






"""
def Euler_Maruyama_no_stop(t0, x0, sigma,thresh, dims, score_function, t_trajectory,  force, dt):
    time = np.arange(t0, t_trajectory, dt)
    vs    = np.zeros((len(time),dims))
    vs[0] = x0
    for i in range(1, len(time)):
        vs[i] = vs[i-1] + force(vs[i-1]) * dt + sigma * dW(dt, dims)
    return time, vs

"""