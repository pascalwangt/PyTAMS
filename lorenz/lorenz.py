# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:14:57 2019

@author: pasca
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:32:19 2019

@author: pasca
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:45:49 2019

@author: pasca
"""

import numpy as np

import sympy as sp
from sympy.abc import x,y,z

import sys
sys.path.append('../')

import ellipsoid_fun
import trajectory_to_score_function
import h5py


#system parameters
sig = 10
beta = 8/3
rho = 10

#%%
# states
initial_state = np.array([np.sqrt(beta*(rho-1)),np.sqrt(beta*(rho-1)), rho-1])
target_state = np.array([-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)), rho-1])
saddle_state = np.array([0., 0, rho-1])

#%%



def force(v):
    x,y,z = v
    return np.array([sig*(y-x),
                     x*(rho-z)-y,
                     x*y-beta*z])
    
noise_matrix = None

#sympy force matrix
force_matrix = sp.Matrix([sig*(y-x),
                          x*(rho-z)-y,
                          x*y-beta*z])

#%%
#reduction
sigma = 0.05

covariance_matrix_start, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma, noise_matrix=noise_matrix)
covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
a = np.linalg.eig(np.linalg.inv(covariance_matrix_target))
spectrum, eigvec = np.linalg.eig(np.linalg.inv(covariance_matrix_start))

#%%
#score functions


def score_function_linear(v):
    score = np.sum((target_state-initial_state)*(v-initial_state)) / np.linalg.norm(target_state-initial_state)**2
    if score >=0:
        return score
    else:
        return 1e-5

def score_function_norm(v):
    x,y=v
    return 1/2*np.sqrt((x+1)**2+1/2*y**2)



def score_function_circle_maker(param = 4):
    """
    param: decay rate of the exponentials
    """
    dist =  np.linalg.norm(target_state-initial_state)
    eta = np.linalg.norm(saddle_state-initial_state)/dist
    
    def score_function(v):
        return eta - eta*np.exp(-param*(np.linalg.norm(v-initial_state)/dist)**2)+(1-eta)*np.exp(-param*(np.linalg.norm(v-target_state)/dist)**2)
    return score_function



def score_function_ellipsoid_maker(param = 0.01, sigma=1):
    """
    param: decay rate of the exponentials
    """
    eta = np.linalg.norm(target_state-saddle_state)/np.linalg.norm(target_state-initial_state)
    
    covariance_matrix_start, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma, noise_matrix=noise_matrix)
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
    
    def score_function(v):
        return eta - eta*np.exp(-param*quad_form_initial(v))+(1-eta)*np.exp(-param*quad_form_target(v))
    
    return score_function

def score_function_custom_maker(filename='trajectory.hdf5', decay=0.2):
    """
    param: trajectory file with key "filled_path"
           decay
    """
    with h5py.File(filename, 'r') as file:
        filled_path = file['filled_path'][:]
        file.close()
    score_function = trajectory_to_score_function.score_function_maker(filled_path.T, decay)
    
    return score_function



