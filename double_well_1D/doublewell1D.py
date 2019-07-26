# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:45:49 2019

@author: pasca
"""

import numpy as np

import sympy as sp
from sympy import exp, tanh, cosh
from sympy.abc import x,y

import h5py

import sys
sys.path.append('../')
import ellipsoid_fun
import trajectory_to_score_function

#%%
#states
initial_state = np.array([-1])
saddle_state = np.array([0])
target_state = np.array([1])

#%%


def force(v):
    [x] = v
    return x-x**3

#sympy force matrix
force_matrix = sp.Matrix([x-x**3])

noise_matrix = None
#%%
#score functions

def score_function_linear(v):
    score = np.sum((target_state-initial_state)*(v-initial_state)) / np.linalg.norm(target_state-initial_state)**2
    if score >=0:
        return score
    else:
        return 1e-5
    
def score_function_linear_simple(v):
    return v[0]/target_state[0]


def score_function_circle_maker(param = 4):
    """
    param: decay rate of the exponentials
    """
    dist =  np.linalg.norm(target_state-initial_state)
    eta = np.linalg.norm(saddle_state-initial_state)/dist
    
    def score_function(v):
        return eta - eta*np.exp(-param*(np.linalg.norm(v-initial_state)/dist)**2)+(1-eta)*np.exp(-param*(np.linalg.norm(v-target_state)/dist)**2)
    return score_function

def score_function_alg_maker(param = 0.05, sigma=1.5):
    """
    param: decay rate of the exponentials
    """
    eta = np.linalg.norm(target_state-saddle_state)/np.linalg.norm(target_state-initial_state)
    
    covariance_matrix_start, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma, noise_matrix=noise_matrix)
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
    
    def score_function(v):
        return eta - eta/(1+param*quad_form_initial(v))+(1-eta)/(1+param*quad_form_target(v))
    return score_function

def score_function_ellipsoid_maker(param = 0.05, sigma=1.5, direction = None, forward=0.3, backward = 0.1):
    """
    param: decay rate of the exponentials
    """
    eta = np.linalg.norm(target_state-saddle_state)/np.linalg.norm(target_state-initial_state)
    
    covariance_matrix_start, quad_form_initial, spectral_radius, level_initial, bound_initial = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma, noise_matrix=noise_matrix)
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
    
    if direction is None:
        def score_function(v):
            return eta - eta*np.exp(-param*quad_form_initial(v))+(1-eta)*np.exp(-param*quad_form_target(v))
        return score_function
    else:
        direction = direction/np.linalg.norm(direction)
        s = np.linspace(-3*bound, 3*bound, 1000)
        
        s = np.linspace(0, 3*bound_initial, 1000)
        
        s = np.expand_dims(s,1)
        line = s*direction+initial_state

        direction_on_border = line[np.argmin(np.apply_along_axis(quad_form_initial, 1, line)<level_initial)]-initial_state
        print(direction_on_border)
        def score_function(v):
            return eta - eta*np.exp(-param*forward*(backward+1+np.tanh(0.2*np.dot(v-initial_state,direction_on_border.T)))*quad_form_initial(v))+(1-eta)*np.exp(-param*quad_form_target(v))
        return score_function
    
def score_function_composite_maker(filename = 'trajectory.hdf5', decay = 4, param=0.01, sigma = 1.5):
    custom_score_function = score_function_custom_maker(filename, decay)
    
    eta = np.linalg.norm(target_state-saddle_state)/np.linalg.norm(target_state-initial_state)
    
    covariance_matrix_start, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma, noise_matrix=noise_matrix)
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
    
    def score_function(v):
        return (custom_score_function(v)+eta - eta*np.exp(-param*quad_form_initial(v)))/1.5
    return score_function



def score_function_custom_maker(filename='trajectory.hdf5', decay=4):
    """
    param: trajectory file with key "filled_path"
           decay
    """
    with h5py.File(filename, 'r') as file:
        filled_path = file['filled_path'][:]
        file.close()
    score_function = trajectory_to_score_function.score_function_maker(filled_path.T, decay)
    
    return score_function

def threshold_simexp_param(param, level):
    dist =  np.linalg.norm(target_state-initial_state)
    eta = np.linalg.norm(saddle_state-initial_state)/dist
    return (1-eta)*(1-np.exp(-level*param))


import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,1, figsize = (1.035*3.19, 1.035*2.61))
x = np.linspace(-1.6,1.6,1000)

def pot(x):
    return -x**2/2+x**4/4
poten = pot(x)
plt.plot(x,poten)
#plt.contour(xx, yy, pot, 30)
plt.xlabel('x')
plt.ylabel('V(x)')
plt.ylim(-0.3, 0.3)
plt.scatter(initial_state[0], pot(initial_state[0]), marker = 'o', label = '$X_A$', color = 'black', s=40, zorder = 40)
plt.scatter(target_state[0], pot(target_state[0]), marker = 'x', label = '$X_B$', color = 'black', s=40, zorder = 40)
plt.text(-0.2, 1,'(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
plt.legend(loc = 'upper center')
plt.savefig('../../../Report/overleaf/1D_pot', bbox_inches = 'tight')
