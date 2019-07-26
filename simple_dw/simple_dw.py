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
from sympy.abc import x,y

import sys
sys.path.append('../')

import ellipsoid_fun
import trajectory_to_score_function
import h5py


#system parameters
beta = -0.3

#%%
# states

initial_state, target_state, saddle_state = np.array([-1, 0]), np.array([1, 0]), np.array([0,0])

def potential(x,y):
    return 0.25*x**4-0.5*x**2+0.5*y**2

def force(v):
    x,y=v
    return np.array([x-x**3,-2*0.5*y])
    
noise_matrix = None

#sympy force matrix
force_matrix = sp.Matrix([x-x**3,
                         -2*0.5*y])

#%%
#reduction
sigma = 0.1

covariance_matrix_start, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma, noise_matrix=noise_matrix)
covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
a = np.linalg.eig(np.linalg.inv(covariance_matrix_target))
spectrum, eigvec = np.linalg.eig(np.linalg.inv(covariance_matrix_start))

#%%
#score functions

def score_function_fred(v):
    da = np.linalg.norm(v-initial_state)
    db = np.linalg.norm(v-target_state)
    if da <= db: 
        return da/2/db
    else:
        return 1-db/2/da
    
def score_function_linear(v):
    return v[0]

def score_function_norm(v):
    x,y=v
    return 1/2*np.sqrt((x+1)**2+1/2*y**2)



def score_function_circle_maker(param = 8):
    """
    param: decay rate of the exponentials
    """
    dist =  np.linalg.norm(target_state-initial_state)
    eta = np.linalg.norm(saddle_state-initial_state)/dist
    
    def score_function(v):
        return eta - eta*np.exp(-param*(np.linalg.norm(v-initial_state)/dist)**2)+(1-eta)*np.exp(-param*(np.linalg.norm(v-target_state)/dist)**2)
    return score_function



def score_function_ellipsoid_maker(param = 0.01, sigma=0.1):
    """
    param: decay rate of the exponentials
    """
    eta = np.linalg.norm(target_state-saddle_state)/np.linalg.norm(target_state-initial_state)
    
    covariance_matrix_start, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma, noise_matrix=noise_matrix)
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
    
    def score_function(v):
        return eta - eta*np.exp(-param*quad_form_initial(v))+(1-eta)*np.exp(-param*quad_form_target(v))
    
    return score_function

def score_function_ellipsoid_maker_plot(param = 2, sigma=0.25):
    """
    param: decay rate of the exponentials
    """
    eta = np.linalg.norm(target_state-saddle_state)/np.linalg.norm(target_state-initial_state)
    
    covariance_matrix_start, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma, noise_matrix=noise_matrix)
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
    
    def score_function(v):
        return eta - eta*np.exp(-param*np.linalg.norm(v-initial_state)**2)+(1-eta)*np.exp(-param*quad_form_target(v)/quad_form_target(saddle_state))
    
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

#%%
potential_well_plot =1
if potential_well_plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    x,y  = np.linspace(-1.5,1.5,100), np.linspace(-1,1,100)

    xx, yy = np.meshgrid(x, y)
    pot = potential(xx,yy)
    im = plt.contourf(xx, yy, pot, 15, cmap = 'RdBu_r')
    plt.contour(xx, yy, pot, 15, linewidths = 0.4, colors = 'grey', linestyles = '-')
    #plt.contour(xx, yy, pot, 30)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    cbar = fig.colorbar(im, format = '%.1f')
    cbar.ax.set_title('$V(x,y)$')
    plt.scatter(initial_state[0],initial_state[1], marker = 'o', label = 'initial state $X_A$', color = 'black', s=40)
    plt.scatter(target_state[0], target_state[1], marker = 'x', label = 'target state $X_B$', color = 'black', s=40)
    plt.legend(loc = 'lower right')
    plt.text(-0.2, 1,'(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    plt.savefig('../../../Report/overleaf/simple_double_well', bbox_inches = 'tight')

