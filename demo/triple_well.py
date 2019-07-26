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


"""
Defines the dynamical system:
dX(t) = b(X(t)) + noise_matrix * dW(t)
where b is the drift (referred as force for numpy calculations and force_matrix for the symbolic calculations)
      noise_matrix is the noise matrix, set to None if sigma is the identity matrix
      
Requires variables:
    initial_state (X_A)
    target_state  (X_B)
    
    force        (vector to vector function)
    force_matrix (using sympy symbols)
    noise_matrix (set to None if identity)
    
    optional: 
        score functions (can be defined in another file)
"""

#%%
#states
initial_state = np.array([-5.77,  0])
saddle_state = np.array([0, -2.9e-02])
target_state = np.array([5.77, 0])

#%% parameters for the force
#general confinement
alpha = 0.1
beta = 0.05
yc = 0

#stable minima
x_min = 6
depth = 10
y_decay = 2
x_decay = 2

#metastable minimum
y_intermediate = 20
depth_intermediate = 0
y_decay_intermediate = 3
x_decay_intermediate = 5

#barrier
y_barrier = 15
y_decay_barrier = 1
x_decay_barrier = 2
barrier = 30


#potential (for plots only)
def potential(x,y):
    return alpha*x**2+beta*(y-yc)**2+barrier*(1+np.tanh(-(y-y_barrier)/y_decay_barrier))*np.exp(-(x/x_decay_barrier)**2)-depth*np.exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-depth*np.exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2)

#numpy force matrix
def force(v):
    x,y=v
    return np.array([-2*alpha*x+barrier*(1+np.tanh(-(y-y_barrier)/y_decay_barrier))*2*x/x_decay_barrier**2*np.exp(-(x/x_decay_barrier)**2)-2*(x-x_min)/x_decay**2*depth*np.exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-2*(x+x_min)/x_decay**2*depth*np.exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2),
                     -2*beta*(y-yc)+barrier/y_decay_barrier*np.exp(-(x/x_decay_barrier)**2)/np.cosh(-(y-y_barrier)/y_decay_barrier)**2-2*y/y_decay**2*depth*np.exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-2*y/y_decay**2*depth*np.exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2)])

#sympy force matrix
force_matrix = sp.Matrix([-2*alpha*x+barrier*(1+tanh(-(y-y_barrier)/y_decay_barrier))*2*x/x_decay_barrier**2*exp(-(x/x_decay_barrier)**2)-2*(x-x_min)/x_decay**2*depth*exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-2*(x+x_min)/x_decay**2*depth*exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2),
                     -2*beta*(y-yc)+barrier/y_decay_barrier*exp(-(x/x_decay_barrier)**2)/cosh(-(y-y_barrier)/y_decay_barrier)**2-2*y/y_decay**2*depth*exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-2*y/y_decay**2*depth*exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2)])

#noise matrix
noise_matrix = None #(identity matrix)

#%%
#score functions collection

def score_function_linear(v):
    score = np.sum((target_state-initial_state)*(v-initial_state)) / np.linalg.norm(target_state-initial_state)**2
    if score >=0:
        return score
    else:
        return 1e-5
    
def score_function_linear_simple(v):
    return v[0]/target_state[0]

def score_function_norm(v):
    x,y=v
    return 1/2*np.sqrt((x+1)**2+1/2*y**2)

def score_function_fred(v):
    da = np.linalg.norm(v-initial_state)
    db = np.linalg.norm(v-target_state)
    if da <= db: 
        return da/2/db
    else:
        return 1-db/2/da
    

def score_function_fred_ell_maker(sigma = 1.5):
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
    
    def score_function(v):
        da = np.linalg.norm(v-initial_state)
        db = np.sqrt(quad_form_target(v))
        if da <= db: 
            return da/2/db
        else:
            return 1-db/2/da
    return score_function



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

def score_function_ellipsoid_maker(param = 0.01, sigma=1.5, direction = None, forward=0.3, backward = 0.1):
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






#%%
#tests
potential_well_plot = 0
    
    
if potential_well_plot:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (1.1*3.19, 1.1*2.61))
    x,y  = np.linspace(-15,15,100), np.linspace(-15,25,100)

    xx, yy = np.meshgrid(x, y)
    pot = potential(xx,yy)
    im = plt.contourf(xx, yy, pot, 20, cmap = 'RdBu_r')
    plt.contour(xx, yy, pot, 20, linewidths = 0.4, colors = 'grey', linestyles = '-')
    #plt.contour(xx, yy, pot, 30)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    cbar = fig.colorbar(im, format = '%.0f')
    cbar.ax.set_title('$V(x,y)$')
    plt.scatter(initial_state[0],initial_state[1], marker = 'o', label = '$X_A$', color = 'black', s=40, zorder = 10)
    plt.scatter(target_state[0], target_state[1], marker = 'x', label = '$X_B$', color = 'black', s=40, zorder = 10)
    plt.text(-0.12, 1,'(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    plt.legend(loc = 'lower right')
    plt.savefig('../../../Report/overleaf/wall_potential', bbox_inches = 'tight')
    #plt.savefig('2D_simple_double_well.png')
    


    
