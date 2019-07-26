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
initial_state = np.array([-5.7715972293490533928661534446291625499725341796875,  6.0870923860507832481997469997736516233999282121658325195312500000000000e-04])
saddle_state = np.array([0, -2.93429453258859905540045787120106979273259639739990234375e-02])
#saddle_state = np.array([0, 18])
target_state = np.array([5.7715972293490533928661534446291625499725341796875, 6.0870923860507832481997469997736516233999282121658325195312500000000000e-04])

#%%
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
y_intermediate = 20/2.5
depth_intermediate = 20/2.5
y_decay_intermediate = 3
x_decay_intermediate = 5

#barrier
y_barrier = 15/2.5
y_decay_barrier = 1
x_decay_barrier = 2
barrier = 20




def potential(x,y):
    return 4.8+alpha*x**2+beta*(y-yc)**2+barrier*(1+np.tanh(-(y-y_barrier)/y_decay_barrier))*np.exp(-(x/x_decay_barrier)**2)-depth_intermediate*np.exp(-(x/x_decay_intermediate)**2-((y-y_intermediate)/y_decay_intermediate)**2)-depth*np.exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-depth*np.exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2)

def force(v):
    x,y=v
    return np.array([-2*alpha*x+barrier*(1+np.tanh(-(y-y_barrier)/y_decay_barrier))*2*x/x_decay_barrier**2*np.exp(-(x/x_decay_barrier)**2)-2*x/x_decay_intermediate**2*depth_intermediate*np.exp(-(x/x_decay_intermediate)**2-((y-y_intermediate)/y_decay_intermediate)**2)-2*(x-x_min)/x_decay**2*depth*np.exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-2*(x+x_min)/x_decay**2*depth*np.exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2),
                     -2*beta*(y-yc)+barrier/y_decay_barrier*np.exp(-(x/x_decay_barrier)**2)/np.cosh(-(y-y_barrier)/y_decay_barrier)**2-2*(y-y_intermediate)/y_decay_intermediate**2*depth_intermediate*np.exp(-(x/x_decay_intermediate)**2-((y-y_intermediate)/y_decay_intermediate)**2)-2*y/y_decay**2*depth*np.exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-2*y/y_decay**2*depth*np.exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2)])

#sympy force matrix
force_matrix = sp.Matrix([-2*alpha*x+barrier*(1+tanh(-(y-y_barrier)/y_decay_barrier))*2*x/x_decay_barrier**2*exp(-(x/x_decay_barrier)**2)-2*x/x_decay_intermediate**2*depth_intermediate*exp(-(x/x_decay_intermediate)**2-((y-y_intermediate)/y_decay_intermediate)**2)-2*(x-x_min)/x_decay**2*depth*exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-2*(x+x_min)/x_decay**2*depth*exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2),
                     -2*beta*(y-yc)+barrier/y_decay_barrier*exp(-(x/x_decay_barrier)**2)/cosh(-(y-y_barrier)/y_decay_barrier)**2-2*(y-y_intermediate)/y_decay_intermediate**2*depth_intermediate*exp(-(x/x_decay_intermediate)**2-((y-y_intermediate)/y_decay_intermediate)**2)-2*y/y_decay**2*depth*exp(-((x-x_min)/x_decay)**2-(y/y_decay)**2)-2*y/y_decay**2*depth*exp(-((x+x_min)/x_decay)**2-(y/y_decay)**2)])

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

def score_function_ellipsoid_maker(param = 0.05, sigma=1.5):
    """
    param: decay rate of the exponentials
    """
    eta = np.linalg.norm(target_state-saddle_state)/np.linalg.norm(target_state-initial_state)
    
    covariance_matrix_start, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma, noise_matrix=noise_matrix)
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix=noise_matrix)
    
    def score_function(v):
        return eta - eta*np.exp(-param*quad_form_initial(v))+(1-eta)*np.exp(-param*quad_form_target(v))
    
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
check_ellipsoid_array = 0
potential_well_plot_3D = 0
potential_well_plot = 0


if check_ellipsoid_array:
    import matplotlib.pyplot as plt
    #ell = ellipsoid_fun.get_ellipsoid_array(target_state, quad_form, level, bound)
    plt.scatter(ell.T[0], ell.T[1])
    CS = ellipsoid_fun.draw_ellipsoid_2D(force_matrix, target_state, noise = sigma)
    foo = ellipsoid_fun.check_ellipsoid(ell, score_function_simexp_ell_param, threshold=threshold_simexp, tolerance=1e-3)
    
    score_level = ellipsoid_fun.get_levelset_array(target_state, score_function_simexp_ell, level = 1-threshold_simexp, bound=2*bound, tolerance = 1e-3)
    plt.scatter(score_level.T[0], score_level.T[1], alpha = 0.5)
    
    print(foo)


if potential_well_plot_3D:
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y  = np.linspace(-12,12,100), np.linspace(-10,25,100)
    xx, yy = np.meshgrid(x, y)
    zz = potential(xx,yy)
    im = ax.plot_surface(xx,yy,zz, cmap = 'RdBu_r')
    ax.set_xlabel('x')
    ax.tick_params(axis='x', which='major', pad=0)
    ax.set_ylabel('y')
    ax.set_zlabel('V(x,y)', labelpad = 10)
    clb = fig.colorbar(im, fraction=0.03, pad=-0.1)
    clb.ax.set_title('V(x,y)', fontsize = 16)
        
    ax.set_facecolor('white')
    #plt.savefig('2D_simple_double_well_ax3D.png')
    
    
if potential_well_plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    x,y  = np.linspace(-15,15,100), np.linspace(-15,25,100)

    xx, yy = np.meshgrid(x, y)
    pot = potential(xx,yy)
    im = plt.contourf(xx, yy, pot, 100, cmap = 'RdBu_r')
    #plt.contour(xx, yy, pot, 30)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    cbar = fig.colorbar(im,)
    cbar.ax.set_title('$V(x,y)$', pad = 15)
    plt.scatter(initial_state[0],initial_state[1], marker = 'o', label = 'start', color = 'black', s=40)
    plt.scatter(target_state[0], target_state[1], marker = 'x', label = 'target', color = 'black', s=40)
    plt.legend(loc = 'lower right')
    plt.savefig('../../figures/potential.png', bbox_inches = 'tight')
    #plt.savefig('2D_simple_double_well.png')
    


    
