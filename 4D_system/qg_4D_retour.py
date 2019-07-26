# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:40:34 2019

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
import sympy.abc

import sys
sys.path.append('../')

import ellipsoid_fun
import trajectory_to_score_function
import h5py



#%%
# states
target_state = np.array([2.559085486905914e-01, 1.783308009875336e+00, -1.475035829596991e+00, -1.825982757733474e+00])
initial_state = np.array([-2.559085486905914e-01, 1.783308009875336e+00, 1.475035829596991e+00, -1.825982757733474e+00])
saddle_state = np.array([0.0000000000E+00, 1.7163920088E+00,  0.0000000000E+00, -2.2021308316E+00])

#%%
#system parameters
s=2 #unused
control_param = 4.9864471442E-01 #control parameter: region with 2 stable equilibria

dist =  np.linalg.norm(target_state-initial_state)
eta = np.linalg.norm(target_state-saddle_state)/np.linalg.norm(target_state-initial_state)

c1 = 0.020736
c2 = 0.018337
c3 = 0.015617
c4 = 0.031977
c5 = 0.036673
c6 = 0.046850 
c7 = 0.314802
l1 = 0.0128616
l2 = 0.0211107
l3 = 0.0318615
l4 = 0.0427787

def force(v):
    return np.array([c1*v[0]*v[1] + c2*v[1]*v[2]  + c3*v[2] *v[3]  - l1*v[0],
                     c4*v[1]*v[3] + c5*v[0]*v[2]   - c1*v[0]*v[0] - l2*v[1] + c7*control_param,
                     c6*v[0]*v[3] - (c2+c5)*v[0]*v[1]  - l3*v[2],
                     -c4*v[1]*v[1] - (c3+c6)*v[0]*v[2]  - l4*v[3]])
    
noise_matrix = np.array([[0., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.]])
    

#sympy force matrix
x1,x2,x3,x4 = sp.abc.symbols('x:4')

force_matrix = sp.Matrix([c1*x1*x2 + c2*x2*x3  + c3*x3 *x4  - l1*x1,
                     c4*x2*x4 + c5*x1*x3   - c1*x1*x1 - l2*x2 + c7*control_param,
                     c6*x1*x4 - (c2+c5)*x1*x2  - l3*x3,
                     -c4*x2*x2 - (c3+c6)*x1*x3  - l4*x4])

#%%
#reduction
sigma = 0.02
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



def score_function_ellipsoid_maker(param = 0.05, sigma=0.05):
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












#%% 
#tests
test = 0
if test:
    dim = 4
    bound = 3
    resolution = 10
    equilibrium_point = np.zeros(4)
    xi = np.linspace(-bound, bound, resolution)
    grids =np.meshgrid(*[xi+equilibrium_point[i] for i in range(dim)])
    grid = np.array(list(grids))
    
    function_values = np.apply_along_axis(score_function_simexp_ell_param, 0, grid)
    #ell = np.where(np.abs(quad_form_values-level)<tolerance, 0,grid)
    print(function_values.max(), function_values.min())

test1=0
if test1:
    ell = ellipsoid_fun.get_ellipsoid_array(target_state, quad_form, level, bound, resolution = 40, tolerance=3e-2 )
    foo = ellipsoid_fun.check_ellipsoid(ell, score_function=score_function_simexp_ell_param, threshold=threshold_simexp, tolerance=1e-3)
    print(f'foo = {foo}')