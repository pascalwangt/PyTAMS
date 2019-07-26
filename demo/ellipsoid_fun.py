# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:31:15 2019

@author: pasca
"""

import numpy as np
import functools

import sympy as sp


import scipy.linalg
import scipy.stats


"""
This module computes covariance matrices, eigenvectors, eigenvalues, associated quadratic form, confidence levels,
functions to test if a state is contained in an ellipsoid, array of points inside the ellispoid"""
    
def get_covariance_matrix(force_matrix, equilibrium_point, noise, noise_matrix=None, warning_threshold = 1e-5):
    """
    Input:
        force_matrix: sympy expression, with variables in ALPHABETICAL ORDER!!!
        noise: standard deviation of the noise, assumes isotropical noise
        
    """
    dims = len(equilibrium_point)       #dimensions of the system
    alphabet = list(sp.ordered(force_matrix.free_symbols))
    
    assert dims == len(alphabet), "Dimensions and number of symbols not equal"  #later: upgrade by auto completing dictionnary if variable muette 
    
    subs_dict = dict(zip(alphabet, equilibrium_point))
    
    J = force_matrix.jacobian(sp.Matrix(alphabet))
    A = J.evalf(subs=subs_dict)
    A = np.asarray(A).astype(np.float)
    
    if noise_matrix is None:
        Q = noise**2*np.identity(dims)
    else:
        Q = noise**2*np.dot(noise_matrix, noise_matrix.T)
    C = scipy.linalg.solve_lyapunov(A,-Q)
    
    residual = np.dot(A,C)+np.dot(C,A.T)+Q
    if np.linalg.norm(residual) > warning_threshold:
        print("Warning! Large residual")
        print(residual)
        
    inv_C = np.linalg.inv(C)
    
    def quad_form(vector):
        return np.linalg.multi_dot(((vector-equilibrium_point).T,inv_C, vector-equilibrium_point))
    
    return C, quad_form

def ingredients_score_function(force_matrix, equilibrium_point, noise, noise_matrix=None, confidence = 0.95, warning_threshold = 1e-3):
    dims = len(equilibrium_point)
    
    C, quad_form = get_covariance_matrix(force_matrix=force_matrix, equilibrium_point=equilibrium_point, noise=noise, warning_threshold=warning_threshold, noise_matrix=noise_matrix)
    
    eigvals = np.linalg.eigvals(C)
    spectral_radius = eigvals.max() #spectral radius of C^-1

    
    level = scipy.stats.chi2.ppf(confidence, dims) #corresponding to confidence level
    bound = np.sqrt(spectral_radius*level)
    
    return C, quad_form, spectral_radius, level, bound
    
    
    
    
    

def draw_ellipsoid_2D(force_matrix, equilibrium_point, noise, noise_matrix=None, color = 'black', confidence = 0.95, warning_threshold = 1e-3, verbose = False, resolution = 200, zorder = 1.5):
    
    """
    Only works in 2D, easily adaptable to 3D
    """
    import matplotlib.pyplot as plt
    dims = len(equilibrium_point)
    
    assert dims == 2, "Equilibrium point is not 2D"
    
    C, quad_form = get_covariance_matrix(force_matrix=force_matrix, equilibrium_point=equilibrium_point, noise=noise, warning_threshold=warning_threshold)
    
    eigvals = np.linalg.eigvals(C)
    spectral_radius = eigvals.max() #spectral radius of C^-1

    
    level = scipy.stats.chi2.ppf(confidence, dims) #corresponding to confidence level
    bound = np.sqrt(spectral_radius*level)
    
    if verbose: 
        print("Eigenvalues of C: ", eigvals)
        print("Bound: ", bound)
        
    xval = np.linspace(equilibrium_point[0]-bound,equilibrium_point[0]+bound,resolution)
    yval = np.linspace(equilibrium_point[1]-bound,equilibrium_point[1]+bound,resolution)

    xx,yy=np.meshgrid(xval,yval)
    field = np.array([xx,yy])
    
    ellipsoid = np.apply_along_axis(quad_form, 0, field)
    
    CS = plt.contour(xx,yy,ellipsoid, levels = [level], colors = [color], zorder = zorder)
    
    return CS


def get_ellipsoid_array(equilibrium_point, quad_form, level, bound, nb_points = 1e6, tolerance_factor=3000):
    """ can be optimized by changing the basis according to the eigenvectors
        returns array of shape (nb of points, nb of dimensions), need to transpose to plot"""
    print(f"Computing ellipsoid array ...")
    
    dim = len(equilibrium_point)
    resolution = nb_points**(1/float(dim))
    
    xi = np.linspace(-bound, bound, resolution)
    grids =np.meshgrid(*[xi+equilibrium_point[i] for i in range(dim)])
    grid = np.array(list(grids))
    
    quad_form_values = np.apply_along_axis(quad_form, 0, grid)
    #ell = np.where(np.abs(quad_form_values-level)<tolerance, 0,grid)
    tolerance = quad_form_values.mean()/tolerance_factor
    print(f"Quadratic form tolerance: {tolerance:.2e}")
    indices = np.where(np.abs(quad_form_values-level)<tolerance)
    
    new_grid = np.moveaxis(grid, 0, dim)
    ell = new_grid[indices]
    print(f"Found {ell.shape[0]} points in the ellipsoid array.")
    return ell

def get_ellipsoid_interior(equilibrium_point, quad_form, level, bound, nb_points = 1e6, sample = 200):
    """ can be optimized by changing the basis according to the eigenvectors
        returns array of shape (nb of points, nb of dimensions), need to transpose to plot"""
    print(f"Computing ellipsoid array ...")
    
    dim = len(equilibrium_point)
    resolution = nb_points**(1/float(dim))
    
    xi = np.linspace(-bound, bound, resolution)
    grids =np.meshgrid(*[xi+equilibrium_point[i] for i in range(dim)])
    grid = np.array(list(grids))
    
    quad_form_values = np.apply_along_axis(quad_form, 0, grid)
    #ell = np.where(np.abs(quad_form_values-level)<tolerance, 0,grid)


    new_grid = np.moveaxis(grid, 0, dim)
    print(new_grid.shape)
    ell = new_grid[quad_form_values<level]
    print(ell.shape)
    print(f"Found {ell.shape[0]} points in the ellipsoid array.")
    subsample = np.random.choice(ell.shape[0], size = sample)
    print(f'Using sample of size: {sample}')
    return ell[subsample]


def get_levelset_array(equilibrium_point, function, level, bound, resolution = 400, tolerance=1e-3 ):
    """ can be optimized by changing the basis according to the eigenvectors
        returns array of shape (nb of points, nb of dimensions), need to transpose to plot"""
        
    dim = len(equilibrium_point)
    xi = np.linspace(-bound, bound, resolution)
    grids =np.meshgrid(*[xi+equilibrium_point[i] for i in range(dim)])
    grid = np.array(list(grids))
    
    function_values = np.apply_along_axis(function, 0, grid)
    #ell = np.where(np.abs(quad_form_values-level)<tolerance, 0,grid)
    indices = np.where(np.abs(function_values-level)<tolerance)
    
    new_grid = np.moveaxis(grid, 0, dim)
    levelset = new_grid[indices]
    
    return levelset


    
    
def check_ellipsoid(ell, score_function, threshold, tolerance):
    difference = np.abs(np.apply_along_axis(score_function, 1, ell)-(1-threshold))
    return np.all(difference<tolerance)

def ellipsoid_test(v, quad_form, level):
    return quad_form(v)<level

def optimize_param(score_function_maker, initial_guess, ell, noise, threshold_param, nmultmax = 50, factor = 1.5, tolerance = 1e-3, reduction_dalpha=1000):
    
    param =  initial_guess
    print(f"Finding optimal value of normal score function decay with initial guess: alpha: {initial_guess:.3e} and tolerance: {tolerance}")
    
    nmult = 0
    bar = True
    while bar and nmult <= nmultmax:
        foo = check_ellipsoid(ell, score_function_maker(param = param, sigma = noise), threshold=threshold_param(param), tolerance=tolerance)
        if foo:
            niter = 0
            dalpha = param/reduction_dalpha
            print(f"New initial guess: param: {param:.3e} and increment {dalpha:.3e}")
            while foo:
                niter += 1
                param -= dalpha
                foo = check_ellipsoid(ell, score_function_maker( param = param, sigma = noise), threshold=threshold_param(param), tolerance=tolerance)
            param += dalpha
            bar = False
            print(f"Optimal value: {param:.3e} after {niter} iterations. \n")
        else:
            param *= factor
            nmult+=1
            print(f"Initial value of parameter is not valid, multiplying by {factor} ...")
    if bar:
        print("Optimal value search has failed.")
    return score_function_maker(param = param, sigma = noise), threshold_param(param)



def optimize_param_normal(score_function_param, initial_guess, ell, threshold_param, nmultmax = 50, factor = 1.5, tolerance = 1e-3, reduction_dalpha=1000):
    
    alpha_s =  initial_guess
    print(f"Finding optimal value of normal score function decay with initial guess: alpha: {initial_guess:.3e} and tolerance: {tolerance}")
    
    nmult = 0
    bar = True
    while bar and nmult <= nmultmax:
        foo = check_ellipsoid(ell, functools.partial(score_function_param, alpha_s = alpha_s), threshold=threshold_param(alpha_s), tolerance=tolerance)
        if foo:
            niter = 0
            dalpha = alpha_s/reduction_dalpha
            print(f"New initial guess: alpha: {alpha_s:.3e} and increment {dalpha:.3e}")
            while foo:
                niter += 1
                alpha_s -= dalpha
                foo = check_ellipsoid(ell, functools.partial(score_function_param, alpha_s = alpha_s), threshold=threshold_param(alpha_s), tolerance=tolerance)
            alpha_s+= dalpha
            bar = False
            print(f"Optimal value: {alpha_s:.3e} after {niter} iterations. \n")
        else:
            alpha_s *= factor
            nmult+=1
            print(f"Initial value of parameter is not valid, multiplying by {factor} ...")
    if bar:
        print("Optimal value search has failed.")
    return functools.partial(score_function_param, alpha_s=alpha_s), threshold_param(alpha_s)






