# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:54:48 2019

@author: pasca
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import mpl_toolkits.axes_grid1.inset_locator as mai 
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../')
import schemes
import functools
import ellipsoid_fun

def plot_ellipsoid(ell_form_array, level, center, ax, color ='black', label = 'covariance ellipsoid'):

    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(ell_form_array)
    radii = np.sqrt(level)/np.sqrt(s)
    
    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
            
    surf = ax.plot_surface(y,x, z,  rstride=4, cstride=4, color=color, alpha=0.1, label = label)
    ax.plot_wireframe(y,x, z,  rstride=10, cstride=10, color=color, alpha=0.4, linewidth = 1)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d


def plot_trajectory(time_array, force, initial_state, target_state, sigma1, noise_matrix, force_matrix,
                    xmin = -15, xmax = 15, ymin = -15, ymax = 15, zmin = 0, zmax = 15, save_path = None, vs = None):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_facecolor('white')
    ax.tick_params(axis='both', which='major', pad=-2)
    
    if vs is None:
        solver_scheme = functools.partial(schemes.Euler_Maruyama_no_stop, force=force, time_array_length=len(time_array), dt=0.01, dims = 3, noise_matrix = noise_matrix) 
        vs = solver_scheme(0,initial_state, sigma1)
    
    
    ax.plot(vs[:,1], vs[:,0], zs=vs[:,2], zdir = 'z', linewidth = 0.2, color = 'C1')
    
    #plt.title(r'$\sigma$'+' = {}'.format(sigma1))
    ax.set_ylabel('x', labelpad = -7)
    ax.set_xlabel('y', labelpad = -7)
    ax.set_zlabel('z', labelpad = -7)
    
    ax.set_xlim(xmin, xmax)
    ax.set_zlim(zmin, zmax)
    ax.set_ylim(ymin, ymax)
    
    
    plt.scatter(initial_state[1], initial_state[0], zs=initial_state[2], marker = 'o', label = '$X_A$', s = 40, color = 'black', zorder = 1000)
    plt.scatter(target_state[1], target_state[0], zs=target_state[2] ,marker = 'x', label = '$X_B$', s= 40, color = 'black', zorder = 1000)
    
    
    if force_matrix is not None:
        covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma1, noise_matrix)
        plot_ellipsoid(np.linalg.inv(covariance_matrix_target), level, target_state, ax, color = 'C0', label = None)
        
        covariance_matrix_initial, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma1, noise_matrix)
        plot_ellipsoid(np.linalg.inv(covariance_matrix_initial), level, initial_state, ax, label = None, color = 'C0')
    
    plt.scatter(initial_state[1], initial_state[0], zs=initial_state[2], marker = 'o', label = '$X_A$', s = 40, color = 'black', zorder = 1000)
    plt.scatter(target_state[1], target_state[0], zs=target_state[2] ,marker = 'x', label = '$X_B$', s= 40, color = 'black', zorder = 1000)
    #plt.legend()
    
    plt.xticks([-5, 0,5])
    plt.yticks([-5, 0,5])
    ax.set_zticks([ 0,5,10,15])
    
    ax.text2D(0.1, 0.9, '(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    if save_path is not None:        
        plt.savefig(save_path)

    plt.show()


def check(time_array, force, initial_state, target_state, sigma1, noise_matrix, force_matrix,
          xmin = -15, xmax = 15, ymin = -15, ymax = 15, zmin = 0, zmax = 15, save_path = None):
    
    
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    solver_scheme = functools.partial(schemes.Euler_Maruyama_no_stop, force=force, time_array_length=len(time_array), dt=0.01, dims = 3, noise_matrix = noise_matrix) 
    
    plt.title(r'$\sigma$'+' = {}'.format(sigma1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    

    
    
    plt.scatter(initial_state[0], initial_state[1], zs=initial_state[2], marker = 'o', label = 'initial', s = 40, color = 'black')
    plt.scatter(target_state[0], target_state[1], zs=target_state[2] ,marker = 'x', label = 'target', s= 40)
    
    
    if force_matrix is not None:
        covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma1, noise_matrix, confidence=0.99)
        plot_ellipsoid(np.linalg.inv(covariance_matrix_target), level, target_state, ax, color = 'C0')
        
        covariance_matrix_initial, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma1, noise_matrix, confidence = 0.99)
        plot_ellipsoid(np.linalg.inv(covariance_matrix_initial), level, initial_state, ax, label = None)
        
        print(level)
        
        ell = ellipsoid_fun.get_ellipsoid_interior(target_state, quad_form_target, level, bound, nb_points = 1e6, sample = 100)
        for elem in ell:
            vs = solver_scheme(0,elem, 0)
            plt.plot(vs[:,0], vs[:,1], vs[:,2], linewidth = 0.1, color = 'C1')
            plt.scatter(vs[-1,0], vs[-1,1], zs = vs[-1:2],  s=10, color = 'red')
            plt.scatter(vs[0,0], vs[0,1], zs = vs[0,2], s=10, color = 'green')
            print(quad_form_target(elem), level)
        
    plt.legend()

    if save_path is not None:        
        plt.savefig(save_path)
    ax.set_xlim(xmin, xmax)
    ax.set_zlim(zmin, zmax)
    ax.set_ylim(ymin, ymax)
    plt.show()
     
     
     
     
     
     


def draw_projection(initial_state, target_state, indices, proj, noise_matrix, lims = [2,2], function=None, sigma = None, force_matrix = None, ):

    dim = len(initial_state)
    
    plt.figure()
    index1, index2 = indices
    [rest1] = [idx for idx in range(dim) if idx not in indices]
    
    binslist = [None]*dim
    
    binslist[rest1] = np.array([proj[rest1]])

    
    binslist[index1] = np.linspace(proj[index1]-lims[0],proj[index1]+lims[0],300)
    binslist[index2] = np.linspace(proj[index2]-lims[1],proj[index2]+lims[1],300)
    
    
    a0, a1, a2 = binslist
    

    
    
    grids = list(np.meshgrid(a0,a1,a2))
    

    
    if function is not None:
        values = np.apply_along_axis(function, 0, np.array(grids))

        values = np.squeeze(values)
        im = plt.contourf(np.squeeze(grids[index1]), np.squeeze(grids[index2]), values, 50)
        plt.colorbar(im)
    
    
    
    if initial_state is not None:
        plt.scatter(initial_state[index1], initial_state[index2], marker = 'o', label = 'initial', s = 40, color = 'black')
    if target_state is not None:
        plt.scatter(target_state[index1], target_state[index2], marker = 'x', label = 'target', s= 40)
        
    if sigma is not None and force_matrix is not None:
        covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma, noise_matrix)
        ell = np.apply_along_axis(quad_form_target, 0, np.array(grids))
        ell = np.squeeze(ell)
        plt.contour(np.squeeze(grids[index1]), np.squeeze(grids[index2]), ell, levels = [level], colors = ['black'], label = 'confidence ellipsoid')
    plt.legend()
    plt.show()



def histogram_proj(histogram, binslist, indices, trajectory = None, instanton = None, initial_state = None, target_state = None, save_path = None):
    index1, index2 = indices
    xbins = binslist[index1]
    ybins = binslist[index2]
    fig, ax = plt.subplots()
    
    proj_histogram = np.sum(histogram, axis = tuple([i for i in range(len(binslist)) if i not in indices]))+1
    #plot histogram
    im = ax.pcolormesh(xbins, ybins, proj_histogram.T, norm=mpl.colors.LogNorm(proj_histogram.min()+1, vmax=proj_histogram.max()))
    locator = mpl.ticker.LogLocator(base=10) 
    fig.colorbar(im, ticks = locator)
    
    if trajectory is not None:
        plt.plot(trajectory[index1], trajectory[index2], label = 'estimation', color = 'red')
        
    if instanton is not None:
        plt.plot(instanton[index1], instanton[index2], label = 'instanton', color = 'C0')
    
    #plot initial and target state
    if initial_state is not None:
        plt.scatter(initial_state[index1], initial_state[index2], marker = 'o', label = '$X_A$', s = 40, color = 'black')
    if target_state is not None:
        plt.scatter(target_state[index1], target_state[index2], marker = 'x', label = '$X_B$', s= 40, color = 'black')
    
    plt.tick_params(which = 'both', direction = 'out')
        
    plt.legend(loc = 'upper left')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


    
def trajectory(time_array, force, initial_state, target_state, sigma1, noise_matrix, force_matrix,
                                     index_time = [0,1], index_traj = [0,1], label_time = ['x', 'y'], label_traj = ['x', 'y'],
                                     score_function = None, xmin = None, xmax = None, ymin = None, ymax = None, save_path = None):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    plt.figure()

    solver_scheme = functools.partial(schemes.Euler_Maruyama_no_stop, force=force, time_array_length=len(time_array), dt=0.01, dims = 4, noise_matrix = noise_matrix) 
    
    #solve first trajectory
    vs = solver_scheme(0,initial_state, sigma1)
    
    
    plt.plot(vs[:,index_traj[0]], vs[:,index_traj[1]], linewidth = 0.1, color = 'C1')
    plt.title(r'$\sigma$'+' = {}'.format(sigma1))
    plt.ylabel(label_traj[1])
    plt.ylim((ymin,ymax))
    plt.xlim(xmin, xmax)
    
    plt.scatter(initial_state[index_traj[0]], initial_state[index_traj[1]], marker = 'o', label = 'initial', s = 40, color = 'black')
    plt.scatter(target_state[index_traj[0]], target_state[index_traj[1]], marker = 'x', label = 'target', s= 40)
    
    
    if force_matrix is not None:
        dim = len(initial_state)
        indices = [0,2]
        index1, index2 = indices
        rest1, rest2 = [idx for idx in range(dim) if idx not in indices]
        
       
        
        
        
        covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma1, noise_matrix)
        proj = target_state
        lims=bound
        binslist = [None]*dim
        
        binslist[rest1] = np.array([proj[rest1]])
        binslist[rest2] = np.array([proj[rest2]])
        
        binslist[index1] = np.linspace(proj[index1]-lims,proj[index1]+lims,300)
        binslist[index2] = np.linspace(proj[index2]-lims,proj[index2]+lims,300)
        
        a0, a1, a2, a3 = binslist
        
        grids = list(np.meshgrid(a0,a1,a2,a3))
        
        ell = np.apply_along_axis(quad_form_target, 0, np.array(grids))
        ell = np.squeeze(ell)
        plt.contour(np.squeeze(grids[index1]), np.squeeze(grids[index2]), ell, levels = [level], colors = ['black'], label = 'confidence ellipsoid')
        
        
        covariance_matrix_initial, quad_form_initial, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma1, noise_matrix)
        proj = initial_state
        lims = bound
        binslist = [None]*dim
        
        binslist[rest1] = np.array([proj[rest1]])
        binslist[rest2] = np.array([proj[rest2]])
        
        binslist[index1] = np.linspace(proj[index1]-lims,proj[index1]+lims,300)
        binslist[index2] = np.linspace(proj[index2]-lims,proj[index2]+lims,300)
        
        a0, a1, a2, a3 = binslist
        
        grids = list(np.meshgrid(a0,a1,a2,a3))
        
        ell = np.apply_along_axis(quad_form_initial, 0, np.array(grids))
        ell = np.squeeze(ell)
        plt.contour(np.squeeze(grids[index1]), np.squeeze(grids[index2]), ell, levels = [level], colors = ['black'], label = 'confidence ellipsoid')
        
    plt.legend()

    if save_path is not None:
        
        plt.savefig(save_path)
    plt.show()
    
    plt.figure()

    
    
    #solve trajectory
    
    
    
    ax = plt.subplot(222)
    ax.plot(time_array, initial_state[2]+np.zeros((len(time_array))), linestyle = '--')
    ax.plot(time_array, target_state[2]+np.zeros((len(time_array))), linestyle = '--')
    ax.plot(time_array, vs[:,2], linewidth = 0.3)
    ax.set_ylabel('$A_3$')
    plt.setp(ax.get_xticklabels(), visible=False)

    
    ax = plt.subplot(221, sharex=ax)
    ax.plot(time_array, initial_state[0]+np.zeros((len(time_array))), linestyle = '--', label = 'initial state')
    ax.plot(time_array, target_state[0]+np.zeros((len(time_array))),  linestyle = '--', label = 'target state')
    ax.plot(time_array, vs[:,0], linewidth = 0.3)
    ax.set_ylabel('$A_1$')
    ax.legend(fontsize = 'xx-small')
    plt.setp(ax.get_xticklabels(), visible=False)
    
    ax = plt.subplot(224)
    ax.plot(time_array, initial_state[3]+np.zeros((len(time_array))), linestyle = '-')
    ax.plot(time_array, target_state[3]+np.zeros((len(time_array))), linestyle = '--',  dashes=(3.5, 3.5))
    ax.plot(time_array, vs[:,3], linewidth = 0.3)
    ax.set_ylabel('$A_4$')
    ax.set_xlabel('time')
    
    ax = plt.subplot(223, sharex = ax)
    ax.plot(time_array, initial_state[1]+np.zeros((len(time_array))), linestyle = '-')
    ax.plot(time_array, target_state[1]+np.zeros((len(time_array))), linestyle = '--',  dashes=(3.5, 3.5))
    ax.plot(time_array, vs[:,1], linewidth = 0.3)
    ax.set_ylabel('$A_2$')
    ax.set_xlabel('time')
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
    print([f'{elem:.15e}' for elem in vs[-1]])

def trajectory_time_plot_compare_sigma(time_array, force, initial_state, target_state, sigma1, sigma2,
                                     index_time = [0,1], index_traj = [0,1], label_time = ['x', 'y'], label_traj = ['x', 'y'],
                                     score_function = None, xmin = None, xmax = None, ymin = None, ymax = None, save_path = None):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    fig, axes = plt.subplots(2, 2, sharex = True)

    solver_scheme = functools.partial(schemes.Euler_Maruyama_no_stop, force=force, time_array_length=len(time_array), dt=0.01)
    
    #solve first trajectory
    ndt, vs = solver_scheme(0,initial_state, sigma1)
    
    axes[0,0].plot(time_array, vs[:,index_time[0]], linewidth = 0.3, label = label_time[0])
    axes[0,0].set_ylabel(label_time[0])
    axes[0,0].twinx().plot(time_array, vs[:,index_time[1]], linewidth = 0.3, label = label_time[1])
    axes[0,0].twinx().set_ylabel(label_time[1])
    
    axes[0,0].set_title(r'$\sigma$'+' = {}'.format(sigma1))
    axes[0,0].set_ylim((ymin,ymax))
    axes[0,0].legend()
    
    axes[0,1].plot(vs[:,index_traj[0]], vs[:,index_traj[1]], linewidth = 0.1, color = 'C1')
    axes[0,1].set_title(r'$\sigma$'+' = {}'.format(sigma1))
    axes[0,1].set_ylabel(label_traj[1])
    axes[0,1].set_ylim((ymin,ymax))
    
    axes[0,1].scatter(initial_state[index_traj[0]], initial_state[index_traj[1]])
    axes[0,1].scatter(target_state[index_traj[0]], target_state[index_traj[1]])

        
        
        
        

    #solve second trajectory
    ndt, vs = solver_scheme(0,initial_state, sigma2)
    
    axes[1,0].plot(time_array, vs[:,index_time[0]], linewidth = 0.3, label = label_time[0])
    axes[1,0].twinx().plot(time_array, vs[:,index_time[1]], linewidth = 0.3, label = label_time[1])
    axes[1,0].set_title(r'$\sigma$'+' = {}'.format(sigma2))
    axes[1,0].set_xlabel('t[s]')
    axes[1,0].set_ylabel(label_time[0])
    axes[1,0].twinx().set_ylabel(label_time[1])
    axes[1,0].set_ylim((xmin,xmax))
    
    axes[1,1].plot(vs[:,index_traj[0]], vs[:,index_traj[1]], linewidth = 0.1, color = 'C1')
    
    axes[1,1].scatter(initial_state[index_traj[0]], initial_state[index_traj[1]])
    axes[1,1].scatter(target_state[index_traj[0]], target_state[index_traj[1]])
    
    axes[1,1].set_title(r'$\sigma$'+' = {}'.format(sigma2))
    axes[1,1].set_ylabel(label_traj[1])
    
    axes[1,1].set_xlabel(label_traj[0])
    
    axes[1,1].set_xlim((xmin,xmax))
    axes[1,1].set_ylim((ymin,ymax))
    
    
   
    
    plt.subplots_adjust(wspace = 0.4)
    if save_path is not None:
        
        plt.savefig(save_path)
    plt.show()

    
def trajectory_plot(time_array, force, initial_state, target_state, sigma, save_path=None):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
    solver_scheme = functools.partial(schemes.Euler_Maruyama_no_stop, force=force, time_array_length=len(time_array), dt=0.01, dims = 4)
    
    vs = solver_scheme(0,initial_state, sigma)
    plt.figure()

    
    
    #solve trajectory
    
    
    
    ax = plt.subplot(223)
    ax.plot(time_array, initial_state[2]+np.zeros((len(time_array))), linestyle = '--')
    ax.plot(time_array, target_state[2]+np.zeros((len(time_array))), linestyle = '--')
    ax.plot(time_array, vs[:,2], linewidth = 0.3)
    ax.set_ylabel('$A_3$')
    ax.set_xlabel('time')
    
    ax = plt.subplot(221, sharex=ax)
    ax.plot(time_array, initial_state[0]+np.zeros((len(time_array))), linestyle = '--', label = 'initial state')
    ax.plot(time_array, target_state[0]+np.zeros((len(time_array))),  linestyle = '--', label = 'target state')
    ax.plot(time_array, vs[:,0], linewidth = 0.3)
    ax.set_ylabel('$A_1$')
    ax.legend(fontsize = 'xx-small')
    plt.setp(ax.get_xticklabels(), visible=False)
    
    ax = plt.subplot(224)
    ax.plot(time_array, initial_state[3]+np.zeros((len(time_array))), linestyle = '-')
    ax.plot(time_array, target_state[3]+np.zeros((len(time_array))), linestyle = '--',  dashes=(3.5, 3.5))
    ax.plot(time_array, vs[:,3], linewidth = 0.3)
    ax.set_ylabel('$A_4$')
    ax.set_xlabel('time')
    
    ax = plt.subplot(222, sharex = ax)
    ax.plot(time_array, initial_state[1]+np.zeros((len(time_array))), linestyle = '-')
    ax.plot(time_array, target_state[1]+np.zeros((len(time_array))), linestyle = '--',  dashes=(3.5, 3.5))
    ax.plot(time_array, vs[:,1], linewidth = 0.3)
    ax.set_ylabel('$A_2$')
    plt.setp(ax.get_xticklabels(), visible=False)
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
    print([f'{elem:.15e}' for elem in vs[-1]])
    
def plot_psi(v, res=25, savepath = '../figures/psi.png'):
    s = 2
    x,y=np.linspace(0,np.pi,100), np.linspace(0,np.pi,100)
    x, y = np.meshgrid(x,y)
    phi1 = np.exp(-s*x)*np.sin(x)*np.sin(y)
    phi2 = np.exp(-s*x)*np.sin(x)*np.sin(2*y)
    phi3 = np.exp(-s*x)*np.sin(x)*np.sin(3*y)
    phi4 = np.exp(-s*x)*np.sin(x)*np.sin(4*y)
    
    psi = v[0]*phi1+v[1]*phi2+v[2]*phi3+v[3]*phi4
    print(psi.max(), psi.min())
    
    fig, ax = plt.subplots()
    bound = np.abs(psi).max()
    levels = np.linspace(-bound, bound, res)
    contf = plt.contourf(x,y,psi,levels=levels, cmap = 'RdBu_r')
    plt.grid()
    
    plt.contour(x,y,psi,levels=levels, colors = 'black')
    cbax = mai.inset_axes(ax, width = "3%", height = "85%", loc = 'right', bbox_to_anchor=(-0.13, -0.03, 1, 1),
                          bbox_transform=ax.transAxes,
                          borderpad=0)
    cbax.set_title('$\psi$')
    plt.colorbar(contf, cax = cbax, ticks = np.arange(-0.6, 1, 0.2), format='%.1f')
    plt.savefig(savepath)
    plt.show()
    

