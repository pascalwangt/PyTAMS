# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:54:48 2019

@author: pasca
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import mpl_toolkits.axes_grid1.inset_locator as mai 
import h5py

import sys
sys.path.append('../')
import schemes
import functools
import ellipsoid_fun

def draw_projection(initial_state, target_state, indices, proj, noise_matrix, lims = [2,2], function=None, sigma = None, force_matrix = None, ):

    dim = len(initial_state)
    
    plt.figure()
    index1, index2 = indices
    rest1, rest2 = [idx for idx in range(dim) if idx not in indices]
    
    binslist = [None]*dim
    
    binslist[rest1] = np.array([proj[rest1]])
    binslist[rest2] = np.array([proj[rest2]])
    
    binslist[index1] = np.linspace(proj[index1]-lims[0],proj[index1]+lims[0],300)
    binslist[index2] = np.linspace(proj[index2]-lims[1],proj[index2]+lims[1],300)
    
    
    a0, a1, a2, a3 = binslist
    

    
    
    grids = list(np.meshgrid(a0,a1,a2,a3))
    

    
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



def histogram_proj(histogram, binslist, indices, trajectory = None, initial_state = None, target_state = None, save_path = None):
    index1, index2 = indices
    xbins = binslist[index1]
    ybins = binslist[index2]
    fig = plt.figure(figsize = (1.05*3.19, 1.05*2.61))
    ax = plt.gca()
    proj_histogram = np.sum(histogram, axis = tuple([i for i in range(len(binslist)) if i not in indices]))+1
    #plot histogram
    im = ax.pcolormesh(xbins, ybins, proj_histogram.T, norm=mpl.colors.LogNorm(proj_histogram.min()+1, vmax=proj_histogram.max()))
    locator = mpl.ticker.LogLocator(base=10) 
    fig.colorbar(im, ticks = locator)
    
    if trajectory is not None:
        plt.plot(trajectory[index1], trajectory[index2], label = None, color = 'red')
    
    #plot initial and target state
    if initial_state is not None:
        plt.scatter(initial_state[index1], initial_state[index2], marker = 'o', label = '$X_A$', s = 40, color = 'black')
    if target_state is not None:
        plt.scatter(target_state[index1], target_state[index2], marker = 'x', label = '$X_B$', s= 40, color = 'black')
    
    plt.xlabel('$A_'+str(index1+1)+'$')
    plt.ylabel('$A_'+str(index2+1)+'$')
    plt.tick_params(which = 'both', direction = 'out')

    if indices == [0,2]:
        plt.xlim(-0.7,0.7)
    plt.legend(loc = 'lower left')
    plt.text(-0.2, 1, '(d)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()

def check(time_array, force, initial_state, target_state, sigma1, noise_matrix, force_matrix,
         index_time = [0,2], index_traj = [0,2], label_time = ['x', 'y'], label_traj = ['x', 'y'],
         score_function = None, xmin = None, xmax = None, ymin = None, ymax = None, save_path = None):
    plt.figure()
    dim = len(initial_state)
    indices = [0,2]
    index1, index2 = indices
    rest1, rest2 = [idx for idx in range(dim) if idx not in indices]
        
        
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma1, noise_matrix)
    
    solver_scheme = functools.partial(schemes.Euler_Maruyama_no_stop, force=force, time_array_length=len(time_array), dt=0.01, dims = 4, noise_matrix = noise_matrix)
    
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
    #plt.contour(np.squeeze(grids[index1]), np.squeeze(grids[index2]), ell, levels = [level], colors = ['black'], label = 'confidence ellipsoid')
    
    ell = ellipsoid_fun.get_ellipsoid_interior(target_state, quad_form_target, level, bound, nb_points = 1e6, sample = 200)
    for elem in ell:
        
        vs = solver_scheme(0,elem, 0)
        plt.plot(vs[:,index_traj[0]], vs[:,index_traj[1]], linewidth = 0.1, color = 'C1')
        plt.scatter(vs[-1,index_traj[0]], vs[-1,index_traj[1]], linewidth = 0.1, s=30, color = 'red', zorder= 10)
        plt.scatter(vs[0,index_traj[0]], vs[0,index_traj[1]], linewidth = 0.1, s=7, color = 'green', zorder =10)
    
    
    #plt.title(r'$\sigma$'+' = {}'.format(sigma1))
    plt.xlabel(label_traj[0])
    plt.ylabel(label_traj[1])
    plt.ylim((ymin,ymax))
    plt.xlim(xmin, xmax)
    
    plt.scatter(initial_state[index_traj[0]], initial_state[index_traj[1]], marker = 'o', label = '$X_A$', s = 20, color = 'black', zorder  = 50)
    plt.scatter(target_state[index_traj[0]], target_state[index_traj[1]], marker = 'x', label = '$X_B$',color = 'black', s= 20, zorder = 50)
    
    plt.legend()
    if save_path is not None:
        
        plt.savefig(save_path)
        
def trajectory(time_array, force, initial_state, target_state, sigma1, noise_matrix, force_matrix, geom_thresh = 1e-2,
                                     index_time = [0,1], index_traj = [0,1], label_time = ['x', 'y'], label_traj = ['x', 'y'],
                                     score_function = None, xmin = None, xmax = None, ymin = None, ymax = None, save_path = None):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    plt.figure()

    solver_scheme = functools.partial(schemes.Euler_Maruyama_no_stop, force=force, time_array_length=len(time_array), dt=0.01, dims = 4, noise_matrix = noise_matrix)
    
    #solve first trajectory
    vs = solver_scheme(0,initial_state, sigma1)
    
    reached = np.apply_along_axis(lambda v: np.linalg.norm(v-target_state)<geom_thresh, 1, vs)
    
    
    
    plt.plot(vs[:,index_traj[0]], vs[:,index_traj[1]], linewidth = 0.1, color = 'C1')
    plt.title(r'$\sigma$'+' = {}'.format(sigma1))
    plt.ylabel(label_traj[1])
    plt.ylim((ymin,ymax))
    plt.xlim(xmin, xmax)
    
    plt.scatter(initial_state[index_traj[0]], initial_state[index_traj[1]], marker = 'o', label = 'initial', s = 40, color = 'black')
    plt.scatter(target_state[index_traj[0]], target_state[index_traj[1]], marker = 'x', label = 'target', s= 40)
    
    if np.any(reached):
        reached_index = np.argmax(reached)
        print(f'Reached target at {time_array[reached_index]}')
        plt.scatter(vs[reached_index,index_traj[0]], vs[reached_index,index_traj[1]], marker = 'o', s = 40, label = 'reached')
    else:
        print(np.apply_along_axis(lambda v: np.linalg.norm(v-target_state), 1, vs).min())
        print(np.apply_along_axis(lambda v: np.linalg.norm(v), 1, vs[1:]-vs[:-1]).max())
        print(np.apply_along_axis(lambda v: np.linalg.norm(v), 1, vs[1:]-vs[:-1]).mean())
        print(np.apply_along_axis(lambda v: np.linalg.norm(v), 1, vs[1:]-vs[:-1]).min())
    
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
    if np.any(reached):
        ax.scatter(time_array[reached_index], vs[reached_index,2], s= 5)
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
    if np.any(reached):
        ax.scatter(time_array[reached_index], vs[reached_index,0], s= 5)
    
    ax = plt.subplot(224)
    ax.plot(time_array, initial_state[3]+np.zeros((len(time_array))), linestyle = '-')
    ax.plot(time_array, target_state[3]+np.zeros((len(time_array))), linestyle = '--',  dashes=(3.5, 3.5))
    ax.plot(time_array, vs[:,3], linewidth = 0.3)
    ax.set_ylabel('$A_4$')
    ax.set_xlabel('time')
    if np.any(reached):
        ax.scatter(time_array[reached_index], vs[reached_index,3], s= 5)
        
    ax = plt.subplot(223, sharex = ax)
    ax.plot(time_array, initial_state[1]+np.zeros((len(time_array))), linestyle = '-')
    ax.plot(time_array, target_state[1]+np.zeros((len(time_array))), linestyle = '--',  dashes=(3.5, 3.5))
    ax.plot(time_array, vs[:,1], linewidth = 0.3)
    ax.set_ylabel('$A_2$')
    ax.set_xlabel('time')
    if np.any(reached):
        ax.scatter(time_array[reached_index], vs[reached_index,1], s= 5)
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    


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
    
    plt.figure()
    ax = plt.gca()
    bound = np.abs(psi).max()
    bound = 0.7
    levels = np.linspace(-bound, bound, res)
    contf = plt.contourf(x,y,psi,levels=levels, cmap = 'RdBu_r')
    
    plt.grid()
    
    plt.xlabel('x', labelpad=-2)
    plt.ylabel('y')
    
    plt.xticks([0, np.pi/3, 2*np.pi/3, np.pi], ['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', '$\pi$'])
    plt.yticks([0, np.pi/3, 2*np.pi/3, np.pi], ['0',  r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', '$\pi$'])
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.text(-0.15, 1, '(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    plt.contour(x,y,psi,levels=levels, colors = 'black', linewidths = 0.5)
    cbax = mai.inset_axes(ax, width = "3%", height = "85%", loc = 'right', bbox_to_anchor=(-0.13, -0.03, 1, 1),
                          bbox_transform=ax.transAxes,
                          borderpad=0)
    cbax.set_title('$\psi$')
    CB = plt.colorbar(contf, cax = cbax, ticks = np.arange(-0.6, 1, 0.2), format='%.1f')
    #CB.ax.get_children()[2].set_linewidths([0.1]*10)
    plt.savefig(savepath, bbox_inches = 'tight')
    plt.show()
    

