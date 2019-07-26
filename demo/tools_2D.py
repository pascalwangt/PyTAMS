# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:46:31 2019

@author: pasca
"""
import numpy as np
import matplotlib.pyplot as plt
import functools

import ellipsoid_fun
import schemes


def visualise_scorefunction(score_function, initial_state, target_state, sigma=None, nb_levels = 25, score_threshold=None, score_thresholds = None, colors = None, force_matrix=None, save_path=None, xmin = -10, xmax = 10, ymin = -10, ymax = 15, new_figure = True):
    
    if new_figure:
         #fig = plt.figure(figsize = (1.18*3.19, 1.18*2.61))
         fig = plt.figure(figsize = (1.02*3.19, 1.02*2.61))
    #coordinates
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,200), np.linspace(ymin,ymax,200))
    
    #score function levels sets
    score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))
    
    #im = plt.contour(xx, yy, score_levels, 50)
    im = plt.contourf(xx, yy, score_levels, levels = np.linspace(0,1,nb_levels))
    plt.contour(xx, yy, score_levels, levels = np.linspace(0,1,nb_levels), linewidths = 0.4, colors = 'grey')
    plt.grid()
    cbar = plt.colorbar(im, format = '%.2f')
    cbar.ax.set_title('$\phi_{ell}^Z$')

    
    if score_threshold is not None:
        #score function threshold for the TAMS algorithm stopping criterion
        CS = plt.contour(xx, yy, score_levels, levels = [1-score_threshold], zorder = 2, linestyles = 'dashed', colors = ['red'])
        #CS.collections[0].set_label('stopping threshold')
        plt.legend()
        
    if score_thresholds is not None:
        for score_threshold, color in zip(score_thresholds, colors):
            #score function threshold for the TAMS algorithm stopping criterion
            CS = plt.contour(xx, yy, score_levels, levels = [1-score_threshold], zorder = 2, linestyles = 'dashed', colors = [color])
            #CS.collections[0].set_label('$\phi_{target}$ = '+str(1-score_threshold))
            plt.legend()
        
        
    if force_matrix is not None and sigma is not None:
        CS = ellipsoid_fun.draw_ellipsoid_2D(force_matrix, equilibrium_point=target_state, noise=sigma, zorder = 1)
        CS.collections[0].set_label('confidence ellipsoid')
        plt.legend(loc = 'lower left')
    plt.scatter(initial_state[0],initial_state[1], marker = 'o',  color = 'black', s=40, zorder = 10)
    plt.scatter(target_state[0], target_state[1], marker = 'x', color = 'black', s=40, zorder = 10)
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.text(-0.2, 1,'(b)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight')
        
    
def visualise_scorefunction_level(score_function, score_threshold, label = 'limit levelset', color = 'blue', save_path=None,xmin = -10, xmax = 10, ymin = -10, ymax = 15, new_figure = True):
    
    if new_figure:
        fig = plt.figure(figsize = (1.05*3.19, 1.05*2.61))
    #coordinates
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,200), np.linspace(ymin,ymax,200))
    
    #score function levels sets
    score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))

    #score function threshold for the TAMS algorithm stopping criterion
    CS = plt.contour(xx, yy, score_levels, levels = [1-score_threshold], zorder = 2, linestyles = 'dashed', colors = [color])
    #CS.collections[0].set_label(label)
    plt.legend()
    
def trajectory_plot_report1(time_array, dt, force, initial_state, target_state, sigma1, save_path, force_matrix=None, score_threshold=None, score_function = None, xmin = -10, xmax = 10, ymin = -10, ymax = 15):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    fig, ax = plt.subplots(1, 1)
    x,y  = np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100)
    xx, yy = np.meshgrid(x, y)
    
    
    #solve first trajectory
    vs = schemes.Euler_Maruyama_no_stop(0,initial_state, sigma1, dt=dt, dims = 2, force=force, time_array_length=len(time_array))
    
    ax.plot(vs[:,0], vs[:,1], linewidth = 0.03, color = 'C0')
    
    print(np.sum(vs[:,0]*vs[:,1])/np.sqrt(np.sum(vs[:,0]*vs[:,0])*np.sum(vs[:,1]*vs[:,1])))
    
    vs2 = schemes.Euler_Maruyama_no_stop(0,target_state, sigma1, dt=dt, dims = 2, force=force, time_array_length=len(time_array))
    ax.plot(vs2[:,0], vs2[:,1], linewidth = 0.03, color = 'C1')
    #ax.set_title(r'$\sigma$'+' = {}'.format(sigma1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim((ymin,ymax))
    ax.set_xlim((xmin,xmax))
    
    plt.scatter(initial_state[0],initial_state[1], marker = 'o',  color = 'black', s=40, zorder = 10)
    plt.scatter(target_state[0], target_state[1], marker = 'x', color = 'black', s=40, zorder = 10)
    
    
    
    #threshold and stopping criterion
    if score_function is not None:
        #score function threshold for the TAMS algorithm stopping criterion
        score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))
        ax.contour(xx, yy, score_levels, levels = [1-score_threshold], zorder = 10)
    
    if force_matrix is not None:
        ellipsoid_fun.draw_ellipsoid_2D(force_matrix, target_state, noise=sigma1, zorder = 20)
        CS = ellipsoid_fun.draw_ellipsoid_2D(force_matrix, initial_state, noise=sigma1, zorder = 20)
        CS.collections[0].set_label('confidence ellipsoid')
        
        covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, initial_state, sigma1)
        target_test = functools.partial(ellipsoid_fun.ellipsoid_test, quad_form=quad_form_target, level=level)
        
        a = np.apply_along_axis(target_test, 1, vs)
        print(len(a[a==True])/len(a))
        
        covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, target_state, sigma1)
        target_test = functools.partial(ellipsoid_fun.ellipsoid_test, quad_form=quad_form_target, level=level)
        
        b = np.apply_along_axis(target_test, 1, vs2)
        print(len(b[b==True])/len(b))
        
        
    plt.legend()
    plt.text(-0.2, 1,'(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()

def trajectory_plot(time_array, dt, force, initial_state, target_state, sigma1,  save_path, sigma2=None, force_matrix=None, score_threshold=None, score_function = None, xmin = -10, xmax = 10, ymin = -10, ymax = 15):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    fig, ax = plt.subplots(1, 1)
    x,y  = np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100)
    xx, yy = np.meshgrid(x, y)
    
    
    #solve first trajectory
   
    if sigma2 is not None:
        vs = schemes.Euler_Maruyama_no_stop(0,initial_state, sigma2, dt=dt, dims = 2, force=force, time_array_length=len(time_array))
        ax.plot(vs[:,0], vs[:,1], linewidth = 0.5, color = 'C1')
    
    vs = schemes.Euler_Maruyama_no_stop(0,initial_state, sigma1, dt=dt, dims = 2, force=force, time_array_length=len(time_array))
    
    ax.plot(vs[:,0], vs[:,1], linewidth = 0.5, color = 'C0')
    
    
    #ax.set_title(r'$\sigma$'+' = {}'.format(sigma1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim((ymin,ymax))
    ax.set_xlim((xmin,xmax))
    
    plt.scatter(initial_state[0],initial_state[1], marker = 'o',  color = 'black', s=40, zorder = 10)
    plt.scatter(target_state[0], target_state[1], marker = 'x', color = 'black', s=40, zorder = 10)
    
    
    
    #threshold and stopping criterion
    if score_function is not None:
        #score function threshold for the TAMS algorithm stopping criterion
        score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))
        ax.contour(xx, yy, score_levels, levels = [1-score_threshold], zorder = 10)
    
    if force_matrix is not None:
        ellipsoid_fun.draw_ellipsoid_2D(force_matrix, target_state, noise=sigma1, zorder = 20)
        #CS.collections[0].set_label('confidence ellipsoid')

        
        
    #plt.legend()
    plt.text(-0.2, 1,'(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()
    
    
def trajectory_plot_report2(time_array, dt, force, initial_state, target_state, sigma1, sigma2, save_path, time_array2 = None, force_matrix=None, score_threshold=None, score_function = None, xmin = -10, xmax = 10, ymin = -10, ymax = 15):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    fig, ax = plt.subplots(1, 1)
    x,y  = np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100)
    xx, yy = np.meshgrid(x, y)
    
    
    #solve first trajectory
    vs = schemes.Euler_Maruyama_no_stop(0,initial_state, sigma1, dt=dt, dims = 2, force=force, time_array_length=len(time_array))
    
    ax.plot(vs[:,0], vs[:,1], linewidth = 0.5, color = 'C0')
    
    if time_array2 is not None:
        vs = schemes.Euler_Maruyama_no_stop(0,initial_state, sigma2, dt=time_array2[1]-time_array2[0], dims = 2, force=force, time_array_length=len(time_array2))
    else:
        vs = schemes.Euler_Maruyama_no_stop(0,initial_state, sigma2, dt=dt, dims = 2, force=force, time_array_length=len(time_array))
        
    ax.plot(vs[:,0], vs[:,1], linewidth = 0.5, color = 'C1')
    
    print(np.sum(vs[:,0]*vs[:,1])/np.sqrt(np.sum(vs[:,0]*vs[:,0])*np.sum(vs[:,1]*vs[:,1])))
    
    #ax.set_title(r'$\sigma$'+' = {}'.format(sigma1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim((ymin,ymax))
    ax.set_xlim((xmin,xmax))
    
    plt.scatter(initial_state[0],initial_state[1], marker = 'o',  color = 'black', s=40, zorder = 10)
    plt.scatter(target_state[0], target_state[1], marker = 'x', color = 'black', s=40, zorder = 10)
    
    
    
    #threshold and stopping criterion
    if score_function is not None:
        #score function threshold for the TAMS algorithm stopping criterion
        score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))
        ax.contour(xx, yy, score_levels, levels = [1-score_threshold], zorder = 10)
    
    if force_matrix is not None:
        #ellipsoid_fun.draw_ellipsoid_2D(force_matrix, target_state, noise=sigma1, zorder = 20)
        #CS.collections[0].set_label('confidence ellipsoid')
        pass
        
        
    #plt.legend()
    plt.text(-0.2, 1,'(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()
    
def trajectory_plot_2D_compare_sigma(time_array, dt, force, initial_state, target_state, sigma1, sigma2, save_path, force_matrix=None, score_threshold=None, score_function = None, xmin = -6, xmax = 6, ymin = -2, ymax = 10):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    fig, axes = plt.subplots(2, 2, sharex = 'col')
    x,y  = np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100)
    xx, yy = np.meshgrid(x, y)
    
    
    #solve first trajectory
    vs = schemes.Euler_Maruyama_no_stop(0,initial_state, sigma1, dt=dt, dims = 2, force=force, time_array_length=len(time_array))
    
    axes[0,0].plot(time_array, vs[:,0], linewidth = 0.3, label = 'x')
    #axes[0,0].plot(time, vs[:,1], linewidth = 0.3, label = 'y')
    axes[0,0].set_title(r'$\sigma$'+' = {}'.format(sigma1))
    axes[0,0].set_ylabel('x')
    axes[0,0].set_ylim((xmin,xmax))
    #axes[0,0].legend()
    axes[0,1].plot(vs[:,0], vs[:,1], linewidth = 0.1, color = 'C1')
    axes[0,1].set_title(r'$\sigma$'+' = {}'.format(sigma1))
    axes[0,1].set_ylabel('y')
    axes[0,1].set_ylim((ymin,ymax))
    
    #threshold and stopping criterion
    if score_function is not None:
        #score function threshold for the TAMS algorithm stopping criterion
        score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))
        axes[0,1].contour(xx, yy, score_levels, levels = [1-score_threshold], zorder = 10)
        axes[1,1].contour(xx, yy, score_levels, levels = [1-score_threshold], zorder =10)
    
    if force_matrix is not None:
        ellipsoid_fun.draw_ellipsoid_2D(force_matrix, target_state, noise=sigma1)
        

    #solve second trajectory
    vs = schemes.Euler_Maruyama_no_stop(0,initial_state, sigma2, dt=dt, dims = 2, force=force, time_array_length=len(time_array))
    
    axes[1,0].plot(time_array, vs[:,0], linewidth = 0.2, label = 'x')
    #axes[1,0].plot(time, vs[:,1], linewidth = 0.3, label = 'y')
    axes[1,0].set_title(r'$\sigma$'+' = {}'.format(sigma2))
    axes[1,0].set_xlabel('t[s]')
    axes[1,0].set_ylabel('x')
    axes[1,0].set_ylim((xmin,xmax))
    
    axes[1,1].plot(vs[:,0], vs[:,1], linewidth = 0.1, color = 'C1')
    axes[1,1].set_title(r'$\sigma$'+' = {}'.format(sigma2))
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    axes[1,1].set_xlim((xmin,xmax))
    axes[1,1].set_ylim((ymin,ymax))
    
    
   
    
    plt.subplots_adjust(wspace = 0.4)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
