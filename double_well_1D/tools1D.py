# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:00:40 2019

@author: pasca
"""
import numpy as np
import matplotlib.pyplot as plt

import schemes

plt.style.use('bbpaper')


def trajectory_plot(time_array, dt, force, initial_state, target_state, sigma1, save_path, ymin = -2, ymax = 2):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    fig, ax = plt.subplots(1, 1)

    
    #solve first trajectory
    vs = schemes.Euler_Maruyama_no_stop(0,initial_state, sigma1, dt=dt, dims = 1, force=force, time_array_length=len(time_array))
    
    ax.plot(time_array, vs, linewidth = 0.3, color = 'black')
    ax.plot(time_array, target_state+np.zeros(len(time_array)), linewidth = 2, linestyle = '--',  color = 'C1', label = '$X_B$' )
    ax.plot(time_array, initial_state+np.zeros(len(time_array)), linewidth = 2, linestyle = '--', color = 'C0', label = '$X_A$')
    plt.text(-0.2, 1,'(b)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('x')
    ax.set_ylim((ymin,ymax))
    plt.legend()
    
    #threshold and stopping criterion
        
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()