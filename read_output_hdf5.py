# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:05:21 2019

@author: pasca
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import scipy.ndimage.filters as snf


from sympy import Matrix, exp
from sympy.abc import x,y
import scipy




def probability_sigma_plot(output_path, save_path, score_function_names, theory_plot = True, monte_carlo_plot = False, ymin = 1e-30, xmin = 0):


    #open and read hdf5 file
    with h5py.File(output_path, 'r') as TAMS_output_file:
        print(f'Opening HDF5 file {output_path} ...\n')
        print('Available entries:')
        for entry in list(TAMS_output_file.keys()):
            print(entry)
        
        
        #transition_probability_grid_TAMS_simexp = np.array(TAMS_output_file['TAMS_simexp_coordinate'])
        
        if monte_carlo_plot:
            transition_probability_DMC = np.array(TAMS_output_file['direct_monte_carlo'])
        
        if theory_plot:
            transition_probability_theory = TAMS_output_file['theory'][:]
            sigma_grid_theory = TAMS_output_file['sigma_grid_theory'][:]
            
        #create plot
        fig, ax = plt.subplots()
        
        
        ax.set_yscale("log", nonposy='clip')
        
        plt.xlabel('noise $\sigma$')
        plt.ylabel('transition probability')
    
        #handle minor ticks
        locmin = mpl.ticker.LogLocator(base=1000.0,subs=(0.001,0.01,0.1),numticks=20)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        
        #theory plot
        if theory_plot:
            plt.plot(sigma_grid_theory, transition_probability_theory, label = 'Eyring-Kramers')
         
        #TAMS data
        for score_function_name in score_function_names:
        
            transition_probability_grid_TAMS = TAMS_output_file[f'transition_probability_grid_TAMS_{score_function_name}_coordinate'][:]
            sigma_grid = TAMS_output_file[f'transition_probability_grid_TAMS_{score_function_name}_coordinate'].attrs.get('sigma_grid')
            median_AMS = np.median(transition_probability_grid_TAMS, axis = 1)
            quartile_error = np.array([median_AMS-np.quantile(transition_probability_grid_TAMS,0.25, axis = 1), 
                                       np.quantile(transition_probability_grid_TAMS,0.75, axis = 1)-median_AMS])
            plt.errorbar(sigma_grid, median_AMS, yerr = quartile_error, label = f'TAMS {score_function_name} coord.',  marker = 'o', markersize = 4, linestyle = '', capsize = 4, capthick = 1, elinewidth = 3, zorder = 2)
        
        #monte carlo data
        if monte_carlo_plot:
            plt.errorbar(sigma_grid, transition_probability_DMC, 
                         yerr = np.sqrt(1./1000*transition_probability_DMC*(1-transition_probability_DMC)), 
                         marker = 'x', markersize = 8,  linestyle = '',markeredgewidth=2, label = 'Monte Carlo')
        
        plt.xlim((xmin, np.max(sigma_grid)+0.1))
        plt.ylim((ymin, 10))
    
        plt.legend()
        
        plt.savefig(save_path, bbox_inches ='tight')
        
        plt.show()
            
        TAMS_output_file.close()
        print('\n'+f'HDF5 file {output_path} closed.')
        print(f'Figure saved to {save_path}.')
        

def probability_density_plot(output_path, save_path, score_function, threshold, xbins, ybins, index, histogram = None):

    if histogram is None:
        TAMS_output_file = h5py.File(output_path, 'r')
        print(f'Opening HDF5 file {output_path} ...\n')
        print('Available entries:')
        for entry in list(TAMS_output_file.keys()):
            print(entry)
        histogram = TAMS_output_file['probability_density'][:]
        TAMS_output_file.close()
        print('\n'+f'HDF5 file {output_path} closed.')
        
    
    fig, ax = plt.subplots()
    xedges = xbins
    yedges = ybins
    
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    
    
    #im = plt.pcolormesh(x,y, histogram[0].T)
    locator = mpl.ticker.LogLocator(base=10)
    im = ax.pcolormesh(xcenters,ycenters,histogram[index].T+1, norm = mpl.colors.LogNorm(vmin=histogram.min()+1, vmax=histogram.max()))
    
    cbar = fig.colorbar(im)
    
    #threshold and stopping criterion
    xx, yy = np.meshgrid(xcenters, ycenters)

    score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))

    ax.contour(xx, yy, score_levels, levels = [1-threshold], zorder = 10, colors = 'red')
    
    plt.tick_params(which = 'both', direction = 'out')
        
    
    plt.savefig(save_path)
    print(f'Figure saved to {save_path}.')
    

def probability_density_plot_target(target, index, output_path, save_path, score_function, threshold, xbins, ybins, histogram = None):

    if histogram is None:
        TAMS_output_file = h5py.File(output_path, 'r')
        print(f'Opening HDF5 file {output_path} ...\n')
        print('Available entries:')
        for entry in list(TAMS_output_file.keys()):
            print(entry)
        histogram = TAMS_output_file['probability_density'][:]
        TAMS_output_file.close()
        print('\n'+f'HDF5 file {output_path} closed.')
        
    
    fig, ax = plt.subplots()
    xedges = xbins
    yedges = ybins
    
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    
    
    #im = plt.pcolormesh(x,y, histogram[0].T)
    locator = mpl.ticker.LogLocator(base=10)
    im = ax.contourf(xcenters,ycenters,histogram[index].T+1, levels = np.logspace(np.log10(np.min(histogram[index].T+1)),np.log10(np.max(histogram[index].T)),100), locator=locator)
    
    cbar = fig.colorbar(im, ticks = locator)
    
    #threshold and stopping criterion
    xx, yy = np.meshgrid(xcenters, ycenters)

    score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))

    plt.scatter(target[0], target[1], marker = 'x', label = 'target')
    
    plt.tick_params(which = 'both', direction = 'out')
        
    plt.legend()
    plt.savefig(save_path)
    print(f'Figure saved to {save_path}.')
    

def trajectory_plot_2D_compare_sigma(time_array, solver_scheme, initial_state, target_state, sigma1, sigma2, save_path, draw_threshold, score_threshold, geom_threshold, score_function = None, xmin = -2, xmax = 2, ymin = -2, ymax = 2):
    """
    plots the trajectories of two particles starting in the initial states, for two values of noise strength sigma
    """
    
   
    fig, axes = plt.subplots(2, 2, sharex = 'col')
    x,y  = np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100)
    xx, yy = np.meshgrid(x, y)
    
    
    #solve first trajectory
    ndt, vs = solver_scheme(0,initial_state, sigma1)
    
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
    if draw_threshold == 'score':
        #score function threshold for the TAMS algorithm stopping criterion
        score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))
        axes[0,1].contour(xx, yy, score_levels, levels = [1-score_threshold], zorder = 10)
        axes[1,1].contour(xx, yy, score_levels, levels = [1-score_threshold], zorder =10)
    elif draw_threshold == 'geom':
        euclid_norm = np.apply_along_axis(np.linalg.norm, 0, np.array([xx, yy])-target_state[:,None, None])
        CS = plt.contour(xx, yy, euclid_norm, levels = [geom_threshold], zorder = 10)
        CS.collections[0].set_label('stopping threshold')
        
        
        
        

    #solve second trajectory
    ndt, vs = solver_scheme(0,initial_state, sigma2)
    
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
    
    plt.savefig(save_path)
    plt.show()
    
    
def trajectory_plot_1D_compare_sigma(initial_state, solver_scheme, sigma1, sigma2, save_path):

    time, vs = solver_scheme(0,initial_state, sigma1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,14), sharex = True)
    ax1.plot(time, vs, linewidth = 0.3)
    ax1.set_title(r'$\sigma$'+f' = {sigma1}')
    ax1.set_ylabel('x [a.u.]')
    
    
    time, vs = solver_scheme(0,initial_state, sigma2)
    
    ax2.plot(time,vs, linewidth = 0.3)
    ax2.set_title(r'$\sigma$'+f' = {sigma2}')
    ax2.set_xlabel('t[s]')
    ax2.set_ylabel('x [a.u.]')
    
    plt.savefig(save_path)
    plt.show()
    
    
def visualise_scorefunction(draw_ellipsoid, sigma, target_state, draw_threshold, geom_threshold, score_threshold, score_function, save_path, xmin = -2, xmax = 2, ymin = -2, ymax = 2):
    fig = plt.figure()
    

    
    """
    def trajectory_maker_1():
        time = np.linspace(0, 1, 1000)
        trajectory = np.zeros((1000, 2))
        x = -1.6734231  + 2*1.6734231 *time
        y = 16*time*(1-time)
        trajectory[:,0] = x
        trajectory[:, 1] = y
        return trajectory
    trajectory = trajectory_maker_1()
    x, y = trajectory[:,0], trajectory[:,1]
    plt.plot(x, y, color ='C1', label = 'fitted trajectory')
    plt.legend()
    """
    
    #plt.plot(b.T[0], b.T[1], color ='C1', label = 'estimated instanton')
    #plt.legend(loc = 'lower right')

    #coordinates
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
    
    #score function levels sets
    score_levels = np.apply_along_axis(score_function, 0, np.array([xx,yy]))
    #im = plt.contour(xx, yy, score_levels, 50)
    im = plt.contourf(xx, yy, score_levels, levels = np.linspace(0,1,30))
    cbar = fig.colorbar(im, format = '%.2f')
    cbar.ax.set_title('$\phi$')

    
    if draw_threshold == 'score':
        #score function threshold for the TAMS algorithm stopping criterion
        CS = plt.contour(xx, yy, score_levels, levels = [1-score_threshold], zorder = 10, linestyles = 'dashed', colors = ['red'])
        CS.collections[0].set_label('stopping threshold')
        plt.legend()
    elif draw_threshold == 'geom':
        euclid_norm = np.apply_along_axis(np.linalg.norm, 0, np.array([xx, yy])-target_state[:,None, None])
        CS = plt.contour(xx, yy, euclid_norm, levels = [geom_threshold], zorder = 10)
        CS.collections[0].set_label('stopping threshold')
        
        
    if draw_ellipsoid:
        "Drawing ellipsoid for the triple well system"
        vA, vB, vC = np.array([-1.6734231 ,  0.04399182]), np.array([1.6734231 ,  0.04399182]), np.array([0, 3.9306]),

        alpha = 0.2
        beta = 1
        y_decay = 3
        
        depth = 7
        depth_intermediate = 18
        barrier = 30
        
        y_intermediate = 4
        x_min = 1.5
        yc = 1/3

        F = Matrix([-4*alpha*x**3+2*x*barrier*exp(-x**2-(y/y_decay)**2)-2*x*depth_intermediate*exp(-x**2-(y-y_intermediate)**2)-2*(x-x_min)*depth*exp(-(x-x_min)**2-y**2)-2*(x+x_min)*depth*exp(-(x+x_min)**2-y**2),
            -2*beta*(y-yc)+2*y/(y_decay)**2*barrier*exp(-x**2-(y/y_decay)**2)-2*(y-y_intermediate)*depth_intermediate*exp(-x**2-(y-y_intermediate)**2)-2*y*depth*exp(-(x-x_min)**2-y**2)-2*y*depth*exp(-(x+x_min)**2-y**2)])


        J = F.jacobian(Matrix([x, y]))
    
        A = J.evalf(subs={x: vB[0] , y: vB[1]})
        A = np.asarray(A).astype(np.float)
        
        noise = sigma**2*np.eye(2)
        
        C = scipy.linalg.solve_lyapunov(A,-noise)
        
        inv_c = np.linalg.inv(C)
        
        w,v = np.linalg.eig(inv_c)

        
        
        xval = np.linspace(xmin,xmax,200)
        yval = np.linspace(ymax,ymin,200)
        
        xx,yy=np.meshgrid(xval,yval)
        field = np.array([xx,yy])

        def quad_form_right(vector):
            return np.linalg.multi_dot(((vector-vB).T,inv_c, vector-vB))
        ell_right = np.apply_along_axis(quad_form_right, 0, field)
        CS = plt.contour(xx,yy,ell_right, levels = [5.99], zorder = 5)
        CS.collections[0].set_label('confidence ellipsoid')
        
        
        
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()
    
def mean_trajectory_plot(index, output_path, save_path, mean_traj = None, sigmax = None, sigmay = None):
    if mean_traj is None:
        TAMS_output_file = h5py.File(output_path, 'r')
        print(f'Opening HDF5 file {output_path} ...\n')
        print('Available entries:')
        for entry in list(TAMS_output_file.keys()):
            print(entry)
        mean_traj = TAMS_output_file['mean_trajectory'][:]
        
    
    x = mean_traj[index,:,0]
    y = mean_traj[index,:,1]
    
    if sigmax is not None:
        x = snf.gaussian_filter(x, sigma = sigmax)
    if sigmay is not None:
        y = snf.gaussian_filter(x, sigma = sigmay)
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y)
    plt.savefig(save_path)
    plt.show()

    
    