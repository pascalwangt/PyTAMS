# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:30:39 2019

@author: pasca
"""

import numpy as np
import h5py

#Eyring Kramer theory (small noise)
def eyring_kramers(lambda_saddle, det_hess_saddle, det_hess_equilibrium, energy_difference, noise_sigma, t_end):
    epsilon = noise_sigma**2/2
    mean_hitting_time = 2*np.pi/lambda_saddle*np.sqrt(det_hess_saddle/det_hess_equilibrium)*np.exp(energy_difference/epsilon)
    transition_probability_theory = -np.expm1(-t_end/mean_hitting_time)
    return transition_probability_theory


def write_theory_1D(t_trajectory, output_path, sigma_grid_theory = np.linspace(0.05, 1, 1000)):
    #1D double_well V(x) = x**4-2*x**2
    transition_probability_1D = eyring_kramers(4, 4, 8, 1, sigma_grid_theory, t_trajectory)
    with h5py.File(output_path, 'a') as file:
            file.create_dataset('sigma_grid_theory', data = sigma_grid_theory)
            theory_dataset = file.create_dataset('theory', data = transition_probability_1D)
            theory_dataset.attrs['trajectory_length'] = t_trajectory
            
            theory_dataset.attrs['system'] = '1D: V(x) = x**4-2*x**2'
            file.close()
    print(f'1D double well theory data written to {output_path}.')


def write_theory_2D(t_trajectory, output_path, sigma_grid_theory = np.linspace(0.05, 1, 10000)):
    #2D simple_well V(x,y) = 0.25*x**4-0.5*x**2+y**2
    transition_probability_simple_2D = eyring_kramers(1, 2, 4, 0.25, sigma_grid_theory, t_trajectory)
    
    with h5py.File(output_path, 'a') as file:
        
            file.create_dataset('sigma_grid_theory', data = sigma_grid_theory)
            theory_dataset = file.create_dataset('theory', data = transition_probability_simple_2D)
            
            theory_dataset.attrs['trajectory_length'] = t_trajectory
            
            
            theory_dataset.attrs['system'] = '2D simple_well V(x,y) = 0.25*x**4-0.5*x**2+y**2'
        
        
#            theory_dataset = file['theory']
#            file['theory'][:] = transition_probability_simple_2D
#            theory_dataset.attrs['sigma_grid_theory'] = sigma_grid_theory
        
            file.close()
    print(f'2D simple_well theory data written to {output_path}.')
