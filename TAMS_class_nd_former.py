# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:11:12 2019

@author: pasca
"""
import numpy as np
import time as tm
import h5py
import functools
import warnings


import ellipsoid_fun
import warp_score_function_ell
import interpolate
import schemes

import types



class TAMS_object:
    def __init__(self, 
                 system, score_function, score_function_name,
                 t_trajectory, dt,
                 interpolate_score, interpolating_grids, 
                 warp, score_threshold = 'auto'):
        
        #dynamical system file
        self.force = system.force
        self.force_matrix = system.force_matrix
        
        self.initial_state = system.initial_state
        self.target_state = system.target_state
        self.saddle_state = system.saddle_state
        
        self.dims = len(self.initial_state)
        
        #score function
        if isinstance(score_function, types.FunctionType):
            self.score_function = score_function
            print('Using user-defined score function...')
        elif isinstance(score_function, str):
            self.score_function = interpolate.hdf5_to_intfunction(score_function)
            self.interpolate_score = False
            print('Using interpolated function from user-defined array...')
            
        self.score_function_name = score_function_name
        
        
        
        #time integration
        self.t_trajectory = t_trajectory #fixed trajectory length
        self.dt = dt
        
        self.time_array = np.arange(0, self.t_trajectory, self.dt)
        self.time_array_length = len(self.time_array)
        
        self.solver_scheme_score_stop = schemes.Euler_Maruyama_score_stop
        
        #process score functions before TAMS
        self.interpolate_score = interpolate_score and self.interpolate_score  #user and non hdf5
        self.interpolating_grids = interpolating_grids
        
        self.warp = warp
        
        if self.warp:
            self.score_threshold = 'auto'
        else:
            self.score_threshold = score_threshold

    
    
    def TAMS_run(self, sigma_grid, N_particles, N_samples, Nitermax, listbins, output_path, histbis = False, verbose = True, store_reactive_trajectories = False):
        

        #transition_probability matrix
        transition_probability_grid_TAMS = np.zeros((len(sigma_grid), N_samples))

        #hitting time matrix, computing time matrix
        mean_hitting_time = np.zeros((len(sigma_grid), N_samples))
        computing_time = np.zeros((len(sigma_grid), N_samples))
        number_timesteps = np.zeros((len(sigma_grid), N_samples))
        
        #same_score_error matrix
        same_score_error = np.zeros((len(sigma_grid), N_samples))
        
        #density probability, mean trajectory
        probability_density = np.zeros([len(sigma_grid)]+[len(elem)-1 for elem in listbins])
        mean_trajectory = np.zeros((len(sigma_grid), len(self.time_array), self.dims))
        
        
        #reactive trajectories
        if store_reactive_trajectories:
            reactive_trajectories = np.zeros((len(self.time_array), len(sigma_grid), self.dims))
            reactive_trajectories = np.expand_dims(reactive_trajectories, axis = 0)
            
            
                
        
        job_start_time = tm.time()
        print('\n######################')
        print("Starting TAMS job ...")
        print('###################### \n\n')
              
    
        for s, sigma in enumerate(sigma_grid):
            print('Parameters: \n noise strength sigma: {:.2f}. \n particles number: {:.1e} \n trajectory length: {} seconds. \n dt = {} seconds \n'.format(sigma, N_particles, self.t_trajectory, self.dt))
            
            """
            if self.adapt_param:
                #covariance matrix and associated quadratic form
                covariance_matrix, quad_form_start, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(self.force_matrix, equilibrium_point=self.initial_state, noise = sigma)
                covariance_matrix, quad_form, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(self.force_matrix, equilibrium_point=self.target_state, noise = sigma)
                
                ell = ellipsoid_fun.get_ellipsoid_array(self.target_state, quad_form, level, bound)
                self.score_function, self.score_threshold = ellipsoid_fun.optimize_param_normal(self.score_function_param, initial_guess= 1/quad_form_start(self.saddle_state)*2,
                                                        ell=ell, threshold_param = functools.partial(self.threshold_param, level = level))
            """

            
            run_score_function = self.score_function
            if self.warp:
                print('Warping the score function...')
                run_score_function, run_threshold = warp_score_function_ell.remap_score_function_ell(run_score_function,
                                                                                                    self.force_matrix,
                                                                                                    self.target_state,
                                                                                                    noise = sigma)
                print(f'Used the ellipsoid to warp the score function threshold. Automatic threshold: {run_threshold:.2e} \n')
                
                
            if self.interpolate_score:
                print('Interpolating the score function...')
                run_score_function = interpolate.function_to_intfunction(run_score_function, self.interpolating_grids)
                print('Using the interpolated score function.\n')
                
            if self.score_threshold != 'auto':
                run_threshold = self.score_threshold
                print('Using user-defined threshold: {run_threshold:.2e}\n')
                
            solver_scheme = functools.partial(self.solver_scheme_score_stop, time_array_length = self.time_array_length, 
                                              dt = self.dt, 
                                              score_function = run_score_function,
                                              force = self.force,
                                              dims = self.dims,
                                              score_thresh = run_threshold)
            
            
            for n in range(N_samples):
                print('Computing sample number {} out of {} ... \n'.format(n+1, N_samples))
           
                
                #initialize variables
                
                k = 1 #iteration number
                weights = [1] #initial weight
                number_discarded_trajectories = [] #number discarded trajectories at each iteration
                local_number_timesteps = 0
                TAMS_start_time = tm.time() #timer
                
                #1. generate N independent trajectories, starting in initial condition
                traj = np.zeros((N_particles, self.time_array_length, self.dims))
                for i in range(N_particles):
                    n_dt, xs = solver_scheme(0,self.initial_state, sigma)
                    traj[i,:,:] = xs
                    local_number_timesteps += n_dt
                
                #2. calculate maximum scores
                scores = np.zeros((N_particles))
                scores = np.nanmax(np.apply_along_axis(run_score_function, 2, traj), axis = 1)
                
                
                min_score = np.min(scores)
                
                try:
                    while k<Nitermax and min_score<1-run_threshold:
                    
                        #3.get  worst trajectories
                        indexes_to_discard = np.array(scores == min_score).nonzero()[0]
                        
                        if len(indexes_to_discard)==N_particles:
                            raise NameError('All trajectories have same the score. :(')
                            
                        
                        #4. update weights and number of discarded trajectories
                        if verbose:
                            print(f'\nNumber of branched trajectories: {len(indexes_to_discard)}. Minimum score: {min_score}')
                        
                        number_discarded_trajectories.append(len(indexes_to_discard))
                        weights.append(weights[-1]*(1-len(indexes_to_discard)/N_particles))
                        
                        #5. branch the discarded trajectories
                        for discarded_index in indexes_to_discard:
                            #select random trajectory in the rest
                            branch_traj_index = np.random.choice([elem for elem in range(N_particles) if not elem in indexes_to_discard])
                            branch_traj = traj[branch_traj_index,:]
                            #get branching time and position
                            branch_time_index, branch_x = next((time_index, x) for time_index, x in enumerate(branch_traj) if run_score_function(x) >= min_score)
                            #compute the branched trajectory and update total computed timesteps
                            n_dt_new, x_new = solver_scheme(branch_time_index, branch_x, sigma)
                            local_number_timesteps += n_dt_new
                            
                            
                            #update the trajectory storage matrix
                            traj[discarded_index,:branch_time_index] = branch_traj[:branch_time_index] 
                            traj[discarded_index, branch_time_index:] = x_new
                            
                            #update the scores list
                            scores[discarded_index] = np.nanmax(np.apply_along_axis(run_score_function, 1, x_new))
                            
                            #verbose
                            if verbose:
                                print(f'Branching index {discarded_index} to {branch_traj_index}. New score: {scores[discarded_index]}')
                                if scores[discarded_index] == min_score:
                                    print('No score improvement')
                                else:
                                    print('Score improved')
                            
                            
                            
                        
                        #update iteration number
                        k += 1
                        min_score = np.min(scores)
                
                    
                
                
                
                
                    #convert lists to numpy arrays
                    weights = np.array(weights)
                    number_discarded_trajectories = np.array(number_discarded_trajectories)
                    
                    #calculate transition probability
                    target_reached_indexes = np.nonzero(scores>1-run_threshold)[0]
                    target_reached_number = len(target_reached_indexes)
                    
                    #print(target_reached,k)
                    weight_normalisation_constant = N_particles*weights[-1]+np.sum(number_discarded_trajectories*weights[:-1])
                    transition_probability = target_reached_number/weight_normalisation_constant*weights[-1]
                    transition_probability_grid_TAMS[s,n] = transition_probability
                    
                    
                    
                    
                    
                    #update reactive trajectories
                    if store_reactive_trajectories:
                        reactive_trajectories = np.concatenate((reactive_trajectories, np.expand_dims(traj[target_reached_indexes], axis = 2)), axis = 0)
                    
                    #update mean_trajectory
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        mean_trajectory[s] = mean_trajectory[s] + np.nanmean(traj[target_reached_indexes], axis = 0)
                    

                    
                    #update histogram
                    if histbis:
                        traj = np.asarray(traj)
                        traj_reshaped = traj.reshape((-1, self.dims))
                        time_weights = np.broadcast_to(np.exp(self.time_array**2), (N_particles, self.time_array_length))
                        weights = time_weights.reshape((-1))
                        hist, bins = np.histogramdd(traj_reshaped, bins = listbins, weights = weights)
                        probability_density[s] += hist
                    else:
                        traj = np.asarray(traj) #convert matrix to array otherwise dimensionnality error in histogram 
                        traj_reshaped = traj.reshape((-1, self.dims))
                        hist, bins = np.histogramdd(traj_reshaped, bins = listbins)
                        probability_density[s] += hist
                    
                    
                    #hitting time update
                    # I expect to see RuntimeWarnings in this block
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        mean_hitting_time[s,n] = self.dt*np.mean(np.argmax(np.isnan(traj[target_reached_indexes,:,0]), axis = 1), axis = 0)
                    
                    #computing time update
                    TAMS_end_time = tm.time()
                    computing_time[s,n] = TAMS_end_time - TAMS_start_time
                    number_timesteps[s,n] = local_number_timesteps
                    
                    #verbose 
                    print(f'TAMS algorithm stopped after {k}/{Nitermax} iterations')
                    print(f'Computed transition probability: {transition_probability:.2e}')
                    
                    print(f'Computed a total of {np.sum(number_discarded_trajectories)} trajectories.')
                    print(f'Average number of trajectories discarded/iteration : {np.mean(number_discarded_trajectories):.2f}')
                    print(f'Computed {target_reached_number} reactive trajectories.')
                    print(f'Number of computed timesteps: {local_number_timesteps:.2e}')

                    print(f'Computed mean first hitting time: {mean_hitting_time[s,n]:.2e} seconds')
                    
                    print(f'TAMS computing time: {self.display_time(TAMS_end_time - TAMS_start_time)} \n')
                    
                
                except NameError:
                    same_score_error[s,n] = 1
                    print('All trajectories have the same score. :( \n')
                
                
                
        job_end_time = tm.time()
        job_elapsed_time = job_end_time-job_start_time
        print("\n"+"TAMS job finished. ")
        print(f'Total timesteps computed: {np.sum(number_timesteps, axis = (0,1)):.2e}')
        print('Total computing time: '+ self.display_time(job_elapsed_time))
        
        if 1 in same_score_error:
            print('WARNING: Same score error occured.')
        
        
        if output_path != None:
            #write data to hdf5 file
            with h5py.File(output_path, 'a') as file:
                transition_proability_dataset_TAMS = file.create_dataset(f'transition_probability_grid_TAMS', data = transition_probability_grid_TAMS)
                
                transition_proability_dataset_TAMS.attrs['score_function_name'] = self.score_function_name
                transition_proability_dataset_TAMS.attrs['sigma_grid'] = sigma_grid
                
                transition_proability_dataset_TAMS.attrs['trajectory_length'] = self.t_trajectory
                transition_proability_dataset_TAMS.attrs['dt'] = self.dt
                transition_proability_dataset_TAMS.attrs['score_threshold'] = run_threshold


                transition_proability_dataset_TAMS.attrs['Initial state'] = self.initial_state
                            
                transition_proability_dataset_TAMS.attrs['N_samples'] = N_samples
                transition_proability_dataset_TAMS.attrs['N_particles'] = N_particles
                transition_proability_dataset_TAMS.attrs['Maximum_number_iterations'] = Nitermax
                
                transition_proability_dataset_TAMS.attrs['plot_label'] = f'TAMS {self.score_function_name} coord.'
                
                file.create_dataset('number_timesteps', data = number_timesteps)
                file.create_dataset('mean_hitting_time', data = mean_hitting_time)
                file.create_dataset('computing_time', data = computing_time)
                
                
                file.create_dataset('probability_density', data = probability_density)
                file.create_dataset('listbins', data = listbins)
                
                mean_trajectory = mean_trajectory/N_samples
                file.create_dataset('mean_trajectory', data = mean_trajectory)
    
                file.close()
                print(f'Wrote data to file {output_path}.')
        return transition_probability_grid_TAMS, mean_hitting_time, computing_time, probability_density, mean_trajectory, number_timesteps

        
    
    def monte_carlo_run(self, sigma_grid, N_particles, freq= 1000, output_path = None):

        #transition_probability matrix
        transition_probability_grid_MC = np.zeros((len(sigma_grid)))

        #hitting time matrix, computing time matrix
        computing_time = np.zeros((len(sigma_grid)))
        number_timesteps = np.zeros((len(sigma_grid)))
        
        
        job_start_time = tm.time()
        print("############################")
        print("Starting Monte Carlo job ...")
        print("############################\n\n")
    
        for s, sigma in enumerate(sigma_grid):
            print('Parameters: \n noise strength sigma: {:.2f}. \n particles number: {:.1e} \n trajectory length: {} seconds. \n dt = {} seconds'.format(sigma, N_particles, self.t_trajectory, self.dt))
            
            sigma_start_time = tm.time()
            #covariance matrix and associated quadratic form
            covariance_matrix, quad_form, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(self.force_matrix, equilibrium_point=self.target_state, noise = sigma)

            for n in range(N_particles):
                ndt, vs, count = schemes.Euler_Maruyama_ellipsoid_stop(0, self.initial_state,sigma=sigma, ellipsoid_test=functools.partial(ellipsoid_fun.ellipsoid_test, quad_form = quad_form, level = level),
                                                                         dims = self.dims, force=self.force, time_array_length=self.time_array_length, dt=self.dt)
                transition_probability_grid_MC[s]+= count
                number_timesteps[s]+=ndt
                
                if n%freq==0:
                    print(f'Estimated probability: {transition_probability_grid_MC[s]}/{n+1} = {transition_probability_grid_MC[s]/(n+1):.2e}')
            transition_probability_grid_MC[s] = transition_probability_grid_MC[s]/N_particles
            sigma_end_time = tm.time()
            computing_time[s] += sigma_end_time-sigma_start_time
            print(f"Computed transition probability: {transition_probability_grid_MC[s]:.3e}")
            
        job_end_time = tm.time()
        job_elapsed_time = job_end_time-job_start_time
        print('Total computing time: '+ self.display_time(job_elapsed_time))
        
        if output_path != None:
            #write data to hdf5 file
            with h5py.File(output_path, 'a') as file:
                transition_proability_dataset_MC = file.create_dataset(f'transition_probability_grid_MC', data = transition_probability_grid_MC)

                
                transition_proability_dataset_MC.attrs['sigma_grid'] = sigma_grid
                
                transition_proability_dataset_MC.attrs['trajectory_length'] = self.t_trajectory
                transition_proability_dataset_MC.attrs['dt'] = self.dt
                
                transition_proability_dataset_MC.attrs['N_particles'] = N_particles

                transition_proability_dataset_MC.attrs['number_timesteps']= number_timesteps
                transition_proability_dataset_MC.attrs['computing_time']= computing_time

    
                file.close()
                print(f'Wrote data to file {output_path}.')
        
        return transition_probability_grid_MC, computing_time, number_timesteps





    @staticmethod
    def display_time(seconds, granularity=5):
        intervals = (
        ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),    # 60 * 60 * 24
        ('hours', 3600),    # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
        )
        result = []
    
        for name, count in intervals:
            value = seconds // count
            if value:
                seconds -= value * count
                if value == 1:
                    name = name.rstrip('s')
                result.append("{} {}".format(int(value), name))
        return ', '.join(result[:granularity])