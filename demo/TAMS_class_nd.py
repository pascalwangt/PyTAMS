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

import interpolate
import schemes
import warp_score_function_ell
import types



class TAMS_object:
    def __init__(self, 
                 system, score_function,
                 t_trajectory, dt,
                 interpolate_score, interpolating_grids, 
                 score_threshold = 'auto'):
        
        #import dynamical system file
        self.force = system.force
        self.force_matrix = system.force_matrix
        self.noise_matrix = system.noise_matrix
        
        self.initial_state = system.initial_state
        self.target_state = system.target_state
        self.saddle_state = system.saddle_state
        
        self.dims = len(self.initial_state)
        
        
        #score function: input string or python function
        if isinstance(score_function, types.FunctionType):
            self.score_function = score_function
            print('Using user-defined score function...')
        elif isinstance(score_function, str):
            self.score_function = interpolate.hdf5_to_intfunction(score_function)
            self.interpolate_score = False
            print('Using interpolated function from user-defined array...')
        
        
        
        #time integration parameters
        self.t_trajectory = t_trajectory #fixed trajectory length
        self.dt = dt
        
        self.time_array = np.arange(0, self.t_trajectory, self.dt)
        self.time_array_length = len(self.time_array)
        
        self.solver_scheme_no_stop = schemes.Euler_Maruyama_no_stop
        
        #if the score function is interpolated from a grid interpolate
        self.interpolate_score = interpolate_score and self.interpolate_score  #user and non hdf5
        self.interpolating_grids = interpolating_grids
        

    
    
    def TAMS_run(self, sigma_grid, N_particles, N_samples, Nitermax, listbins, output_path, histbis = False, quadrant = False, branching_points=False, verbose = True, quadrant_factor = 1000, geom_stopping = False, score_stopping = False, warp = False):
        
        #transition_probability matrix
        transition_probability_grid_TAMS = np.zeros((len(sigma_grid), N_samples))

        #computing time matrix
        computing_time = np.zeros((len(sigma_grid), N_samples))
        number_timesteps = np.zeros((len(sigma_grid), N_samples))
        
        #same_score_error matrix (0 if no error, 1 if same score error is encountered)
        same_score_error = np.zeros((len(sigma_grid), N_samples))
        
        #density probability or histogram, mean trajectory
        probability_density = np.zeros([len(sigma_grid)]+[len(elem)-1 for elem in listbins])
        mean_trajectory = np.zeros((len(sigma_grid), len(self.time_array), self.dims))
        
        #samples for the preferred initial direction WARNING: this only works if running N_samples=1 sample
        quadrant_samples = None
        direction = None
        
        #branching points at each iteration WARNING: this only works if running N_samples=1 sample
        branching_points_list = []
        pre_branching_points_list = []
        branching_scores_list = []
        pre_branching_scores_list = []
        
            

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

            #adapt the score function
            run_score_function = self.score_function
            
            #warp the score function so that the target set is the confidence ellipsoid
            if warp:
                print('Warping the score function...')
                run_score_function, run_threshold = warp_score_function_ell.remap_score_function_ell(run_score_function,
                                                                                                    self.force_matrix,
                                                                                                    self.target_state,
                                                                                                    noise = sigma)
                print(f'Used the ellipsoid to warp the score function threshold. Automatic threshold: {run_threshold:.2e} \n')
            
            # if you want to sample the exit direction of the ellipsoid, calculate the ellipsoid and the function that tests if a point is in the ellipsoid
            if quadrant:
                covariance_matrix_initial, quad_form_initial, spectral_radius_initial, level_initial, bound_initial = ellipsoid_fun.ingredients_score_function(self.force_matrix, self.initial_state, sigma, self.noise_matrix, confidence=0.99)
                initial_test = functools.partial(ellipsoid_fun.ellipsoid_test, quad_form=quad_form_initial, level=level_initial)
            
            
            # defines the criterion for knowing if a trajectory has converged or not: either score (automatic or manual) or geometric
            if score_stopping:
                if warp:
                    score_stopping_level = 1 - run_threshold
                else:
                    score_stopping_level = float(score_stopping)
                print(f'Using score stopping with level : {score_stopping_level: .2e}')
                target_test = lambda v: run_score_function(v) > score_stopping_level  #stopping if score > score_stopping_level
            else:
                if not geom_stopping :
                    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(self.force_matrix, self.target_state, sigma, self.noise_matrix)
                    target_test = functools.partial(ellipsoid_fun.ellipsoid_test, quad_form=quad_form_target, level=level)
                else:
                    geom_stopping_threshold = float(geom_stopping)
                    print(f'Using geometrical stopping with  threshold : {geom_stopping_threshold: .2e}')
                    target_test = lambda v: np.linalg.norm(v-self.target_state)<geom_stopping_threshold  #stopping if the trajectory is close to the target state by geom_stopping_threshold
                
            
            

            #generate Euler-Maruyama solver
            solver_scheme = functools.partial(self.solver_scheme_no_stop,
                                              dims = self.dims,
                                              force = self.force,
                                              time_array_length = self.time_array_length,
                                              dt = self.dt,
                                              noise_matrix = self.noise_matrix)
            
            
            for n in range(N_samples):
                print('Computing sample number {} out of {} ... \n'.format(n+1, N_samples))
           
                
                #initialize variables
                
                k = 1 #iteration number
                weights = [1] #initial weight
                number_discarded_trajectories = [] #number discarded trajectories at each iteration

                TAMS_start_time = tm.time() #timer
                
                #1. generate N independent trajectories, starting in initial condition
                traj = np.empty((N_particles, self.time_array_length, self.dims))
                for i in range(N_particles):
                    xs = solver_scheme(0,self.initial_state, sigma)
                    traj[i,:,:] = xs
                    
                local_number_timesteps = N_particles*self.time_array_length #stores number of computed steps
                
                #2. calculate scores
                scores_array = np.apply_along_axis(run_score_function, 2, traj)
                scores = np.max(scores_array, axis = 1)
                min_score = np.min(scores)
                
                #3. target reached based on defined criterion
                target_reached_array = np.any(np.apply_along_axis(target_test, 2, traj), axis = 1)
                
                
                while k<Nitermax and np.count_nonzero(target_reached_array)<N_particles and not same_score_error[s,n]:
                
                    #3.get  worst trajectories
                    indexes_to_discard = np.array(scores == min_score).nonzero()[0]
                    
                    if len(indexes_to_discard)==N_particles:
                        same_score_error[s,n] = 1
                        break
                        
                    
                    #4. update weights and number of discarded trajectories
                    if verbose:
                        print(f'\n Iteration # {k}')
                        print(f'Number of branched trajectories: {len(indexes_to_discard)}. Minimum score: {min_score}')
                    
                    number_discarded_trajectories.append(len(indexes_to_discard))
                    weights.append(weights[-1]*(1-len(indexes_to_discard)/N_particles))
                    
                    indexes_choice_pool = [elem for elem in range(N_particles) if not elem in indexes_to_discard]
                    
                    #5. branch the discarded trajectories
                    for discarded_index in indexes_to_discard:
                        #select random trajectory in the rest
                        branch_traj_index = np.random.choice(indexes_choice_pool)
                        
                        #get branching time and position
                        branch_time_index = np.argmax(scores_array[branch_traj_index,:]>= min_score)
                        branch_x = traj[branch_traj_index, branch_time_index]
                        
                        #store branching points if asked
                        if branching_points:
                            reached_max = np.argmax(scores_array[discarded_index])
                            assert scores_array[discarded_index, reached_max] == min_score
                            
                            pre_branch_x = traj[discarded_index][reached_max,:]
                            
                            pre_branching_points_list.append(pre_branch_x)
                            branching_points_list.append(branch_x)
                            
                            pre_branching_scores_list.append(run_score_function(pre_branch_x))
                            branching_scores_list.append(run_score_function(branch_x))
                            
                            
                        #compute the branched trajectory and update total computed timesteps
                        x_new = solver_scheme(branch_time_index, branch_x, sigma)
                        scores_new =  np.apply_along_axis(run_score_function, 1, x_new)
                        local_number_timesteps += self.time_array_length-branch_time_index
                        
                        #update the trajectory storage matrix
                        traj[discarded_index,:branch_time_index] = traj[branch_traj_index, :branch_time_index] #necessary only for the histogram?
                        traj[discarded_index, branch_time_index:] = x_new
                        
                        #update the scores list and target reached
                        scores_array[discarded_index,:branch_time_index] = scores_array[branch_traj_index, :branch_time_index] #unnecessary?
                        scores_array[discarded_index, branch_time_index:] = scores_new
                        
                        scores[discarded_index] = np.max(scores_new)
                        
                        #update target reached
                        target_reached_array[discarded_index] = np.any(np.apply_along_axis(target_test, 1, x_new))
                        
                        #verbose
                        if verbose:
                            print(f'Branching index {discarded_index} to {branch_traj_index}. New score: {scores[discarded_index]}')
                            if scores[discarded_index] == min_score:
                                print('No score improvement')
                            else:
                                print('Score improved')
                            print(f'{np.count_nonzero(target_reached_array)}/{N_particles} reached the target.')
                        
                        
                        
                    
                    #update iteration number
                    k += 1
                    min_score = np.min(scores)
                        
            
                if same_score_error[s,n]:
                    print('All trajectories have the same score! :(')
                else:
                    #convert lists to numpy arrays
                    weights = np.array(weights)
                    number_discarded_trajectories = np.array(number_discarded_trajectories)
                    
                    #calculate transition probability
                    target_reached_indexes = np.nonzero(target_reached_array)[0]
                    target_reached_number = np.count_nonzero(target_reached_array)
                    
                    #print(target_reached,k)
                    weight_normalisation_constant = N_particles*weights[-1]+np.sum(number_discarded_trajectories*weights[:-1])
                    transition_probability = target_reached_number/weight_normalisation_constant*weights[-1]
                    transition_probability_grid_TAMS[s,n] = transition_probability
                        
                        

                    
                    #update mean_trajectory
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        mean_trajectory[s]+= np.mean(traj[target_reached_indexes], axis = 0)/N_samples
                    
    
                    
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
                    
                    #calculates the preferred directions
                    if quadrant:
                        traj = np.asarray(traj) #matrix to arrat
                        tested = np.apply_along_axis(initial_test, 2, traj) #tests if the positions are in the initial confidence ellipsoid
                        indices_start = self.time_array_length - np.apply_along_axis(np.argmax, 1, tested[:,::-1]) #gets the last position at which all the positions after are outside the ellipsoid
                        indices_end = indices_start + int(self.time_array_length//quadrant_factor) #defines the sample window (int(self.time_array_length//quadrant_factor) )
    
                        r = np.arange(self.time_array_length)
                        mask = (indices_start[:, None] <= r) & (r<=indices_end[:, None])
                        quadrant_samples = traj[mask, :] #get the defined samples
                        
                        #get the first two eigenvectors of the covariance matrix (can be easily generalized to more)
                        spectrum, eigvec = np.linalg.eig(np.linalg.inv(covariance_matrix_initial))
                        eigvec1 = eigvec[:,0]
                        eigvec2 = eigvec[:,1]

                        #define projections onto the eigenvectors
                        def proj_1(v):
                            return np.sum((eigvec1)*(v-self.initial_state))
                        
                        def proj_2(v):
                            return np.sum((eigvec2)*(v-self.initial_state))
                        
                        #compute the average projections
                        avg_1 = np.mean(np.apply_along_axis(proj_1, 1, quadrant_samples))
                        avg_2 = np.mean(np.apply_along_axis(proj_2, 1, quadrant_samples))
                            
                        #compute the direction
                        direction = avg_1*eigvec1+avg_2*eigvec2
                        vector_direction = avg_1*eigvec1+avg_2*eigvec2 + self.initial_state
                            
                        
                        

                    #computing time update
                    TAMS_end_time = tm.time()
                    computing_time[s,n] = TAMS_end_time - TAMS_start_time
                    number_timesteps[s,n] = local_number_timesteps
                    
                    #verbose 
                    print(f'\nTAMS algorithm stopped after {k}/{Nitermax} iterations')
                    print(f'Computed transition probability: {transition_probability:.2e}')
                    print(f'Number of computed timesteps: {local_number_timesteps:.2e}\n')
                    
                    
                    print(f'Average number of trajectories discarded/iteration : {np.mean(number_discarded_trajectories):.2f}')
                    print(f'Computed a total of {np.sum(number_discarded_trajectories)} trajectories.')
                    print(f'Computed {target_reached_number} reactive trajectories.')
                    
                    if quadrant:
                        print(f"Collected {quadrant_samples.shape[0]} points for direction calculation.")
                        print(f'Computed starting direction: {direction} and direction vector: {vector_direction}')
                    
                    print(f'TAMS computing time: {self.display_time(TAMS_end_time - TAMS_start_time)} \n')
                
                
                
        job_end_time = tm.time()
        job_elapsed_time = job_end_time-job_start_time
        print("\n"+"TAMS job finished. ")
        print(f'Total timesteps computed: {np.sum(number_timesteps, axis = (0,1)):.2e}')
        print('Total computing time: '+ self.display_time(job_elapsed_time))
        
        if 1 in same_score_error:
            print('WARNING: Same score error occured at some point.')
        
        
        if output_path is not None:
            #write data to hdf5 file
            with h5py.File(output_path, 'a') as file:
                transition_proability_dataset_TAMS = file.create_dataset('transition_probability_grid_TAMS', data = transition_probability_grid_TAMS)
                
                if quadrant:
                    file.create_dataset('quadrant_samples', data = quadrant_samples)
                    file.create_dataset('direction', data = direction)
                    
                if branching_points:
                    file.create_dataset('branching_points', data = np.array(branching_points_list))
                    file.create_dataset('pre_branching_points', data = np.array(pre_branching_points_list))
                    
                    file.create_dataset('branching_scores', data = np.array(branching_scores_list))
                    file.create_dataset('pre_branching_scores', data = np.array(pre_branching_scores_list))
                    
                    
                transition_proability_dataset_TAMS.attrs['sigma_grid'] = sigma_grid
                
                transition_proability_dataset_TAMS.attrs['trajectory_length'] = self.t_trajectory
                transition_proability_dataset_TAMS.attrs['dt'] = self.dt
                #transition_proability_dataset_TAMS.attrs['score_threshold'] = run_threshold


                transition_proability_dataset_TAMS.attrs['Initial state'] = self.initial_state
                            
                transition_proability_dataset_TAMS.attrs['N_samples'] = N_samples
                transition_proability_dataset_TAMS.attrs['N_particles'] = N_particles
                transition_proability_dataset_TAMS.attrs['Maximum_number_iterations'] = Nitermax
                
                
                file.create_dataset('number_timesteps', data = number_timesteps)
                file.create_dataset('computing_time', data = computing_time)
                
                
                file.create_dataset('probability_density', data = probability_density)
                file.create_dataset('listbins', data = listbins)
                
                mean_trajectory = mean_trajectory/N_samples
                file.create_dataset('mean_trajectory', data = mean_trajectory)
    
                file.close()
                print(f'Wrote data to file {output_path}.')
        return transition_probability_grid_TAMS, probability_density, quadrant_samples

        
    
    def monte_carlo_run(self, sigma_grid, N_particles, freq= 1000, geom_stopping = False, output_path = None, listbins = None):

        #transition_probability matrix
        transition_probability_grid_MC = np.zeros((len(sigma_grid)))

        #hitting time matrix, computing time matrix
        computing_time = np.zeros((len(sigma_grid)))
        number_timesteps = np.zeros((len(sigma_grid)))
        
        if listbins is not None:
            probability_density = np.zeros([len(sigma_grid)]+[len(elem)-1 for elem in listbins])
        
        job_start_time = tm.time()
        print("############################")
        print("Starting Monte Carlo job ...")
        print("############################\n\n")
    
        for s, sigma in enumerate(sigma_grid):
            print('Parameters: \n noise strength sigma: {:.2f}. \n particles number: {:.1e} \n trajectory length: {} seconds. \n dt = {} seconds'.format(sigma, N_particles, self.t_trajectory, self.dt))
            
            sigma_start_time = tm.time()
            #covariance matrix and associated quadratic form
            
            if not geom_stopping:
                covariance_matrix, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(self.force_matrix, equilibrium_point=self.target_state, noise = sigma)
                target_test = functools.partial(ellipsoid_fun.ellipsoid_test, quad_form=quad_form_target, level=level)
            else:
                geom_stopping_threshold = float(geom_stopping)
                print(f'Using geometrical stopping with  threhsold : {geom_stopping_threshold: .2e}')
                target_test = lambda v: np.linalg.norm(v-self.target_state)<geom_stopping_threshold
                
            solver_scheme = functools.partial(self.solver_scheme_no_stop,
                                              dims = self.dims,
                                              force = self.force,
                                              time_array_length = self.time_array_length,
                                              dt = self.dt,
                                              noise_matrix = self.noise_matrix)

            for n in range(N_particles):
                vs = solver_scheme(0, self.initial_state,sigma=sigma)
                reached = np.any(np.apply_along_axis(target_test,1,vs))
                transition_probability_grid_MC[s]+= reached
                
                
                if reached and listbins is not None:
                    hist, bins = np.histogramdd(vs, bins = listbins)
                    probability_density[s] += hist
                    
                    
                number_timesteps[s]+=self.time_array_length
                
                if n%freq==0 and n!=0:
                    print(f'Estimated probability: {transition_probability_grid_MC[s]}/{n+1} = {transition_probability_grid_MC[s]/(n+1):.2e}')
            transition_probability_grid_MC[s] = transition_probability_grid_MC[s]/N_particles
            sigma_end_time = tm.time()
            computing_time[s] += sigma_end_time-sigma_start_time
            print("Computing time: " +self.display_time(computing_time[s]))
            print(f"Computed transition probability: {transition_probability_grid_MC[s]:.3e} \n")
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

                if listbins is not None:
                    file.create_dataset('probability_density', data = probability_density)
    
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