import functools
import ellipsoid_fun
import numpy as np




sigma = 3
covariance_matrix_initial, quad_form_initial, spectral_radius_initial, level_initial, bound_initial = ellipsoid_fun.ingredients_score_function(run.force_matrix, run.initial_state, sigma, run.noise_matrix)
initial_test = functools.partial(ellipsoid_fun.ellipsoid_test, quad_form=quad_form_initial, level=level_initial)

tested = np.apply_along_axis(initial_test, 2, traj)
                        
indices_start = run.time_array_length - np.apply_along_axis(np.argmax, 1, tested[:,::-1])
indices_end = indices_start + 1 #TOOOO CHANGE §§§
r = np.arange(run.time_array_length)

mask = (indices_start[:, None] <= r) & (r<=indices_end[:, None])



print(mask.shape)
samples = traj[mask, :]
print(traj[mask, :].shape)

import matplotlib.pyplot as plt


plt.scatter(samples[:,0], samples[:,1], label = 'last exit points')
plt.scatter(run.initial_state[0], run.initial_state[1], marker = 'o', s=20)
CS = ellipsoid_fun.draw_ellipsoid_2D(run.force_matrix, run.initial_state, noise=sigma, confidence=0.95)
CS.collections[0].set_label('confidence ellipsoid')


spectrum, eigvec = np.linalg.eig(np.linalg.inv(covariance_matrix_initial))
eigvec1 = eigvec[:,0]
eigvec2 = eigvec[:,1]

print(f'eigvec_1: {eigvec1}')
print(f'eigvec_2: {eigvec2}')

def proj_1(v):
    return np.sum((eigvec1)*(v-run.initial_state))

def proj_2(v):
    return np.sum((eigvec2)*(v-run.initial_state))
    
avg_1 = np.mean(np.apply_along_axis(proj_1, 1, samples))
avg_2 = np.mean(np.apply_along_axis(proj_2, 1, samples))


print(f'avg_1: {avg_1:.2e}')
print(f'avg_2: {avg_2:.2e}')

direction = avg_1*eigvec1+avg_2*eigvec2
vector_direction = run.initial_state + avg_1*eigvec1+avg_2*eigvec2

s = np.linspace(0, 3*bound_initial, 1000)
        
s = np.expand_dims(s,1)
line = s*direction+run.initial_state

direction_on_border = line[np.argmin(np.apply_along_axis(quad_form_initial, 1, line)<level_initial)]-run.initial_state

eta=0.5
param = 0.05

covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(run.force_matrix, run.target_state, sigma, noise_matrix=run.noise_matrix)

def score_function(v):
    #return 0.5+0.5*np.tanh(-0.5*np.dot(v-run.initial_state,direction_on_border.T))
    return eta - eta*np.exp(-param*quad_form_initial(v))*(0.5+0.5*np.tanh(-0.2*np.dot(v-initial_state,direction_on_border.T)))+(1-eta)*np.exp(-param*quad_form_target(v))

print(f'Computed starting direction: {direction} and direction vector: {vector_direction}')
print(f'Computed border direction: {direction_on_border}')


plt.scatter(vector_direction[0], vector_direction[1], marker = 'o', s=50, label = "mean", color = 'blue')
plt.scatter((direction_on_border+run.initial_state)[0],(direction_on_border+run.initial_state)[1], marker = 'o', s=50, label = "border", color = 'blue')


s = np.linspace(0,2,100)
s = np.expand_dims(s,1)
line = vector_direction*s+(1-s)*run.initial_state
plt.plot(line[:,0], line[:,1], color = 'green', label = 'good direction')

s = np.linspace(-2,0,100)
s = np.expand_dims(s,1)
line = vector_direction*s+(1-s)*run.initial_state
plt.plot(line[:,0], line[:,1], color = 'red', label = 'wrong direction')

plt.legend(loc = 'lower left')
