import functools
import ellipsoid_fun
import numpy as np
import matplotlib.pyplot as plt
import h5py
import triple_well_2D.triple_well_0 as tw
import triple_well_2D.tools_2D as too

plotit = 1
make_score_function = 1
output_path = '../outputs/report_quadrant.hdf5' 
index = -1


with h5py.File(output_path, 'r') as TAMS_output_file:
    print(f'Opening HDF5 file {output_path} ...\n')
    print('Available entries:')
    for entry in list(TAMS_output_file.keys()):
        print(entry)
    sigma_grid = TAMS_output_file[f'transition_probability_grid_TAMS'].attrs.get('sigma_grid')
    sigma = sigma_grid[index]
    samples = TAMS_output_file['quadrant_samples'][:]
    direction_output = TAMS_output_file['direction'][:]
    TAMS_output_file.close()


print(f'Samples has shape {samples.shape}')

covariance_matrix_initial, quad_form_initial, spectral_radius_initial, level_initial, bound_initial = ellipsoid_fun.ingredients_score_function(tw.force_matrix, tw.initial_state, sigma, tw.noise_matrix)
initial_test = functools.partial(ellipsoid_fun.ellipsoid_test, quad_form=quad_form_initial, level=level_initial)
spectrum, eigvec = np.linalg.eig(np.linalg.inv(covariance_matrix_initial))
eigvec1 = eigvec[:,0]
eigvec2 = eigvec[:,1]

print(f'eigvec_1: {eigvec1}')
print(f'eigvec_2: {eigvec2}')

def proj_1(v):
    return np.sum((eigvec1)*(v-tw.initial_state))

def proj_2(v):
    return np.sum((eigvec2)*(v-tw.initial_state))
    
avg_1 = np.mean(np.apply_along_axis(proj_1, 1, samples))
avg_2 = np.mean(np.apply_along_axis(proj_2, 1, samples))


print(f'avg_1: {avg_1:.2e}')
print(f'avg_2: {avg_2:.2e}')

direction = avg_1*eigvec1+avg_2*eigvec2
vector_direction = tw.initial_state + avg_1*eigvec1+avg_2*eigvec2


direction = direction_output

s = np.linspace(0, 3*bound_initial, 1000)
        
s = np.expand_dims(s,1)
line = s*direction+tw.initial_state

direction_on_border = line[np.argmin(np.apply_along_axis(quad_form_initial, 1, line)<level_initial)]-tw.initial_state

print(f'Computed starting direction: {direction} and direction vector: {vector_direction}')
print(f'Output direction: {direction_output}')
print(f'Computed border direction: {direction_on_border}')

if make_score_function:
    eta=0.5
    param = 0.05
    
    covariance_matrix_target, quad_form_target, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(tw.force_matrix, tw.target_state, sigma, noise_matrix=tw.noise_matrix)
    
    def score_function(v):
        #return 0.5+0.5*np.tanh(-0.5*np.dot(v-tw.initial_state,direction_on_border.T))
        return eta - eta*np.exp(-param*quad_form_initial(v))*(0.5+0.5*np.tanh(-0.2*np.dot(v-tw.initial_state,direction_on_border.T)))+(1-eta)*np.exp(-param*quad_form_target(v))
    
    def score_function2(v):
        #return 0.5+0.5*np.tanh(-0.5*np.dot(v-tw.initial_state,direction_on_border.T))
        factor_good = 1
        factor_bad = 0.6
        scalar_prod = np.dot(v-tw.initial_state,direction_on_border.T)
        
        if scalar_prod >= 0:
            arg = -param*factor_good*quad_form_initial(v)
        else:
            arg = -param*factor_bad*quad_form_initial(v)
        return eta - eta*np.exp(arg)+(1-eta)*np.exp(-param*quad_form_target(v))
    
    def score_function3(v):
        #return 0.5+0.5*np.tanh(-0.5*np.dot(v-tw.initial_state,direction_on_border.T))
        return eta - eta*np.exp(-param*0.3*(1.5+np.tanh(0.2*np.dot(v-tw.initial_state,direction_on_border.T)))*quad_form_initial(v))+(1-eta)*np.exp(-param*quad_form_target(v))



if plotit:
    plt.figure()
    plt.grid()
    plt.grid(zorder= -50)

    plt.scatter(tw.initial_state[0], tw.initial_state[1], marker = 'o', s=40, color = 'black', zorder = 50)
    CS = ellipsoid_fun.draw_ellipsoid_2D(tw.force_matrix, tw.initial_state, noise=sigma, confidence=0.99)
    CS.collections[0].set_label('confidence ellipsoid')
    
    print(samples.shape)
    plt.scatter(samples[:,0], samples[:,1], label = 'last exit points')
        
    plt.scatter(tw.initial_state[0]+direction[0], tw.initial_state[1]+direction[1], marker = 'o', s=50, color = 'green')
    plt.text(tw.initial_state[0]+direction[0], tw.initial_state[1]+direction[1]+0.8, '$Z$', fontsize=11)
    #plt.scatter((direction_on_border+tw.initial_state)[0],(direction_on_border+tw.initial_state)[1], marker = 'o', s=50, label = "border", color = 'blue')
    
    print(f'Computed starting direction: {direction} and direction vector: {vector_direction}')
    arrow = plt.arrow(tw.initial_state[0], tw.initial_state[1], direction[0]*0.8, direction[1]*0.8, color = 'red', width = 0.1, zorder = 10)
    print(direction[0])
    s = np.linspace(0,2,100)
    s = np.expand_dims(s,1)
    line = vector_direction*s+(1-s)*tw.initial_state
    #plt.plot(line[:,0], line[:,1], color = 'green', label = 'good direction')
    
    s = np.linspace(-2,0,100)
    s = np.expand_dims(s,1)
    #line = vector_direction*s+(1-s)*tw.initial_state
    #plt.plot(line[:,0], line[:,1], color = 'red', label = 'wrong direction')
    
    plt.legend(loc = 'upper left')
    plt.xlim(-10,-2)
    plt.ylim(-4,8)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.text(-0.15, 1,'(a)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
    plt.text(-5.77, -1.5,'$X_A$', fontsize=11)
    plt.savefig('../../Report/overleaf/direction', bbox_inches = 'tight')
    plt.show()
    
    
    fig = plt.figure(figsize = (1.05*3.19, 1.05*2.61))
    too.visualise_scorefunction(tw.score_function_ellipsoid_maker(param = 0.01, direction = direction_on_border, backward = 0.5, forward = 0.5), tw.initial_state, tw.target_state, sigma=sigma, new_figure=False, nb_levels = 15)
    arrow = plt.arrow(tw.initial_state[0], tw.initial_state[1], direction[0], direction[1], color = 'red', width = 0.1)
    

    
    import matplotlib.patches as mpatches
    
    class AnyObject(object):
        pass
    
    class AnyObjectHandler(object):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            patch = plt.arrow(x0, y0+3,  15, 0, color ='red', width = 0.8, head_width = 5*0.8,
                                   transform=handlebox.get_transform(), zorder = 50)
            return patch
    
    plt.legend([AnyObject()], ['favoured direction'], handler_map={AnyObject: AnyObjectHandler()})
    

    plt.savefig('../../Report/overleaf/score_direction', bbox_inches = 'tight')