"""
Created on Mon May 13 11:31:15 2019

@author: pasca
"""

import numpy as np
import itertools
import scipy.ndimage.filters
import collections



def sub_histogram(histogram, binslist, binlengths, factors = None):
    #hypothesis: evenly spaced bins
    print(f'Input histogram shape: {np.array(histogram.shape)}')
    if factors is None:
        factors = [int(binlength//(bins[1]-bins[0])) for binlength, bins in zip(binlengths, binslist)]
    if all([x==1 for x in factors]):
        return histogram, binslist
    indices_to_keep = [np.append(np.arange(0,len(bins)-1,factor), len(bins)-1) for factor, bins in zip(factors, binslist)]
    sub_binslist = [bins[idx_to_keep] for bins, idx_to_keep in zip(binslist, indices_to_keep)]

    
    shape = np.array(np.shape(histogram))
    sub_shape = np.ceil(shape/np.array(factors)).astype(int)
    
    print(f"New histogram shape: {sub_shape}")
    
    sub_histogram = np.zeros(sub_shape)
    
    ranges = [range(length) for length in sub_shape]
    
    for indices in itertools.product(*ranges):
        slices = [slice(factors[dim]*idx,factors[dim]*(idx+1)) for dim, idx in enumerate(indices)]
        sub_histogram[tuple(indices)] = np.sum(histogram[tuple(slices)])
        
    return sub_histogram, sub_binslist



    
def vector_to_indices(vector, binslist): #equivalent to np.digitize
    return [np.max([i for i in range(len(binslist[j])) if binslist[j][i] <vector[j]]) for j in range(len(vector))]


def neighbour_candidates_nd(coordinates,forbidden_indices, limits, dead_ends=[]):
    candidates = []
    ranges = [range(xi-1,xi+2) for xi in coordinates]
    for neighbour in itertools.product(*ranges):
        test_lower = [idx >= 0 for idx in neighbour]
        test_upper = [idx<limit for (idx, limit) in zip(neighbour, limits)]
        
        if  all(test_lower) and all(test_upper) and not forbidden_indices[neighbour] and list(neighbour) not in dead_ends:
            candidates.append(list(neighbour))
    return candidates


#%%
def get_path_v1(histogram, initial_state, target_state, binslist, freq = 1000):
    start = vector_to_indices(initial_state, binslist)
    target = vector_to_indices(target_state, binslist)
    limits = [len(bins)-1 for bins in binslist]
    
    path =[start]
    forbidden_indices = np.zeros(histogram.shape)
    forbidden_indices[tuple(start)] = 1
    
    if histogram[tuple(start)]==0:
        print('WARNING starting step has 0 points in histogram.')
    if histogram[tuple(target)]==0:
        print('WARNING target step has value 0 in histogram.')
        
    while path[-1] != target:
        path, forbidden_indices = next_step_new(histogram, path, forbidden_indices, limits)
        #print(np.linalg.norm(np.array(path[-1])-np.array(target), ord = 1))
        if len(path)%freq==0:
            print(f"Path has {len(path)} steps.")
    return np.array(path)



def next_step_new(histogram, path, forbidden_indices, limits):
    current_step = path[-1]
    candidates = neighbour_candidates_nd(current_step, forbidden_indices, limits)
    
    if candidates == []:
        forbidden_indices[tuple(current_step)] = 1
        path.pop()
        print(f"Length {len(path)}. No suitable candidates, backing up one step...")
        return path, forbidden_indices
    else:
        step_index = np.argmax([histogram[tuple(candidate)] for candidate in candidates])
        step = candidates[step_index]
        
        if histogram[tuple(step)] == 0:
            print(f"Length {len(path)}. Histogram = 0, backing up one step...")
            forbidden_indices[tuple(current_step)] = 1
            path.pop()
            return path, forbidden_indices
        else:
            path.append(step)
            forbidden_indices[tuple(step)] = 1
            return path, forbidden_indices

#%%
def get_path_v2(histogram, initial_state, target_state, binslist, freq = 1000):
    start = vector_to_indices(initial_state, binslist)
    target = vector_to_indices(target_state, binslist)

    limits = [len(bins)-1 for bins in binslist]
    
    path =[start]
    forbidden_indices = np.zeros(histogram.shape)
    forbidden_indices[tuple(start)] = 1
    dead_ends = []
    
    if histogram[tuple(start)]==0:
        print('WARNING starting step has 0 points in histogram.')
    if histogram[tuple(target)]==0:
        print('WARNING target step has value 0 in histogram.')
        
    while path[-1] != target:
        path, forbidden_indices, dead_ends = next_step_rec(histogram, path, forbidden_indices, limits, dead_ends)
        #print(np.linalg.norm(np.array(path[-1])-np.array(target), ord = 1))
        if len(path)%freq==0:
            print(f"Path has {len(path)} steps.")
    return np.array(path)

def next_step_rec(histogram, path, forbidden_indices, limits, dead_ends, ):
    current_step = path[-1]
    candidates = neighbour_candidates_nd(current_step, forbidden_indices, limits, dead_ends)
    print(dead_ends)
    if candidates == []:
        path.pop()
        forbidden_indices[tuple(current_step)] = 0
        dead_ends.append(current_step)
        print(f"Length {len(path)}. No suitable candidates, backing up {len(dead_ends)} step(s)...")
        return next_step_rec(histogram, path, forbidden_indices, limits, dead_ends)
    else:
        step_index = np.argmax([histogram[tuple(candidate)] for candidate in candidates])
        step = candidates[step_index]
        
        if histogram[tuple(step)] == 0:
            path.pop()
            forbidden_indices[tuple(current_step)] = 0
            dead_ends.append(current_step)
            print(f"Length {len(path)}. Histogram = 0, backing up {len(dead_ends)} step(s)...")
            return next_step_rec(histogram, path, forbidden_indices, limits,  dead_ends)
        else:
            path.append(step)
            print(step)
            forbidden_indices[tuple(step)] = 1
            return path, forbidden_indices, []

#%%
def get_path_v3(histogram, initial_state, target_state, binslist, freq = 1000):
    start = vector_to_indices(initial_state, binslist)
    target = vector_to_indices(target_state, binslist)

    limits = [len(bins)-1 for bins in binslist]


    if histogram[tuple(start)]==0:
        print('WARNING starting step has value 0 in histogram.')
    if histogram[tuple(target)]==0:
        print('WARNING target step has value 0 in histogram.')
    
    forbidden_indices = np.zeros(histogram.shape)
    forbidden_indices[tuple(start)] = 1
    
    stack = collections.deque()
    stack.append((start, [start], forbidden_indices))
    
    
    
    while stack:
        (current_step, path, forbidden_indices) = stack.pop()
        #forbidden_indices[tuple(current_step)] = 1
        if len(path)%freq==0:
            print(f"Path has {len(path)} steps.")
        for neighbour in ordered_neighbours(current_step, histogram, forbidden_indices, limits):
            if neighbour == target:
                yield path + [neighbour]
            else:
                forbidden_indices[tuple(neighbour)] = 1
                stack.append((neighbour, path + [neighbour], forbidden_indices))



def ordered_neighbours(current_step, histogram, forbidden_indices, limits):
    candidates = []
    histogram_values = []
    
    ranges = [range(xi-1,xi+2) for xi in current_step]
    
    for neighbour in itertools.product(*ranges):
        test_lower = [idx >= 0 for idx in neighbour]
        test_upper = [idx<limit for (idx, limit) in zip(neighbour, limits)]
        if all(test_lower) and all(test_upper):
            histogram_value = histogram[neighbour]
            if not forbidden_indices[tuple(neighbour)] and histogram_value>0:
                candidates.append(list(neighbour))
                histogram_values.append(histogram_value)
            
    ordered_candidates = [candidate for _, candidate in sorted(zip(histogram_values,candidates), key=lambda pair: pair[0], reverse=False)]
    return ordered_candidates


#%%



def cleanup(path):
    index = 0
    while index < len(path):
        current_step = path[index]
        neighbours_indices = [n+index+1 for n, step in enumerate(path[index+1:]) if np.linalg.norm(step-current_step, ord = 1) < 1.5]
        if len(neighbours_indices) > 0:
            next_index = np.max(neighbours_indices)
            path = np.delete(path, slice(index+1, next_index), axis = 0)
        index += 1
    return path

def cleanupv2(binslist, path):
    limits = [len(bins)-1 for bins in binslist]
    mat = np.zeros(tuple(limits), dtype = int)
    
    to_keep = np.array([True]*len(path))
    
    for i, elem in enumerate(path):
        mat[tuple(elem)] = i

    i = 0
    while i<len(path)-1:
        values=[]
        ranges = [range(xi-1,xi+2) for xi in path[i]]
        for neighbour in itertools.product(*ranges):
            test_lower = [idx >= 0 for idx in neighbour]
            test_upper = [idx<limit for (idx, limit) in zip(neighbour, limits)]
            if  all(test_lower) and all(test_upper) and mat[tuple(neighbour)]>i:
                values.append(mat[tuple(neighbour)])
        if len(values)>0:
            next_step = max(values)
            to_keep[i+1:int(next_step)] = False
            i = next_step
        else:
            i+=1
    
    cleaned_path = path[to_keep]
    return cleaned_path



def path_to_positions(path, binslist, initial_state, target_state):
    centerslist = [(bins[:-1] + bins[1:])/2 for bins in binslist]
    positions = [[centerslist[dim][index] for dim, index in enumerate(state)] for state in path]
    positions[0] = initial_state
    positions[-1] = target_state
    return np.array(positions).T


def positions_filler(positions, res):
    filled_path = []
    for coordinate_path in positions:
        for i in range(len(coordinate_path)-1):
            coordinate_path = np.insert(coordinate_path, 1+i*(res+1), np.linspace(coordinate_path[(res+1)*i], coordinate_path[(res+1)*i+1], res))
        filled_path.append(coordinate_path)
    return np.array(filled_path)

def positions_filler_new(positions, ds):
    positions = list(positions)
    vectors = np.array(positions).T
    
    res = np.linalg.norm(vectors[:-1]-vectors[1:], axis = 1)//ds
    res = res.astype(int)
    cum_res = np.insert(np.cumsum(res),0,0)
    for i in range(len(vectors)-1):
        for j in range(len(positions)):
            positions[j] = np.insert(positions[j], 1+cum_res[i]+i, np.linspace(positions[j][cum_res[i]+i], positions[j][cum_res[i]+i+1], res[i]))
    return np.array(positions)
    
def refill(path, initial_state, target_state, ds):
    gap_beg = np.linalg.norm(initial_state-path[:,0])
    gap_end = np.linalg.norm(target_state-path[:,-1])
    res_beg = np.int(gap_beg//ds)
    res_end = np.int(gap_end//ds)
    
    path = list(path)
    for j in range(len(path)):
        beginning = np.linspace(initial_state[j], path[j][0], res_beg)
        end = np.linspace(path[j][-1], target_state[j], res_end)
        path[j] = np.insert(path[j], 0, beginning)
        path[j] = np.append(path[j], end)
    return np.array(path)
        




def get_positions(histogram, initial_state, target_state, binslist, binlegnths = None, factors = None, cleaned = True, filled = True, smoothed = True, smooth = 5, res = 100, undersample = 200, version = 1):
    if binlegnths is not None or factors is not None:
        histogram, binslist = sub_histogram(histogram, binslist, binlegnths, factors)
    print("Calculating path...")
    if version == 1:
        path = get_path_v1(histogram, initial_state, target_state, binslist)
    elif version == 2:
        path = get_path_v2(histogram, initial_state, target_state, binslist)
    elif version == 3:
        path = next(get_path_v3(histogram, initial_state, target_state, binslist))
        path = np.array(path)
    print(f"Path calculated. Length : {len(path)}")
    if cleaned:
        print("Cleaning up the path...")
        path = cleanupv2(binslist,path)
        print(f"Cleaned path has length {len(path)}.")
    path = path_to_positions(path, binslist, initial_state, target_state)
    
    if filled:
        print("Filling up the path")
        ds=np.linalg.norm(initial_state-target_state)/res
        filled_path = positions_filler_new(path, ds)
        print(f"Filled path has length {len(filled_path.T)}")
        if smoothed:
            print("Smoothing path")
            mean_increment = np.mean(np.linalg.norm(filled_path.T[:-1]-filled_path.T[1:], axis = 1))
            sigma = np.int(np.linalg.norm(initial_state-target_state)/smooth//mean_increment)
            filled_path = scipy.ndimage.filters.gaussian_filter1d(filled_path, sigma = sigma, mode = 'nearest')
            print(f'Smoothed path with sigma: {sigma}')
            
            filled_path = refill(filled_path, initial_state, target_state, ds)
            print('Refilled path')
    else:
        filled_path = path
    
    if undersample:
        print('Undersampling path...')
        skip = int(len(filled_path.T)//undersample)
        if skip==0:
            skip = 1
        print(f'Skipping every {skip} steps.')
        filled_path = filled_path[:,::skip]
        
        filled_path[:,-1] = target_state
        print(f'Undersampled path has length {len(filled_path.T)}')
        
        
    return path, filled_path, histogram, binslist
    


def next_step(histogram, path, forbidden_indices, limits):
    current_step = path[-1]
    candidates = neighbour_candidates_nd(current_step, forbidden_indices, limits)
    if candidates == []:
        raise Exception("No suitable candidates")
        
    step_index = np.argmax([histogram[tuple(candidate)] for candidate in candidates])
    
    step = candidates[step_index]
    
    path.append(step)
    forbidden_indices.append(step)
    
    return path, forbidden_indices

