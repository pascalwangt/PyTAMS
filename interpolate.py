# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:47:09 2019

@author: pasca
"""
import numpy as np
import scipy.interpolate
import h5py




#%%
#conversion functions
def function_to_hdf5(score_function, listbins, filename = 'score_values'):
    points_grid, values = function_to_grid(score_function, listbins)
    #write data to hdf5 file
    with h5py.File(filename, 'a') as score_file:
        score_file.create_dataset('listbins', data = listbins)
        score_file.create_dataset('values', data = values)
        score_file.close()

def hdf5_to_intfunction(filepath = 'score_values'):
    with h5py.File(filepath, 'r') as score_file:
        listbins = score_file['listbins'][:]
        values = score_file['values'][:]
        score_file.close()
    return un_array(scipy.interpolate.RegularGridInterpolator(tuple(listbins), values, bounds_error = False, fill_value = 0))
    



def function_to_intfunction(score_function, listbins):
    points_grid, values = function_to_grid(score_function, listbins)
    return un_array(scipy.interpolate.RegularGridInterpolator(tuple(listbins), values, bounds_error = False, fill_value = 0))

#%%
    
def function_to_grid(score_function, listbins):
    points_grid= np.array(np.meshgrid(*listbins, indexing = 'ij'))
    values = np.apply_along_axis(score_function,0,points_grid)
    return points_grid, values

def un_array(func):
    def wrapper(v):
        assert func(v).size==1
        return func(v)[0]
    return wrapper

