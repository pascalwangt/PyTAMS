# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:01:40 2019

@author: pasca
"""
import numpy as np

def curvilinear_coordinate(trajectory):
    """ trajectory in space-dimension N stores time in axis = 0 and space coordinates in axis = 1 of length N """
    
    nb_points = np.shape(trajectory)[0]
    
    ds_2 = np.zeros(nb_points-1)
    
    for i in range(1, trajectory.ndim):
        dxi = trajectory[1:,i]-trajectory[:-1,i]
        ds_2 = ds_2 + dxi**2
        
    curvilinear_coordinate = np.zeros(nb_points)
    curvilinear_coordinate[1:] = np.cumsum(np.sqrt(ds_2))
    normalised_curvilinear_coordinate = curvilinear_coordinate/curvilinear_coordinate[-1]
    
    return normalised_curvilinear_coordinate


def score_function_maker(trajectory, decay):
    normalised_curvilinear_coordinate = curvilinear_coordinate(trajectory)
    
    def score_function(v):
        distances = np.linalg.norm(trajectory-v, axis = 1)
        closest_point_index = np.argmin(distances)
    
        d = distances[closest_point_index] #distance between vector and trajectory,, normalised
        s = normalised_curvilinear_coordinate[closest_point_index]    #curvilinear coordinate corresponding to closest point
        
        score = s*np.exp(-(d/decay)**2)
        return score
    return score_function