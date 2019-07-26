# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:03:09 2019

@author: pasca
"""

import numpy as np
import ellipsoid_fun



def remap_score_function_ell(score_function, force_matrix, equilibrium_point, noise):
    """
    This function warps the score function so that the target set is the 0.95-confidence ellipsoid.
    return the score function and the associated threshold
    """
    C, quad_form, spectral_radius, level, bound = ellipsoid_fun.ingredients_score_function(force_matrix, equilibrium_point, noise, confidence = 0.95, warning_threshold = 1e-3)
    
    ellipsoid_array = ellipsoid_fun.get_ellipsoid_array(equilibrium_point, quad_form, level, bound)
    
    score_level = np.min(np.apply_along_axis(score_function, 1, ellipsoid_array))
    def remapped_score_function(v):
        score = score_function(v)
        if score>=score_level:
            if ellipsoid_fun.ellipsoid_test(v, quad_form, level):
                return 1
            else:
                return score_level
        else:
            return score
    return remapped_score_function, (1-score_level)/2

