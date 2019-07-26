# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:19:44 2019

@author: pasca
"""

import numpy as np
import sympy as sp
import scipy as sci
import matplotlib.pyplot as plt

import sys
sys.path.append('4D_system')
import qg_4D as qg


#dynamical system
force_matrix = qg.force_matrix

alpha = 1e-3
sigma_matrix = sp.diag(alpha, 1, alpha, alpha)

initial_state = qg.initial_state
target_state = qg.target_state


#gmam parameters
N = 1000 #number of points
delta_tau = 0.0001

kmax = 100
threshold = 0.001

phi_0 = np.linspace(initial_state, target_state, N).T #shape (dimension, N)


#%%
#diffusion
a = sigma_matrix*sigma_matrix.T
a_inv = a**-1
#drift
b = force_matrix

#variables
dim = len(force_matrix)

alphabet_x = list(sp.ordered(b.free_symbols))
new_alphabet_x = sp.symbols(f'x:{dim}')
b = b.subs(list(zip(alphabet_x, new_alphabet_x)))

alphabet_x = new_alphabet_x
x = sp.Matrix(new_alphabet_x)

alphabet_p = sp.symbols(f'p:{dim}') #p0, p1 ...
p = sp.Matrix(alphabet_p)


#%%
ham = (b.T*p+1/2*p.T*a*p) #scalar

ham_p = ham.jacobian(p) #vector
ham_x = ham.jacobian(x) #vector

ham_px = ham_p.jacobian(x) # nxn matrix 
ham_pp = ham_p.jacobian(p) # nxn matrix

theta = a_inv*(b.T.dot(a_inv*b)/p.T.dot(a_inv*p)*p-b) #theta is in fact p and is a vector
lamda = b.T.dot(a_inv*b)/p.T.dot(a_inv*p) #typo so no conflict

#%%
ham_p_np = sp.lambdify(alphabet_x+alphabet_p, ham_p, "numpy")
ham_x_np = sp.lambdify(alphabet_x+alphabet_p, ham_x, "numpy")

ham_px_np = sp.lambdify(alphabet_x+alphabet_p, ham_px, "numpy")
ham_pp_np = sp.lambdify(alphabet_x+alphabet_p, ham_pp, "numpy")

ham_pp_x_np = sp.lambdify(alphabet_x+alphabet_p, ham_pp*ham_x.T, "numpy")

theta_np = sp.lambdify(alphabet_x+alphabet_p, theta, "numpy")
lamda_np = sp.lambdify(alphabet_x+alphabet_p, lamda, "numpy")

#%% outer loop



plot = 1
plot_index = None

freq = 1



incr = np.inf
k = 1



fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.patch.set_facecolor('white')
ax.tick_params(axis='both', which='major', pad=-2)
ax.plot(phi_0[0], phi_0[1], zs=phi_0[2], zdir = 'z', linewidth = 0.2, color = plt.cm.plasma(k/kmax))

    
while k<kmax and incr > threshold:
    
    if k==1:
        phi_i = phi_0
    
    phi_i_prime = (phi_i[:,2:]-phi_i[:,:-2])/(2/N) # 1 to N-1
    theta_i = np.squeeze(theta_np(*phi_i[:,1:-1], *phi_i_prime)) #sympy matrix is inherently 2D 
    
    
    lambda_i = np.einsum('ij,ij -> j',  np.squeeze(ham_p_np(*phi_i[:,1:-1], *theta_i)), phi_i_prime) /np.linalg.norm(phi_i_prime, axis = 0)
    lambda_0 = 3*lambda_i[0]-3*lambda_i[1]+lambda_i[2]
    lambda_N = 3*lambda_i[-1]-3*lambda_i[-2]+lambda_i[-3]
    lambda_i_full = np.insert(np.append(lambda_i, lambda_N), 0, lambda_0) #[0] is important because of the newaxis [[x],[y]] NVM
    lambda_i_prime = (lambda_i_full[2:]-lambda_i_full[:-2])/(2/N)
    
    
    
    term_2_i = lambda_i*np.einsum('ijk, ki -> ji', np.array([ham_px_np(*x,*p) for x,p in zip(phi_i[:,1:-1].T, theta_i.T)]), phi_i_prime)
    term_3_i = np.squeeze(ham_pp_x_np(*phi_i[:,1:-1], *theta_i))
    term_4_i = lambda_i*lambda_i_prime*phi_i_prime
    
    left_hand = phi_i[:,1:-1]/delta_tau
    
    B = np.empty((dim, N))
    
    B[:,0], B[:,-1], B[:,1:-1] = initial_state, target_state, -term_2_i+term_3_i+term_4_i+left_hand
    #B[:,0], B[:,-1], B[:,1:-1] = ham_p_np(*phi_i[:,0], 0,0)+phi_i[:,0]/delta_tau, ham_p_np(*phi_i[:,-1],0,0)+phi_i[:,-1]/delta_tau, -term_2_i+term_3_i+term_4_i+left_hand
    
    """
    diag, band, ab = np.ones(N), np.zeros(N), np.zeros((2, N))
    diag[1:-1]= 1/delta_tau + 2*N**2*lambda_i**2
    band[:-2] = lambda_i**2*N**2
    ab[0], ab[1] =  diag, band
    phi_tilde = sci.linalg.solveh_banded(ab, B.T)
    """
    
    #solving the matrix equation: A phi_tilde =B
    diag, band, ab = np.ones(N), np.zeros(N), np.zeros((3, N))
    diag[1:-1]= 1/delta_tau + 2*N**2*lambda_i**2
    band = -lambda_i**2*N**2
    ab[0, 2:], ab[1], ab[2,:-2] =  band, diag, band
    phi_tilde = sci.linalg.solve_banded((1,1), ab, B.T) # (N, dim)
    
    

    #interpolation reparametrisation
    tck, u = sci.interpolate.splprep(phi_tilde.T, k=1,s=0)
    phi_i_update = np.array(sci.interpolate.splev(np.linspace(0,1,N), tck))
    
    #2nd method
    func = sci.interpolate.CubicSpline(u, phi_tilde)
    phi_i_update = func(np.linspace(0,1,N)).T
    
    #print(np.max(np.linalg.norm(phi_i_update[:-1]-phi_i_update[1:])), np.min(np.linalg.norm(phi_i_update[:-1]-phi_i_update[1:]))) #check if equidistant
    
    if np.linalg.norm(initial_state-phi_i_update[:,0]) > 0.1 or  np.linalg.norm(target_state-phi_i_update[:,-1])>0.1:
        print('WARNING BOUNDARY CONDITIONS MOVED')
        print(f'xA: {phi_i_update[0,0]:.2e}, {phi_i_update[1,0]:.2e} \nxB: {phi_i_update[0,-1]:.2e}, {phi_i_update[1,-1]:.2e} ')
        
        
    incr = np.sum(np.linalg.norm(phi_i_update-phi_i, axis = 0))
    phi_i = phi_i_update
    
    if k%freq==0:
        print(f"Iteration {k}: {incr:.1f}")
        
    if plot:

        
        if k%freq==0:
            plt.figure(3)
            vs = phi_i
            ax.plot(vs[0], vs[1], zs=vs[2], zdir = 'z', linewidth = 0.2, color = plt.cm.plasma(k/kmax))
    
            #plt.title(r'$\sigma$'+' = {}'.format(sigma1))
            ax.set_ylabel('x', labelpad = -7)
            ax.set_xlabel('y', labelpad = -7)
            ax.set_zlabel('z', labelpad = -7)
            
            xmin = -10
            xmax = 10
            ymin = 10
            ymax = -10
            zmin = 0
            zmax = 15

#            ax.set_xlim(xmin, xmax)
#            ax.set_zlim(zmin, zmax)
#            ax.set_ylim(ymin, ymax)
            
            plt.scatter(initial_state[0], initial_state[1], zs=initial_state[2], marker = 'o', label = '$X_A$', s = 40, color = 'black', zorder = 1000)
            plt.scatter(target_state[0], target_state[1], zs=target_state[2] ,marker = 'x', label = '$X_B$', s= 40, color = 'black', zorder = 1000)
            

            
    k+=1
    

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.patch.set_facecolor('white')
ax.tick_params(axis='both', which='major', pad=-2)
ax.plot(phi_i[0], phi_i[1], phi_i[2], label = f'instanton', linewidth = 2)
ax.legend()
ax.set_xlabel('$A_1$')
ax.set_ylabel('$A_2$')
ax.set_zlabel('$A_3$')
plt.scatter(initial_state[0], initial_state[1], zs=initial_state[2], marker = 'o', label = '$X_A$', s = 40, color = 'black', zorder = 1000)
plt.scatter(target_state[0], target_state[1], zs=target_state[2] ,marker = 'x', label = '$X_B$', s= 40, color = 'black', zorder = 1000)
plt.savefig('../../Report/update11/4D/instanton')

plt.figure(3)
plt.savefig('../../Report/update11/4D/evolution')




