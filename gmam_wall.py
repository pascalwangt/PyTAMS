# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:19:44 2019

@author: pasca
"""

import numpy as np
import sympy as sp
import scipy as sci
import matplotlib.pyplot as plt

import triple_well_2D.triple_well as tw

plt.close('all')
#input

force_matrix = tw.force_matrix
sigma_matrix = sp.eye(2)
initial_state = tw.initial_state
target_state = tw.target_state


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

N = 1000
delta_tau = 0.01

kmax = 200
threshold = 0.5

phi_0 = np.linspace(initial_state, target_state, N).T #shape (dimension, N)

s = np.linspace(-5.77, 5.77, N)
phi_0 = np.array([s, -(s+5.77)*(s-5.77)])


plot = 1
plot_index = None

freq = 1



incr = np.inf
k = 1
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
    
    
    if plot:
        plt.figure(1)
        if k == plot_index:
            sc = plt.scatter(phi_tilde[:,0], phi_tilde[:,1], c = np.linspace(0,1, N), cmap = plt.cm.jet, label = f'{plot_index} preinterp')
            plt.colorbar()
            pass
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

        if k==1:
            plt.figure(1)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(phi_0[0], phi_0[1], label = '0')
            
            plt.figure(3)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(phi_0[0], phi_0[1], label = '0')
            pass
        if k==plot_index:
            plt.figure(1)
            plt.plot(phi_i[0], phi_i[1], label = f'{plot_index} after interp', linewidth = 0.5)
            plt.legend()
        
        if k%freq==0:
            plt.figure(3)
            plt.plot(phi_i[0], phi_i[1], label = f'{k}', color = plt.cm.plasma(k/kmax), linewidth = 0.5)
            plt.title('all')
            plt.ylim(-10, 35)
            plt.xlim(-10, 10)
            
    k+=1
    
plt.figure(4)
plt.plot(phi_i[0], phi_i[1], label = f'instanton', linewidth = 2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-10, 20)
plt.xlim(-10, 10)
plt.savefig('../../Report/update11/wall/instanton')

plt.figure(3)
plt.savefig('../../Report/update11/wall/evolution')

#plt.plot(phi_i[0], phi_i[1])
    
    
#temp = np.array([ham_px_np(*x,*p) for x,p in zip(phi_i[:,1:-1].T, theta_i.T)]) #can be sped up by vectorizing
#lambda_i = np.sum(np.squeeze(ham_p_np(*phi_i[:,1:-1], *theta_i))* phi_i_prime, axis = 0)/np.linalg.norm(phi_i_prime, axis = 0)
#term_2_i = lambda_i*np.squeeze(np.matmul(np.array([ham_px_np(*x,*p) for x,p in zip(phi_i[:,1:-1].T, theta_i.T)]), phi_i_prime.T[:,:,np.newaxis])).T
#%%
"""
alphabet_a = list(sp.ordered(a.free_symbols))
alphabet_b = list(sp.ordered(b.free_symbols))

h
def ham(state,p):

    b_x = b.evalf(subs=dict(zip(alphabet_b, state)))
    b_x = np.asarray(b_x).astype(np.float)
    
    a_x = a.evalf(subs=dict(zip(alphabet_a, state)))
    a_x = np.asarray(a_x).astype(np.float)
    
    return np.dot(b_x.T,p)+1/2*np.linalg.multi_dot((p.T,a_x,p))
    
def theta(state,p):
    a_inv_x = a.evalf(subs=dict(zip(alphabet_a, state)))
    a_inv_x = np.asarray(a_inv_x).astype(np.float)
    
    b_x = b.evalf(subs=dict(zip(alphabet_b, state)))
    b_x = np.asarray(b_x).astype(np.float)
    
    return b_x
"""