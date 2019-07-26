# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:55:45 2019

@author: pasca
"""
 

save_path=None
xmin = -15
xmax =15
ymin = -15
ymax = 25
                                         

plt.figure(figsize = (3.19, 2.61))
ax = plt.gca()
x,y  = np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100)
xx, yy = np.meshgrid(x, y)



    
ax.plot(vs1[:-1350,0], vs1[:-1350,1], linewidth = 0.5, color = 'C1')


#vs = schemes.Euler_Maruyama_no_stop(0, run.initial_state, 30, dt=0.0001, dims = 2, force=run.force, time_array_length=1550)
    
ax.plot(vs[:-1000,0], vs[:-1000,1], linewidth = 0.5, color = 'C0')
    
    
#ax.set_title(r'$\sigma$'+' = {}'.format(sigma1))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim((ymin,ymax))
ax.set_xlim((xmin,xmax))

plt.scatter(run.initial_state[0],run.initial_state[1], marker = 'o',  color = 'black', s=40, zorder = 10)
plt.scatter(run.target_state[0], run.target_state[1], marker = 'x', color = 'black', s=40, zorder = 10)

CS = ellipsoid_fun.draw_ellipsoid_2D(run.force_matrix, equilibrium_point=target_state, noise=3, zorder = 10)


    
#plt.legend()
plt.text(-0.2, 1,'(b)',horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes, fontsize=9)
if save_path is not None:
    plt.savefig(save_path, bbox_inches = 'tight')
plt.show()