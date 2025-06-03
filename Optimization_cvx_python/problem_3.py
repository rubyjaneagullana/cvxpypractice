#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 15:34:10 2025

@author: ruby
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

num_antenna = 40
N=np.arange(num_antenna)
c= 3e8
f=2.4e9
wavelen = c/f
d = wavelen/2
theta_l = 0
theta_u = 40
angle_des = 20


phase  = (-1j*2*np.pi*d*N/wavelen)[:,None]#40

steering_des = np.exp(phase*np.sin(np.deg2rad(angle_des)))


#P_matrix = np.zeros((num_antenna, num_antenna), dtype =np.complex128)

t = cp.Variable()
w = cp.Variable(num_antenna,complex = True)

constraints = []

constraints += [w.conj().T @ steering_des == 1]
constraints += [t >= 0]
for i in range(-90,theta_l):
   a = np.exp(-1j*2*np.pi*d*N*np.sin(np.deg2rad(i))/wavelen) 
   P_matrix = np.outer(a,a.conj())
   x=cp.quad_form(w,P_matrix)
   constraints += [x <= t]
   
   
for i in range(theta_u,90):
    b= np.exp(-1j*2*np.pi*d*N*np.sin(np.deg2rad(i))/wavelen)
    P_matrix = np.outer(b,b.conj())
    y = cp.quad_form(w,P_matrix)
    constraints += [y <= t]
    

objective = cp.Minimize(t)
problem = cp.Problem(objective,constraints)
problem.solve()
print(problem.solver_stats.solver_name, problem.value)

print("optimal value",problem.value)
print("optimal w",w.value)

angles = (np.arange(-90,90)) #180,1

W_opt = w.value  #40
si= np.sin(np.deg2rad(angles))[None,:]#1,180
steer_vector = np.exp(phase @ si) #40,180

power = W_opt.conj() @ steer_vector

power_db = 10 * np.log10(np.abs(power)**2) #1,180

plt.figure(figsize=(7.16, 4.0))   # 7.16 in wide × 4 in tall (you can tweak height)
plt.plot(angles,power_db)
plt.title('Minimizing the worst side-lobe')
plt.xlabel('Angles ($^\circ$)')
plt.ylabel('Power (dB)')
#plt.legend([r'$ M = 40 \ \theta_l = 10 \ \theta_u = 30 $'],loc = 2)
plt.grid()
plt.xlim(min(angles),max(angles))
plt.axvline(x=angle_des,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Beam Center ({angle_des}°)')
plt.axvline(x=theta_l,
            color='blue',
            linestyle='-.',
            linewidth=2,
            label=f'Beam Edges ({theta_l}°, {theta_u}°)')
plt.axvline(x=theta_u,
            color='blue',
            linestyle='-.',
            linewidth=2)
plt.legend()
plt.show()




