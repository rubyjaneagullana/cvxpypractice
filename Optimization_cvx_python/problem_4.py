#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 13:23:52 2025

@author: ruby
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


N = 8
num_antenna = np.arange(N)
K = 2
c = 3e8
f = 2.4e9
wavelen = c/f
d = wavelen /2
gain = np.ones(K) #(2)

angles = np.array([-33, 10]) #(2)

sigma = np.sqrt(.01)*np.ones(K)
gamma_db = 10 #db
gamma = 10**(gamma_db/10)

H = np.zeros((N,K), dtype = np.complex128)
for i in range(K):
    H[:,i] = gain[i] * np.exp(-1j * 2*np.pi* d*num_antenna*np.sin(np.deg2rad(angles[i]))/wavelen) 

#make individual w_(1 to Num antennas) for each user
w = [cp.Variable((N,1), complex = True) for i in range(K)]

t = cp. Variable(nonneg=True)

constraints = []

#sum 1 to K ||w_i||_2

sum_of_squares = 0

for i in range(K):
    sum_of_squares += cp.sum_squares(w[i])
    
constraints += [sum_of_squares <= t]
    
#h_i^T w_i
for i in range(K):
    h_i = H[:,i][:,None] #8,1
    h_H= h_i.conj().T @ w[i] # 1x1
    
    #Phase alignment
    constraints += [cp.real(h_H) >= 0]
    constraints += [cp.imag(h_H) == 0]
    
    interference = 0
    for j in range(K):
        if j == i:
            continue
        interference += (h_i.conj().T @ w[j]) # (N,1).T N,1
    interference += (sigma[i])
    
    interference_vec = cp.hstack(interference)
    interference_norm = cp.norm(interference_vec,2)
    
    lhs = cp.real(h_H)
    rhs = cp.sqrt(gamma) * interference_norm
    constraints += [lhs >= rhs]
    
    
objective = cp.Minimize(t)
problem = cp.Problem(objective,constraints)
problem.solve(solver = cp.MOSEK)
print("problem_status",problem.status)
print("optimal total power t*=",t.value)
print(problem.solver_stats.solver_name,problem.value)

for i in range(K):
    print("w{",i,"} = ",w[i].value)
    

#plotting

W_opt = np.hstack([w[i].value for i in range(K)]) #8,2

angle_plot = np.arange(-90,90)[:,None] #angle_plot(180,1) num_antenna(1,8) a_theta(180,8)                                                                           )

a_theta = np.exp(-1j*2*np.pi*d*num_antenna[None,:]*np.sin(np.deg2rad(angle_plot))/wavelen)

    
total_power = np.zeros((angle_plot.size),dtype = np.complex128) #180

for i in range(K):    
    pattern_i = a_theta.conj() @ W_opt[:,i] #180,8  8,1 = 180,1
    total_power += pattern_i
    
power_db = 10 * np.log10(np.abs(total_power)**2 + 1e-12)
    
plt.plot(angle_plot,power_db)
plt.xlabel("Angle (degrees)")
plt.ylabel("Array Power (dB)")
plt.title("Total Beam Pattern")
plt.ylim([-60, np.max(power_db)+3])   

for angle in angles:
    plt.axvline(x=angle, color='r', linestyle='--', linewidth=1.5)


plt.grid(True)
plt.show()

    
    
    
    

    
    
    
    
    
    
    
    

    




