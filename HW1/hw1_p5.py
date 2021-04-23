#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 22:22:29 2021
LQR model for linear feedback control of a cartpole problem.
@author: kmoy14
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
from copy import deepcopy
my_path = os.path.dirname(os.path.realpath(__file__))

mp = 2
mc = 10
l = 1
g = 9.81
del_t = 0.1

Q = np.identity(4)
R = np.identity(1)

P = np.zeros((4,4))

A = np.zeros((4,4))
B = np.zeros((4,1))

A[0,2] = 1
A[1,3] = 1
A[2,1] = (mp*g)/(mc)
A[3,1] = (mc+mp)*g/(mc*l)

A = A*del_t + np.identity(4)

B[2,0] = 1/(mc)
B[3,0] = 1/(mc*l)

B = B*del_t

P_k1 = np.ones((4,4))
while np.amax(np.absolute(P_k1-P)) > 1e-5:
    P_k1 = deepcopy(P)
    K = -np.linalg.inv(R + B.T.dot(P_k1).dot(B)).dot(B.T).dot(P_k1).dot(A)
    P = Q + A.T.dot(P_k1).dot(A + B.dot(K))
    
# %%
s_star = np.array([0, np.pi, 0, 0])
def cartpole(s, t, u):
    u = K.dot(s-s_star)
    x, theta, x_dot, theta_dot = s
    f0 = x_dot
    f1 = theta_dot
    f2 = (mp*np.sin(l*theta**2 + g*np.cos(theta))+u)/(mc+mp*np.sin(theta)**2)
    f3 = -((mc+mp)*g*np.sin(theta)+mp*l*theta_dot**2*np.sin(theta)*np.cos(theta)+u*np.cos(theta))/((mc+mp*np.sin(theta)**2)*l)
    return[f0, f1, f2, f3]

s_init = np.array([0, np.pi*3/4, 0, 0])

t = np.linspace(0,30,305)

s = np.zeros((301,4))
s[0] = s_init
u = np.zeros((301,1))
u[0] = K.dot(s_init-s_star)
for k in range(0,300):
    s[k+1] = odeint(cartpole, s[k], t[k:k+2], (u[k],))[1]
    s[k+1] += np.random.normal(0, [0, 0, 1e-2, 1e-2])

ts = np.linspace(0,30,301)

fig, ax = plt.subplots()
lines = ax.plot(ts,s[:,:])
ax.legend(lines, ["x", "theta", "x_dot", "theta_dot"])
plt.xlabel('t')
plt.grid()
# plt.savefig(os.path.join(my_path, 'cartpole.png'), dpi=300)
plt.savefig(os.path.join(my_path, 'cartpole_w_noise.png'), dpi=300)

# %%
