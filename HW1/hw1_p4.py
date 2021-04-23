#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation of a Markovian drone flight!
Guide a drone to its goal destination through a storm.
Produce a value function plot and optimal policy.
Plot the value function, optimal policy, and sample trajectory.
Created on Tue Apr 20 22:22:29 2021

@author: kmoy14
"""
#%%
import numpy as np 
import matplotlib.pyplot as plt
import os
from copy import deepcopy
my_path = os.path.dirname(os.path.realpath(__file__))

n = 20              # grid size
sig = 10            # distribution of storm
gamma = 0.95        # discount factor
x_eye = [15, 15]    # location of storm center
x_goal = (19, 9)    # location of goal state

# Initialize matrices
R = np.zeros((n, n))
V = np.zeros((n, n))
w = np.zeros((n, n))
# %%
# fill in W, R
R[x_goal] = 1

for i in range(n):
    for j in range(n):
        x = np.array((i,j))
        w[i,j] = np.exp(-(np.linalg.norm(x-np.array(x_eye)))**2/(2*sig**2))

# plot w for fun
plt.imshow(w.T, cmap='hot', origin='lower')
plt.show()

plt.imshow(R.T, cmap='hot', origin='lower')
plt.show()

# %%
# loop through all states
# assign next states probability given by a and w
# if next state not possible, reassign to current state
# then calculate max value from next state reward

actions = np.array([(0,1), (0, -1), (-1, 0), (1, 0)])
V_prev = np.ones((n,n))
policy = np.zeros((n,n))
while np.amax(np.absolute(V_prev-V)) > 1e-9:
    V_prev = deepcopy(V)
    for i in range(n):
        for j in range(n):
            sum_a = np.zeros(4)
            for a in range(4):
                act_a = actions[a]
                w_ij = w[i,j]  

                j_up = j+1
                j_down = j-1
                i_left = i-1
                i_right = i+1
                
                moves_allowed = [j_up < n, j_down >= 0, i_left >= 0, i_right < n]

                next_state = [(i, j_up), (i, j_down), (i_left, j), (i_right, j)]
                prob_next_state = np.zeros(4) 
                v_next_state = np.zeros(4) 
                r_next_state = np.zeros(4) 

                for b in range(4):
                    prob_next_state[b] = w_ij/4
                    if b == a:
                        prob_next_state[b] += 1 - w_ij
                    if not moves_allowed[b]:
                        next_state[b] = (i,j)
                    v_next_state[b] = V[next_state[b]]
                    r_next_state[b] = R[next_state[b]]


                rtogo_next_state = gamma*v_next_state + r_next_state

                sum_a[a] = np.dot(prob_next_state, rtogo_next_state)
            V[i,j] = np.max(sum_a)
            policy[i,j] = np.argmax(sum_a)
    # print(np.amax(np.absolute(V_prev-V)))


plt.figure(1)
plt.imshow(V.T, cmap='hot', origin='lower')
plt.colorbar()
plt.savefig(os.path.join(my_path, 'value_function.png'), dpi=300)
plt.show()

plt.figure(2)
plt.imshow(policy.T, cmap='hot', origin='lower')
plt.colorbar()
plt.savefig(os.path.join(my_path, 'optimal_policy.png'), dpi=300)
plt.show()

# %% simulate trajectory

x = [(0,19)]
for n in range(200):
    # print(x[-1])
    a = policy[x[-1]]
    # print(a)
    i = x[-1][0]
    j = x[-1][1]

    j_up = j+1
    j_down = j-1
    i_left = i-1
    i_right = i+1

    moves_allowed = [j_up < n, j_down >= 0, i_left >= 0, i_right < n]

    next_state = [(i, j_up), (i, j_down), (i_left, j), (i_right, j)]
    prob_next_state = np.zeros(4) 

    for b in range(4):
        prob_next_state[b] = w_ij/4
        if b == a:
            prob_next_state[b] += 1 - w_ij
        if not moves_allowed[b]:
            next_state[b] = (i,j)

    # print(prob_next_state)
    
    # choose stochastic action based on probabilities
    ns = np.random.choice(np.arange(0,4), p = prob_next_state)
    # print(ns)
    x.append(next_state[ns])
    # print(x)

x_traj = zip(*x)

plt.figure(2)
plt.imshow(policy.T, cmap='hot', origin='lower')
plt.colorbar()
plt.scatter(*x_traj, c=np.arange(0,201))
plt.savefig(os.path.join(my_path, 'sim_traj.png'), dpi=300)
plt.show()

# %%
