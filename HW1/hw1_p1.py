#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of Model Reference Adaptive Control (MRAC) for a first-order differential equation.
Plots evolution of true plant, reference model, and gains.
Created on Tue Apr 20 22:22:29 2021

@author: kmoy14
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
my_path = os.path.dirname(os.path.realpath(__file__))

def model1(yy, t):
    y, ym, kr, ky = yy

    alpha = -1
    gamma = 2
    beta = 3
    r = 4

    dy_dt = -alpha*y + beta*(kr*r + ky*y)
    dym_dt = -4*ym + 4*r
    e = y - ym
    dkr_dt = -gamma * e * r
    dky_dt = -gamma * e * y
    return [dy_dt, dym_dt, dkr_dt, dky_dt]

def model2(yy, t):
    # one day I will figure out how to pass a function as an argument
    # but that day is not today
    y, ym, kr, ky = yy

    alpha = -1
    gamma = 2
    beta = 3
    r = 4*np.sin(3*t)

    dy_dt = -alpha*y + beta*(kr*r + ky*y)
    dym_dt = -4*ym + 4*r
    e = y - ym
    dkr_dt = -gamma * e * r
    dky_dt = -gamma * e * y
    return [dy_dt, dym_dt, dkr_dt, dky_dt]


t = np.linspace(0, 10, 1001)
kr_star = (4/3)*np.ones(t.size)
ky_star = ((1-4)/3)*np.ones(t.size)

yy0 = [0.0, 0.0, 0.0, 0.0]

sol1 = odeint(model1, yy0, t)
sol2 = odeint(model2, yy0, t)

plt.figure(1)
plt.plot(t, sol1[:, 0], label='y')
plt.plot(t, sol1[:, 1], label='y_m')
plt.legend(loc='best')
plt.xlabel('t')
plt.title('r(t) = 4')
plt.grid()
plt.savefig(os.path.join(my_path, 'y_ym_4.png'), dpi=300)

plt.figure(2)
plt.plot(t, kr_star, 'k--', label='k_r*')
plt.plot(t, ky_star, 'g--', label='k_y*')
plt.plot(t, sol1[:, 2], label='k_r')
plt.plot(t, sol1[:, 3], label='k_y')
plt.legend(loc='best')
plt.xlabel('t')
plt.title('r(t) = 4')
plt.grid()
plt.savefig(os.path.join(my_path, 'kr_ky_4.png'), dpi=300)

plt.figure(3)
plt.plot(t, sol2[:, 0], label='y')
plt.plot(t, sol2[:, 1], label='y_m')
plt.legend(loc='best')
plt.xlabel('t')
plt.title('r(t) = 4sin(3t)')
plt.grid()
plt.savefig(os.path.join(my_path, 'y_ym_4sin3t.png'), dpi=300)


plt.figure(4)
plt.plot(t, kr_star, 'k--', label='k_r*')
plt.plot(t, ky_star, 'g--', label='k_y*')
plt.plot(t, sol2[:, 2], label='k_r')
plt.plot(t, sol2[:, 3], label='k_y')
plt.legend(loc='best')
plt.xlabel('t')
plt.title('r(t) = 4sin(3t)')
plt.grid()
plt.savefig(os.path.join(my_path, 'kr_ky_4sin3t.png'), dpi=300)


# plt.show()