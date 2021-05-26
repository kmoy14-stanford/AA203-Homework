#%% Imports
import numpy as np
import scipy.linalg as sla
import cvxpy as cvx
import copy
import matplotlib.pyplot as plt
from copy import deepcopy

# Riccati recursion
def Riccati(A,B,Q,R):

    # TODO implement infinite horizon riccati recursion
    P = np.zeros(np.shape(A))
    P_k1 = np.ones(np.shape(A))
    while np.amax(np.absolute(P_k1-P)) > 1e-5:
        P_k1 = deepcopy(P)
        L = -np.linalg.inv(R + B.T.dot(P_k1).dot(B)).dot(B.T).dot(P_k1).dot(A)
        P = Q + A.T.dot(P_k1).dot(A + B.dot(L))

    return L,P


#%% Define problem dynamics and cost functions

A = np.array([[0.95, 0.5],[0., 0.95]])
B = np.array([[0.],[1.]])
Q = np.eye(2)
R = np.eye(1)

M = np.array([[0.04, 0.],[0., 1.06]])
test = M - A.T.dot(M).dot(A)
# %%
L_inf, P_inf = Riccati(A, B, Q, R)