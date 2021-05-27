#%%
from model import dynamics, cost
import numpy as np
from copy import deepcopy

dynfun = dynamics(stochastic=False)
# dynfun = dynamics(stochastic=True) # uncomment for stochastic dynamics

costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

# Riccati recursion
def Riccati(A,B,Q,R):

    # implement infinite horizon riccati recursion
    P = np.zeros(np.shape(A))
    P_k1 = np.ones(np.shape(A))
    while np.amax(np.absolute(P_k1-P)) > 1e-1:
        P_k1 = deepcopy(P)
        L = -np.linalg.inv(R + B.T.dot(P_k1).dot(B)).dot(B.T).dot(P_k1).dot(A)
        P = Q + A.T.dot(P_k1).dot(A + B.dot(L))
        P *= gamma

    return L,P


A = dynfun.A
B = dynfun.B
Q = costfun.Q
R = costfun.R

L,P = Riccati(A,B,Q,R)

total_costs = []

for n in range(N):
    costs = []
    
    x = dynfun.reset()
    for t in range(T):
        
        # policy 
        u = (L @ x)
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
    
        # dynamics step
        x = dynfun.step(u)
        
    total_costs.append(sum(costs))
    
print(np.mean(total_costs))
# %%
