from model import dynamics, cost
import numpy as np


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

total_costs = []

#TODO: Import A, B, Q, R here but don't use them -- only use them to create our estimates
# A_s, B_s, Q_s, and R_s (where A_s = \hat{A} etc.
# A_s, B_s initialized to random variables of size A,B 
# P, Q, R initialized to the identity of size A, Q, R
# Also need P_k1 as before to be able to recursively update

for n in range(N):
    costs = []
    
    x = dynfun.reset()
    for t in range(T):
        

        # TODO compute policy
        # L -np.linalg.inv(R + B.T.dot(P_k1).dot(B)).dot(B.T).dot(P_k1).dot(A)
        
        # TODO compute action
        # u = L @ x
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)
        
        # TODO implement recursive least squares update


        x = xp.copy()
        
    total_costs.append(sum(costs))