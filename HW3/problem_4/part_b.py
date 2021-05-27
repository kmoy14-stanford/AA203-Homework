#%%
from model import dynamics, cost
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

total_costs = []
total_loss = []

# Riccati recursion
def Riccati(A,B,Q,R):

    # implement infinite horizon riccati recursion
    P = np.zeros(np.shape(A))
    P_k1 = np.ones(np.shape(A))
    while np.amax(np.absolute(P_k1-P)) > 1e-1:
        # print(np.amax(np.absolute(P_k1-P)))
        P_k1 = deepcopy(P)
        L = -np.linalg.inv(R + B.T.dot(P_k1).dot(B)).dot(B.T).dot(P_k1).dot(A)
        P = Q + A.T.dot(P_k1).dot(A + B.dot(L))
        P *= gamma

    return L,P
#%%
# Import A, B, Q, R here but don't use them -- only use them to create our estimates
A = dynfun.A
B = dynfun.B
Q = costfun.Q
R = costfun.R

# Optimal policy from part (a)
L_star = np.array([[-2.51210964,  1.03523376, -3.10840661, -0.11485777],
       [-0.12845041, -0.95608079, -0.07756695, -1.17061573]])

# Create estimates:
# A_s, B_s, Q_s, and R_s (where A_s = \hat{A} etc.
# A_s, B_s initialized to random variables of size A, B 
# Q_s, R_s initialized to the identity of size Q, R
A_s = np.random.rand(*A.shape)
eig, _ = np.linalg.eig(A_s)
A_s = A_s/np.max(eig)
B_s = np.random.rand(*B.shape)
Q_s = np.eye(*Q.shape)
R_s = np.eye(*R.shape)

# Create supporting arrays for iterative least-squares:
# For dynamics (A, B):
c_ab = np.hstack([A_s, B_s])
p_ab = np.eye(A.shape[1] + B.shape[1])
# For cost function (Q, R):
c_qr = np.hstack([Q_s.flatten(), R_s.flatten()])
p_qr = np.eye(Q_s.size + R_s.size)

#%%
for n in tqdm(range(N)):
    costs = []
    
    x = dynfun.reset()
    for t in range(T):

        # get policy
        L, _ = Riccati(A_s, B_s, Q_s, R_s)
                
        # compute action
        u = L @ x
        
        # get reward
        c = costfun.evaluate(x,u)
        print(c)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)
        
        # implement recursive least squares update
        # update to C_hat = [A_hat, B_hat]
        z_ab = np.hstack([x, u])
        p_ab1 = deepcopy(p_ab)
        p_ab = p_ab1 - np.outer(p_ab1@z_ab.T, p_ab1@z_ab.T) / (1 + z_ab.dot(p_ab1).dot(z_ab.T))
        c_ab = c_ab + np.outer(xp.T - c_ab@z_ab.T, p_ab1@z_ab) / (1 + z_ab.dot(p_ab1).dot(z_ab.T))

        # update Q and R
        z_qr = np.hstack([np.outer(x, x).flatten(), np.outer(u, u).flatten()])
        p_qr1 = deepcopy(p_qr)
        p_qr = p_qr1 - np.outer(p_qr1@z_qr.T, p_qr1@z_qr.T) / (1 + z_qr.dot(p_qr1).dot(z_qr.T))
        c_qr = c_qr + np.outer(c - c_qr@z_qr.T, p_qr1@z_qr) / (1 + z_qr.dot(p_qr1).dot(z_qr.T))

        # retrieve new A_s, B_s, Q_s, R_s
        A_s = c_ab[:, :B_s.shape[0]]
        B_s = c_ab[:, B_s.shape[0]:]

        Q_s = c_qr[0, :Q_s.size].reshape(*Q_s.shape)
        R_s = c_qr[0, Q_s.size:].reshape(*R_s.shape)

        x = xp.copy()
    total_costs.append(sum(costs))
    total_loss.append(np.linalg.norm(L_star - L))
# %%
plt.plot(total_costs)
plt.xlabel('Iteration')
plt.ylabel('Total Cost per Iteration')

# %%
plt.plot(total_loss)
plt.xlabel('Iteration')
plt.ylabel('2-Norm Loss between optimal and derived policy')
# %%
