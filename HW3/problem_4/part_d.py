#%%
from copy import deepcopy
from model import dynamics, cost
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L_star = np.array([[-2.51210964,  1.03523376, -3.10840661, -0.11485777],
       [-0.12845041, -0.95608079, -0.07756695, -1.17061573]])

stochastic_dynamics = True # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100
N = 10000 
gamma = 0.95 # discount factor

total_costs = []
total_loss = []

W = np.zeros([2,4])
Sigma = 0.1*np.eye(2)
alpha = 1e-13
#%%
for n in tqdm(range(N)):
    costs = []
    W_n = deepcopy(W)
    Sig_inv = np.linalg.inv(Sigma)
    
    x_hist = []
    u_hist = []
    r_hist = []

    x = dynfun.reset()
    for t in range(T):

        # compute action
        u = np.random.multivariate_normal(W@x, Sigma) 

        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)

        # save x, u, r
        x_hist.append(x)
        u_hist.append(u)
        r_hist.append(c)

        x = xp.copy()

    # update policy

    for t in range(T):
        x_t = x_hist[t]
        u_t = u_hist[t]
        G_t = np.sum(r_hist[t:])
        grad = np.outer((Sig_inv + Sig_inv.T)@(W@x_t - u_t), x_t)
        W_t = deepcopy(W_n)
        W_n = W_t + alpha*G_t*grad
    
    W = deepcopy(W_n)

    total_costs.append(sum(costs))
    total_loss.append(np.linalg.norm(L_star - W))

plt.plot(total_costs)
plt.xlabel('Iteration')
plt.ylabel('Total Cost per Iteration')

# %%
plt.plot(total_loss)
plt.xlabel('Iteration')
plt.ylabel('2-Norm Loss between optimal and derived policy')
# %%

# %%
