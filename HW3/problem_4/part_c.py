#%%
from model import dynamics, cost
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100
N = 100
gamma = 0.95 # discount factor

total_costs = []
total_loss = []

# Optimal policy from part (a)
L_star = np.array([[-2.51210964,  1.03523376, -3.10840661, -0.11485777],
       [-0.12845041, -0.95608079, -0.07756695, -1.17061573]])

# Supporting matrices
# just hardcode for now that len(x) = 4, len(u) = 2, len(x) + len(u) = 6
theta = 2*np.eye(6).flatten()
# theta = np.random.rand(6,6).flatten()
# theta = np.ones([6,6]).flatten()
sigma = np.eye(2)

P = 1e2*np.eye(6**2)

# Initialize policy
H = theta.reshape(6,6)

H = 0.5*(H + H.T)

H_22 = H[4:6, 4:6]
H_21 = H[4:6, 0:4]

L = np.linalg.pinv(H_22) @ H_21

for n in range(N):
    costs = []

    # P = 1e2*np.eye(6**2)
    
    x = dynfun.reset()
    for t in range(T):
        
        # TODO compute action
        u = np.random.multivariate_normal(-L @ x, sigma)

        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)

        # TODO recursive least squares policy evaluation step

        # Current x_t, u_t
        xu = np.hstack([x,u])
        xu_bar = np.outer(xu, xu).flatten()
        
        # Next step x_t+1, u_t+1
        up = -L@xp
        xup = np.hstack([xp, up])
        xup_bar = np.outer(xup, xup).flatten()

        # phi_t
        phi = xu_bar - gamma*xup_bar
        # phi = phi / np.amax(phi)

        P1 = deepcopy(P)
        theta1 = deepcopy(theta)
        P = P1 - np.outer(P1@phi.T, P1@phi.T) / (1 + phi.dot(P1).dot(phi.T))
        theta = theta1 + ((P1@phi)*(c.T - phi.T@theta1)) / (1 + phi.dot(P1).dot(phi.T))
        
        # P1 = P - np.outer(P@phi.T, P@phi.T) / (1 + phi.dot(P).dot(phi.T))
        # theta1 = theta + ((P1@phi) * (c.T - phi.T@theta)) / (1 + phi.dot(P1).dot(phi.T))

        # P = P1.copy()
        # theta = theta1.copy()
        x = xp.copy()
    # print(c.T - phi.T@theta)
    # TODO policy improvement step
    H = theta.reshape(6,6)

    H = 0.5*(H + H.T)

    H_22 = H[4:6, 4:6]
    H_21 = H[4:6, 0:4]

    L = np.linalg.pinv(H_22) @ H_21

    # Save iter results
    total_loss.append(np.linalg.norm(L_star - L))

    total_costs.append(sum(costs))

plt.plot(total_costs)
plt.xlabel('Iteration')
plt.ylabel('Total Cost per Iteration')

# %%
plt.plot(total_loss)
plt.xlabel('Iteration')
plt.ylabel('2-Norm Loss between optimal and derived policy')
# %%# %%
# fig, axs = plt.subplots(2)
# axs[0].plot(total_loss)
# axs[0].set(ylabel='Loss')
# axs[1].plot(total_costs[1:])
# plt.xlabel('Iteration')
# %%
