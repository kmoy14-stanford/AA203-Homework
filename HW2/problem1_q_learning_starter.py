#%%
from problem1_q_learning_env import *
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
my_path = os.path.dirname(os.path.realpath(__file__))

# initialize simulator
sim = simulator()

T = 5*365 # simulation duration
gamma = 0.95 # discount factor
alpha = 0.1 # TODO: Scale this alpha to something that makes sense

# get historical data
data = generate_historical_data(sim)
# historical dataset: 
# shape is 3*365 x 4
# k'th row contains (x_k, u_k, r_k, x_{k+1})

#%% # write Q-learning to yield Q values,

# Q matrix is 6 states x 3 actions:
Q = np.zeros((6,3))
Q_prev = np.ones((6,3))
Q_store = np.zeros((6,3))
eps = 1e-4

while np.linalg.norm(Q-Q_prev, np.inf) > eps:
    print(np.linalg.norm(Q-Q_prev, np.inf))
    Q_prev = deepcopy(Q)
# for i in range(50):
    j = 0
    for i in range(data.shape[0]):
        x_k, u_k, r_k, x_k1 = data[i]
        s_k = int(x_k)
        a_k = int(u_k/2)
        s_k1 = int(x_k1)
        nextq = (1 - alpha) * Q[s_k, a_k] + alpha*( r_k + gamma*max(Q[s_k1]))
        # print(nextq)
        Q[s_k, a_k] = nextq
        # print(Q)
        Q_store = np.dstack((Q_store, Q))

#%% Plot Q-values for each state
for widgets_stored in range(6):
    fig, ax = plt.subplots()
    lines = ax.plot(Q_store[widgets_stored,:,:].T)
    ax.legend(lines, ["No order", "Half order", "Full order"])
    plt.xlabel('Iterations')
    plt.ylabel('Q-value')
    plt.title(r'{} Widgets Stored, $\alpha$ = {}'.format(widgets_stored, alpha))
    plt.grid()
    plt.savefig(os.path.join(my_path, 'Q_values_{}_widgets_alpha_{}.png'.format(widgets_stored, alpha)), dpi=300)
#%% use Q values in policy 
def policy(state,Q):
    s = int(state)
    action = np.argmax(Q[s])
    return 2*action
    # TODO fill in 
    
#%% Forward simulating the system 
s = sim.reset()
reward = 0
reward_store = np.zeros(T)
for t in range(T):
    a = policy(s,Q)
    sp,r = sim.step(a)
    s = sp
    reward += r
    reward_store[t] = reward

#%% Plot cumulative rewards over time
plt.plot(reward_store)
plt.xlabel('Iterations')
plt.ylabel('Cumulative Profit')
plt.grid()
plt.savefig(os.path.join(my_path, 'sim_reward_alpha_{}.png'.format(alpha)), dpi=300)
# wow this looks bad is this correct??

#%% TODO: write value iteration to compute true Q values
# use functions:
# - sim.transition (dynamics)
# - sim.get_reward 
# plus sim.demand_probs for the probabilities associated with each demand value

# TODO: For each s,a pair, calculate the transition probability to create the T matrix
# up front: dimension s x a x s (state, action, next state)
T = np.zeros((6,3,6))
R = np.zeros((6,3,6))
for s in range(6):
    for a in range(3):
        next_states = np.zeros(6)
        for d_p in range(6):
            # Calculate next state given s and a for each d. 
            next_states[d_p] = sim.transition(s,2*a,d_p)
        # print(next_states)
        for ns in range(6):
            # Then the next states occur with probability determined by the next states.
            T[s,a,ns] = np.count_nonzero(next_states == ns)/6


#%%
Q_VI = np.zeros((6,3))
Q_VI_prev = np.ones((6,3))
V = np.zeros(6)
eps = 1e-3
niter = 0
while np.linalg.norm(Q_VI-Q_VI_prev, np.inf) > eps:
    Q_VI_prev = deepcopy(Q_VI)
    for s in range(6):
        # s = sim.init_state # initialize state
        s_next = np.zeros((3,6)) # index by action, demand
        r = np.zeros((3,6))
        V_a = np.zeros(3)
        V_next = np.zeros((3,6))
        probs = np.array(sim.demand_probs)
        for a in range(3):
            act = 2*a
            for d in range(6):
                nexts = sim.transition(s,act,d)
                s_next[a,d] = nexts
                r[a,d] = sim.get_reward(s,act,d)
                V_next[a,d] = V[int(nexts)]

        V_a = r*probs + gamma*V_next*probs
        V[s] = max(np.sum(V_a, axis=1))
        Q_VI[s] = np.sum(V_a, axis=1)
    niter += 1
# %%
