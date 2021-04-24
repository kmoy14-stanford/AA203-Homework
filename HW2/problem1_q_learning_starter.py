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
alpha = 0.8 # TODO: Scale this alpha to something that makes sense

# get historical data
data = generate_historical_data(sim)
# historical dataset: 
# shape is 3*365 x 4
# k'th row contains (x_k, u_k, r_k, x_{k+1})

#%% # write Q-learning to yield Q values,

# Q matrix is 6 states x 3 actions:
Q = np.zeros((6,3))
Q_store = np.zeros((6,3,data.shape[0]))

for i in range(data.shape[0]):
    x_k, u_k, r_k, x_k1 = data[i]
    s_k = int(x_k)
    a_k = int(u_k/2)
    s_k1 = int(s_k)
    Q[s_k, a_k] += alpha*(r_k + gamma*max(Q[s_k1]) - Q[s_k,a_k] )
    Q_store[:,:,i] = Q

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
plt.show()

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
while np.amax(np.absolute(Q_VI-Q_VI_prev)) > 1e-3:
    Q_VI_prev = deepcopy(Q_VI)
    for s in range (6):
        for a in range(3):
            # TODO: Handle stochasticity in reward !!
            # Is this just an expectation over demand?? instead of next state??
            
            # # stochastically determine demand
            # d = np.random.choice(sim.valid_demands, p = sim.demand_probs)
            # r = sim.get_reward(s,2*a,d)
            # tq = np.dot(np.array(sim.demand_probs), np.amax(Q_VI,1))
            tq = np.dot(T[s,a,:], np.amax(Q_VI,1))
            Q_VI[s,a] = r + gamma*tq
    print(np.amax(np.absolute(Q_VI-Q_VI_prev)))
    


# x = sim.init_state # Starting x value
# # stochastically determine demand
# demand = np.random.choice(sim.valid_demands, p = sim.demand_probs)
# # find next state
# x_next = sim.transition()
# %%

# %%
