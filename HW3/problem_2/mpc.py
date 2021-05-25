#%% Imports
import numpy as np
import scipy.linalg as sla
import cvxpy as cvx
import copy
import matplotlib.pyplot as plt
#%% Define problem dynamics and cost functions

A = np.array([[1., 1.],[0., 1.]])
B = np.array([[0.],[1.]])
Q = np.eye(2)
R = 0.01*np.eye(1)

#%% Implement one step of MPC using cvxpy
# Input: x(t), A, B, Q, R, P, N, x_bar, u_bar, X_f
# Output: u(t)

def opt_finite_traj(A, B, Q, R, P, N, x_bar, u_bar, x0, X_f0=False):
    converged = False
    n = Q.shape[0] # state dimension
    m = R.shape[0] # control dimension  
    # Initialize variables
    x = cvx.Variable((N+1, n))
    u = cvx.Variable((N, m))
    # Initialize cost, constraint forms
    cost = []
    cost.append(cvx.quad_form(x[N],P))
    constraints = []
    constraints.append(x[0] == x0)
    for k in range(N):
        cost.append(cvx.quad_form(x[k], Q))
        cost.append(cvx.quad_form(u[k], R))
        # constraints.append(u[k] <= uUB)
        # constraints.append(u[k] >= uLB)
        constraints.append(x[k+1] == (A @ (x[k])  + B @ (u[k])))
    constraints.append(u <= u_bar)
    constraints.append(u >= -u_bar)
    constraints.append(x <= x_bar)
    constraints.append(x >= -x_bar)
    if X_f0:
        constraints.append(x[N] == np.array([0, 0]))
    objective = cvx.Minimize(cvx.sum(cost))
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    u_new = u.value
    if prob.status == cvx.OPTIMAL:
        converged = True
    # Note this is either -Inf or Inf if infeasible or unbounded
    return converged, u_new

#%% Implement entire receding horizon control
def mpc(A, B, Q, R, P, N, x_bar, u_bar, x0, X_f0=False):
    x = copy.deepcopy(x0)
    x_hist = copy.deepcopy(x0)
    feas = True
    converged, u = opt_finite_traj(A, B, Q, R, P, N, x_bar, u_bar, x0, X_f0)
    if converged:
        x1 = A @ x0 + B @ u[0]
        x_hist = np.vstack([x_hist, x1])
        round = 0
        eps = 1e-5
        # TODO: Collapse this into a while function
        for i in range(100):
            if (np.linalg.norm(x-x1) > eps):
                # print("round: %s, x update: %s" % (round, x))
                x = x1
                converged, u = opt_finite_traj(A, B, Q, R, P, N, x_bar, u_bar, x)
                if converged:
                    x1 = A @ x + B @ u[0]
                    x_hist = np.vstack([x_hist, x1])
                else:
                    feas = False
                    break
                round += 1
            else:
                break
    else:
        feas = False

    if feas == False:
        feasible = False
    else:
        feasible = True
    
    return feasible, x_hist

#%% Part (b)
# Test quantities
x_bar = 5
u_bar = 0.5
N = 3
P = np.eye(2)
Rb = 10.0*np.eye(1)

feas1, x_hist1 = mpc(A, B, Q, Rb, P, N, x_bar, u_bar, np.array([-4.5, 2]))
feas2, x_hist2 = mpc(A, B, Q, Rb, P, N, x_bar, u_bar, np.array([-4.5, 3]))

# Plot results
plt.plot(x_hist1[:,0], x_hist1[:,1], marker='o')
plt.plot(x_hist2[:,0], x_hist2[:,1],  marker='o')
plt.axis('square')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.axis('square')

#%% Part (c), (d), (e), (f)
#TODO: Construct function to find feasible points
x_bar = 10.0
u_bar = 1.0
N = 6

P_inf = sla.solve_discrete_are(A, B, Q, R)

# discretize state space
x1 = np.linspace(-10, 10, 21)
x2 = np.linspace(-10, 10, 21)

# Find feasible points
x_feas = np.array([0, 0])
eps = 1e-4
for i in x1:
    for j in x2:
        coords = np.array([i, j])
        feas, x_hist = mpc(A, B, Q, R, P_inf, N, x_bar, u_bar, coords)
        if feas & (np.linalg.norm(x_hist1[-1] - np.zeros(2)) < eps):
            x_feas = np.vstack([x_feas, coords])

# %%
plt.scatter(x_feas[:,0], x_feas[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(r'$N = {}, X_f = \mathbb{{R}}^2$'.format(N))
# plt.title(r'$N = {}, X_f = 0$'.format(N))
# %% Part (h)

x0 = np.array([-0.1, 0.01])
feas1, x_hist1 = mpc(A, B, Q, R, P, 2, x_bar, u_bar, x0, True)
feas2, x_hist2 = mpc(A, B, Q, R, P, 3, x_bar, u_bar, x0, True)
feas3, x_hist3 = mpc(A, B, Q, R, P, 6, x_bar, u_bar, x0, True)
feas3, x_hist4 = mpc(A, B, Q, R, P, 12, x_bar, u_bar, x0, True)

#%%
plt.plot(x_hist1[:,0], x_hist1[:,1], marker='o')
plt.plot(x_hist2[:,0], x_hist2[:,1],  marker='o')
plt.plot(x_hist3[:,0], x_hist3[:,1], marker='o')
plt.plot(x_hist4[:,0], x_hist4[:,1],  marker='o')
plt.legend(['N=2', 'N=3', 'N=6', 'N=12'])
# %%
