#%%
import numpy as np
import scipy as sp
import cvxpy as cvx
import copy
import matplotlib.pyplot as plt
#%% Define problem dynamics and cost functions

A = np.array([[1., 1.],[0., 1.]])
B = np.array([[0.],[1.]])
Q = np.eye(2)
R = 0.01*np.eye(1)

#%% TODO: Implement one step of MPC using cvxpy
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

#%% TODO: Implement entire receding horizon control

def mpc(A, B, Q, R, P, N, x_bar, u_bar, x0):
    x = copy.deepcopy(x0)
    x_hist = copy.deepcopy(x0)
    feas = True
    converged, u = opt_finite_traj(A, B, Q, R, P, N, x_bar, u_bar, x0)
    if converged:
        x1 = A @ x0 + B @ u[0]
        x_hist = np.vstack([x_hist, x1])
    else:
        feas = False

    round = 0
    eps = 1e-5
    # TODO: Collapse this into a while function
    for i in range(100):
        if (np.linalg.norm(x-x1) > eps):
            print("round: %s, x update: %s" % (round, x))
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
R = 10.0*np.eye(1)

feas1, x_hist1 = mpc(A, B, Q, R, P, N, x_bar, u_bar, np.array([-4.5, 2]))
feas2, x_hist2 = mpc(A, B, Q, R, P, N, x_bar, u_bar, np.array([-4.5, 3]))

#%% Plot results

plt.plot(x_hist1[:,0], x_hist1[:,1], marker='o')
plt.plot(x_hist2[:,0], x_hist2[:,1],  marker='o')
plt.axis('square')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.axis('square')
#%%
t = cvx.Variable()

# An infeasible problem.
prob = cvx.Problem(cvx.Minimize(t), [t >= 1, t <= 0])
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)

# An unbounded problem.
prob = cvx.Problem(cvx.Minimize(t))
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
# %%
