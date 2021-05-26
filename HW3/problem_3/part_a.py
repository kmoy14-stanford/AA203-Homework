#%% Imports
import numpy as np
import scipy.linalg as sla
import cvxpy as cvx
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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

#%% Implement one step of MPC using cvxpy
# Input: x(t), A, B, Q, R, P, N, x0, M, X_f
# Output: u(t)

def opt_finite_traj(A, B, Q, R, P, N, x0, M, X_f0, Pf):
    converged = False
    n = Q.shape[0] # state dimension
    m = R.shape[0] # control dimension  
    # Initialize variables
    x = cvx.Variable((N+1, n))
    u = cvx.Variable((N, m))
    # Initialize cost, constraint forms
    cost = []
    if Pf:
        cost.append(cvx.quad_form(x[N],P))
    constraints = []
    constraints.append(x[0] == x0)
    for k in range(N):
        cost.append(cvx.quad_form(x[k], Q))
        cost.append(cvx.quad_form(u[k], R))
        constraints.append(cvx.norm(u[k]) <= 1)
        constraints.append(cvx.norm(x[k]) <= 5)
        constraints.append(x[k+1] == (A @ (x[k])  + B @ (u[k])))
    constraints.append(cvx.norm(x[N]) <= 5)
    if X_f0:
        constraints.append(cvx.quad_form(x[N], M) <= 1)
    objective = cvx.Minimize(cvx.sum(cost))
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    u_new = u.value
    if prob.status == cvx.OPTIMAL:
        converged = True
    # Note this is either -Inf or Inf if infeasible or unbounded
    return converged, u_new

#%% Implement entire receding horizon control
def mpc(A, B, Q, R, P, N, x0, M, X_f0, Pf):
    x = copy.deepcopy(x0)
    x_hist = copy.deepcopy(x0)
    feas = True
    converged, u = opt_finite_traj(A, B, Q, R, P, N, x0, M, X_f0, Pf)
    if converged:
        x1 = A @ x0 + B @ u[0]
        x2 = A @ x1 + B @ u[1]
        x3 = A @ x2 + B @ u[2]
        x4 = A @ x3 + B @ u[3]
        x_hist = np.vstack([x_hist, x1])
        u_hist = u[0]
        x_traj = np.vstack([x0, x1, x2, x3, x4])
        round = 0
        eps = 1e-5
        # TODO: Collapse this into a while function
        for i in range(100):
            if (np.linalg.norm(x-x1) > eps):
                x = x1
                converged, u = opt_finite_traj(A, B, Q, R, P, N, x, M, X_f0, Pf)
                if converged:
                    x1 = A @ x + B @ u[0]
                    x2 = A @ x1 + B @ u[1]
                    x3 = A @ x2 + B @ u[2]
                    x4 = A @ x3 + B @ u[3]
                    x_hist = np.vstack([x_hist, x1])
                    u_hist = np.vstack([u_hist, u[0]])
                    x_traj_i = np.vstack([x, x1, x2, x3, x4])
                    x_traj = np.dstack([x_traj, x_traj_i])
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
    
    return feasible, x_hist, u_hist, x_traj
#%% Define problem dynamics and cost functions
A = np.array([[0.95, 0.5],[0., 0.95]])
B = np.array([[0.],[1.]])
Q = np.eye(2)
R = np.eye(1)

M = np.array([[0.04, 0.],[0., 1.06]])
test = M - A.T.dot(M).dot(A)
test_eig, _ = np.linalg.eig(test)
# %%
L_inf, P_inf = Riccati(A, B, Q, R)
# %%
x0 = np.array([-3.0, -2.5])
N = 4
feas, x_hist, u_hist, x_traj = mpc(A, B, Q, R, P_inf, N, x0, M, X_f0=True, Pf=False)

#%% plot trajectory at each timestep
num_plots = np.size(u_hist)
colormap = plt.cm.gist_ncar
fig, ax = plt.subplots(figsize=(6, 6))
M_eig, M_ev = np.linalg.eig(M)
r1 = 1/np.sqrt(M_eig[0])
r2 = 1/np.sqrt(M_eig[1])
ellipse = Ellipse((0,0), width=r1*2, height=r2*2, fill=False)
ellipse2 = Ellipse((0,0), width=10, height=10, fill=False, color = 'red')
ax.add_patch(ellipse)
ax.add_patch(ellipse2)
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
for i in range(num_plots):
    x_tj = x_traj[:,:,i]
    plt.plot(x_tj[:,0], x_tj[:,1])
    plt.scatter(x_hist[i,0], x_hist[i,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Terminal Constraint, No Terminal Cost')
#%% Plot control
plt.plot(u_hist)
plt.xlabel('iteration')
plt.ylabel('control')
plt.title('No Terminal Constraint, No Terminal Cost')

# %%
