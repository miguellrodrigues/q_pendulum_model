import numpy as np
import cvxpy as cvx
from system import linear_space_system


A, B, C, D = linear_space_system()

n = A.shape[0]
nu = 1

W = cvx.Variable((n, n), symmetric=True)
Z = cvx.Variable((nu, n))

constraints = [
  W >> np.eye(n) * 1e-9,
  W @ A.T + A @ W + B @ Z + Z.T @ B.T << -np.eye(nu*n) * 1e-9,
]


def find_controller(verbose=False):
  prob = cvx.Problem(
    cvx.Minimize(0),
    constraints
  )

  prob.solve(verbose=verbose, solver='MOSEK')

  W_arr = np.array(W.value)
  Z_arr = np.array(Z.value)

  np.save('./data/controller/W.npy', W_arr)
  np.save('./data/controller/Z.npy', Z_arr)

  K = Z_arr @ np.linalg.inv(W_arr)

  return K


