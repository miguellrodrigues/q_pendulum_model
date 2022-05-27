import numpy as np
import cvxpy as cvx
from system import load_matrices


A, B, C, D = load_matrices(continuous=False)

n = A.shape[0]
m = B.shape[1]


W = cvx.Variable((n, n), symmetric=True)
S = cvx.Variable((m, m), diag=True)
Z = cvx.Variable((m, n))
L = cvx.Variable((m, n))
rho = 10

# # LMI 1
LMI1_11 = -W
LMI1_12 = -Z.T
LMI1_13 = L.T@B.T + W.T@A.T

LMI1_21 = -Z
LMI1_22 = -2*S
LMI1_23 = S@B.T

LMI1_31 = A@W+B@L
LMI1_32 = B@S
LMI1_33 = -W

LMI1_1 = cvx.hstack([LMI1_11, LMI1_12, LMI1_13])
LMI1_2 = cvx.hstack([LMI1_21, LMI1_22, LMI1_23])
LMI1_3 = cvx.hstack([LMI1_31, LMI1_32, LMI1_33])

LMI1 = cvx.vstack([LMI1_1, LMI1_2, LMI1_3])

# # END LMI 1

# # LMI 2

LMI2_11 = W
LMI2_12 = L.T - Z.T
LMI2_21 = L - Z
LMI2_22 = np.array([
  [rho ** 2]
])

LMI2_1 = cvx.hstack([LMI2_11, LMI2_12])
LMI2_2 = cvx.hstack([LMI2_21, LMI2_22])

LMI2 = cvx.vstack([LMI2_1, LMI2_2])

# # END LMI 2

constraints = [
  LMI1 << -np.eye(9) * 1e-9,
  LMI2 >> np.eye(5) * 1e-9,
]


def find_controller():
  prob = cvx.Problem(
    cvx.Minimize(0),
    constraints
  )

  prob.solve(verbose=False, solver='MOSEK')

  W_arr = np.array(W.value)
  L_arr = np.array(L.value)

  K = L_arr @ np.linalg.inv(W_arr)

  return K
