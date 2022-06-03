import numpy as np

Mp = .027
lp = .153
r = .08260
g = 9.810
Jp = 1.10e-4
Jeq = 1.23e-4
Rm = 3.30
Kt = .02797
Km = .02797

Mp2 = Mp ** 2
lp2 = lp ** 2
r2 = r ** 2


def A(Z1, Z2, Z3, Z4):
  beta = (Mp * r2 * Z2 - Jeq - Mp * r2) * Jp - Mp * lp2 * Jeq

  return np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, -(Mp2 * g * lp2 * Z1) / beta, -((Jp * Mp * r2 * Z3) / beta) - ((Kt * Km * (-Jp + Mp * lp2)) / (beta * Rm)), 0],
    [0, -(lp * Mp * (-Jeq * g * Mp * r2 * Z2 * g - Mp * r2 * g)) / beta,
     - ((lp * Mp * r * Jeq * Z4) / beta) - ((lp * Mp * r * Kt * Km * Z1) / (beta * Rm)), 0]
  ])


def B(Z1, Z2):
  beta = (Mp * r2 * Z2 - Jeq - Mp * r2) * Jp - Mp * lp2 * Jeq

  return np.array([
    [0],
    [0],
    [(Kt * (-Jp + Mp * lp2)) / (beta * Rm)],
    [(Mp * lp * Kt * r * Z1) / (beta * Rm)]
  ]).T


def var_dot(prev_dot, prev_var, var):
  return (.9048 * prev_dot) + (95.24 * var) - (95.24 * prev_var)


def load_matrices(continuous=False):
  f = './data/'

  if continuous:
    f += 'continuous_system/'
  else:
    f += 'discrete_system/'

  _A = np.load(f + 'A.npy')
  _B = np.load(f + 'B.npy')
  _C = np.load(f + 'C.npy')
  _D = np.load(f + 'D.npy')

  return _A, _B, _C, _D
