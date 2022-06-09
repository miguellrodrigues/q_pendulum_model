import numpy as np

Jeq = .000184
Mp = .0270
r = .0826
lp = .153
Jp = .000170
g = 9.81
Beq = .004
Bp = .0
Kt = .03334
Km = .03334
Rm = 8.7

Mp2 = Mp ** 2
lp2 = lp ** 2
r2 = r ** 2

a = Jeq + Mp * r2
b = Mp * lp * r
c = (Jp + Mp * lp2)
d = Mp * g * lp

G = (Kt * Km) / Rm


def A(alpha, alpha_dot):
  E = a * c - ((b ** 2) * (np.cos(alpha) ** 2))

  return np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (b * d * np.sin(alpha) * np.cos(alpha)) / (E * alpha), -((Beq + G) * c) / E,
     -(b * Bp + b * c * np.sin(alpha) * alpha_dot) / E],
    [0, (a * d * np.sin(alpha)) / (E * alpha), -(b * (G + Beq) * np.cos(alpha)) / E,
     -((b ** 2) * alpha_dot * np.sin(alpha) * np.cos(alpha) + Bp) / E]
  ])


def B(alpha):
  E = a * c - b ** 2 * np.cos(alpha) ** 2

  return np.array([
    [0],
    [0],
    [((c * Kt) / Rm) / E],
    [((b * Kt) / (Rm * np.cos(alpha))) / E]
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
