import numpy as np

'''
Mp = .027
lp = .200
r = .08260
Jm = 3e-5
Marm = .028
g = 9.810
Jp = 2.0e-4
Jeq = 2.3e-4
Beq = .0
Bp = .0
Rm = 3.30
Kt = .028
Km = .028

Mp2 = Mp ** 2
lp2 = lp ** 2
r2 = r ** 2
'''


def var_dot(prev_dot, prev_var, var):
  return (.9048 * prev_dot) + (95.24 * var) - (95.24 * prev_var)


def load_matrices(continuous=False):
  f = './data/'

  if continuous:
    f += 'continuous_system/'
  else:
    f += 'discrete_system/'

  A = np.load(f + 'A.npy')
  B = np.load(f + 'B.npy')
  C = np.load(f + 'C.npy')
  D = np.load(f + 'D.npy')

  return A, B, C, D
