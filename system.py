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
e = Beq + (Kt * Km) / Rm
f = Kt / Km

G = (Kt * Km) / Rm


def A(Z1, Z2, Z3):
  E = a * c - ((b ** 2) * (Z1 ** 2))

  return np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (b * d * Z1 * Z2) / E, -(c * e) / E, -(b * c * Z3) / E],
    [0, (a * d * Z2) / E, -(b * e * Z1) / E, -((b ** 2) * Z1 * Z3) / E]
  ])


def B(Z1):
  E = a * c - ((b ** 2) * (Z1 ** 2))

  return np.array([
    [0],
    [0],
    [(c * f) / E],
    [(b * f * Z1) / E]
  ]).T


def var_dot(prev_dot, prev_var, var):
  return (.9048 * prev_dot) + (95.24 * var) - (95.24 * prev_var)

