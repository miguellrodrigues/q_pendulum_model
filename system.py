import numpy as np


class Quantity:
  def __init__(self, value, unit):
    self.value = value
    self.unit = unit

  def __repr__(self):
    return f"{self.value}{self.unit}"


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

A = np.array([
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  [0, (r * Mp2 * lp2 * g) / (Jp * Jeq + Mp * lp2 * Jeq + Jp * Mp * r2),
   -(Kt * Km * (Jp + Mp * lp2)) / ((Jp * Jeq + Mp * lp2 * Jeq + Jp * Mp * r2) * Rm), 0],
  [0, (Mp * lp * g * (Jeq + Mp * r2)) / (Jp * Jeq + Mp * lp2 * Jeq + Jp * Mp * r2),
   -(Mp * lp * Kt * r * Km) / ((Jp * Jeq + Mp * lp2 * Jeq + Jp * Mp * r2) * Rm), 0]
])

B = np.array([
  [0],
  [0],
  [(Kt * (Jp + Mp * lp2)) / ((Jp * Jeq + Mp * lp2 * Jeq + Jp * Mp * r2) * Rm)],
  [(Mp * lp * Kt * r) / ((Jp * Jeq + Mp * lp2 * Jeq + Jp * Mp * r2) * Rm)]
])

C = np.eye(4)

D = np.array([
  0, 0, 0, 0
]).T


def linear_space_system():
  return A, B, C, D

