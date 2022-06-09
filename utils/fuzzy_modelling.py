import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

plt.style.use([
  'science',
  'nature',
  'grid',
])

np.set_printoptions(precision=3, suppress=True)


# defining premisses
def z1(alpha): return np.cos(alpha)


def z2(alpha): return np.sin(alpha) / alpha


def z3(alpha, ad): return np.sin(alpha) * ad


interval = np.array([
  -np.pi, np.pi
])

# defining |alpha_dot|
alpha_dot = 2 * np.pi

x = np.linspace(interval[0], interval[1], 1000)

y_1 = z1(x)
y_2 = z2(x)
y_3 = z3(x, alpha_dot)

min_z1 = np.min(y_1)
max_z1 = np.max(y_1)

min_z2 = np.min(y_2)
max_z2 = np.max(y_2)

min_z3 = np.min(y_3)
max_z3 = np.max(y_3)

print(' ')
print('Interval:', interval)
print(' ')

print(' ')
print('min/max z1:', min_z1, max_z1)
print('min/max z2:', min_z2, max_z2)
print('min/max z3:', min_z3, max_z3)
print(' ')

# pertinence functions
# z1 = M1(Z1) * max_z1 + M2(Z1) * min_z1
# z2 = M1(Z2) * max_z2 + M2(Z2) * min_z2
# z3 = M1(Z3) * max_z3 + M2(Z3) * min_z3

# M1(Zn) + M2(Zn) = 1

# create symbolic variables for pertinence functions
Z1 = sp.Symbol('Z1')
Z2 = sp.Symbol('Z2')
Z3 = sp.Symbol('Z3')

M1 = sp.Symbol('M1')
M2 = 1 - M1

# solve for z1
z1 = M1 * max_z1 + M2 * min_z1
z1_expr = sp.Eq(z1, Z1)
z1_solution = sp.solve(z1_expr, M1)[0]

# solve for z2
z2 = M1 * max_z2 + M2 * min_z2
z2_expr = sp.Eq(z2, Z2)
z2_solution = sp.solve(z2_expr, M1)[0]

# solve for z3
z3 = M1 * max_z3 + M2 * min_z3
z3_expr = sp.Eq(z3, Z3)
z3_solution = sp.solve(z3_expr, M1)[0]

# # # # #

M1 = z1_solution
M2 = 1 - M1

N1 = z2_solution
N2 = 1 - N1

P1 = z3_solution
P2 = 1 - P1

# # # # #

print(' ')
print('M1(Z1):', M1)
print('N1(Z2):', N1)
print('P1(Z3):', P1)
print(' ')
print('M2(Z1):', M2)
print('N2(Z2):', N2)
print('P2(Z3):', P2)
print(' ')

_M1 = sp.lambdify(Z1, M1)
_N1 = sp.lambdify(Z2, N1)
_P1 = sp.lambdify(Z3, P1)


def _M2(a): 1 - _M1(a)
def _N2(a): 1 - _N1(a)
def _P2(a): 1 - _P1(a)


# # # # Plotting

M1_values = _M1(y_1)
N1_values = _N1(y_2)
P1_values = _P1(y_3)

M2_values = 1 - _M1(y_1)
N2_values = 1 - _N1(y_2)
P2_values = 1 - _P1(y_3)

fig, axs = plt.subplots(3, 1, figsize=(8, 8))

axs[0].plot(y_1, M1_values, label='M1')
axs[0].plot(y_1, M2_values, label='M2')

axs[1].plot(y_2, N1_values, label='N1')
axs[1].plot(y_2, N2_values, label='N2')

axs[2].plot(y_3, P1_values, label='P1')
axs[2].plot(y_3, P2_values, label='P2')

# # # #

plt.show()
