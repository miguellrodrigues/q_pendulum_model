import matplotlib.pyplot as plt
import numpy as np

from system import A, B

plt.style.use([
  'science',
  'nature',
  'grid',
])

# system data

z1_min, z1_max = -1, 1
z2_min, z2_max = 0, 1
z3_min, z3_max = -6.2831775400855925, 6.2831775400855925

A_matrices = [
  A(z1_min, z2_min, z3_min),
  A(z1_min, z2_min, z3_max),
  A(z1_min, z2_max, z3_min),
  A(z1_min, z2_max, z3_max),
  A(z1_max, z2_min, z3_min),
  A(z1_max, z2_min, z3_max),
  A(z1_max, z2_max, z3_min),
  A(z1_max, z2_max, z3_max),
]

B_matrices = [
  B(z1_min),
  B(z1_min),
  B(z1_min),
  B(z1_min),
  B(z1_max),
  B(z1_max),
  B(z1_max),
  B(z1_max),
]

n = len(A_matrices)

A_rows = A_matrices[0].shape[0]
A_cols = A_matrices[0].shape[1]

B_rows = B_matrices[0].shape[0]
B_cols = B_matrices[0].shape[1]


# premisses
def z1(a): return np.cos(a)
def z2(a): return np.sin(a) / a
def z3(a, ad): return np.sin(a) * ad


# pertinence functions
def M1(a): return .5*z1(a) + .5
def N1(a): return z2(a)
def P1(a, ad): return .0795775699174638*z3(a, ad) + .5


# complements
def M2(a): return 1 - M1(a)
def N2(a): return 1 - N1(a)
def P2(a, ad): return 1 - P1(a, ad)


pertinence_functions = np.array([
  [M1, M2],
  [N1, N2],
  [P1, P2],
])

# simulation

initial_conditions = np.array([
  [.0],
  [.00001],
  [.0],
  [.0]
])

x = np.copy(initial_conditions)

samples = 10000
simulation_step = 1e-3

time_values = np.arange(0, samples*simulation_step, simulation_step)

theta_values = np.zeros((samples, 2))
alpha_values = np.zeros((samples, 2))

theta_values[0] = np.array([initial_conditions[0], initial_conditions[2]]).T
alpha_values[0] = np.array([initial_conditions[1], initial_conditions[3]]).T

u = np.array([
  [.0],
  [.0],
  [.0],
  [.0]
])

for i in range(1, samples):
  alpha = alpha_values[i - 1, 0]
  alpha_dot = np.clip(alpha_values[i - 1, 1], z3_min, z3_max)

  # calculating the pertinence functions activation values

  pertinence_values = np.array([
    [pertinence_functions[0, 0](alpha), pertinence_functions[0, 1](alpha)],
    [pertinence_functions[1, 0](alpha), pertinence_functions[1, 1](alpha)],
    [pertinence_functions[2, 0](alpha, alpha_dot), pertinence_functions[2, 1](alpha, alpha_dot)],
  ])

  # calculating weights
  weights = np.array([
    pertinence_values[0, 1] * pertinence_values[1, 1] * pertinence_values[2, 1],
    pertinence_values[0, 1] * pertinence_values[1, 1] * pertinence_values[2, 0],
    pertinence_values[0, 1] * pertinence_values[1, 0] * pertinence_values[2, 1],
    pertinence_values[0, 1] * pertinence_values[1, 0] * pertinence_values[2, 0],
    pertinence_values[0, 0] * pertinence_values[1, 1] * pertinence_values[2, 1],
    pertinence_values[0, 0] * pertinence_values[1, 1] * pertinence_values[2, 0],
    pertinence_values[0, 0] * pertinence_values[1, 0] * pertinence_values[2, 1],
    pertinence_values[0, 0] * pertinence_values[1, 0] * pertinence_values[2, 0],
  ])

  # Calculating Ai and Bi
  # Ai = w1*A1 + w2*A2 + w3*A3 + w4*A4 ...
  Ai = sum(weights[j] * A_matrices[j] for j in range(n))
  Bi = sum(weights[j] * B_matrices[j] for j in range(n))

  delta_system = Ai @ x
  x += delta_system * simulation_step

  theta_values[i] = np.array([x[0], x[2]]).T
  alpha_values[i] = np.array([x[1], x[3]]).T

fig1, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0][0].plot(time_values, theta_values[:, 0], color='blue')

axs[0][0].legend(['theta'])
axs[0][0].set_xlabel('time [s]')
axs[0][0].set_ylabel('angle [rad]')

axs[0][1].plot(time_values, theta_values[:, 1], color='blue')

axs[0][1].legend(['theta dot'])
axs[0][1].set_xlabel('time [s]')
axs[0][1].set_ylabel('angle [rad/s]')

# # # # # # # # # # # # # # # # # # # #

axs[1][0].plot(time_values, alpha_values[:, 0], color='red')

axs[1][0].legend(['alpha'])
axs[1][0].set_xlabel('time [s]')
axs[1][0].set_ylabel('angle [rad]')

axs[1][1].plot(time_values, alpha_values[:, 1], color='red')

axs[1][1].legend(['alpha dot', 'alpha dot approximation'])
axs[1][1].set_xlabel('time [s]')
axs[1][1].set_ylabel('angle [rad/s]')

plt.savefig('./figures/sim_states.png', dpi=300)
plt.show()
