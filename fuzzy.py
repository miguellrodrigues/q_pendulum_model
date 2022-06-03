import numpy as np
import matplotlib.pyplot as plt
from system import A, B


# system data

z1_min, z1_max = 0, 1
z2_min, z2_max = 0, 1
z3_min, z3_max = -np.pi/2, np.pi/2
z4_min, z4_max = -np.pi, np.pi


A_matrices = [
    A(z1_min, z2_min, z3_min, z4_min),
    A(z1_min, z2_min, z3_min, z4_max),
    A(z1_min, z2_min, z3_max, z4_min),
    A(z1_min, z2_min, z3_max, z4_max),
    A(z1_min, z2_max, z3_min, z4_min),
    A(z1_min, z2_max, z3_min, z4_max),
    A(z1_min, z2_max, z3_max, z4_min),
    A(z1_min, z2_max, z3_max, z4_max),
    A(z1_max, z2_min, z3_min, z4_min),
    A(z1_max, z2_min, z3_min, z4_max),
    A(z1_max, z2_min, z3_max, z4_min),
    A(z1_max, z2_min, z3_max, z4_max),
    A(z1_max, z2_max, z3_min, z4_min),
    A(z1_max, z2_max, z3_min, z4_max),
    A(z1_max, z2_max, z3_max, z4_min),
    A(z1_max, z2_max, z3_max, z4_max)
]

B_matrices = [
    B(z1_min, z2_min),
    B(z1_min, z2_min),
    B(z1_min, z2_min),
    B(z1_min, z2_min),

    B(z1_min, z2_max),
    B(z1_min, z2_max),
    B(z1_min, z2_max),
    B(z1_min, z2_max),

    B(z1_max, z2_min),
    B(z1_max, z2_min),
    B(z1_max, z2_min),
    B(z1_max, z2_min),

    B(z1_max, z2_max),
    B(z1_max, z2_max),
    B(z1_max, z2_max),
    B(z1_max, z2_max)
]

n = len(A_matrices)

A_rows = A_matrices[0].shape[0]
A_cols = A_matrices[0].shape[1]

B_rows = B_matrices[0].shape[0]
B_cols = B_matrices[0].shape[1]

# premisses

z1 = lambda theta: np.cos(theta)
z2 = lambda theta: np.sin(theta) ** 2
z3 = lambda theta, theta_dot: np.cos(theta) * np.sin(theta) * theta_dot
z4 = lambda theta, theta_dot: np.sin(theta) * theta_dot

# pertinence functions

M1 = lambda theta: 1 - z1(theta)
M2 = lambda theta: 1 - M1(theta)

N1 = lambda theta: 1 - z2(theta)
N2 = lambda theta: 1 - N1(theta)

P1 = lambda theta, theta_dot: (-4*z3(theta, theta_dot) + 2*np.pi) / (4*np.pi)
P2 = lambda theta, theta_dot: 1 - P1(theta, theta_dot)

Q1 = lambda theta, theta_dot: (2*np.pi - z4(theta, theta_dot)) / (2*np.pi)
Q2 = lambda theta, theta_dot: 1 - Q1(theta, theta_dot)

# M1 = lambda theta: (1 - z1(theta)) / 2.0
# M2 = lambda theta: (1 + z1(theta)) / 2.0

# N1 = lambda theta: 1 - z2(theta)
# N2 = lambda theta: z2(theta)

# P1 = lambda theta, theta_dot: (-z3(theta, theta_dot) + np.pi) / (2*np.pi)
# P2 = lambda theta, theta_dot: (z3(theta, theta_dot) + np.pi) / (2*np.pi)

# Q1 = lambda theta, theta_dot: (2*np.pi - z4(theta, theta_dot)) / (4*np.pi)
# Q2 = lambda theta, theta_dot: (2*np.pi + z4(theta, theta_dot)) / (4*np.pi)

pertinence_functions = np.array([
    [M1, M2],
    [N1, N2],
    [P1, P2],
    [Q1, Q2]
])

# simulation

initial_conditions = np.array([
    [.0],
    [np.deg2rad(45)],
    [.0],
    [.0]
])

x = np.copy(initial_conditions)

simulation_time = 10  # seconds
simulation_step = 1e-3

iterations = int(simulation_time / simulation_step)
time_values = np.arange(0, simulation_time, simulation_step)

theta_values = np.zeros((iterations, 2))
alpha_values = np.zeros((iterations, 2))

theta_values[0] = np.array([initial_conditions[0], initial_conditions[2]]).T
alpha_values[0] = np.array([initial_conditions[1], initial_conditions[3]]).T

u = np.array([
    [.0],
    [.0],
    [.0],
    [.0]
])

for i in range(1, iterations):
    theta = theta_values[i - 1, 0]
    theta_dot = np.clip(theta_values[i - 1, 1], -np.pi, np.pi)

    # calculating the pertinence functions acvation values
    pertinence_values = np.array([
        [pertinence_functions[0, 0](theta),            pertinence_functions[0, 1](theta)],
        [pertinence_functions[1, 0](theta),            pertinence_functions[1, 1](theta)],
        [pertinence_functions[2, 0](theta, theta_dot), pertinence_functions[2, 1](theta, theta_dot)],
        [pertinence_functions[3, 0](theta, theta_dot), pertinence_functions[3, 1](theta, theta_dot)]
    ])

    # calculating weights
    weights = np.array([
        pertinence_values[0, 0] * pertinence_values[1, 0] * pertinence_values[2, 0] * pertinence_values[3, 0],
        pertinence_values[0, 0] * pertinence_values[1, 0] * pertinence_values[2, 0] * pertinence_values[3, 1],
        pertinence_values[0, 0] * pertinence_values[1, 0] * pertinence_values[2, 1] * pertinence_values[3, 0],
        pertinence_values[0, 0] * pertinence_values[1, 0] * pertinence_values[2, 1] * pertinence_values[3, 1],
        pertinence_values[0, 0] * pertinence_values[1, 1] * pertinence_values[2, 0] * pertinence_values[3, 0],
        pertinence_values[0, 0] * pertinence_values[1, 1] * pertinence_values[2, 0] * pertinence_values[3, 1],
        pertinence_values[0, 0] * pertinence_values[1, 1] * pertinence_values[2, 1] * pertinence_values[3, 0],
        pertinence_values[0, 0] * pertinence_values[1, 1] * pertinence_values[2, 1] * pertinence_values[3, 1],
        pertinence_values[0, 1] * pertinence_values[1, 0] * pertinence_values[2, 0] * pertinence_values[3, 0],
        pertinence_values[0, 1] * pertinence_values[1, 0] * pertinence_values[2, 0] * pertinence_values[3, 1],
        pertinence_values[0, 1] * pertinence_values[1, 0] * pertinence_values[2, 1] * pertinence_values[3, 0],
        pertinence_values[0, 1] * pertinence_values[1, 0] * pertinence_values[2, 1] * pertinence_values[3, 1],
        pertinence_values[0, 1] * pertinence_values[1, 1] * pertinence_values[2, 0] * pertinence_values[3, 0],
        pertinence_values[0, 1] * pertinence_values[1, 1] * pertinence_values[2, 0] * pertinence_values[3, 1],
        pertinence_values[0, 1] * pertinence_values[1, 1] * pertinence_values[2, 1] * pertinence_values[3, 0],
        pertinence_values[0, 1] * pertinence_values[1, 1] * pertinence_values[2, 1] * pertinence_values[3, 1],
    ])
    
    # Calculating Ai and Bi
    # Ai = w1*A1 + w2*A2 + w3*A3 + w4*A4 ...
    Ai = sum(weights[j] * A_matrices[j] for j in range(n))
    Bi = sum(weights[j] * B_matrices[j] for j in range(n))
    
    delta_system = Ai@x
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
