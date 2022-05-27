import numpy as np
import matplotlib.pyplot as plt
from system import linear_space_system, alpha_dot
from controller import find_controller

plt.style.use([
  'science',
  'nature',
  'grid',
])

plt.rcParams["font.family"] = "FreeSerif, Regular"
plt.rcParams['font.size'] = 12

sine_signal = np.load('../data/sine_signal.npy')

A, B, C, D = linear_space_system()
K = find_controller()

initial_condition = np.array([
  [.0],
  [np.radians(10)],
  [.0],
  [.0]
])

x = np.copy(initial_condition)

simulation_time = 10
simulation_step = 1e-3

iterations = int(simulation_time / simulation_step)

time = np.arange(0, simulation_time, simulation_step)

theta_values = np.zeros((iterations, 2))
alpha_values = np.zeros((iterations, 2))

theta_values[0] = np.array([x[0], x[2]]).T
alpha_values[0] = np.array([x[1], x[3]]).T

us = np.zeros((iterations, 1))

ad = np.zeros((iterations, 1))

for i in range(1, iterations):
  u = K @ x

  delta_system = (A @ x) + (B @ u)
  x += (delta_system * simulation_step)

  theta_values[i] = np.array([x[0], x[2]]).T
  alpha_values[i] = np.array([x[1], x[3]]).T

  alpha_dot_k = alpha_dot(alpha_values[i - 1, 1], alpha_values[i - 1, 0], alpha_values[i, 0])
  ad[i] = alpha_dot_k

  us[i] = u


fig1, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0][0].plot(time, theta_values[:, 0], color='blue')

axs[0][0].legend(['theta'])
axs[0][0].set_xlabel('time [s]')
axs[0][0].set_ylabel('angle [rad]')

axs[0][1].plot(time, theta_values[:, 1], color='blue')

axs[0][1].legend(['theta dot'])
axs[0][1].set_xlabel('time [s]')
axs[0][1].set_ylabel('angle [rad/s]')

# # # # # # # # # # # # # # # # # # # #

axs[1][0].plot(time, alpha_values[:, 0], color='red')

axs[1][0].legend(['alpha'])
axs[1][0].set_xlabel('time [s]')
axs[1][0].set_ylabel('angle [rad]')

axs[1][1].plot(time, alpha_values[:, 1], color='red')
axs[1][1].plot(time, ad, '--', color='black')

axs[1][1].legend(['alpha dot', 'alpha dot approximation'])
axs[1][1].set_xlabel('time [s]')
axs[1][1].set_ylabel('angle [rad/s]')

plt.savefig('./figures/sim_states.png', dpi=300)

fig2, axs = plt.subplots(1, 1, figsize=(10, 10))

axs.plot(time, us, color='green')
axs.set_xlabel('time [s]')
axs.set_ylabel('control input [V]')

plt.savefig('./figures/sim_u.png', dpi=300)
plt.show()
