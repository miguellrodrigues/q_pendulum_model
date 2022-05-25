import numpy as np
import matplotlib.pyplot as plt
from system import linear_space_system

plt.style.use([
  'science',
  'nature',
  'grid',
])

plt.rcParams["font.family"] = "FreeSerif, Regular"
plt.rcParams['font.size'] = 12

sine_signal = np.load('./data/sine_signal.npy')

A, B, C, D = linear_space_system()

initial_condition = np.array([
  .0, np.pi, .0, .0
]).T

x = np.copy(initial_condition)

simulation_time = 15
simulation_step = 1e-3

iterations = int(simulation_time / simulation_step)

time = np.arange(0, simulation_time, simulation_step)

theta_values = np.zeros((iterations, 2))
alpha_values = np.zeros((iterations, 2))

theta_errors = np.zeros((iterations, 1))

theta_values[0] = [x[0], x[2]]
alpha_values[0] = [x[1], x[3]]

u = np.array([
  0, 0, 0, 0
])

for i in range(1, iterations):
  current_time = time[i]
  delta_time = current_time - time[i - 1]

  delta_system = (A @ x) + (B @ u)
  x += delta_system * simulation_step

  y = C @ x

  theta_values[i] = [y[0], y[2]]
  alpha_values[i] = [y[1], y[3]]

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

axs[1][1].legend(['alpha dot'])
axs[1][1].set_xlabel('time [s]')
axs[1][1].set_ylabel('angle [rad/s]')

plt.savefig('./figures/sine_signal.png', dpi=300)
plt.show()
