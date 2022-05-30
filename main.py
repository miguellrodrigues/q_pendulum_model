import numpy as np
import matplotlib.pyplot as plt
import control as cnt
from system import load_matrices, var_dot
from controller import find_controller

plt.style.use([
  'science',
  'nature',
  'grid',
])

plt.rcParams["font.family"] = "FreeSerif, Regular"
plt.rcParams['font.size'] = 12

A, B, C, D = load_matrices()
K = find_controller()

print("A eigenvalues:", np.linalg.eigvals(A))
print('A + BK eigenvalues:', np.linalg.eigvals(A + B @ K))

initial_condition = np.array([
  [.0],
  [np.radians(10)],
  [.0],
  [.0]
])

xk = np.copy(initial_condition)
x = np.zeros((4, 1))

samples = 10000
simulation_step = 1e-3

time = np.arange(0, samples*simulation_step, simulation_step)

theta_values = np.zeros((samples, 2))
alpha_values = np.zeros((samples, 2))

theta_values[0] = np.array([xk[0], xk[2]]).T
alpha_values[0] = np.array([xk[1], xk[3]]).T

control_signal = np.zeros((samples, 1))


for i in range(1, samples):
  u = K @ xk

  x = A @ xk + B @ u
  xk = x

  theta_dot = var_dot(theta_values[i - 1, 1], theta_values[i - 1, 0], x[0])
  alpha_dot = var_dot(alpha_values[i - 1, 1], alpha_values[i - 1, 0], x[1])

  theta_values[i] = np.array([x[0], theta_dot]).T
  alpha_values[i] = np.array([x[1], alpha_dot]).T

  control_signal[i] = u


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

axs[1][1].legend(['alpha dot', 'alpha dot approximation'])
axs[1][1].set_xlabel('time [s]')
axs[1][1].set_ylabel('angle [rad/s]')

plt.savefig('./figures/sim_states.png', dpi=300)

fig2, axs = plt.subplots(1, 1, figsize=(10, 10))

axs.plot(time, control_signal, color='green')
axs.set_xlabel('time [s]')
axs.set_ylabel('control input [V]')

plt.savefig('./figures/sim_u.png', dpi=300)
plt.show()
