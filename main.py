import numpy as np
import matplotlib.pyplot as plt
import control
from system import linear_space_system, controller

plt.style.use([
  'science',
  'nature',
  'grid',
])

plt.rcParams["font.family"] = "FreeSerif, Regular"
plt.rcParams['font.size'] = 12

sine_signal = np.load('./data/sine_signal.npy')

A, B, C, D = linear_space_system()
K = controller()

initial_condition = np.array([
  [.0],
  [np.radians(0.0005)],
  [.0],
  [.0]
])

x = np.copy(initial_condition)

simulation_time = 15
simulation_step = 1e-3

iterations = int(simulation_time / simulation_step)

time = np.arange(0, simulation_time, simulation_step)

theta_values = np.zeros((iterations, 2))
alpha_values = np.zeros((iterations, 2))

theta_errors = np.zeros((iterations, 1))

thetas = np.zeros((iterations, 1))
alphas = np.zeros((iterations, 1))

alpha_dot = np.zeros((iterations, 1))

us = np.zeros((iterations, 1))

c = .01
s = control.tf([1, 0], [c, 1])

Gd = control.c2d(s, .001, method='tustin')

for i in range(1, iterations):
  u = K @ x

  delta_system = (A @ x) + (B @ u)
  x += (delta_system * simulation_step)

  thetas[i] = x[0]
  alphas[i] = x[1]
  alpha_dot[i] = Gd(x[1])
  us[i] = u


fig, axs = plt.subplots(4, 1)

axs[0].plot(time, thetas, label='theta')
axs[1].plot(time, alphas, label='alpha')
axs[2].plot(time, us, label='theta')
axs[3].plot(time, alpha_dot, label='alpha')

plt.show()

# fig1, axs = plt.subplots(2, 2, figsize=(10, 10))
#
# axs[0][0].plot(time, theta_values[:, 0], color='blue')
#
# axs[0][0].legend(['theta'])
# axs[0][0].set_xlabel('time [s]')
# axs[0][0].set_ylabel('angle [rad]')
#
# axs[0][1].plot(time, theta_values[:, 1], color='blue')
#
# axs[0][1].legend(['theta dot'])
# axs[0][1].set_xlabel('time [s]')
# axs[0][1].set_ylabel('angle [rad/s]')
#
# # # # # # # # # # # # # # # # # # # # #
#
# axs[1][0].plot(time, alpha_values[:, 0], color='red')
#
# axs[1][0].legend(['alpha'])
# axs[1][0].set_xlabel('time [s]')
# axs[1][0].set_ylabel('angle [rad]')
#
# axs[1][1].plot(time, alpha_values[:, 1], color='red')
#
# axs[1][1].legend(['alpha dot'])
# axs[1][1].set_xlabel('time [s]')
# axs[1][1].set_ylabel('angle [rad/s]')
#
# plt.savefig('./figures/sine_signal.png', dpi=300)
# plt.show()
