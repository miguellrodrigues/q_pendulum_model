import numpy as np
import matplotlib.pyplot as plt


Mp = .027
lp = .153
r = .08260
g = 9.810
Jp = 1.10e-4
Jeq = 1.23e-4
Rm = 3.30
Kt = .02797
Km = .02797

Mp2 = Mp ** 2
lp2 = lp ** 2
r2 = r ** 2


def A(a):
  beta = (Mp * r2 * (np.sin(a) ** 2) - Jeq - Mp * r2) * Jp - Mp * lp2 * Jeq  # (Mp * r2 * (np.sin(a) ** 2) + Jeq) * Jp + Mp * lp2 * Jeq

  return np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (-Mp2 * g * lp2 * r * np.cos(a)) / beta, ((-Kt * Km * (-Jp + Mp * lp2)) / (beta * Rm)), 0],
    [0, -(lp * Mp * (-Jeq * g + Mp * r2 * (np.sin(a) ** 2) * g)) / beta,
     ((-lp * Mp * r * Kt * Km * np.cos(a)) / (beta * Rm)), 0]
  ])


initial_conditions = np.array([
  [.0],
  [.1],
  [.0],
  [.0]
])

x = np.copy(initial_conditions)

simulation_time = 30
simulation_step = 1e-3

iterations = int(simulation_time / simulation_step)

time = np.arange(0, simulation_time, simulation_step)

theta_values = np.zeros((iterations, 2))
alpha_values = np.zeros((iterations, 2))

theta_values[0] = np.array([initial_conditions[0], initial_conditions[2]]).T
alpha_values[0] = np.array([initial_conditions[1], initial_conditions[3]]).T

for i in range(1, iterations):
  theta = theta_values[i - 1, 0]

  delta_sys = A(theta) @ x
  # print(np.linalg.eigvals(A(theta)))
  x += delta_sys * simulation_step

  theta_values[i] = np.array([x[0], x[2]]).T
  alpha_values[i] = np.array([x[1], x[3]]).T

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
plt.show()
