from system import load_matrices
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


A, B, C, D = load_matrices()

theta = sp.Symbol('theta')

z1 = sp.sin(2*theta)
z3 = sp.cos(theta)
z4 = sp.cos(theta) ** 2

t = np.arange(-np.pi, np.pi, 0.01)
x = np.arange(np.pi/4, (3*np.pi)/4, 0.01)

plt.plot(t, np.sin(2*t))
plt.plot(x, np.sin(2*x))

plt.plot(x, 2.0 - 1.27323954473516*x, '--')
plt.plot(x, 3.6 - 2.29183118052329*x, '--')


plt.axvline(x=np.pi/4, color='r')
plt.axvline(x=(3*np.pi)/4, color='r')

# plt.axvline(x=np.pi/2, color='black', linestyle='--')
# plt.axhline(y=0, color='black', linestyle='--')

plt.savefig('z2.png', dpi=300)
plt.show()
