from system import load_matrices
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


A, B, C, D = load_matrices()

theta = sp.Symbol('theta')

z2 = sp.sin(2*theta)  # b = [4/pi, 36/(5*pi)]
z3 = sp.cos(theta)
z4 = sp.cos(theta) ** 2

t = np.arange(-np.pi, np.pi, 0.01)
x = np.arange(-np.pi/4, np.pi/4, 0.01)

plt.plot(t, np.sin(2*t))

plt.axvline(x=-np.pi/4, color='r')
plt.axvline(x=np.pi/4, color='r')

plt.savefig('z2.png', dpi=300)
plt.show()
