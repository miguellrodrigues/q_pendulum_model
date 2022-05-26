import numpy as np
import matplotlib.pyplot as plt
import control as ct
import scipy.signal as signal

c = .01
G = ct.tf([1, 0], [c, 1])
Gd = ct.c2d(G, .001, 'tustin')

t = np.arange(0, 10, .001)
y = np.sin(t)
dy = np.cos(t)

# sys = signal.dlti([95.24, -95.24], [1, -.9048], dt=.001)
sys = signal.dlti(Gd.num[0][0], Gd.den[0][0], dt=.001)

[T, Y] = signal.dlsim(sys, [1])

plt.plot(t, y)
plt.plot(t, dy)
plt.plot(T, Y, 'b')
plt.show()



