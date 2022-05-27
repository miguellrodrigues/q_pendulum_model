import numpy as np


# generate a step signal
def step_signal(t, t_step, t_step_duration):
    return np.array([
        np.heaviside(t - t_step, t_step_duration),
        np.heaviside(t - t_step, t_step_duration)
    ]).T


# generate a sinewave signal
def sine_wave_signal(t, t_step):
    time = np.arange(0, t, t_step)
    return np.sin(time)


# generate a sinewave signal with 60 Hz and amplitude of 1
signal = sine_wave_signal(t=10, t_step=1e-3)
np.save('../data/sine_signal.npy', signal)
