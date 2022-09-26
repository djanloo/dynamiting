"""Generates the spectrum figure used in explanation"""
import numpy as np
import matplotlib.pyplot as plt

NOISE_AMPLITUDE = 0.1

# Generates 100ms of sound sampled at 40 kHz
t = np.linspace(0, 0.1, int(40_000 * 0.1)) # time in seconds
sound = np.zeros(len(t))

# Generates a noisy signal that has more harmonics 
freqs = [1.1, 2.2, 2.8, 3.9 , 5.1] # frequencies in kHz
amps = [1, 0.7, 0.6, 0.4, 0.2]  # amplitudes for eqch frequency

for f,a in zip(freqs, amps):

    omega = 2*np.pi*1000*f # pulsation in rad/s
    phi = np.random.uniform(0, 2*np.pi) # adds a random phase

    sound += np.sin(omega * t + phi)

sound += NOISE_AMPLITUDE*np.random.normal(0,1, size=t.shape)

plt.plot(t, sound)
plt.show()

