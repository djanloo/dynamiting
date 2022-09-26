"""Generates the spectrum figure used in explanation"""
import numpy as np
import matplotlib.pyplot as plt

NOISE_AMPLITUDE =0.0

# Generates 100ms of sound sampled at 40 kHz
t = np.linspace(0, 0.1, int(40_000 * 0.1)) # time in seconds
sound = np.zeros(len(t))

# Generates a noisy signal that has more harmonics 
freqs = [1.1, 2.2, 2.8, 3.9 , 5.1] # frequencies in kHz
amps = [1, 0.5, 0.2, 0.1, 0.05]  # amplitudes for eqch frequency


PHASE_KICKS = 2
for f,a in zip(freqs, amps):
    omega = 2*np.pi*1000*f # pulsation in rad/s

    kick_index = np.random.randint(len(t), size=PHASE_KICKS)
    phi = np.zeros(len(t))
    phi[kick_index] = np.random.uniform(0, 6.28, size=PHASE_KICKS)
    phi = np.cumsum(phi)

    sound += a*np.sin(omega * t + phi)

sound += NOISE_AMPLITUDE*np.random.normal(0,1, size=t.shape)

f_ = np.fft.rfftfreq(t.shape[-1])

plt.plot(1000*f_[0:len(f_)//2], np.abs(np.fft.rfft(sound))[0:len(f_)//2])
plt.yscale("log")
plt.show()

