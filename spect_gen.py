"""Generates the spectrum figure used in explanation"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from rich import print

NOISE_AMPLITUDE = 0.0

def pitch_to_freq(p):
    return 440*2**((p-69)/12)

def freq_to_pitch(freq):
    return 69 + np.log2(freq/440)

# Generates 100ms of sound sampled at 40 kHz
t = np.linspace(0, 0.1, int(40_000 * 0.1)) # time in seconds
sound = np.zeros(len(t))

print(f"Using {len(t)} timesteps of {np.diff(t)[0]:.2} secs")

# Generates a noisy signal that has more harmonics 
freqs = [0.44, 1.1, 2.2, 2.8, 3.9 , 5.1] # frequencies in kHz
amps = [1, 0.5, 0.2, 0.1, 0.05]  # amplitudes for eqch frequency

PHASE_KICKS = 1
for f,a in zip(freqs, amps):
    omega = 2*np.pi*1000*f # pulsation in rad/s

    kick_index = np.random.randint(len(t), size=PHASE_KICKS)
    phi = np.zeros(len(t))
    phi[kick_index] = np.random.uniform(0, 0.1, size=PHASE_KICKS)
    phi = np.cumsum(phi)

    sound += a*np.sin(omega * t + phi)

sound += NOISE_AMPLITUDE*np.random.normal(0,1, size=t.shape)

f_ = np.fft.rfftfreq(len(t), d=0.1/len(t))
f_ = f_[0:len(f_)//2]
power_spectrum = np.abs(np.fft.rfft(sound))[0:len(f_)]**2
plt.plot(f_, power_spectrum, color='k', alpha=0.7)
plt.yscale("log")

cmap = matplotlib.cm.get_cmap('plasma')
colors = cmap(np.linspace(0,1,12))
pitches = range(69-12,120)
for i,p in enumerate(pitches):
    f1, f2 = pitch_to_freq(p - 0.5), pitch_to_freq(p + 0.5)
    masked_spectrum = power_spectrum[(f_>f1)&(f_<f2)]
    dummy_f = f_[(f_>f1)&(f_<f2)]
    plt.fill_between(dummy_f, 1e-3*np.ones(len(dummy_f)), masked_spectrum, alpha=0.3, color=colors[i%12])


plt.ylim((1e-3, 1e7))
plt.xlim((100, 5000))
plt.xlabel(r"$f$ [Hz]")
plt.ylabel(r"$S(f)$")
plt.show()

