import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set font to serif
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.size"] = 11

data = pd.read_csv("ravdess_features.csv")
intensity = data['intensity'].to_numpy()
std = data['std'].to_numpy()

# Excludes where the intensity is not given
i_nan_filter = np.where(np.logical_not(np.isnan(intensity)))

intensity = intensity[i_nan_filter]
std = std[i_nan_filter]

plt.figure(figsize=(4,4))
plt.plot(intensity[:500], std[:500], ls="", marker=".", color='k')
plt.xlabel("Intensity [dB]")
plt.ylabel("Signal std [a.u.]")
plt.yscale("log")


from scipy.optimize import curve_fit

def line(x, m, q):
    return m*x+q

pars, popt = curve_fit(line, intensity, np.log(std))

x_ = np.linspace(min(intensity), max(intensity))
plt.plot(x_ , np.exp(line(x_, *pars)), color='r')
print(pars)
corr = np.corrcoef(np.log(std), intensity)
print(f"Correlation is {corr[0,1]}")
plt.show()
