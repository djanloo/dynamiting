from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set font to serif
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.size"] = 11

# Gets the data
data = pd.read_csv("ravdess_features.csv")

# Converts to categorical
data.sex = (-2*pd.Categorical(data.sex).codes + 1) # +- 1 for simplicity
data.emotion = pd.Categorical(data.emotion).codes

excluded = [
    "modality",
    "vocal_channel",
    "emotional_intensity",
    "statement",
    "repetition",
    "actor",
    "channels",
    "sample_width",
    "frame_rate",
    "frame_width",
    "stft_max",
    "frame_count",
    "length_ms",
    "mean",
    "max","min",
    # "kur","skew",
]
data = data.drop(columns=excluded)

### Fills missing values
def I(std):
    # Values obtained by log-linear regression
    return 20*np.log10(std/0.993)

i = data.intensity.to_numpy()
std = data['std'].to_numpy()
# Computes only the missing values
i[np.isnan(i)] = I(std[np.isnan(i)])
# Refills the dataframe
data.intensity = i

# Correlation plot
plt.figure(figsize=(8,8))
plt.imshow(np.corrcoef(data.T), cmap='RdBu')
plt.xticks(range(len(data.keys())), labels=data.keys(), rotation=90)
plt.yticks(range(len(data.keys())), labels=data.keys(), rotation=0)
plt.colorbar()


# Scatter plot
plt.figure(figsize=(8,8))

for sex, label in zip([-1,1], ['male', 'female']):
    plt.scatter(data.mfcc_max[data.sex==sex], data.stft_mean[data.sex==sex], 
                                marker="+", 
                                label=label,
                                c=data.emotion[data.sex==sex])

plt.colorbar()
plt.legend()
plt.show()