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
data.sex = (-2*pd.Categorical( data.sex).codes + 1)
print(data.sex)
qualitative = [
    "modality",
    "vocal_channel",
    "emotion",
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
    "kur","skew",
]
labels = list(data.keys())

for p in labels:
    if p in qualitative:
        data=data.drop(columns=[p])
labels = list(data.keys())
label_to_index = dict(zip(labels, range(len(data))))
data = data.to_numpy()
# Fill I values
def I(std):
    return 20*np.log10(std/0.993)

fill_index = np.where(np.isnan(data[:,label_to_index["intensity"]]))
fill_std = data[:,label_to_index["std"]][fill_index]
data[:,label_to_index["intensity"]][fill_index] = I(fill_std)

plt.figure(figsize=(8,8))
plt.imshow(np.corrcoef(data.T),cmap='RdBu')
plt.xticks(range(len(labels)), labels=labels, rotation=90)
plt.yticks(range(len(labels)), labels=labels, rotation=0)

plt.colorbar()
plt.show()