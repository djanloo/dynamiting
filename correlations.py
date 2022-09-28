from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rich import print

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, alpha=0.5, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Set font to serif
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.size"] = 11

# Gets the data
data = pd.read_csv("ravdess_features.csv")

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

data.emotion = pd.Categorical(data.emotion)
data.sex = pd.Categorical(data.sex)

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
# Converts to categorical
quantitative = data.copy()
quantitative.sex = (-2*quantitative.sex.cat.codes + 1) # +- 1 for simplicity
quantitative.emotion = quantitative.emotion.cat.codes

plt.figure(figsize=(8,8))
plt.imshow(np.corrcoef(quantitative.T), cmap='RdBu')
plt.xticks(range(len(quantitative.keys())), labels=quantitative.keys(), rotation=90)
plt.yticks(range(len(quantitative.keys())), labels=quantitative.keys(), rotation=0)
plt.colorbar()


# Scatter plot
plt.figure(figsize=(5,5))

for sex, marker in zip( ['M', 'F'], ["+",">"]):
    plt.scatter(data.mfcc_max[data.sex==sex], data.stft_mean[data.sex==sex], 
                                marker=marker, 
                                label=sex,
                                c=data.emotion[data.sex==sex].cat.codes)

plt.xlabel("mfcc_max")
plt.ylabel("stft_mean")

cbar = plt.colorbar()
cbar.ax.set_yticklabels(data.emotion.cat.categories)
plt.legend()


# Ellipse plot

# Normalization
for field in ["mfcc_max", "stft_mean", "stft_std"]:
    data[field] -= np.mean(data[field])
    data[field] /= np.std(data[field])


means_property1 = {"M":[], "F":[]}
means_property2 = {"M":[], "F":[]}

fig, ax = plt.subplots(1)
cmap = cm.get_cmap("Set2", 8)
colors = dict(zip(data.emotion.cat.categories, cmap(np.linspace(0,1,8))))
for emo in data.emotion.cat.categories:
    for sex in data.sex.cat.categories:
        print(f"Doing emotion {emo} for sex {sex}")
        condition = (data.sex==sex)&(data.emotion==emo)
        property1 = data.mfcc_max[condition]
        property2 = 0*data.stft_mean[condition] - data.stft_std[condition]

        means_property1[sex].append( np.mean(property1))
        means_property2[sex].append( np.mean(property2))
        confidence_ellipse(property1, property2, ax, n_std=0.3, facecolor=colors[emo], edgecolor=colors[emo])

for sex, marker in zip(["M", "F"],["+","."]):
    mappable = ax.scatter(means_property1[sex], means_property2[sex], 
                c=range(8), 
                label=sex,
                cmap=cmap,
                marker=marker)
cbar=fig.colorbar(mappable)
plt.legend()
cbar.ax.set_yticklabels(data.emotion.cat.categories)

plt.xlabel("mfcc_max [normalized]")
plt.ylabel("stft_mean - stft_std [normalized]")
plt.show()