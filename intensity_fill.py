import pandas as pd
import numpy as np

def I_from_std(std):
    return 20*np.log10(std/0.993)

def fill_intensity(dataframe):
    df = dataframe.copy()
    df.intensity[df.intensity.isna()] = I_from_std(df["std"][df.intensity.isna()])
    return df