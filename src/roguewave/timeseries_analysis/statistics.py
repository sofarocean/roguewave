import numpy as np
from scipy.signal import hilbert
import pandas as pd
def zero_mean(signal):
    return signal - np.nanmean(signal)
def third_order_moment(signal):
    z = hilbert(zero_mean(signal))
    return np.nanmean(z*z*np.conj(z))*3/4
def normalized_third_order_moment(signal):
    moment = third_order_moment(signal)
    norm = variance(signal)**(3/2)
    return moment / norm
def skewness(signal):
    return np.real(third_order_moment(signal))
def asymmetry(signal):
    return np.imag(third_order_moment(signal))

def normalized_skewness(signal):
    return np.real(normalized_third_order_moment(signal))

def normalized_asymmetry(signal):
    return np.imag(normalized_third_order_moment(signal))
def variance(signal):
    return np.nanmean(signal**2)

def bulk_properties(signal) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "variance",
            "skewness",
            "asymmetry",
            "normalized skewness",
            "normalized asymmetry",
        ],
        index=[0],
    )
    df['mean'] = np.nanmean(signal)
    df["variance"] = variance(signal)
    df["hs"] = 4*np.sqrt(variance(signal))
    df["skewness"] = skewness(signal)
    df["asymmetry"] = asymmetry(signal)
    df["normalized skewness"] = normalized_skewness(signal)
    df["normalized asymmetry"] = normalized_asymmetry(signal)
    return df