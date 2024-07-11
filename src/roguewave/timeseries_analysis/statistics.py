import numpy as np
from scipy.signal import hilbert
import pandas as pd

def zero_mean(signal):
    return signal - np.nanmean(signal)

def third_order_moment(signal):
    z = hilbert(zero_mean(signal))
    return np.nanmean(z*z*np.conj(z))*3/4

def fourth_order_moment(signal):
    z = hilbert(zero_mean(signal))
    return 3*np.nanmean(z*z*np.conj(z)*np.conj(z))/8 + 4*np.nanmean(z*z*z*np.conj(z))/8

def normalized_third_order_moment(signal):
    moment = third_order_moment(signal)
    norm = variance(signal)**(3/2)
    return moment / norm

def skewness(signal):
    return np.real(third_order_moment(signal))

def asymmetry(signal):
    return np.imag(third_order_moment(signal))

def kurtosis(signal):
    return np.real(fourth_order_moment(signal))

def normalized_kurtosis(signal):
    return kurtosis(signal) / variance(signal)**2

def excess_kurtosis(signal):
    return normalized_kurtosis(signal) - 3

def im_kurtosis(signal):
    return np.imag(fourth_order_moment(signal))

def normalized_im_kurtosis(signal):
    return im_kurtosis(signal) / variance(signal)**2

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
    df["kurtosis"] = kurtosis(signal)
    df["excess kurtosis"] = excess_kurtosis(signal)
    df["normalized kurtosis"] = normalized_kurtosis(signal)
    df["im kurtosis"] = im_kurtosis(signal)
    df["normalized im kurtosis"] = normalized_im_kurtosis(signal)
    return df