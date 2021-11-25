# extra modules
import numpy as np
import scipy.signal
import librosa


def compute_mfcc(audio_path, trim=False):
    y, sr = librosa.load(audio_path)
    y = librosa.effects.remix(y, intervals=librosa.effects.split(y)) if trim else y
    return librosa.feature.mfcc(y=y, sr=sr).T


def compute_chroma(audio_path, trim=False):
    y, sr = librosa.load(audio_path)
    y = librosa.effects.remix(y, intervals=librosa.effects.split(y)) if trim else y
    return librosa.feature.chroma_stft(y=y, sr=sr).T


def compute_delta(mfcc_or_chroma, order):
    return librosa.feature.delta(mfcc_or_chroma, order=order)


def display_mfcc_delta_delta2(mfcc, mfcc_delta, mfcc_delta2):
    import librosa.display
    import matplotlib.pyplot as plt
    import seaborn
    seaborn.set(style='ticks')
    
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfcc)
    plt.title('MFCC')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(mfcc_delta)
    plt.title(r'MFCC-$\Delta$')
    plt.colorbar()
    
    plt.subplot(3, 1, 3)
    librosa.display.specshow(mfcc_delta2) #, x_axis='time')
    plt.title(r'MFCC-$\Delta^2$')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def get_min_mfcc(mfcc):
    return np.min(mfcc, axis=0)


def get_max_mfcc(mfcc):
    return np.max(mfcc, axis=0)


def get_mean_mfcc(mfcc):
    return np.mean(mfcc, axis=0)


def get_var_mfcc(mfcc):
    return np.var(mfcc, axis=0)


def get_histogram_mfcc(mfcc, n_bins):
    hist, bin_edges = np.histogram(mfcc, bins=n_bins)
    return hist


def get_resample_mfcc(mfcc, target_size):
    return scipy.signal.resample(mfcc, target_size)
