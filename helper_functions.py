import SpectralSynthesis as ss
import numpy as np

def noisy_nonperiodic_xfbm(N,E,H,S, noise=0.05, seed = None, normalise = None):
    X = ss.fBm(N,E,H,S, exp=True, centred = True, periodic = False, seed=seed)
    noise_fraction = noise*np.std(X.flatten())
    noise_field = ss.fBm(N=N, E=E, H=-1, exp=False)*noise_fraction

    X = X+noise_field

    X /= np.max(X.flatten())

    return X

def noisy_xfbm(N,E,H,S, noise=0.05, seed = None, normalise=None):
    X = ss.fBm(N,E,H,S, exp=True, centred = True, periodic = True, seed=seed)
    noise_fraction = noise*np.std(X.flatten())
    noise_field = ss.fBm(N=N, E=E, H=-1, exp=False)*noise_fraction

    X = X+noise_field
    X /= np.max(X.flatten())

    return X

def root_mean_square_error(y_actual,y_predict):
    y_actual = np.array(y_actual)
    y_predict = np.array(y_predict)
    return np.sqrt(np.sum((y_actual-y_predict)**2)/len(y_actual))