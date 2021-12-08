import numpy as np

def add_noise(dspl, N):
    dsplN = np.zeros(shape=dspl.shape)
    S = dspl.shape[0]
    stdev = N / np.sqrt(2)
    noise = np.random.normal(loc=0, scale=stdev, size=(S, S))
    dsplN[:, :, 1] = dspl[:, :, 1] + noise
    noise = np.random.normal(loc=0, scale=stdev, size=(S, S))
    dsplN[:, :, 2] = dspl[:, :, 2] + noise
    return dsplN

