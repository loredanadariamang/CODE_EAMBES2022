import numpy as np


def normalizeTestTRainSets(train, val, test):
    train_data_mean, train_data_std = [], []
    tam = np.shape(train)
    for idx in range(tam[1]):
        train_data_mean.append(np.mean(train[:, idx]))
        train_data_std.append(np.std(train[:, idx]))

    for idx in range(tam[1]):
        train[:, idx] = (train[:, idx] - train_data_mean[idx]) / train_data_std[idx]
        val[:, idx] = (val[:, idx] - train_data_mean[idx]) / train_data_std[idx]
        test[:, idx] = (test[:, idx] - train_data_mean[idx]) / train_data_std[idx]

    return train, val, test

