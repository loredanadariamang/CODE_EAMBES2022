import numpy as np  # linear algebra
import librosa as lb
from scipy.io import loadmat, savemat
from scipy import signal


def DBFeatures(filename, variable_x, variable_y, root, reqLen, feature):
    if feature == 'cocleogram':
        cocleogramSpec = []
        for idx in np.arange(np.shape(variable_x)[0]):
            path_mat = root + variable_x[idx][0]
            data_mat = loadmat(path_mat)
            data_2 = data_mat['ouotput']
            #data_2 = data_2[0:128,:] # esta linea es para caso de 4C_... para coger solamente 2 cocleogramas anidados
            padded_data = lb.util.pad_center(data_2, reqLen)
            cocleogramSpec.append(padded_data)
        cocleogramSpec_variable = np.array(cocleogramSpec)
        np.savez(filename, cocleogramSpec_variable, variable_y)
    elif feature == 'stft':
        stftSpec = []
        for idx in np.arange(np.shape(variable_x)[0]):
            path_wav = root + variable_x[idx][0]
            soundArr, sample_rate = lb.load(path_wav, sr=4000)
            # if (np.size(soundArr) < 1024):
            #     soundArr = lb.util.pad_center(soundArr, 1026)
            feature = signal.stft(x=soundArr, fs=sample_rate, nperseg=128, nfft=256, noverlap=96, window='blackmanharris', boundary='zeros')[2]
            padded_data_a = lb.util.pad_center(feature, reqLen)
            stftSpec.append(padded_data_a)
        stftSpec_variable = np.array(stftSpec)
        np.savez(filename, stftSpec_variable, variable_y)
    elif feature == 'mel':
        melSpec = []
        for idx in np.arange(np.shape(variable_x)[0]):
            path_wav = root + variable_x[idx][0]
            soundArr, sample_rate = lb.load(path_wav, sr=4000)
            feature = lb.feature.melspectrogram(y=soundArr, sr=sample_rate, n_mels=64, n_fft=256, hop_length=32, win_length=128, window='blackmanharris')
            padded_data_b = lb.util.pad_center(feature, reqLen)
            melSpec.append(padded_data_b)
        melSpec_variable = np.array(melSpec)
        np.savez(filename, melSpec_variable, variable_y)
    else:
        errno('NO EXISTE FEATURE: PIMIENTO')


# Main Function in this file
def featureExtracting(ICBHI_dataset_file, root_mat, root_wav):
    # Variables para COCLEOGRAMAS -> necesitan la lista de los .mat
    dataset = loadmat(ICBHI_dataset_file)

    # # CRACKLES COCLEOGRAM
    data = dataset['Xdata_mat'][0]
    data_y = dataset['ydata'][0]
    file = './RAW_DATA/ICBHI_dataset_cocleogram_crackles_and_others'
    DBFeatures(file, data, data_y, root_mat, 3000, 'cocleogram')

    # # # CRACKLES STFT
    data = dataset['Xdata_wav'][0]
    data_y = dataset['ydata'][0]
    file = './RAW_DATA/ICBHI_dataset_stft_crackles_and_others'
    DBFeatures(file, data, data_y, root_wav, 4000, 'stft')

    # # # CRACKLES MEL
    data = dataset['Xdata_wav'][0]
    data_y = dataset['ydata'][0]
    file = './RAW_DATA/ICBHI_dataset_mel_crackles_and_others'
    DBFeatures(file, data, data_y, root_wav, 4000, 'mel')


