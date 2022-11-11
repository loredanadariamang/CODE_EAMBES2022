import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from scipy.io import savemat


def DBcreateDictionary(csvfile):
    # Handeling Class Imbalance
    diagnosis=pd.read_csv(csvfile, names=['start', 'end', 'crackles', 'weezels', 'filename_mat', 'filename_wav', 'sound_class', 'pid','mode','filename','disease'])
    # input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv',
    diagnosis.head()

    patients_split = random.sample(range(101,226+1), 126)
    patients = list(map(str, patients_split[:]))
    Xdata = diagnosis.loc[diagnosis['pid'].isin(patients)]

    Xdata_mat = np.array(Xdata['filename_mat'])
    Xdata_wav = np.array(Xdata['filename_wav'])

    ydata_crackles = np.array(Xdata['weezels']).astype(int)

    mdic = {"Xdata_mat": Xdata_mat, "Xdata_wav": Xdata_wav, "ydata": ydata_crackles}
    savemat("ICBHI_dataset_wheezes_and_others.mat", mdic)
