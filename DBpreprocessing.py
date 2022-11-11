#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
import librosa as lb
import soundfile as sf


# FUNCTIONS HERE!
def getFilenameInfo(file):
    return file.split('_')


def getPureSample(raw_data, start, end, sr=22050):
    '''
    Takes a numpy array and splits its using start and end args
    raw_data=numpy array of audio sample
    start=time
    end=time
    sr=sampling_rate
    mode=mono/stereo
    '''

    max_ind = len(raw_data)
    start_ind = min(int(start * sr), max_ind)
    end_ind = min(int(end * sr), max_ind)
    return raw_data[start_ind: end_ind]


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
patient_data = pd.read_csv('kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv',names=['pid', 'disease'])
patient_data.head()

# * Here we have Patient Ids and Disease info
# > Lets check out what is in annoted '.txt' files of audio files.
# Ejemplo paciente 106
# df = pd.read_csv('kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/160_1b3_Al_mc_AKGC417L.txt', sep='\t')
# df.head()

# * These are very usefull information namely (Start , End ( time of respiratory cycles) ,crackles,weezels)
# > So lets get them into a dataset
# > Note:- i use sep **' \t '** cause we are reading data from text file which is sperated by tabs here
path = 'kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
files = [s.split('.')[0] for s in os.listdir(path) if '.txt' in s]
# files[:5]

# > As mentioned in **filename_format.txt** , '.txt' files of annotated audio files have various information. So we will try to extract that too.
# getFilenameInfo('160_1b3_Al_mc_AKGC417L')
files_data = []
for file in files:
    # We take the decision of asigninng a value to each sound class: crackles = 0 / wheezes = 1 / both = 2 / none = 3
    data = pd.read_csv(path + file + '.txt', sep='\t', names=['start', 'end', 'crackles', 'weezels', 'filename_mat', 'filename_wav', 'sound_class'])
    cont = 0
    name_data = getFilenameInfo(file)
    data['pid'] = name_data[0]
    data['mode'] = name_data[-2]
    data['filename'] = file
    for cont in np.arange(data.shape[0]):
        data['filename_mat'][cont] = str(data['filename'][cont]) + '_' + str(cont) + '.mat'
        data['filename_wav'][cont] = str(data['filename'][cont]) + '_' + str(cont) + '.wav'
        if (data['crackles'][cont] == 1) & (data['weezels'][cont] == 0):
            data['sound_class'][cont] = 0
        elif (data['crackles'][cont] == 0) & (data['weezels'][cont] == 1):
            data['sound_class'][cont] = 1
        elif (data['crackles'][cont] == 1) & (data['weezels'][cont] == 1):
            data['sound_class'][cont] = 2
        else:
            data['sound_class'][cont] = 3

    files_data.append(data)
files_df = pd.concat(files_data)
files_df.reset_index()
files_df.head()

# > Now we can join both **patient_data** and **files_df**
patient_data.info()
files_df.info()

# > Lets get **pid** and **101** to same type and merge both dataframes on pid
patient_data.pid = patient_data.pid.astype('int32')
files_df.pid = files_df.pid.astype('int32')
files_df.sound_class = files_df.sound_class.astype('int32')
data = pd.merge(files_df, patient_data, on='pid')
data.head()

# Crea fichero data
os.makedirs('csv_data')
data.to_csv('csv_data/data.csv', index=False)


# # Step 2
# # Processing Audio files
# > Now we only want that parts from whole audio file which contains **Respiratory Cycles**
# > We can do this by utilizing the start and end time specifiles for these cycles in our **data dataframe**

# * we multiplied start with sampling rate cause start is time and raw_data is array sampled acc. to sampling rate
# > Now we also want our input images to our cnn to be of same size for that audio files must be of **same length** i.e **(start - end)** must be same
# > Lets find the best length we can have
sns.scatterplot(x=(data.end - data.start), y=data.pid)
sns.boxplot(y=(data.end - data.start))

# > From these plots we can conclude that best length is **~6**
# > Also if difference is <6 we must **Zero Pad** it to get it to required length
# > Zero Padding means **silent**

# > Lets create a directory for storing our files
os.makedirs('processed_audio_files')

# * We can iterate over dataset using iterrows, its output is as shown
for index, row in data.iterrows():
    print("Index ->", index)
    print("Data->\n", row)
    break

# > Also a single sample of audio file can have **Many Respiratory Cycles** so we might have to same multiple files for a simple audio file
# > I will be using **Librosa** module for loading audio files and **Soundfile** module for writing to output path
# > **Study The following Function Carefully**

i, c = 0, 0
for index, row in data.iterrows():
    maxLen = 6
    start = row['start']
    end = row['end']
    filename = row['filename']

    # If len > maxLen , change it to maxLen
    if end - start > maxLen:
        end = start + maxLen

    audio_file_loc = path + filename + '.wav'

    if index > 0:
        # check if more cycles exits for same patient if so then add i to change filename
        if data.iloc[index - 1]['filename'] == filename:
            i += 1
        else:
            i = 0
    filename = filename + '_' + str(i) + '.wav'

    save_path = 'processed_audio_files/' + filename
    c += 1

    audioArr, sampleRate = lb.load(audio_file_loc)
    pureSample = getPureSample(audioArr, start, end, sampleRate)

    pureSample = lb.resample(pureSample,sampleRate,4000)
    sampleRate = 4000
    # pad audio if pureSample len < max_len
    #reqLen = 6 * sampleRate
    #padded_data = lb.util.pad_center(pureSample, reqLen)

    sf.write(file=save_path, data=pureSample, samplerate=sampleRate)

print('Total Files Processed: ', c)

