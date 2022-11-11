# EJECUCION DE LAS 3 CLASES: COCLEOGRAMAS / STFT / MELSPEC
print('===============================================================================================')
print('=====================================CRACKLES MAT==============================================')
print('===============================================================================================')
#import DBsplitTrainTest                         # En este paso dividimos la bse de datos entre un conjunto de entrenamiento/validacion y testeo
from DBcreateDictionary import DBcreateDictionary
from DBcomputeFeatures import featureExtracting


root_mat = './cocleogramas-OUTPUT_1C_20_2-ujaen/'
#root_mat = './RAW_DATA/'
root_wav = 'processed_audio_files/'
ICBHI_dataset_file = "ICBHI_dataset_crackles_and_others.mat"
csv_file = 'csv_data/data.csv'

#DBcreateDictionary(csv_file)
featureExtracting(ICBHI_dataset_file, root_mat, root_wav)

