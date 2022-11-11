import numpy as np  # linear algebra
from sklearn.model_selection import train_test_split
from normalizeTestTRainSets import normalizeTestTRainSets
from saveJPG import createImg
import os.path


def borrador(kfold_no, feature, disease, kfold_dir):
    # Files to analize
    file = './RAW_DATA/OLD/ICBHI_' + feature + '_' + disease + '.npz'


    # CREATE K-Fold -> comment this after 10
    #kfold(file)


    # load data for each model
    npzfile = np.load(file)
    inputs = np.abs(npzfile['arr_0'])
    targets = npzfile['arr_1']

    # Start with each fold per data type
    # GETTING TRAINING IDX
    file_kFold = './KFOLD_IDX/'+kfold_dir+'ICBHI_Train_kFold_' + str(kfold_no) + '.npz'
    npzfile = np.load(file_kFold)
    train_idx = np.abs(npzfile['arr_0'])
    # GETTING TESTING IDX
    file_kFold = './KFOLD_IDX/'+kfold_dir+'ICBHI_Test_kFold_' + str(kfold_no) + '.npz'
    npzfile = np.load(file_kFold)
    test_idx = np.abs(npzfile['arr_0'])

    # --------- DEPENDE DE KFOLD -----
    # SHAPE needed
    training_data = np.abs(inputs[train_idx])
    training_data[~np.isfinite(training_data)]=0
    training_data_y = targets[train_idx]
    training_data, validation_data = train_test_split(training_data, random_state=42, test_size=0.25)
    training_data_y, validation_data_y = train_test_split(training_data_y, random_state=42, test_size=0.25)

    testing_data = np.abs(inputs[test_idx])
    testing_data[~np.isfinite(testing_data)]=0
    testing_data_y = targets[test_idx]

    # Normalization of the data
    training_data, validation_data, testing_data = normalizeTestTRainSets(training_data, validation_data, testing_data)

    path = './aria/KFOLD'+str(1) + '/'

    filename = path + 'Training_img/'+feature+'/ICBHI_training_' + feature + '_' + disease + '_' + str(kfold_no)
    createImg(filename, training_data, training_data_y)

    filename = path + 'Validation_img/'+feature+'/ICBHI_validation_' + feature + '_' + disease + '_' + str(kfold_no)
    createImg(filename, validation_data, validation_data_y)

    filename = path + 'Testing_img/'+feature+'/ICBHI_testing_' + feature + '_' + disease + '_' + str(kfold_no)
    createImg(filename, testing_data, testing_data_y)

    print('Saved ' + filename)


