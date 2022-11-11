import numpy as np  # linear algebra
from sklearn.model_selection import train_test_split
from VGG_16 import createModel4Training, createModel4Testing
from normalizeTestTRainSets import normalizeTestTRainSets
from KFold import kfold


def borrador(kfold_no, feature, disease, results_dir, kfold_dir):
    # Files to analize
    file = './RAW_DATA/ICBHI_' + feature + '_' + disease + '.npz'



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

    # Create model and test
    cnn_model_json = './MODELS/model_' + feature + str(kfold_no) + '.json'
    cnn_weights_h5 = './MODELS/modelweights_' + feature + str(kfold_no) + '.h5'
    scores = createModel4Training(training_data, training_data_y, validation_data, validation_data_y, cnn_model_json, cnn_weights_h5, kfold_no)
    test_eval, precision, recall = createModel4Testing(testing_data, testing_data_y, cnn_model_json, cnn_weights_h5, kfold_no)

    # load file and append
    if kfold_no == 1:
        RESULTS = [[], [], [], []]
    else:
        npzfile = np.load('./RESULTS/'+results_dir+'RESULTS_' + disease + '_' + feature + '.npz')
        RESULTS = np.abs(npzfile['arr_0'])
        RESULTS = RESULTS.tolist()

    RESULTS[0].append(scores[1] * 100)
    RESULTS[1].append(test_eval[1] * 100)
    RESULTS[2].append(precision * 100)
    RESULTS[3].append(recall * 100)

    np.savez('./RESULTS/'+results_dir+'RESULTS_' + disease + '_' + feature,  RESULTS)


