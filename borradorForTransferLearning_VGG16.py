import numpy as np  # linear algebra
from VGG_16 import createModel4Training, createModel4Testing

def borrador(kfold_no, feature, disease, results_dir, kfold_dir):
    # Files to analize
    file_train = './aria/KFOLD'+str(kfold_no)+'/Training_img/'+feature+'/ICBHI_training_' + feature + '_' + disease + '_' + str(kfold_no) + '.npz'
    file_val = './aria/KFOLD'+str(kfold_no)+'/Validation_img/'+feature+'/ICBHI_validation_' + feature + '_' + disease + '_' + str(kfold_no) + '.npz'
    file_test = './aria/KFOLD'+str(kfold_no)+'/Testing_img/'+feature+'/ICBHI_testing_' + feature + '_' + disease + '_' + str(kfold_no) + '.npz'


    # CREATE K-Fold -> comment this after 10
    #kfold(file)


    # load data for each model
    npzfile = np.load(file_train)
    training_data = np.abs(npzfile['arr_0'])
    training_data_y = npzfile['arr_1']

    # plt.imshow(training_data[0])
    # plt.show()

    npzfile = np.load(file_val)
    validation_data = np.abs(npzfile['arr_0'])
    validation_data_y = npzfile['arr_1']

    npzfile = np.load(file_test)
    testing_data = np.abs(npzfile['arr_0'])
    testing_data_y = npzfile['arr_1']


    # Create model and test
    cnn_model_json = './MODELS/model_' + feature + str(kfold_no) + '.json'
    cnn_weights_h5 = './MODELS/modelweights_' + feature + str(kfold_no) + '.h5'
    scores = createModel4Training(training_data, training_data_y, validation_data, validation_data_y, cnn_model_json, cnn_weights_h5, kfold_no)
    test_eval, precision, recall, precisiony, speficicity, sensitivity, acc, tn, fp, fn, tp = createModel4Testing(testing_data, testing_data_y, cnn_model_json, cnn_weights_h5, kfold_no)

    # load file and append
    if kfold_no == 1:
        RESULTS = [[], [], [], [], [], [], [], [], [], [], [], []]
    else:
        npzfile = np.load('./RESULTS/' + results_dir + 'RESULTS_' + disease + '_' + feature + '_VGG16.npz', allow_pickle=True)
        RESULTS = np.abs(npzfile['arr_0'])
        RESULTS = RESULTS.tolist()

    RESULTS[0].append(scores[1] * 100)
    RESULTS[1].append(test_eval[1] * 100)
    RESULTS[2].append(precision * 100)
    RESULTS[3].append(recall * 100)
    RESULTS[4].append(precisiony * 100)
    RESULTS[5].append(speficicity * 100)
    RESULTS[6].append(sensitivity * 100)
    RESULTS[7].append(acc * 100)
    RESULTS[8].append(tn * 100)
    RESULTS[9].append(fp * 100)
    RESULTS[10].append(fn * 100)
    RESULTS[11].append(tp * 100)

    np.savez('./RESULTS/' + results_dir + 'RESULTS_' + disease + '_' + feature + '_VGG16')


