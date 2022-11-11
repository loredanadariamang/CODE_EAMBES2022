import numpy as np  # linear algebra
from sklearn.model_selection import KFold



def kfold (file1):
    val_1 = 'arr_0'
    val_2 = 'arr_1'

    # load data for each model
    file = file1
    npzfile = np.load(file)
    inputs = np.abs(npzfile[val_1])
    targets  = npzfile[val_2]

    # Número de folds para el K-FOLD
    num_folds = 10

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1

    for train, test in kfold.split(inputs, targets):
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Creating fold {fold_no} ...')

        np.savez('ICBHI_Train_kFold_' + str(fold_no), train)
        np.savez('ICBHI_Test_kFold_' + str(fold_no), test)
        # Increase fold number
        fold_no = fold_no + 1


