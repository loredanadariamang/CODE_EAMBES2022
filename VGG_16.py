import numpy as np  # linear algebra
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD
from keras import backend as K
from keras.models import model_from_json
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from sklearn import metrics

def createModel4Training(training_data, training_data_y, validation_data, validation_data_y, cnn_model_json, cnn_weights_h5, kfold_no):

    # We need teh CNN model for the following step
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    ]
    # SHAPE needed for the model
    first_value = np.size(training_data, 1)
    second_value = np.size(training_data, 2)
    l2_reg = 0.

    model = Sequential()
    model.add(VGG16(include_top=False, pooling='avg', input_shape=(first_value, second_value, 3), weights='imagenet'))
    model.add(Dense(4, activation='softmax'))
    model.add(Flatten())
    opt = SGD(0.002)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    K.set_value(model.optimizer.learning_rate, 0.001)

    # CREATE AND SAVE MODEL
    history = model.fit(
        {"vgg16_input": training_data},
        training_data_y,
        batch_size=16,
        validation_data=({"vgg16_input": validation_data}, validation_data_y),
        epochs=3, verbose=1,
        callbacks=my_callbacks
    )

    # Generate generalization metrics
    model_json = model.to_json()
    with open(cnn_model_json, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(cnn_weights_h5)

    scores = model.evaluate(validation_data, validation_data_y, verbose=2)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./Graphs/VGG16/ACC/Acc_KFold' + str(kfold_no) + '.png', dpi=300)
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./Graphs/VGG16/LOSS/Loss_KFold' + str(kfold_no) + '.png', dpi=300)
    plt.close()

    return scores

# -------------

def createModel4Testing(testing_data, testing_data_y, cnn_model_json, cnn_weights_h5, kfold_no):

    # load json and create model
    json_file = open(cnn_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(cnn_weights_h5)
    print("Loaded model from disk")
    #first_value = np.size(testing_data, 1)
    #second_value = np.size(testing_data, 2)
    #data_test = testing_data.reshape(-1, first_value, second_value, 3)
    #data_test_y = testing_data_y.reshape(data_test.shape[0], 3)
    #loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Testeo
    out = loaded_model.predict(testing_data)
    test_eval = loaded_model.evaluate(testing_data, testing_data_y, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    lr_auc = metrics.roc_auc_score(testing_data_y, out, multi_class='ovr')
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}
    n_class = 4

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = metrics.roc_curve(testing_data_y, out[:, i], pos_label=i)

    # plotting
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--', color='yellow', label='Class 3 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('./Graphs/VGG16/RoC/Roc_KFold'+str(kfold_no)+'.png', dpi=300)
    plt.close()


    out = np.argmax(out, axis=1)
    #conf_matrix = metrics.confusion_matrix(y_true=testing_data_y, y_pred=out)
    precision = metrics.precision_score(testing_data_y, out, average='micro')
    recall = metrics.recall_score(testing_data_y, out, average='micro')
    print('Precision: %.3f' % metrics.precision_score(testing_data_y, out, average='micro'))
    print('Recall: %.3f' % metrics.recall_score(testing_data_y, out, average='micro'))

    tn, fp, fn, tp = metrics.confusion_matrix(y_true=testing_data_y, y_pred=out)
    precisiony = tp/(tp+fp)
    speficicity = tn/(tn+fp)
    sensitivity = tp/(tp+fn)
    acc = (tp+tn)/(tp+fp+fn+tn)

    return test_eval, precision, recall, precisiony, speficicity, sensitivity, acc, tn, fp, fn, tp