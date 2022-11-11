import numpy as np  # linear algebra
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import model_from_json
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

    # CNN MODEL
    #input = tf.keras.models.Sequential()
    input = keras.layers.Input(shape=(first_value, second_value, 1), name="input")
    x = keras.layers.Conv2D(32, 3, strides=(1, 3), padding='same')(input)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(64, 3, strides=(1, 3), padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)
    x = keras.layers.Dropout(0.5)(x)
    hidden1 = keras.layers.Flatten()(x)

    hidden1 = keras.layers.Dense(1000)(hidden1)
    hidden1 = keras.layers.LeakyReLU(alpha=0.1)(hidden1)
    hidden1 = keras.layers.Dropout(0.5)(hidden1)

    hidden2 = keras.layers.Dense(500)(hidden1)
    hidden2 = keras.layers.LeakyReLU(alpha=0.1)(hidden2)
    hidden2 = keras.layers.Dropout(0.5)(hidden2)

    hidden3 = keras.layers.Dense(100)(hidden2)
    hidden3 = keras.layers.LeakyReLU(alpha=0.1)(hidden3)
    hidden3 = keras.layers.Dropout(0.5)(hidden3)

    output = keras.layers.Dense(4, activation='softmax')(hidden3)
    net = keras.Model(input, output, name="Net")
    net.summary()

    # COMPILE
    net.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    K.set_value(net.optimizer.learning_rate, 0.001)

    # CREATE AND SAVE MODEL
    history = net.fit(
        {"input": training_data},
        training_data_y,
        batch_size=16,
        validation_data=({"input": validation_data}, validation_data_y),
        epochs=500, verbose=1,
        callbacks=my_callbacks
    )

    # Generate generalization metrics
    model_json = net.to_json()
    with open(cnn_model_json, "w") as json_file:
        json_file.write(model_json)
    net.save_weights(cnn_weights_h5)

    scores = net.evaluate(validation_data, validation_data_y, verbose=2)
    # summarize history for accuracy
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./Graphs/BASELINE_CNN/ACC/Acc_KFold' + str(kfold_no) + '.png', dpi=300)
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./Graphs/BASELINE_CNN/LOSS/Loss_KFold' + str(kfold_no) + '.png', dpi=300)
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
    plt.savefig('./Graphs/BASELINE_CNN/RoC/Roc_KFold' + str(kfold_no) + '.png', dpi=300)
    plt.close()

    out = np.argmax(out, axis=1)
    # conf_matrix = metrics.confusion_matrix(y_true=testing_data_y, y_pred=out)
    precision = metrics.precision_score(testing_data_y, out, average='micro')
    recall = metrics.recall_score(testing_data_y, out, average='micro')
    print('Precision: %.3f' % metrics.precision_score(testing_data_y, out, average='micro'))
    print('Recall: %.3f' % metrics.recall_score(testing_data_y, out, average='micro'))

    tn, fp, fn, tp = metrics.confusion_matrix(y_true=testing_data_y, y_pred=out)
    precisiony = tp / (tp + fp)
    speficicity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    acc = (tp + tn) / (tp + fp + fn + tn)

    return test_eval, precision, recall, precisiony, speficicity, sensitivity, acc, tn, fp, fn, tp


























