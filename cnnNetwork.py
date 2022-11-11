import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import librosa as lb
import librosa.display as lbd
import os
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from scipy.io import loadmat, savemat
from scipy import signal
import statistics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import json



def createModel4Training(training_data, training_data_y, validation_data, validation_data_y, cnn_model_json, cnn_weights_h5):

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
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(64, 3, strides=(1, 3), padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)
    x = keras.layers.Dropout(0.25)(x)
    hidden = keras.layers.Flatten()(x)
    hidden = keras.layers.Dense(200)(hidden)
    hidden = keras.layers.LeakyReLU(alpha=0.1)(hidden)
    hidden = keras.layers.Dropout(0.5)(hidden)
    output = keras.layers.Dense(4, activation='softmax')(hidden)
    net = keras.Model(input, output, name="Net")
    net.summary()

    # PLOT
    keras.utils.plot_model(net, "net.png", show_shapes=True)
    # COMPILE
    net.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    K.set_value(net.optimizer.learning_rate, 0.001)

    # CREATE AND SAVE MODEL
    history = net.fit(
        {"input": training_data},
        training_data_y,
        batch_size=16,
        validation_data=({"input": validation_data}, validation_data_y),
        epochs=30, verbose=1,
        callbacks=my_callbacks
    )

    # Generate generalization metrics
    model_json = net.to_json()
    with open(cnn_model_json, "w") as json_file:
        json_file.write(model_json)
    net.save_weights(cnn_weights_h5)

    # Testeo
    scores = net.evaluate(validation_data, validation_data_y, verbose=0)
    # test_eval = net.evaluate(testing_data, testing_data_y, verbose=0)

    return scores

# -------------

def createModel4Testing(testing_data, testing_data_y, cnn_model_json, cnn_weights_h5):

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

    return test_eval