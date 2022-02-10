import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy, RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import LearningRateScheduler
from train.tracnet_model import build_tracnet
import os
from matplotlib import pyplot as plt


# set seed for reproducability
np.random.seed(1)
tf.compat.v1.set_random_seed(1)


# create input and target datasets
def create_datasets(size, input_data_filepath, target_data_filepath):
    # number_samples = os.path.getsize
    dspl_array = np.empty([1472, size, size, 2], dtype='float64')
    for i, filename in enumerate(os.listdir(input_data_filepath)):
        f = os.path.join(input_data_filepath, filename)
        if os.path.isfile(f):
            file = loadmat(f)
            dspl_array[i, :, :, :] = file['dspl']

    trac_array = np.empty([1472, size, size, 2], dtype='float64')
    for i, filename in enumerate(os.listdir(target_data_filepath)):
        f = os.path.join(target_data_filepath, filename)
        if os.path.isfile(f):
            file = loadmat(f)
            trac_array[i, :, :, :] = file['trac']

    return dspl_array, trac_array


# step-wise decay
def lr_step_decay(epoch):
    initial_lr = 6e-4
    drop_rate = 0.7943
    epochs_drop = 10.0
    return initial_lr * np.power(drop_rate, np.floor(epoch/epochs_drop))


# build and train a tracnet from scratch
def train_tracnet(dspl_train, trc_train, dspl_test, trc_test, input_shape=(104, 104, 2), epochs=23):
    batch_size = 32
    model = build_tracnet(input_shape, batch_size)
    model.compile(
        optimizer=SGD(momentum=0.9),
        loss=MeanSquaredError(),
        metrics=[Accuracy(), RootMeanSquaredError(), MeanAbsoluteError()]
    )
    model.summary()
    history = model.fit(
        dspl_train,
        trc_train,
        validation_data=(dspl_test, trc_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[LearningRateScheduler(lr_step_decay, verbose=2)],
        verbose=2
    )
    # plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('accuracy_1.png')
    plt.close()

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss_1.png')
    plt.close()

    # plot rmse
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('rmse_1.png')
    plt.close()

    # save model
    model.save(
        "'model_1.tf'",
        save_format='tf',
        include_optimizer=True
    )


dspl_train, trc_train = create_datasets(104, '/home/r/richard/Desktop/trainData104/dspl',
                                        '/home/r/richard/Desktop/trainData104/trac')
dspl_test, trc_test = create_datasets(104, '/home/r/richard/Desktop/test/generic/testData104/dspl',
                                      '/home/r/richard/Desktop/test/generic/testData104/trac')
train_tracnet(dspl_train, trc_train, dspl_test, trc_test)