import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import LearningRateScheduler
from tracnet_model import build_tracnet
import os
#from customized_lr_schedules import StepDecay
from math import pow, floor
from matplotlib import pyplot
import sklearn
from sklearn.model_selection import train_test_split

np.random.seed(0)
tf.compat.v1.set_random_seed(0)

# create input and target datasets
def create_datasets(size, input_data_filepath, target_date_filepath):
    dspl_array = np.empty([1472, size, size, 2], dtype='float64')
    for i, filename in enumerate(os.listdir(input_data_filepath)):
        f = os.path.join(input_data_filepath, filename)
        if os.path.isfile(f):
            file = loadmat(f)
            dspl_array[i, :, :, :] = file['dspl']

    trac_array = np.empty([1472, size, size, 2], dtype='float64')
    for i, filename in enumerate(os.listdir(target_date_filepath)):
        f = os.path.join(target_date_filepath, filename)
        if os.path.isfile(f):
            file = loadmat(f)
            trac_array[i, :, :, :] = file['trac']


    return dspl_array, trac_array


def lr_step_decay(epoch):
    initial_lr = 6e-4
    drop_rate = 0.7943
    epochs_drop = 10.0
    return initial_lr * pow(drop_rate, floor(epoch/epochs_drop))


# build and train a tracnet from scratch
def train_tracnet(X_train, y_train, input_shape=(104, 104, 2), epochs=5):
    model = build_tracnet(input_shape)
    model.compile(
        optimizer=SGD(momentum=0.9),
        loss=MeanSquaredError(),
        metrics=[RootMeanSquaredError(), 'mse', 'mae']
    )
    model.summary()
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=epochs,
        callbacks=[LearningRateScheduler(lr_step_decay, verbose=2)],
        verbose=2
    )
    pyplot.plot(history.history['root_mean_squared_error'])
#    pyplot.plot(history.history['mean_squared_error'])
#    pyplot.plot(history.history['mean_absolute_percentage_error'])
    pyplot.show()
    pyplot.savefig('fig.png')
    model.save(
        "'model.tf'",
        save_format='tf',
        include_optimizer=True
    )
    pyplot.close()



X_train, y_train = create_datasets(104, '/home/r/richard/Desktop/trainData104/dspl',
                '/home/r/richard/Desktop/trainData104/trac')
train_tracnet(X_train, y_train)