import math

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, MaxPooling1D, Flatten
from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from keras.optimizers import *
from sklearn.metrics import mean_squared_error


def load_close(data):
    f = open(data, 'r').readlines()[1:]
    raw_data = []
    raw_dates = []
    for line in f:
        try:
            close_price = float(line.split(',')[4])
            raw_data.append(close_price)
            raw_dates.append(line.split(',')[0])
        except:
            continue
    return raw_data, raw_dates


def load_returns(data):
    f = open(data, 'r').readlines()[1:]
    raw_data = []
    raw_dates = []
    for line in f:
        try:
            open_price = float(line.split(',')[1])
            close_price = float(line.split(',')[4])
            raw_data.append(close_price - open_price)
            raw_dates.append(line.split(',')[0])
        except:
            continue

    return raw_data[::-1], raw_dates[::-1]


def make_timeseries_regressor(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
    model = Sequential((
        Conv1D(input_shape=(window_size, nb_input_series),
                      kernel_size=filter_length, activation="relu", filters=nb_filter,
               padding='causal'),
        MaxPooling1D(),
        Conv1D(kernel_size=filter_length, activation="relu", filters=nb_filter,
               padding='causal'),
        MaxPooling1D(),
        Flatten(),
        Dense(nb_outputs, activation='linear'),
    ))
    opt = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    return model


def make_timeseries_instances(timeseries, window_size):
    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_3d(
        np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    y = timeseries[window_size:]
    q = np.atleast_3d([timeseries[-window_size:]])
    return X, y, q


def evaluate_timeseries(timeseries, window_size, epochs, batch_size):
    filter_length = 5
    nb_filter = 4
    timeseries = np.atleast_2d(timeseries)
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T

    nb_samples, nb_series = timeseries.shape
    # print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=nb_series,
                                      nb_outputs=nb_series, nb_filter=nb_filter)
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape,
                                                                                              model.output_shape,
                                                                                              nb_filter, filter_length))
    model.summary()

    X, y, q = make_timeseries_instances(timeseries, window_size)
    print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', q, sep='\n')
    test_size = int(0.1 * nb_samples)
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    pred = model.predict(X_test)
    print('\n\nactual', 'predicted', sep='\t')
    for actual, predicted in zip(y_test, pred.squeeze()):
        print(actual.squeeze(), predicted, sep='\t')
    print('next', model.predict(q).squeeze(), sep='\t')

"""Prepare input data, build model, evaluate."""
np.set_printoptions(threshold=25)
window_size = 50
epochs = 25
batch_size = 2

timeseries = load_close('stockdatas/VTI.csv')
evaluate_timeseries(timeseries[0], window_size, epochs, batch_size)
