import os

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, ConvLSTM2D, Flatten, RepeatVector, Reshape, TimeDistributed, Lambda, Dense, Dropout
from tensorflow.keras.optimizers import Adagrad

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

from load import load_links
from DQR.models import tilted_loss, convLSTM

# Search space:
num_filters = Integer(low=10, high=128, name='num_filters')
kernel_length = Integer(low=1, high=20, name='kernel_length')
prob = Real(low=0, high=0.6, prior='uniform', name='prob')
dimensions = [num_filters, kernel_length, prob]
default_parameters = [64, 10, 0.10]

def roll(ix, ts, lags, preds):
    X = np.stack([np.roll(ts, i, axis=0) for i in range(lags, 0, -1)], axis=1)[lags:-preds,]
    Y = np.stack([np.roll(ts, -i, axis=0) for i in range(0, preds, 1)], axis=1)[lags:-preds,]
    Y_ix = ix[lags:-preds]
    return X, Y, Y_ix

def load_data(line, freq, lags, preds):
    train = pd.read_csv(f'Data/{line}_{freq}_train_norm.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    train_counts = pd.read_csv(f'Data/{line}_{freq}_train_sample_count.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    X_train, Y_train, Y_train_ix = roll(train.index, train.values, lags, preds)
    
    val = pd.read_csv(f'Data/{line}_{freq}_val_norm.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    val_counts = pd.read_csv(f'Data/{line}_{freq}_val_sample_count.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    X_val, Y_val, Y_val_ix = roll(val.index, val.values, lags, preds)

    test = pd.read_csv(f'Data/{line}_{freq}_test_norm.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    test_counts = pd.read_csv(f'Data/{line}_{freq}_test_sample_count.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    test_mean = pd.read_csv(f'Data/{line}_{freq}_test_mean.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    test_std = pd.read_csv(f'Data/{line}_{freq}_test_std.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    X_test, Y_test, Y_test_ix = roll(test.index, test.values, lags, preds)
    _, mu_test, _ = roll(test_mean.index, test_mean.values, lags, preds)
    _, sigma_test, _ = roll(test_std.index, test_std.values, lags, preds)
        
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test, Y_test_ix, mu_test, sigma_test)

# Tilted loss for both mean and quantiles
def joint_tilted_loss(quantiles, y, f):
    loss = K.mean(K.square(y[:,:,:,0,0]-f[:,:,:,0,0]), axis = -1)
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,:,:,0,k+1]-f[:,:,:,0,k+1])
        loss += K.mean(K.maximum(q*e, (q-1)*e))
    return loss

def joint_convLstm(num_filters, kernel_length, input_timesteps, num_links, output_timesteps, quantiles, prob, loss, opt):
    model = Sequential()
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_links, 1, 1)))
    model.add(ConvLSTM2D(name ='conv_lstm_1',
                         filters = num_filters, kernel_size = (kernel_length, 1), 
                         padding='same',
                         return_sequences=False))

    model.add(Dropout(prob, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(Flatten())
    model.add(RepeatVector(output_timesteps))
    model.add(Reshape((output_timesteps, num_links, 1, num_filters)))

    model.add(ConvLSTM2D(name ='conv_lstm_2',filters = num_filters, kernel_size = (kernel_length, 1), padding='same',return_sequences = True))
    model.add(Dropout(prob, name = 'dropout_2'))

    model.add(TimeDistributed(Dense(units = len(quantiles) + 1, name = 'dense_1')))
    model.compile(loss = loss, optimizer = opt)
    return model

@use_named_args(dimensions=dimensions)
def train_DQR(num_filters, kernel_length, prob):
    opt = Adagrad(learning_rate=1e-3)
    loss = lambda y, f: joint_tilted_loss(quantiles, y, f)
    net = joint_convLstm(num_filters, kernel_length, lags, num_links, preds, quantiles, prob, loss, opt)
    mc = tf.keras.callbacks.ModelCheckpoint(filepath=f"DQR/Model Checkpoint/joint/gp_{num_filters}_{kernel_length}_{prob}",
                                            save_weights_only=True,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    history = net.fit(X_train, y_traink, validation_data=(X_val, y_valk), batch_size=50, callbacks=[es, mc], epochs=200)
    print(np.min(history.history['val_loss']))
    return np.min(history.history['val_loss'])

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found')
else:
    print('Found GPU at: {}'.format(device_name))

line = '300S'
freq = '15min'
lags = 32
preds = 3

data_train, data_val, data_test = load_data(line, freq, lags, preds)

# Unpack tensors
(X_train, y_train) = data_train
(X_val, y_val) = data_val
(X_test, y_test, y_ix_test, y_mean_test, y_std_test) = data_test

X_train = X_train[..., np.newaxis, np.newaxis]
y_train = y_train[..., np.newaxis, np.newaxis]
X_val = X_val[..., np.newaxis, np.newaxis]
y_val = y_val[..., np.newaxis, np.newaxis]
X_test = X_test[..., np.newaxis, np.newaxis]
y_test = y_test[..., np.newaxis, np.newaxis]

quantiles = np.sort([0.025, 0.975, 0.05, 0.95, 0.10, 0.90, 0.20, 0.8, 0.4, 0.6])
y_traink = np.zeros((y_train.shape[0], y_train.shape[1], y_train.shape[2], 1, len(quantiles)+1))
y_valk  = np.zeros((y_val.shape[0], y_train.shape[1], y_train.shape[2], 1, len(quantiles)+1))
for i in range(len(quantiles)+1):
    y_traink[:,:,:,:,i] = y_train[:,:,:,:,0]
    y_valk[:,:,:,:,i] = y_val[:,:,:,:,0]

num_links = X_train.shape[2]
num_samples = 50    
    
search_result = gp_minimize(func=train_DQR,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=num_samples,
                            x0=default_parameters,
                            random_state=46)
print("Best parameters", search_result.x)
print("Validation Loss", search_result.fun)
