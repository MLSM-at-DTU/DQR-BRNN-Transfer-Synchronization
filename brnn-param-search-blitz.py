import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from blitz.modules import BayesianLinear, BayesianLSTM
from blitz.utils import variational_estimator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from skopt import gp_minimize

# Search space:
units = Integer(low=10, high=50, name='units')
prob = Real(low=.75, high=1, prior='uniform', name='prob')
sigma1 = Real(low=1, high=3, prior='uniform', name='sigma1')
sigma2 = Real(low=.001, high=1, prior='uniform', name='sigma2')
dimensions = [units, prob, sigma1, sigma2]
default_parameters = [10, 1, 1, 0.1]

@variational_estimator
class BRNN(nn.Module):
    def __init__(self, n_links, hidden_size=20, prior_pi=1.0, prior_sigma_1=1.0, prior_sigma_2=0.01):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_1 = BayesianLSTM(n_links, hidden_size, prior_pi=prior_pi, prior_sigma_1 = prior_sigma_1, prior_sigma_2 = prior_sigma_2, posterior_rho_init=1.0, peephole=False)
        self.lstm_2 = BayesianLSTM(hidden_size, hidden_size, prior_pi=prior_pi, prior_sigma_1 = prior_sigma_1, prior_sigma_2 = prior_sigma_2, posterior_rho_init=1.0, peephole=False)
        self.linear = BayesianLinear(hidden_size, n_links, prior_pi=prior_pi, prior_sigma_1 = prior_sigma_1, prior_sigma_2 = prior_sigma_2, posterior_rho_init=3.0)
            
    def forward(self, x):
        # encoder
        x_, (c_t, h_t) = self.lstm_1(x)        
        # gathering only the latent end-of-sequence for the decoder layer
        x_ = x_[:, -3:, :]
        # decoder
        x_, _ = self.lstm_2(x_, hidden_states = (c_t, h_t)) 

        # time invariant dense layer
        x_ = x_.reshape(-1, self.hidden_size)
        x_ = self.linear(x_)
        x_ = x_.reshape(-1, 3, 39)
        return x_

def roll(ix, ts, lags, preds):
    X = np.stack([np.roll(ts, i, axis=0) for i in range(lags, 0, -1)], axis=1)[lags:-preds,]
    Y = np.stack([np.roll(ts, -i, axis=0) for i in range(0, preds, 1)], axis=1)[lags:-preds,]
    Y_ix = ix[lags:-preds]
    return X, Y, Y_ix

def load_data(line, freq, lags, preds):
    train = pd.read_csv(f'Data/{line}_{freq}_train_raw.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    train_counts = pd.read_csv(f'Data/{line}_{freq}_train_sample_count.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    X_train, Y_train, Y_train_ix = roll(train.index, train.values, lags, preds)
    
    val = pd.read_csv(f'Data/{line}_{freq}_val_raw.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    val_counts = pd.read_csv(f'Data/{line}_{freq}_val_sample_count.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    X_val, Y_val, Y_val_ix = roll(val.index, val.values, lags, preds)

    test = pd.read_csv(f'Data/{line}_{freq}_test_raw.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    test_counts = pd.read_csv(f'Data/{line}_{freq}_test_sample_count.csv.gz', index_col=0, compression='gzip', parse_dates=True)
    X_test, Y_test, Y_test_ix = roll(test.index, test.values, lags, preds)
    
    # Descale
    df_mean = train.groupby([ train.index.weekday, train.index.hour]).mean()
    df_mean.index.names = ['DoW', 'ToD']
    
    X_train_mu, y_train_mu, _ = roll(train.index, df_mean.loc[zip(train.index.weekday, train.index.hour)].values, lags, preds)
    X_val_mu, y_val_mu, _ = roll(val.index, df_mean.loc[zip(val.index.weekday, val.index.hour)].values, lags, preds)
    X_test_mu, y_test_mu, _ = roll(test.index, df_mean.loc[zip(test.index.weekday, test.index.hour)].values, lags, preds)
    
    std = train.values.std(axis=0)
    
    return (X_train, Y_train, Y_train_ix, X_train_mu, y_train_mu, std, std), (X_val, Y_val, Y_val_ix, X_val_mu, y_val_mu, std, std), (X_test, Y_test, Y_test_ix, X_test_mu, y_test_mu, std, std)

@use_named_args(dimensions=dimensions)
def train_BRNN(units, prob, sigma1, sigma2):
    print("*************************************************************")
    print("Parameters:", units, prob, sigma1, sigma2)
    print("*************************************************************")

    net = BRNN(num_links, units, prob, sigma1, sigma2).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,15,30], gamma=0.1)
    
    losses = []
    val_losses = []
    for epoch in range(50):
        running_loss = 0

        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            loss = net.sample_elbo(inputs=datapoints,
                                   labels=labels,
                                   criterion=criterion,
                                   sample_nbr=3,
                                   complexity_cost_weight=datapoints.shape[0]/X_train.shape[0])
            loss.backward()
            optimizer.step()
            
            if np.isnan(loss.item()):
                print('Nan detected.')
                if len(val_losses) > 0:
                    return np.min(val_losses)
                else:
                    return 1e6
            
            running_loss += loss.item()

        preds_val = net(X_val_)
        loss_val = net.sample_elbo(inputs=X_val_,
                                   labels=y_val_,
                                   criterion=criterion,
                                   sample_nbr=3,
                                   complexity_cost_weight=X_val_.shape[0]/X_train.shape[0])
        print(f"Epoch: {epoch}, Train-loss {running_loss:.4f} Val-loss: {loss_val:.4f}")
        scheduler.step()
        losses.append(running_loss)
        val_losses.append(loss_val.item())
        
    return np.min(val_losses)
    
line = '300S'
freq = '15min'
lags = 32
preds = 3

data_train, data_val, data_test = load_data(line, freq, lags, preds)

# Unpack tensors
(X_train, y_train, y_ix_train, X_train_mu, y_train_mu, X_train_std, y_train_std) = data_train
(X_val, y_val, y_ix_val, X_val_mu, y_val_mu, X_val_std, y_val_std) = data_val
(X_test, y_test, y_ix_test, X_test_mu, y_test_mu, X_val_std, y_val_std) = data_test

X_train = X_train
y_train = y_train
X_val = X_val
y_val = y_val
X_test = X_test
y_test = y_test

print('X_train', X_train.shape, 'y_train', y_train.shape)
print('X_val', X_val.shape, 'y_val', y_val.shape)
print('X_test', X_test.shape, 'y_test', y_test.shape)

X_train_scaled = (X_train - X_train_mu) / X_train_std
y_train_scaled = (y_train - y_train_mu) / y_train_std
X_val_scaled = (X_val - X_val_mu) / X_val_std
y_val_scaled = (y_val - y_val_mu) / y_val_std
X_test_scaled = (X_test - X_test_mu) / X_val_std
y_test_scaled = (y_test - y_test_mu) / y_val_std

X_train_, y_train_ = torch.tensor(X_train_scaled).float().cuda(), torch.tensor(y_train_scaled).float().cuda()
X_val_, y_val_  = torch.tensor(X_val_scaled).float().cuda(), torch.tensor(y_val_scaled).float().cuda()
X_test_, y_test_  = torch.tensor(X_test_scaled).float().cuda(), torch.tensor(y_test_scaled).float().cuda()

ds_train = torch.utils.data.TensorDataset(X_train_, y_train_)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=50, shuffle=True)

num_links = X_train.shape[2]

num_samples = 50

search_result = gp_minimize(func=train_BRNN,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=num_samples,
                            x0=default_parameters,
                            random_state=46)

print("Best parameters", search_result.x)
print("Validation Loss", search_result.fun)
