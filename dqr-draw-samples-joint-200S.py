import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

# Config
quantiles = np.sort([0.025, 0.975, 0.05, 0.95, 0.10, 0.90, 0.20, 0.8, 0.4, 0.6])
num_links_tst = 29
preds=3
S = 500

# Load predictions / scaling
y_pred = np.load('DQR/y_test_pred_joint_200s.npy')
y_mean_test = np.load('DQR/y_test_mean_200s.npy')
y_std_test = np.load('DQR/y_test_std_200s.npy')

print('Loaded data/shapes:')
print('y_pred', y_pred.shape)
print('y_mean_test', y_mean_test.shape)
print('y_std_test', y_std_test.shape)

y_pred_s = np.empty((len(y_pred), y_pred.shape[1], num_links_tst, S))

for n in range(y_pred_s.shape[0]):
    if (n + 1) % 100 == 0:
        print(f"{n + 1} / {len(y_pred)}")
    for t in range(preds):
        for l in range(num_links_tst):
            mu = y_pred[n, t, l, 0, 0]
            (sigma), _ = curve_fit(lambda x, sigma: norm.cdf(x, mu, sigma), y_pred[n, t, l, 0, 1:], quantiles)
            y_pred_s[n, t, l, :] = norm.rvs(loc=mu, scale=sigma, size=S) * y_std_test[n, t, l] + y_mean_test[n, t, l]

np.save('DQR/y_test_pred_joint_samples_200s.npy', y_pred_s)
                