# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements Gaussian-process regression and compares it with scikit-learn.
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

rng = np.random.default_rng(15)
X_train = np.linspace(0.0, 8.0, 12)[:, None]
y_train = np.sin(X_train[:, 0]) + rng.normal(0.0, 0.12, X_train.shape[0])
X_test = np.linspace(0.0, 8.0, 250)[:, None]

length_scale = 1.0
noise_variance = 0.12**2
rbf = lambda A, B: np.exp(-0.5 * (A - B.T) ** 2 / length_scale**2)
K = rbf(X_train, X_train) + noise_variance * np.eye(X_train.shape[0])
K_cross = rbf(X_train, X_test)
K_test = rbf(X_test, X_test)
K_inverse = np.linalg.inv(K)
mean_manual = K_cross.T @ K_inverse @ y_train
covariance_manual = K_test - K_cross.T @ K_inverse @ K_cross
std_manual = np.sqrt(np.maximum(np.diag(covariance_manual), 0.0))

kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(noise_level=noise_variance)
model = GaussianProcessRegressor(kernel=kernel, random_state=15).fit(X_train, y_train)
mean_sklearn, std_sklearn = model.predict(X_test, return_std=True)
print("Fitted scikit-learn kernel:", model.kernel_)

plt.scatter(X_train[:, 0], y_train, label="observations")
plt.plot(X_test[:, 0], mean_manual, label="manual GP mean")
plt.fill_between(X_test[:, 0], mean_manual - 1.96 * std_manual, mean_manual + 1.96 * std_manual, alpha=0.2)
plt.plot(X_test[:, 0], mean_sklearn, "--", label="scikit-learn GP mean")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
