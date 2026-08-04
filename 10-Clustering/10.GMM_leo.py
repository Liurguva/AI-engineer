# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements a full-covariance Gaussian mixture with EM.
"""


import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X, true_labels = make_blobs(n_samples=450, centers=3, cluster_std=[0.7, 1.0, 0.6], random_state=46)
rng = np.random.default_rng(46)
components = 3
means = X[rng.choice(X.shape[0], components, replace=False)]
covariances = np.array([np.cov(X.T) for _ in range(components)])
weights = np.full(components, 1.0 / components)

for _ in range(100):
    weighted_density = np.column_stack(
        [weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k], allow_singular=False) for k in range(components)]
    )
    responsibilities = weighted_density / (weighted_density.sum(axis=1, keepdims=True) + 1e-15)
    effective_counts = responsibilities.sum(axis=0)
    new_means = (responsibilities.T @ X) / effective_counts[:, None]
    new_covariances = []
    for k in range(components):
        centered = X - new_means[k]
        covariance = (responsibilities[:, k, None] * centered).T @ centered / effective_counts[k]
        new_covariances.append(covariance + 1e-6 * np.eye(X.shape[1]))
    new_covariances = np.array(new_covariances)
    new_weights = effective_counts / X.shape[0]
    if np.allclose(new_means, means, atol=1e-5):
        means, covariances, weights = new_means, new_covariances, new_weights
        break
    means, covariances, weights = new_means, new_covariances, new_weights

labels = responsibilities.argmax(axis=1)
print("Mixture weights:", np.round(weights, 3))
print(f"Adjusted Rand index: {adjusted_rand_score(true_labels, labels):.3f}")
