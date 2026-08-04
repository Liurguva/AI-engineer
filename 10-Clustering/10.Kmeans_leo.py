# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements K-means clustering from scratch.
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

X, true_labels = make_blobs(n_samples=350, centers=4, cluster_std=0.75, random_state=42)
rng = np.random.default_rng(42)
centroids = X[rng.choice(X.shape[0], size=4, replace=False)]

for _ in range(100):
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    new_centroids = np.vstack([X[labels == cluster].mean(axis=0) for cluster in range(4)])
    if np.allclose(new_centroids, centroids):
        break
    centroids = new_centroids

inertia = np.sum((X - centroids[labels]) ** 2)
print(f"Inertia: {inertia:.3f}")
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=180, label="centroids")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
