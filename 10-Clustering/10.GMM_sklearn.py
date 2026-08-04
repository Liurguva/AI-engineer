# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies scikit-learn Gaussian mixture clustering.
"""


from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

X, true_labels = make_blobs(n_samples=450, centers=3, cluster_std=[0.7, 1.0, 0.6], random_state=47)
model = GaussianMixture(n_components=3, covariance_type="full", random_state=47).fit(X)
labels = model.predict(X)
print("Mixture weights:", model.weights_)
print(f"Adjusted Rand index: {adjusted_rand_score(true_labels, labels):.3f}")
