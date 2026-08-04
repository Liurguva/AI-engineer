# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies scikit-learn DBSCAN clustering.
"""


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X, true_labels = make_moons(n_samples=350, noise=0.08, random_state=45)
X = StandardScaler().fit_transform(X)
labels = DBSCAN(eps=0.28, min_samples=5).fit_predict(X)
print(f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Noise points: {np.sum(labels == -1)}")
print(f"Adjusted Rand index: {adjusted_rand_score(true_labels, labels):.3f}")
