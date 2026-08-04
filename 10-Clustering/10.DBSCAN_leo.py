# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements DBSCAN clustering from scratch.
"""


from collections import deque
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X, true_labels = make_moons(n_samples=350, noise=0.08, random_state=44)
X = StandardScaler().fit_transform(X)
epsilon = 0.28
minimum_points = 5
labels = np.full(X.shape[0], -99, dtype=int)  # -99 unvisited, -1 noise
cluster_id = 0

neighbors = [np.flatnonzero(np.linalg.norm(X - X[i], axis=1) <= epsilon) for i in range(X.shape[0])]
for point in range(X.shape[0]):
    if labels[point] != -99:
        continue
    if neighbors[point].size < minimum_points:
        labels[point] = -1
        continue
    labels[point] = cluster_id
    queue = deque(neighbors[point].tolist())
    while queue:
        neighbor = queue.popleft()
        if labels[neighbor] == -1:
            labels[neighbor] = cluster_id
        if labels[neighbor] != -99:
            continue
        labels[neighbor] = cluster_id
        if neighbors[neighbor].size >= minimum_points:
            queue.extend(neighbors[neighbor].tolist())
    cluster_id += 1

print(f"Clusters found: {cluster_id}")
print(f"Noise points: {np.sum(labels == -1)}")
print(f"Adjusted Rand index: {adjusted_rand_score(true_labels, labels):.3f}")
