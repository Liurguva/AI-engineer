# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies scikit-learn K-means clustering.
"""


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=350, centers=4, cluster_std=0.75, random_state=43)
model = KMeans(n_clusters=4, n_init=10, random_state=43).fit(X)
print(f"Inertia: {model.inertia_:.3f}")
print(f"Silhouette score: {silhouette_score(X, model.labels_):.3f}")
