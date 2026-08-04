# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements principal component analysis from scratch.
"""


import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_standardized = StandardScaler().fit_transform(X)
covariance = np.cov(X_standardized, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
X_projected = X_standardized @ eigenvectors[:, :2]

model = PCA(n_components=2).fit(X_standardized)
print("Manual explained-variance ratios:", eigenvalues[:2] / eigenvalues.sum())
print("scikit-learn explained-variance ratios:", model.explained_variance_ratio_)
print("Projected shape:", X_projected.shape)
