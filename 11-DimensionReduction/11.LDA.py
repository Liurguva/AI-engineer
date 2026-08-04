# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements linear discriminant analysis for dimension reduction.
"""


import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
overall_mean = X.mean(axis=0)
within_scatter = np.zeros((X.shape[1], X.shape[1]))
between_scatter = np.zeros_like(within_scatter)

for class_name in np.unique(y):
    X_class = X[y == class_name]
    class_mean = X_class.mean(axis=0)
    centered = X_class - class_mean
    within_scatter += centered.T @ centered
    mean_difference = (class_mean - overall_mean)[:, None]
    between_scatter += X_class.shape[0] * (mean_difference @ mean_difference.T)

eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(within_scatter) @ between_scatter)
order = np.argsort(eigenvalues.real)[::-1]
projection = eigenvectors[:, order[:2]].real
X_projected = X @ projection

model = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
print("Manual projected shape:", X_projected.shape)
print("scikit-learn explained-variance ratios:", model.explained_variance_ratio_)
