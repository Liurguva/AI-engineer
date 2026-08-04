# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example introduces core NumPy operations used throughout the repository.
"""


import numpy as np

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
w = np.array([0.5, -0.25])
b = 1.0

y = X @ w + b
column_means = X.mean(axis=0)
centered = X - column_means
covariance = centered.T @ centered / (X.shape[0] - 1)

print("Predictions:", y)
print("Column means:", column_means)
print("Covariance matrix:\n", covariance)
