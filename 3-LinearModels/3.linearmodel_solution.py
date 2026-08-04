# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example solves linear regression with the normal equation.
"""


import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(2)
X = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [1.5, 2.5], [2.0, 4.0], [1.0, 3.0], [3.0, 1.5]])
true_weights = np.array([1.0, 2.0])
true_intercept = 3.0
y = X @ true_weights + true_intercept + rng.normal(0.0, 0.25, X.shape[0])

X_augmented = np.column_stack((X, np.ones(X.shape[0])))
parameters = np.linalg.pinv(X_augmented) @ y
weights = parameters[:-1]
intercept = parameters[-1]
y_fitted = X @ weights + intercept

print("Estimated weights:", weights)
print(f"Estimated intercept: {intercept:.3f}")
print(f"Mean squared error: {np.mean((y - y_fitted) ** 2):.4f}")

figure = plt.figure()
axis = figure.add_subplot(projection="3d")
axis.scatter(X[:, 0], X[:, 1], y, label="observations")
axis.scatter(X[:, 0], X[:, 1], y_fitted, label="fitted values")
axis.set_xlabel("x1")
axis.set_ylabel("x2")
axis.set_zlabel("y")
axis.legend(frameon=False)
plt.tight_layout()
plt.show()
