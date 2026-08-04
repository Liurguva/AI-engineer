# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example solves linear regression by minimizing mean squared error.
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(3)
X = rng.uniform(-2.0, 2.0, size=(80, 2))
y = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.75 + rng.normal(0.0, 0.35, X.shape[0])

objective = lambda parameters: np.mean((X @ parameters[:-1] + parameters[-1] - y) ** 2)
result = minimize(objective, x0=np.zeros(X.shape[1] + 1), method="BFGS")
if not result.success:
    raise RuntimeError(result.message)

weights = result.x[:-1]
intercept = result.x[-1]
y_fitted = X @ weights + intercept
print("Estimated weights:", weights)
print(f"Estimated intercept: {intercept:.3f}")
print(f"Mean squared error: {objective(result.x):.4f}")

figure = plt.figure()
axis = figure.add_subplot(projection="3d")
axis.scatter(X[:, 0], X[:, 1], y, label="observations")
axis.scatter(X[:, 0], X[:, 1], y_fitted, label="optimized fit")
axis.set_xlabel("x1")
axis.set_ylabel("x2")
axis.set_zlabel("y")
axis.legend(frameon=False)
plt.tight_layout()
plt.show()
