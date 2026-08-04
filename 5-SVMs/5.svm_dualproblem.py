# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example solves a hard-margin linear SVM in the dual form.
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

X = np.array([[-1.0, -1.5], [0.0, -1.0], [1.0, -0.5], [-0.5, 0.0], [1.0, -1.0], [1.5, 1.0], [1.0, 2.0], [0.5, 1.5], [2.5, 1.5], [2.0, 2.0]])
y = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

gram = X @ X.T
objective = lambda alpha: 0.5 * np.sum(np.outer(alpha * y, alpha * y) * gram) - np.sum(alpha)
constraints = {"type": "eq", "fun": lambda alpha: np.dot(alpha, y)}
bounds = [(0.0, None)] * X.shape[0]
result = minimize(objective, x0=np.zeros(X.shape[0]), method="SLSQP", bounds=bounds, constraints=constraints)
if not result.success:
    raise RuntimeError(result.message)

alpha = result.x
support_mask = alpha > 1e-5
support_vectors = X[support_mask]
weights = np.sum((alpha * y)[:, None] * X, axis=0)
intercept = np.mean(y[support_mask] - support_vectors @ weights)
print("Nonzero alpha:", alpha[support_mask])
print("Weights:", weights)
print(f"Intercept: {intercept:.3f}")

x_line = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
y_line = -(weights[0] * x_line + intercept) / weights[1]
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors="none", edgecolors="black", s=140, label="support vectors")
plt.plot(x_line, y_line, label="decision boundary")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
