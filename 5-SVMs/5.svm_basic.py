# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example solves a hard-margin linear SVM in the primal form.
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

X = np.array([[-1.0, -1.5], [0.0, -1.0], [1.0, -0.5], [-0.5, 0.0], [1.0, -1.0], [1.5, 1.0], [1.0, 2.0], [0.5, 1.5], [2.5, 1.5], [2.0, 2.0]])
y = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

objective = lambda parameters: 0.5 * np.dot(parameters[:-1], parameters[:-1])
constraints = {"type": "ineq", "fun": lambda parameters: y * (X @ parameters[:-1] + parameters[-1]) - 1.0}
result = minimize(objective, x0=np.zeros(X.shape[1] + 1), method="SLSQP", constraints=constraints)
if not result.success:
    raise RuntimeError(result.message)

weights = result.x[:-1]
intercept = result.x[-1]
margin_values = y * (X @ weights + intercept)
support_vectors = X[np.isclose(margin_values, 1.0, atol=1e-4)]
print("Weights:", weights)
print(f"Intercept: {intercept:.3f}")
print("Support vectors:\n", support_vectors)

x_line = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
y_line = -(weights[0] * x_line + intercept) / weights[1]
y_margin_positive = -(weights[0] * x_line + intercept - 1.0) / weights[1]
y_margin_negative = -(weights[0] * x_line + intercept + 1.0) / weights[1]

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors="none", edgecolors="black", s=140, label="support vectors")
plt.plot(x_line, y_line, label="decision boundary")
plt.plot(x_line, y_margin_positive, "--")
plt.plot(x_line, y_margin_negative, "--")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
