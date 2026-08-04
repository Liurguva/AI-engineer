# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example uses polynomial basis functions for nonlinear regression.
"""


import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(5)
X = np.linspace(-3.0, 3.0, 70)
y = 0.5 * X**3 - 1.2 * X**2 + 0.8 * X + 2.0 + rng.normal(0.0, 2.0, X.size)

# Explicit polynomial basis: [1, x, x^2, x^3].
Phi = np.column_stack((np.ones(X.size), X, X**2, X**3))
weights = np.linalg.pinv(Phi) @ y
y_fitted = Phi @ weights

print("Polynomial coefficients [1, x, x^2, x^3]:", weights)
print(f"Mean squared error: {np.mean((y - y_fitted) ** 2):.3f}")

plt.scatter(X, y, label="observations")
plt.plot(X, y_fitted, label="cubic basis fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
