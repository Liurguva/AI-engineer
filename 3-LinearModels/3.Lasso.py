# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example demonstrates L1-regularized linear regression (Lasso).
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(6)
X = rng.normal(size=(120, 10))
true_weights = np.array([3.0, -2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.5, 0.0])
y = X @ true_weights + rng.normal(0.0, 0.6, X.shape[0])
X_scaled = StandardScaler().fit_transform(X)

model = Lasso(alpha=0.08, max_iter=10000).fit(X_scaled, y)
print("Estimated coefficients:", np.round(model.coef_, 3))
print(f"Nonzero coefficients: {np.count_nonzero(model.coef_)}")

plt.stem(np.arange(model.coef_.size), model.coef_)
plt.xlabel("feature index")
plt.ylabel("estimated coefficient")
plt.tight_layout()
plt.show()
