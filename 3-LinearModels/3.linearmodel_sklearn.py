# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies scikit-learn linear regression.
"""


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(7)
X = rng.normal(size=(150, 3))
y = X @ np.array([1.5, -2.0, 0.75]) + 4.0 + rng.normal(0.0, 0.4, X.shape[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
model = LinearRegression().fit(X_train, y_train)
y_predicted = model.predict(X_test)

print("Coefficients:", model.coef_)
print(f"Intercept: {model.intercept_:.3f}")
print(f"Test MSE: {mean_squared_error(y_test, y_predicted):.4f}")
print(f"Test R^2: {r2_score(y_test, y_predicted):.3f}")
