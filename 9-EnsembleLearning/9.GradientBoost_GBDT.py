# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements gradient boosting for squared-error regression.
"""


import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)
initial_prediction = y_train.mean()
train_prediction = np.full(y_train.shape, initial_prediction)
test_prediction = np.full(y_test.shape, initial_prediction)
learning_rate = 0.05
learners = []

for _ in range(100):
    residual = y_train - train_prediction
    tree = DecisionTreeRegressor(max_depth=2, random_state=32)
    tree.fit(X_train, residual)
    train_prediction += learning_rate * tree.predict(X_train)
    test_prediction += learning_rate * tree.predict(X_test)
    learners.append(tree)

print(f"Test RMSE: {mean_squared_error(y_test, test_prediction) ** 0.5:.3f}")
