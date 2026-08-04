# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements binary logistic regression from scratch.
"""


import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=4, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train_augmented = np.column_stack((X_train, np.ones(X_train.shape[0])))
X_test_augmented = np.column_stack((X_test, np.ones(X_test.shape[0])))

sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def loss_and_gradient(parameters):
    probability = sigmoid(X_train_augmented @ parameters)
    epsilon = 1e-12
    loss = -np.mean(y_train * np.log(probability + epsilon) + (1 - y_train) * np.log(1 - probability + epsilon))
    gradient = X_train_augmented.T @ (probability - y_train) / y_train.size
    return loss, gradient

result = minimize(
    fun=lambda p: loss_and_gradient(p)[0],
    x0=np.zeros(X_train_augmented.shape[1]),
    jac=lambda p: loss_and_gradient(p)[1],
    method="BFGS",
)
if not result.success:
    raise RuntimeError(result.message)

probability = sigmoid(X_test_augmented @ result.x)
y_predicted = (probability >= 0.5).astype(int)
accuracy = np.mean(y_predicted == y_test)
print(f"Test accuracy: {accuracy:.3f}")
