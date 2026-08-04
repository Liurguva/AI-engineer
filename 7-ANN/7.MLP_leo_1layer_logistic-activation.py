# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example trains a one-hidden-layer logistic MLP for Iris classification.
"""


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(22)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=22, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
Y_train = np.eye(3)[y_train]

W1 = rng.normal(scale=0.2, size=(X_train.shape[1], 10))
b1 = np.zeros((1, 10))
W2 = rng.normal(scale=0.2, size=(10, 3))
b2 = np.zeros((1, 3))
learning_rate = 0.2
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

for _ in range(2500):
    hidden = sigmoid(X_train @ W1 + b1)
    logits = hidden @ W2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    error_output = (probability - Y_train) / X_train.shape[0]
    gradient_W2 = hidden.T @ error_output
    gradient_b2 = error_output.sum(axis=0, keepdims=True)
    error_hidden = (error_output @ W2.T) * hidden * (1.0 - hidden)
    gradient_W1 = X_train.T @ error_hidden
    gradient_b1 = error_hidden.sum(axis=0, keepdims=True)

    W2 -= learning_rate * gradient_W2
    b2 -= learning_rate * gradient_b2
    W1 -= learning_rate * gradient_W1
    b1 -= learning_rate * gradient_b1

hidden_test = sigmoid(X_test @ W1 + b1)
y_predicted = np.argmax(hidden_test @ W2 + b2, axis=1)
print(f"Test accuracy: {np.mean(y_predicted == y_test):.3f}")
