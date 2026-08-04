# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements k-nearest-neighbor classification from scratch.
"""


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
k = 5
predictions = []

for sample in X_test:
    distances = np.linalg.norm(X_train - sample, axis=1)
    neighbors = y_train[np.argsort(distances)[:k]]
    prediction = np.bincount(neighbors).argmax()
    predictions.append(prediction)

print(f"Test accuracy: {np.mean(np.array(predictions) == y_test):.3f}")
