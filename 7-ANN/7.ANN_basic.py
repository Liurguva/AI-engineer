# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example trains one logistic neuron by gradient descent.
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(20)
X, y = make_classification(
    n_samples=250,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    class_sep=1.5,
    random_state=20,
)
X = StandardScaler().fit_transform(X)
weights = rng.normal(scale=0.1, size=X.shape[1])
intercept = 0.0
learning_rate = 0.1
loss_history = []

for _ in range(500):
    logits = X @ weights + intercept
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
    error = probability - y
    weights -= learning_rate * (X.T @ error) / X.shape[0]
    intercept -= learning_rate * error.mean()
    loss = -np.mean(y * np.log(probability + 1e-12) + (1 - y) * np.log(1 - probability + 1e-12))
    loss_history.append(loss)

prediction = (probability >= 0.5).astype(int)
print("Weights:", weights)
print(f"Intercept: {intercept:.3f}")
print(f"Training accuracy: {np.mean(prediction == y):.3f}")

plt.plot(loss_history)
plt.xlabel("iteration")
plt.ylabel("cross-entropy loss")
plt.tight_layout()
plt.show()
