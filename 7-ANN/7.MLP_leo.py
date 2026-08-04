# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements a two-layer MLP and backpropagation from scratch.
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(21)
X, y = make_moons(n_samples=400, noise=0.18, random_state=21)
X = StandardScaler().fit_transform(X)
y = y[:, None]

n_hidden = 12
W1 = rng.normal(scale=np.sqrt(2.0 / X.shape[1]), size=(X.shape[1], n_hidden))
b1 = np.zeros((1, n_hidden))
W2 = rng.normal(scale=np.sqrt(2.0 / n_hidden), size=(n_hidden, 1))
b2 = np.zeros((1, 1))
learning_rate = 0.05
loss_history = []

for _ in range(2000):
    hidden_linear = X @ W1 + b1
    hidden = np.tanh(hidden_linear)
    output_linear = hidden @ W2 + b2
    probability = 1.0 / (1.0 + np.exp(-np.clip(output_linear, -35.0, 35.0)))

    error_output = (probability - y) / X.shape[0]
    gradient_W2 = hidden.T @ error_output
    gradient_b2 = error_output.sum(axis=0, keepdims=True)
    error_hidden = (error_output @ W2.T) * (1.0 - hidden**2)
    gradient_W1 = X.T @ error_hidden
    gradient_b1 = error_hidden.sum(axis=0, keepdims=True)

    W2 -= learning_rate * gradient_W2
    b2 -= learning_rate * gradient_b2
    W1 -= learning_rate * gradient_W1
    b1 -= learning_rate * gradient_b1

    loss = -np.mean(y * np.log(probability + 1e-12) + (1 - y) * np.log(1 - probability + 1e-12))
    loss_history.append(loss)

prediction = (probability.ravel() >= 0.5).astype(int)
print(f"Training accuracy: {np.mean(prediction == y.ravel()):.3f}")
print(f"Final loss: {loss_history[-1]:.4f}")

plt.plot(loss_history)
plt.xlabel("iteration")
plt.ylabel("cross-entropy loss")
plt.tight_layout()
plt.show()
