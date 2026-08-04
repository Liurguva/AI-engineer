# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example trains a simple recurrent neural network from scratch.
"""


import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(24)
sequence = np.sin(np.linspace(0.0, 16.0 * np.pi, 500))
window = 12
X = np.array([sequence[i : i + window] for i in range(sequence.size - window)])
y = sequence[window:, None]

hidden_size = 16
Wxh = rng.normal(scale=0.2, size=(hidden_size, 1))
Whh = rng.normal(scale=0.2, size=(hidden_size, hidden_size))
Why = rng.normal(scale=0.2, size=(1, hidden_size))
bh = np.zeros((hidden_size, 1))
by = np.zeros((1, 1))
learning_rate = 0.01
loss_history = []

for epoch in range(120):
    total_loss = 0.0
    for sample, target in zip(X[::4], y[::4]):
        states = [np.zeros((hidden_size, 1))]
        for value in sample:
            states.append(np.tanh(Wxh * value + Whh @ states[-1] + bh))
        prediction = Why @ states[-1] + by
        error = prediction - target
        total_loss += float(error.item() ** 2)

        gradient_Why = error @ states[-1].T
        gradient_by = error
        gradient_Wxh = np.zeros_like(Wxh)
        gradient_Whh = np.zeros_like(Whh)
        gradient_bh = np.zeros_like(bh)
        gradient_state = Why.T @ error
        for time_index in reversed(range(window)):
            local_gradient = (1.0 - states[time_index + 1] ** 2) * gradient_state
            gradient_Wxh += local_gradient * sample[time_index]
            gradient_Whh += local_gradient @ states[time_index].T
            gradient_bh += local_gradient
            gradient_state = Whh.T @ local_gradient

        for gradient in (gradient_Why, gradient_by, gradient_Wxh, gradient_Whh, gradient_bh):
            np.clip(gradient, -1.0, 1.0, out=gradient)
        Why -= learning_rate * gradient_Why
        by -= learning_rate * gradient_by
        Wxh -= learning_rate * gradient_Wxh
        Whh -= learning_rate * gradient_Whh
        bh -= learning_rate * gradient_bh
    loss_history.append(total_loss / len(X[::4]))

predictions = []
for sample in X:
    state = np.zeros((hidden_size, 1))
    for value in sample:
        state = np.tanh(Wxh * value + Whh @ state + bh)
    predictions.append((Why @ state + by).item())
print(f"Final mean squared error: {np.mean((np.array(predictions) - y.ravel()) ** 2):.4f}")

plt.plot(y[:200], label="target")
plt.plot(predictions[:200], label="RNN prediction")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
