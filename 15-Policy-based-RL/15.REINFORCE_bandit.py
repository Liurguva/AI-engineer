# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies REINFORCE to a multi-armed bandit.
"""


import numpy as np

rng = np.random.default_rng(70)
true_reward_probabilities = np.array([0.15, 0.30, 0.65, 0.45])
preferences = np.zeros(true_reward_probabilities.size)
learning_rate = 0.05
baseline = 0.0

for episode in range(5000):
    shifted = preferences - preferences.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    action = rng.choice(preferences.size, p=probabilities)
    reward = float(rng.random() < true_reward_probabilities[action])
    baseline += 0.01 * (reward - baseline)
    gradient = -probabilities
    gradient[action] += 1.0
    preferences += learning_rate * (reward - baseline) * gradient

probabilities = np.exp(preferences - preferences.max())
probabilities /= probabilities.sum()
print("Learned action probabilities:", np.round(probabilities, 3))
print("Best true arm:", int(np.argmax(true_reward_probabilities)))
