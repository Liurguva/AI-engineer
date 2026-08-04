# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements policy-gradient learning in a one-dimensional environment.
"""


import numpy as np

rng = np.random.default_rng(71)
number_of_states = 7
actions = np.array([-1, 1])
preferences = np.zeros((number_of_states, actions.size))
learning_rate = 0.02
discount = 0.97

for episode in range(3000):
    trajectory = []
    state = 0
    for _ in range(50):
        logits = preferences[state] - preferences[state].max()
        probabilities = np.exp(logits) / np.exp(logits).sum()
        action_index = rng.choice(actions.size, p=probabilities)
        next_state = int(np.clip(state + actions[action_index], 0, number_of_states - 1))
        reward = 1.0 if next_state == number_of_states - 1 else -0.01
        trajectory.append((state, action_index, reward))
        state = next_state
        if state == number_of_states - 1:
            break

    return_value = 0.0
    for state, action_index, reward in reversed(trajectory):
        return_value = reward + discount * return_value
        logits = preferences[state] - preferences[state].max()
        probabilities = np.exp(logits) / np.exp(logits).sum()
        gradient = -probabilities
        gradient[action_index] += 1.0
        preferences[state] += learning_rate * return_value * gradient

policy = np.argmax(preferences, axis=1)
print("Greedy actions (0=left, 1=right):", policy)
