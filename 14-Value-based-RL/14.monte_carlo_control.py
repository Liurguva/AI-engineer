# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements first-visit Monte Carlo control.
"""


from collections import defaultdict
import numpy as np

rng = np.random.default_rng(62)
number_of_states = 7
actions = np.array([-1, 1])
Q = np.zeros((number_of_states, actions.size))
returns = defaultdict(list)
discount = 0.95
epsilon = 0.2

for episode in range(2000):
    trajectory = []
    state = 0
    for _ in range(100):
        action_index = rng.integers(actions.size) if rng.random() < epsilon else np.argmax(Q[state])
        next_state = int(np.clip(state + actions[action_index], 0, number_of_states - 1))
        reward = 1.0 if next_state == number_of_states - 1 else -0.01
        trajectory.append((state, action_index, reward))
        state = next_state
        if state == number_of_states - 1:
            break
    return_value = 0.0
    visited = set()
    for state, action_index, reward in reversed(trajectory):
        return_value = reward + discount * return_value
        pair = (state, action_index)
        if pair not in visited:
            returns[pair].append(return_value)
            Q[pair] = np.mean(returns[pair])
            visited.add(pair)

print("Q table:\n", np.round(Q, 3))
