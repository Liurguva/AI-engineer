# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements tabular Q-learning in a one-dimensional environment.
"""


import numpy as np

rng = np.random.default_rng(60)
number_of_states = 7
actions = np.array([-1, 1])
Q = np.zeros((number_of_states, actions.size))
learning_rate = 0.2
discount = 0.95
epsilon = 0.2

for episode in range(500):
    state = 0
    for _ in range(100):
        if rng.random() < epsilon:
            action_index = rng.integers(actions.size)
        else:
            action_index = np.argmax(Q[state])
        next_state = int(np.clip(state + actions[action_index], 0, number_of_states - 1))
        reward = 1.0 if next_state == number_of_states - 1 else -0.01
        target = reward if next_state == number_of_states - 1 else reward + discount * np.max(Q[next_state])
        Q[state, action_index] += learning_rate * (target - Q[state, action_index])
        state = next_state
        if state == number_of_states - 1:
            break

state = 0
path = [state]
while state != number_of_states - 1 and len(path) < 30:
    state = int(np.clip(state + actions[np.argmax(Q[state])], 0, number_of_states - 1))
    path.append(state)
print("Learned path:", path)
print("Q table:\n", np.round(Q, 3))
