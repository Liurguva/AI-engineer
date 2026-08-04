# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements tabular SARSA in a one-dimensional environment.
"""


import numpy as np

rng = np.random.default_rng(61)
number_of_states = 7
actions = np.array([-1, 1])
Q = np.zeros((number_of_states, actions.size))
learning_rate = 0.2
discount = 0.95
epsilon = 0.2

choose_action = lambda state: rng.integers(actions.size) if rng.random() < epsilon else np.argmax(Q[state])
for episode in range(500):
    state = 0
    action_index = choose_action(state)
    for _ in range(100):
        next_state = int(np.clip(state + actions[action_index], 0, number_of_states - 1))
        reward = 1.0 if next_state == number_of_states - 1 else -0.01
        if next_state == number_of_states - 1:
            target = reward
            Q[state, action_index] += learning_rate * (target - Q[state, action_index])
            break
        next_action_index = choose_action(next_state)
        target = reward + discount * Q[next_state, next_action_index]
        Q[state, action_index] += learning_rate * (target - Q[state, action_index])
        state, action_index = next_state, next_action_index

state = 0
path = [state]
while state != number_of_states - 1 and len(path) < 30:
    state = int(np.clip(state + actions[np.argmax(Q[state])], 0, number_of_states - 1))
    path.append(state)
print("Learned path:", path)
print("Q table:\n", np.round(Q, 3))
