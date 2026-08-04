# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies discretized tabular Q-learning to Gymnasium CartPole.
"""


import gymnasium as gym
import numpy as np

environment = gym.make("CartPole-v1")
rng = np.random.default_rng(63)
bins = [
    np.linspace(-2.4, 2.4, 7),
    np.linspace(-3.0, 3.0, 7),
    np.linspace(-0.21, 0.21, 9),
    np.linspace(-3.5, 3.5, 9),
]
Q = np.zeros(tuple(len(boundary) + 1 for boundary in bins) + (environment.action_space.n,))
discretize = lambda observation: tuple(np.digitize(observation[i], bins[i]) for i in range(4))

for episode in range(800):
    observation, _ = environment.reset(seed=63 + episode)
    state = discretize(observation)
    epsilon = max(0.05, 0.8 * (1.0 - episode / 800))
    for _ in range(500):
        action = int(rng.integers(environment.action_space.n)) if rng.random() < epsilon else int(np.argmax(Q[state]))
        next_observation, reward, terminated, truncated, _ = environment.step(action)
        next_state = discretize(next_observation)
        done = terminated or truncated
        target = reward if done else reward + 0.99 * np.max(Q[next_state])
        Q[state + (action,)] += 0.15 * (target - Q[state + (action,)])
        state = next_state
        if done:
            break
environment.close()
print("Training completed.")
