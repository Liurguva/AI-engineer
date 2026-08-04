# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example trains a PyTorch REINFORCE policy on Gymnasium CartPole.
"""


import gymnasium as gym
import numpy as np
import torch
from torch import nn

torch.manual_seed(72)
environment = gym.make("CartPole-v1")
policy = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2), nn.Softmax(dim=-1))
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
discount = 0.99

for episode in range(300):
    observation, _ = environment.reset(seed=72 + episode)
    log_probabilities = []
    rewards = []
    for _ in range(500):
        probability = policy(torch.tensor(observation, dtype=torch.float32))
        distribution = torch.distributions.Categorical(probability)
        action = distribution.sample()
        log_probabilities.append(distribution.log_prob(action))
        observation, reward, terminated, truncated, _ = environment.step(int(action.item()))
        rewards.append(reward)
        if terminated or truncated:
            break

    returns = []
    return_value = 0.0
    for reward in reversed(rewards):
        return_value = reward + discount * return_value
        returns.append(return_value)
    returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    loss = -torch.stack([log_probability * value for log_probability, value in zip(log_probabilities, returns)]).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

environment.close()
print("Training completed.")
