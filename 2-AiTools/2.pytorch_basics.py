# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example demonstrates PyTorch tensors and automatic differentiation.
"""


import torch

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])
w = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.05)

for _ in range(200):
    prediction = x @ w + b
    loss = torch.mean((prediction - y) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"w = {w.item():.3f}")
print(f"b = {b.item():.3f}")
print(f"loss = {loss.item():.6f}")
