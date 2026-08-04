# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements the equations of one LSTM cell with NumPy.
"""


import numpy as np

rng = np.random.default_rng(25)
sequence = np.array([[0.2], [0.4], [0.1], [0.8], [0.5]])
input_size = 1
hidden_size = 3
combined_size = input_size + hidden_size

W_forget = rng.normal(scale=0.3, size=(hidden_size, combined_size))
W_input = rng.normal(scale=0.3, size=(hidden_size, combined_size))
W_candidate = rng.normal(scale=0.3, size=(hidden_size, combined_size))
W_output = rng.normal(scale=0.3, size=(hidden_size, combined_size))
b_forget = np.zeros((hidden_size, 1))
b_input = np.zeros((hidden_size, 1))
b_candidate = np.zeros((hidden_size, 1))
b_output = np.zeros((hidden_size, 1))

sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
hidden = np.zeros((hidden_size, 1))
cell = np.zeros((hidden_size, 1))

for value in sequence:
    combined = np.vstack((hidden, value.reshape(1, 1)))
    forget_gate = sigmoid(W_forget @ combined + b_forget)
    input_gate = sigmoid(W_input @ combined + b_input)
    candidate = np.tanh(W_candidate @ combined + b_candidate)
    output_gate = sigmoid(W_output @ combined + b_output)
    cell = forget_gate * cell + input_gate * candidate
    hidden = output_gate * np.tanh(cell)

print("Final cell state:\n", cell)
print("Final hidden state:\n", hidden)
