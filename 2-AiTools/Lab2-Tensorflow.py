# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example demonstrates TensorFlow 2 tensors and automatic differentiation.
"""


import tensorflow as tf

x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y = tf.constant([[3.0], [5.0], [7.0], [9.0]])
w = tf.Variable([[0.0]])
b = tf.Variable([0.0])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

for _ in range(200):
    with tf.GradientTape() as tape:
        prediction = x @ w + b
        loss = tf.reduce_mean((prediction - y) ** 2)
    gradients = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(zip(gradients, [w, b]))

print(f"w = {float(w.numpy()[0, 0]):.3f}")
print(f"b = {float(b.numpy()[0]):.3f}")
print(f"loss = {float(loss.numpy()):.6f}")
