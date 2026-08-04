# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example trains a LeNet-5-style TensorFlow/Keras model on MNIST.
"""


import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.pad(x_train[:12000][..., None] / 255.0, [[0, 0], [2, 2], [2, 2], [0, 0]])
x_test = tf.pad(x_test[:2000][..., None] / 255.0, [[0, 0], [2, 2], [2, 2], [0, 0]])
y_train = y_train[:12000]
y_test = y_test[:2000]

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(6, 5, activation="tanh"),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(16, 5, activation="tanh"),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation="tanh"),
        tf.keras.layers.Dense(84, activation="tanh"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.1, verbose=2)
_, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.3f}")
