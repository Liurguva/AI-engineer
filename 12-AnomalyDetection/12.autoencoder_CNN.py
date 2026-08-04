# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example trains a convolutional autoencoder on MNIST.
"""


import numpy as np
import tensorflow as tf

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:12000].astype("float32") / 255.0
x_test = x_test[:2000].astype("float32") / 255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2, padding="same"),
        tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2, padding="same"),
        tf.keras.layers.Conv2DTranspose(8, 3, strides=2, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same"),
    ]
)
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, x_train, epochs=3, batch_size=128, validation_split=0.1, verbose=2)
reconstruction = model.predict(x_test, verbose=0)
error = np.mean((x_test - reconstruction) ** 2, axis=(1, 2, 3))
print("Highest-error sample indices:", np.argsort(error)[-10:])
