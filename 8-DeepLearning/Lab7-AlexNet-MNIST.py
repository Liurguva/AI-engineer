# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example trains a scaled AlexNet-style TensorFlow/Keras model on MNIST.
"""


import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:12000].astype("float32") / 255.0
x_test = x_test[:2000].astype("float32") / 255.0
y_train = y_train[:12000]
y_test = y_test[:2000]
x_train = tf.image.resize(x_train[..., None], (64, 64))
x_test = tf.image.resize(x_test[..., None], (64, 64))

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(64, 64, 1)),
        tf.keras.layers.Conv2D(32, 7, strides=2, activation="relu"),
        tf.keras.layers.MaxPooling2D(3, strides=2),
        tf.keras.layers.Conv2D(64, 5, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(3, strides=2),
        tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.1, verbose=2)
_, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.3f}")
