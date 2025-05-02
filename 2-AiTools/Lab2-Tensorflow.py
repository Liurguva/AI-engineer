# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at www.AI-engineer.org.
This script file is for implementing a DNN using TensorFlow and Keras.
"""

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# TensorFlow
X = tf.placeholder(dtype=tf.float64)
Y = tf.placeholder(dtype=tf.float64)
num_hidden=128

# Build a hidden layer
W_hidden = tf.Variable(np.random.randn(784, num_hidden))
b_hidden = tf.Variable(np.random.randn(num_hidden))
p_hidden = tf.nn.sigmoid( tf.add(tf.matmul(X, W_hidden), b_hidden) )

# Build another hidden layer
W_hidden2 = tf.Variable(np.random.randn(num_hidden, num_hidden))
b_hidden2 = tf.Variable(np.random.randn(num_hidden))
p_hidden2 = tf.nn.sigmoid( tf.add(tf.matmul(p_hidden, W_hidden2), b_hidden2) )

# Build the output layer
W_output = tf.Variable(np.random.randn(num_hidden, 10))
b_output = tf.Variable(np.random.randn(10))
p_output = tf.nn.softmax( tf.add(tf.matmul(p_hidden2, W_output), b_output) )

loss = tf.reduce_mean(tf.losses.mean_squared_error(
        labels=Y,predictions=p_output))
accuracy=1-tf.sqrt(loss)
minimization_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

feed_dict = {
    X: x_train.reshape(-1,784),
    Y: pd.get_dummies(y_train)
}
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(10000):
        J_value = session.run(loss, feed_dict)
        acc = session.run(accuracy, feed_dict)
        if step % 100 == 0:
            print("Step:", step, " Loss:", J_value," Accuracy:", acc)

            session.run(minimization_op, feed_dict)
    pred00 = session.run([p_output], feed_dict={X: x_test.reshape(-1,784)})
    

# # Keras
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense
# from keras.models import Model

# l = tf.keras.layers

# model = tf.keras.Sequential([
#     l.Flatten(input_shape=(784,)),
#     l.Dense(128, activation='relu'),
#     l.Dense(128, activation='relu'),
#     l.Dense(10, activation='softmax')
# ])

# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

# model.summary()

# model.fit(x_train.reshape(-1,784),pd.get_dummies(y_train),nb_epoch=15,batch_size=128,verbose=1)