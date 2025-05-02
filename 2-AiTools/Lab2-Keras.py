# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at www.AI-engineer.org.
This script file is for implementing a DNN using Keras.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from keras.models import Model

l = tf.keras.layers

model = tf.keras.Sequential([
    l.Flatten(input_shape=(784,)),
    l.Dense(128, activation='relu'),
    l.Dense(128, activation='relu'),
    l.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.summary()

model.fit(x_train.reshape(-1,784),pd.get_dummies(y_train),nb_epoch=15,batch_size=128,verbose=1)
    
    