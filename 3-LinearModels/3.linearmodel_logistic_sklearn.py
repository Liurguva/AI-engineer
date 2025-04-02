# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This file gives a simple example for using a logistic regression model from sklearn.
"""

from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y = True)
LRG = linear_model.LogisticRegression(
    random_state = 0, solver = 'liblinear', multi_class='auto'
    )
LRG.fit (X, y)
print(LRG.score(X, y))