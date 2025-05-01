# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This file gives a simple example for using decision trees from sklearn.
"""

from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

X_original, y_original = load_iris(return_X_y=True) # Load data from Dateset Iris
Data_original = np.hstack((X_original,y_original.reshape(y_original.size,1))) # Merge the orignal data for shuffling
Data = Data_original.copy() # Create data for shuffling
# np.random.shuffle(Data) # Shuffle data
X = Data[:,0:-1].copy() # Extract X from the Shuffled data, 150 X 4
y = Data[:,-1].copy() # Extract y from the Suffled data, 150 X 1

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X[0:-30], y[0:-30]) # Select Samples 1-120 for traning

print(clf.score(X[-31:-1], y[-31:-1])) # Use the rest of samples, i.e., Sample 121-150, for testing
tree.plot_tree(clf) 