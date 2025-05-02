# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 22:40:21 2020
This code works for continuous feature only.
@author: leo-desk
"""

from sklearn.datasets import load_iris
import numpy as np
import math

X_original, y_original = load_iris(return_X_y=True) # Load data from Dateset Iris
Data_original = np.hstack((X_original,y_original.reshape(y_original.size,1))) # Merge the orignal data for shuffling
Data = Data_original.copy() # Create data for shuffling
np.random.shuffle(Data) # Shuffle data
X = Data[:,0:-1].copy() # Extract X from the Shuffled data, 150 X 4
y = Data[:,-1].copy() # Extract y from the Suffled data, 150 X 1

# Ajust the value of n_test to adjust the number of testing samples
n_test = 50
X_train = X[0:-n_test]
y_train = y[0:-n_test]
Data_train = np.hstack((X_train,y_train.reshape(y_train.size,1))) # Sort X, y based on y
Data_train = Data_train[Data_train[:,-1].argsort()]
X_test = X[-(n_test+1):]
y_test = y[-(n_test+1):]


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
accuracy =  1-(y_test != y_pred).sum()/X_test.shape[0]
        

    

    
    




