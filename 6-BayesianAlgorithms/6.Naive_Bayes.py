# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This script file is for Naive Bayes for discrete feature values (e.g., non-numeric).
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

# X[-31:-1], y[-31:-1] # Use the rest of samples, i.e., Sample 121-150, for testing
# tree.plot_tree(clf) 

# Training: Calculate P(c)
C,Counts = np.unique(y_train,return_counts=True) # Matrix of classificatioin labels and matrix of numbers of samples in the categories
Pc = Counts / y_train.size


# Training: The following loop calculate the mean and standard variance of individual features for each category
Data_c = np.array([])
Mu = np.array([])
Sigma = np.array([])
for n_c in range(0,C.size):
    Data_c = Data_train[Data_train[:,-1] == C[n_c]] # Extract data based on classification label
    Mu = np.append(Mu, Data_c[:,0:-1].mean(axis=0))
    Sigma = np.append(Sigma,Data_c[:,0:-1].std(axis=0))
Mu = Mu.reshape(C.size,X_train.shape[1])
Sigma = Sigma.reshape(C.size,X_train.shape[1])
    
# Testing: Calculate P(x|c) and and P(x|c)
Pxc = np.empty(shape=[0, X_test.shape[1]])
Pcx = np.empty(shape=[y_test.size,C.size])
for n_test in range(0,y_test.size):
    for n_c in range(0,C.size):
        sigma = Sigma[n_c]
        mu = Mu[n_c]
        pxc = 1/((2*np.pi)**0.5* sigma) * np.exp(-(X_test[n_test]-mu)**2/(2*sigma**2))
        pxc = pxc.reshape(1,pxc.size)

        Pcx[n_test,n_c] = Pc[n_c] * pxc.prod(axis=1)            
    
y_pred = np.argmax(Pcx,axis=1)
accuracy = 1-np.count_nonzero(y_pred-y_test)/y_test.size
    

    
    




