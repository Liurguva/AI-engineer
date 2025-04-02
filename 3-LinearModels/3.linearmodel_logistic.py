# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This file impplements logistic regression in linear models.
"""

from sklearn import datasets
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.datasets import load_iris
X, y_label = load_iris(return_X_y = True)

# Convert label array to the onehot format
y_onehot = np.zeros((y_label.size, y_label.max()+1))
y_onehot[np.arange(y_label.size),y_label] = 1
y=y_onehot

def LogisitcRegression(X,y):
 
    X_bar = np.hstack((X,np.ones([X.shape[0],1]))) # I = 8 by J+1 = 3
    W_hat = Optimize(X_bar,y)
    return W_hat, X_bar

def Optimize(X_bar,y):
    from scipy.optimize import minimize
    e = 1e-10 # Tolerance
    fun = lambda W_hat : - np.sum( y * np.log( np.exp( X_bar@W_hat.reshape(X_bar.shape[1],y.shape[1]) ) / np.sum( np.exp(  X_bar@W_hat.reshape(X_bar.shape[1],y.shape[1]) ) ) ) )# Logistic regression cost function
    
    W_hat0 = np.ones(X_bar.shape[1]*y.shape[1]) # Set initial values
    res = minimize(fun, W_hat0, method='SLSQP', tol=e)
    W_hat = res.x
    return W_hat
    print('The minimum value is：',res.fun)
    print('The parameter for getting the minimum value is：',res.x)

W_hat, X_bar= LogisitcRegression(X,y)
W = W_hat[:-2]
b = W_hat[-2:]

y_fit = np.dot(X_bar,W_hat.reshape(X_bar.shape[1],y.shape[1]))

result_probability =  ( np.exp(  X_bar@W_hat.reshape(X_bar.shape[1],y.shape[1]) ) / np.sum( np.exp( X_bar@W_hat.reshape(X_bar.shape[1],y.shape[1]) ) ) ) 
y_pred = np.argmax(result_probability,axis=1) # Find the column number (axis=1) with the highest probability in each row
accuracy = 1-np.count_nonzero(y_pred-y_label)/y_label.size

