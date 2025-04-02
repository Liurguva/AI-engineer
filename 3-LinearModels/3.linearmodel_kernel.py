# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This file gives the implementation of a basic linear model (for regression) with kernels.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

X_list = [[1.,1.],[1.,2.],[2.,2.],[2.,3.],[1.5,2.5],[2.,4.],[1.,3.],[3.,1.5]]
X = np.array(X_list)
Proj = np.array([1,2])
d = 3 # Order of the polinomial for testing; to be used for both generating data and in the testing of the model
y = (X**d).sum(axis=1)+np.dot(X,Proj)+3 
y = y + np.random.randint(-3,3,size=y.size)


def KernelMatrix(X_bar,d):
    # Use Polynomial kernal with the order J+1
    global K
    K = ((X_bar@X_bar.T)**d)
    return K


def LinearRegression(X,y):
 
    X_bar = np.hstack((X,np.ones([X.shape[0],1]))) # I = 8 by J+1 = 3
    K = KernelMatrix(X_bar,d)
    z = np.linalg.inv(K)@y
    return z

z = LinearRegression(X,y)
y_fit = K@z

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X[:,0],X[:,1],y,'b*')
ax.plot_trisurf(X[:,0],X[:,1],y_fit)
# ax.show()
