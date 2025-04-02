# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
Extended from linearmodel; it introduced optimization for loss/objective function, 
which is needed when a closed-form solution is not available
"""

# from sklearn import linear_model as LM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

X_list = [[1.,1.],[1.,2.],[2.,2.],[2.,3.],[1.5,2.5],[2.,4.],[1.,3.],[3.,1.5]]
X = np.array(X_list)
Proj = np.array([1,2])
y = np.dot(X,Proj)+3 
y = y + np.random.rand(y.size)



def LinearRegression(X,y):
 
    X_bar = np.hstack((X,np.ones([X.shape[0],1]))) # I = 8 by J+1 = 3
    W_hat = Optimize(X_bar,y)
    return W_hat, X_bar

def Optimize(X_bar,y):
    from scipy.optimize import minimize
    e = 1e-10 # Tolerance
    fun = lambda W_hat : np.sum( (np.dot(X_bar,W_hat) - y)**2 )  # Linear regression cost function

    W_hat0 = np.ones(X_bar.shape[1]) # Initial value
    res = minimize(fun, W_hat0, method='SLSQP')
    W_hat = res.x
    return W_hat
    print('The minimum value is：',res.fun)
    print('The parameter for getting the minimum value is：',res.x)

W_hat, X_bar= LinearRegression(X,y)
W = W_hat[:-2]
b = W_hat[-2:]

y_fit = np.dot(X_bar,W_hat)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X[:,0],X[:,1],y,'b*')
ax.scatter(X[:,0],X[:,1],y_fit,'rd')
# ax.show()
