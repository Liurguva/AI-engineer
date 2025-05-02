# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This file shows the basic SVM implemented using the original formulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

X = np.array([[-1, -1.5], [0, -1], [1, -0.5], [-0.5, 0], [1, -1], [1.5, 1], [1, 2], [0.5, 1.5], [2.5, 1.5], [2, 2]])
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])


# Prepare optimizer function
tole = 1e-8 # Tolerance
fun = lambda wb: wb[:-1]@wb[:-1] # Optimize that work with lambda function with one argument (general function can have more arguments)
# Add equality constraints
cons = (#{'type': 'eq', 'fun': lambda alpha: alpha @ y}, # func =0
        {'type': 'ineq', 'fun': lambda wb: y*(X@wb[:-1]+wb[-1])-1} # func > 0
        )
# Optimization
w0 = np.ones(X.shape[1])/X.shape[1] # Initial value; alpha0 must satisfy the above constraints
b0 = 1
wb0 = np.append(w0,b0)
res = minimize(fun, wb0, method='SLSQP',constraints=cons,tol=tole) # SLSQP  BFGS
wb = res.x 
print(wb)


w_star = wb[:-1]
b_star = wb[-1] # [0] gives out the first support vector
print('The parameter for getting the minimum loss is:',w_star,b_star)

# Generate data for the SVM separation curve  
a = -w_star[0] / w_star[1]
xx = np.linspace(X[:,0].min(), X[:,0].max(),100)
yy = a * xx + (b_star) #/ w_star[1]

fig = plt.figure()
# Plot original data with label
plt.scatter(X[:,0],X[:,1],marker='o',color='b')
for X1,y1 in zip(X,y):
    label = "{:.2f}".format(y1)
    plt.annotate(label, # this is the text
                 (X1[0],X1[1]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
# Plot the SVM separation curve
plt.scatter(xx,yy,marker='.',color='r')



