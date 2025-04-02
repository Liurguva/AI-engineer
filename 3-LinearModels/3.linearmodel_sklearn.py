# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This file gives a simple example for using a linear model from sklearn.
"""

from sklearn import linear_model as LM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

X_list = [[1.,1.],[1.,2.],[2.,2.],[2.,3.],[1.5,2.5],[2.,4.],[1.,3.],[3.,1.5]]
X = np.array(X_list)
Proj = np.array([1,2]) # Projection direction vector to get inclined plane in a 3D space
# y = X@Proj # Use this equation or the following one.
y = np.dot(X,Proj)+3  # Get the plane
y = y + np.random.rand(y.size) # Use rand to add noise to the "meausured" data
# regr = LM.LinearRegression(n_jobs=2)
regr = LM.Lasso(alpha = 0.5)
regr.fit(X,y)
regr.predict(np.array([[3,5]]))


y_fit = np.dot(X,regr.coef_)+3

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X[:,0],X[:,1],y,'b*')
ax.scatter(X[:,0],X[:,1],y_fit,'rd')
# ax.show()



