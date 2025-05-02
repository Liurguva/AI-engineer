# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This file shows the basic SVM implemented using functions from sklearn.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

X = np.array([[-1, -1.5], [0, -1], [1, -0.5], [-0.5, 0], [1, -1], [1.5, 1], [1, 2], [0.5, 1.5], [2.5, 1.5], [2, 2]])
y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

from sklearn.svm import SVC
SVCClf = SVC(kernel = 'linear',gamma = 'scale', shrinking = False,)
SVCClf.fit(X, y)
#SVCClf.predict([[-0.5,-0.8]])


# 0 = np.dot(X,SVCClf.coef_.T)+SVCClf.intercept_
w = SVCClf.coef_[0]
# Obtain arrays for ploting the fitting curve
a = -w[0] / w[1]
xx = np.linspace(X[:,0].min(), X[:,1].max(),100)
yy = a * xx - (SVCClf.intercept_[0]) / w[1]

fig = plt.figure()
# ax = plt.axes() # (projection='3d')
plt.scatter(X[:,0],X[:,1],marker='*',color='b')
# ax.cla()
# ax.scatter(X[:,0],X[:,1],'rd')
plt.scatter(xx,yy,marker='.',color='r')
# ax.show()



