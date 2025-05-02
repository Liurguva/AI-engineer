# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This file shows the comparison of data-driven method against traditional engineering methods for an example given in Chapter 1.
"""

import matplotlib.pyplot as plt
import numpy as np

v_0 = 5
t = 10
delta_t = 0.5
g = 9.81
n_step = int(t/delta_t)
t = np.linspace(0,10,n_step)

# Physics-Based Method 1: Analytical solution
x_ana = t * v_0 
y_ana = g/(2*v_0**2) * x_ana**2 # 0.01962 = g/(2*v_0^2)
plt.plot(x_ana,y_ana,'g',label='Analytical')

# Physics-Based Method 1: Numerical solution
x_num = np.zeros_like(x_ana)
y_num = np.zeros_like(x_ana) # For storing y_i
v_xi = v_0
v_yi = 0
for i in range(1,n_step ):
    v_yi = v_yi + g*delta_t
    x_num[i] = x_num[i-1] + v_xi * delta_t
    y_num[i] = y_num[i-1] + v_yi * delta_t   
plt.plot(x_num,y_num,'r-*',label='Numerical')

# Data-Driven Method
rdm = np.random.RandomState(1) # default_rng(12345) 
noise = rdm.randint(-100,100, size=np.shape(y_ana))/10.0
y_exp = y_ana + noise
plt.plot(x_ana,y_exp,'ko',label='Experimental')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# KernelRidge(alpha=1.0,kernel='poly') # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} 
ml = LinearRegression()
pf = PolynomialFeatures(degree=2)
ml.fit(pf.fit_transform(x_ana.reshape(-1, 1)), y_exp.reshape(-1, 1)) # ml.coef_
xx = pf.transform(x_ana.reshape(-1, 1))
y_ml=ml.predict(xx) 
plt.plot(x_ana,y_ml,'c*',label='Data-Driven')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(frameon=False)