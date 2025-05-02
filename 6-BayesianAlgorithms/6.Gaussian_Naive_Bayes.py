# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This script file is for Naive Bayes for continuous attribute values (called Gaussian Naive Bayes).
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB

# ------------------ Load data ------------------ #
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20190308, test_size=0.3)

# Calculate the probabilities of every attribute value P(x_j|c) for P(x|c)
def gaussion_pdf(x_test, x):
    return np.exp(-(x_test-x.mean(0))**2 / (2 * x.std(0)**2)) / np.sqrt(2 * np.pi * x.std(0)**2)

# -------------- Define Naive Bayes Model -------------- #
classes = np.unique(np.concatenate([y_train, y_test], 0))
pred_probs = []


# Loop over all classes to obtain the probabilities of all testing samples belonging to different classes
for i in classes:
    idx_i = y_train == i # idx_i stores all indices of samples in Class c (y_train == i)
    p_c = len(idx_i) / len(y_train) # Calculate P(c)
    p_x_c = np.prod(gaussion_pdf(x_test, x_train[idx_i]), 1) # Use Gaussian to calculate P(x|c) for continuous attribute values
    prob_i = p_c * p_x_c # Joint probability prob_i is the probability of sample i belonging to Class c
    pred_probs.append(prob_i)
# Array for probabilities of samples belonging to different classes: N_samples x N_classes
pred_probs = np.vstack(pred_probs).T 
# Index of the class with the highest probability as the classification for each sample
label_idx = pred_probs.argmax(1) 
y_pred = classes[label_idx] # Select the class based on the index

score = np.count_nonzero(y_test == y_pred)/len(y_test) # Accuracy: correct when the predicted class is the same as the actual label
print('The accuracy of self-developed Gaussian Naive Bayes is: ',score)

# ------------------ Apply Gaussian Naive Bayes from Scikit-learn ------------------ #   
model = GaussianNB()
model.fit(x_train, y_train)
print('The accuracy of Gaussian Naive Bayes from Scikit-learn is: ', model.score(x_test, y_test))