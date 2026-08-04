# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements Gaussian Naive Bayes from scratch.
"""


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=11, stratify=y
)
classes = np.unique(y_train)
means = np.vstack([X_train[y_train == class_name].mean(axis=0) for class_name in classes])
variances = np.vstack([X_train[y_train == class_name].var(axis=0) + 1e-9 for class_name in classes])
priors = np.array([np.mean(y_train == class_name) for class_name in classes])

log_posteriors = []
for class_index, class_name in enumerate(classes):
    log_likelihood = -0.5 * np.sum(
        np.log(2.0 * np.pi * variances[class_index])
        + (X_test - means[class_index]) ** 2 / variances[class_index],
        axis=1,
    )
    log_posteriors.append(np.log(priors[class_index]) + log_likelihood)

y_predicted = classes[np.argmax(np.column_stack(log_posteriors), axis=1)]
print(f"Test accuracy: {np.mean(y_predicted == y_test):.3f}")
