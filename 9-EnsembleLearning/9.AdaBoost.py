# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements binary AdaBoost with decision stumps.
"""


import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, labels = make_classification(
    n_samples=500, n_features=5, n_informative=4, n_redundant=0, class_sep=1.2, random_state=31
)
y = np.where(labels == 1, 1.0, -1.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=31, stratify=y)
weights = np.full(X_train.shape[0], 1.0 / X_train.shape[0])
learners = []
learner_weights = []

for _ in range(30):
    stump = DecisionTreeClassifier(max_depth=1, random_state=31)
    stump.fit(X_train, y_train, sample_weight=weights)
    prediction = stump.predict(X_train)
    weighted_error = np.sum(weights[prediction != y_train])
    weighted_error = np.clip(weighted_error, 1e-12, 1.0 - 1e-12)
    alpha = 0.5 * np.log((1.0 - weighted_error) / weighted_error)
    weights *= np.exp(-alpha * y_train * prediction)
    weights /= weights.sum()
    learners.append(stump)
    learner_weights.append(alpha)

score = sum(alpha * learner.predict(X_test) for alpha, learner in zip(learner_weights, learners))
y_predicted = np.where(score >= 0.0, 1.0, -1.0)
print(f"Test accuracy: {np.mean(y_predicted == y_test):.3f}")
