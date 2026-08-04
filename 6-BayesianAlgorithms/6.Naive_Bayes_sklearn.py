# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies scikit-learn Gaussian Naive Bayes.
"""


from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=12, stratify=y
)
model = GaussianNB().fit(X_train, y_train)
y_predicted = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_predicted):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_test, y_predicted))
