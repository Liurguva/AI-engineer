# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example trains and visualizes a scikit-learn decision tree.
"""


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=9, stratify=y
)
classifier = DecisionTreeClassifier(max_depth=3, random_state=9)
classifier.fit(X_train, y_train)
y_predicted = classifier.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_predicted):.3f}")

plot_tree(classifier, filled=True, feature_names=load_iris().feature_names, class_names=load_iris().target_names)
plt.tight_layout()
plt.show()
