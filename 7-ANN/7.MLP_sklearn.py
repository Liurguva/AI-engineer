# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies a scikit-learn multilayer perceptron classifier.
"""


from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23, stratify=y
)
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, early_stopping=True, random_state=23),
)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_predicted):.3f}")
