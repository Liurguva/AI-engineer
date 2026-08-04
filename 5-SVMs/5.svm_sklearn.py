# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example compares linear and RBF support vector classifiers.
"""


from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=400, noise=0.2, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10, stratify=y
)

for kernel in ("linear", "rbf"):
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, gamma="scale"))
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    print(f"{kernel} kernel accuracy: {accuracy_score(y_test, y_predicted):.3f}")
