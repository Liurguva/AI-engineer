# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example demonstrates bagging and voting ensembles.
"""


from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

X_regression, y_regression = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.25, random_state=30)
bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=50, random_state=30)
bagging_regressor.fit(X_train, y_train)
print(f"Bagging regression RMSE: {mean_squared_error(y_test, bagging_regressor.predict(X_test)) ** 0.5:.3f}")

X_classification, y_classification = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_classification, y_classification, test_size=0.25, random_state=30, stratify=y_classification
)
bagging_classifier = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=30)
bagging_classifier.fit(X_train, y_train)
print(f"Bagging classification accuracy: {accuracy_score(y_test, bagging_classifier.predict(X_test)):.3f}")

voting = VotingClassifier(
    estimators=[
        ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=30)),
    ],
    voting="soft",
)
voting.fit(X_train, y_train)
print(f"Voting accuracy: {accuracy_score(y_test, voting.predict(X_test)):.3f}")
