# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements out-of-fold stacking for classification.
"""


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33, stratify=y)
base_models = [
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    DecisionTreeClassifier(max_depth=5, random_state=33),
    RandomForestClassifier(n_estimators=100, random_state=33),
]
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
meta_train = np.zeros((X_train.shape[0], len(base_models)))
meta_test = np.zeros((X_test.shape[0], len(base_models)))

for model_index, model in enumerate(base_models):
    fold_test_predictions = []
    for fit_index, validation_index in folds.split(X_train, y_train):
        model.fit(X_train[fit_index], y_train[fit_index])
        meta_train[validation_index, model_index] = model.predict_proba(X_train[validation_index])[:, 1]
        fold_test_predictions.append(model.predict_proba(X_test)[:, 1])
    meta_test[:, model_index] = np.mean(fold_test_predictions, axis=0)

meta_model = LogisticRegression().fit(meta_train, y_train)
y_predicted = meta_model.predict(meta_test)
print(f"Stacked test accuracy: {accuracy_score(y_test, y_predicted):.3f}")
