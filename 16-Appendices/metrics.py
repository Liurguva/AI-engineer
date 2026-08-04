# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example calculates common regression and classification metrics.
"""


import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score

observed_regression = np.array([3.0, 5.0, 2.5, 7.0, 6.0])
predicted_regression = np.array([2.8, 5.3, 2.1, 6.8, 6.4])
manual_mae = np.mean(np.abs(observed_regression - predicted_regression))
manual_rmse = np.sqrt(np.mean((observed_regression - predicted_regression) ** 2))
print(f"Manual MAE: {manual_mae:.3f}")
print(f"scikit-learn MAE: {mean_absolute_error(observed_regression, predicted_regression):.3f}")
print(f"Manual RMSE: {manual_rmse:.3f}")
print(f"scikit-learn RMSE: {mean_squared_error(observed_regression, predicted_regression) ** 0.5:.3f}")
print(f"R^2: {r2_score(observed_regression, predicted_regression):.3f}")

observed_classification = np.array([0, 1, 1, 0, 1, 0, 1, 1])
predicted_classification = np.array([0, 1, 0, 0, 1, 1, 1, 1])
print(f"Accuracy: {accuracy_score(observed_classification, predicted_classification):.3f}")
print(f"Precision: {precision_score(observed_classification, predicted_classification):.3f}")
print(f"Recall: {recall_score(observed_classification, predicted_classification):.3f}")
print("Confusion matrix:\n", confusion_matrix(observed_classification, predicted_classification))
