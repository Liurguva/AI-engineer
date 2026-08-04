# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies local outlier factor to multivariate anomaly detection.
"""


import numpy as np
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.default_rng(53)
normal = rng.normal(size=(300, 2))
anomalies = np.array([[6.0, 6.0], [-6.0, 5.0], [5.0, -6.0]])
X = np.vstack((normal, anomalies))
model = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
prediction = model.fit_predict(X)
print("Detected anomaly indices:", np.flatnonzero(prediction == -1))
