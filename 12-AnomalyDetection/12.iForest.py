# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example applies an isolation forest to multivariate anomaly detection.
"""


import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(52)
normal = rng.normal(size=(300, 2))
anomalies = np.array([[6.0, 6.0], [-6.0, 5.0], [5.0, -6.0]])
X = np.vstack((normal, anomalies))
model = IsolationForest(contamination=0.02, random_state=52).fit(X)
prediction = model.predict(X)
print("Detected anomaly indices:", np.flatnonzero(prediction == -1))
