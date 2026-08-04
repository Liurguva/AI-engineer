# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example detects univariate anomalies with the three-sigma rule.
"""


import numpy as np

rng = np.random.default_rng(50)
data = np.concatenate((rng.normal(10.0, 1.2, 200), np.array([2.0, 18.0, 21.0])))
mean = data.mean()
standard_deviation = data.std(ddof=1)
lower = mean - 3.0 * standard_deviation
upper = mean + 3.0 * standard_deviation
outliers = data[(data < lower) | (data > upper)]
print(f"Bounds: [{lower:.3f}, {upper:.3f}]")
print("Detected outliers:", outliers)
