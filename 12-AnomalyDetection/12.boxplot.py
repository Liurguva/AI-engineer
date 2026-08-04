# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example detects univariate anomalies with the IQR rule.
"""


import numpy as np

rng = np.random.default_rng(51)
data = np.concatenate((rng.normal(10.0, 1.2, 200), np.array([2.0, 18.0, 21.0])))
first_quartile, third_quartile = np.percentile(data, [25, 75])
interquartile_range = third_quartile - first_quartile
lower = first_quartile - 1.5 * interquartile_range
upper = third_quartile + 1.5 * interquartile_range
outliers = data[(data < lower) | (data > upper)]
print(f"Bounds: [{lower:.3f}, {upper:.3f}]")
print("Detected outliers:", outliers)
