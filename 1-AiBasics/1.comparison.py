# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example compares physics-based, numerical, and data-driven solutions.
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(1)
initial_horizontal_velocity = 5.0
gravity = 9.81
final_time = 5.0
time_step = 0.25
time = np.arange(0.0, final_time + time_step, time_step)

# Physics-based analytical solution for a horizontally launched object.
x_analytical = initial_horizontal_velocity * time
y_analytical = 0.5 * gravity * time**2

# Physics-based numerical solution using explicit time integration.
x_numerical = np.zeros_like(time)
y_numerical = np.zeros_like(time)
vertical_velocity = 0.0
for i in range(1, time.size):
    x_numerical[i] = x_numerical[i - 1] + initial_horizontal_velocity * time_step
    y_numerical[i] = y_numerical[i - 1] + vertical_velocity * time_step
    vertical_velocity = vertical_velocity + gravity * time_step

# Data-driven solution fitted to noisy observations.
noise = rng.normal(loc=0.0, scale=2.5, size=time.size)
y_observed = y_analytical + noise
polynomial = PolynomialFeatures(degree=2, include_bias=False)
X_polynomial = polynomial.fit_transform(x_analytical.reshape(-1, 1))
model = LinearRegression().fit(X_polynomial, y_observed)
y_data_driven = model.predict(X_polynomial)

print(f"Data-driven R^2: {model.score(X_polynomial, y_observed):.3f}")

plt.plot(x_analytical, y_analytical, label="Analytical")
plt.plot(x_numerical, y_numerical, "o--", label="Numerical")
plt.scatter(x_analytical, y_observed, label="Observed data")
plt.plot(x_analytical, y_data_driven, label="Data-driven")
plt.xlabel("Horizontal distance, x")
plt.ylabel("Vertical drop, y")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
