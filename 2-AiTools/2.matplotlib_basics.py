# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example introduces basic Matplotlib plotting operations.
"""


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.0, 2.0 * np.pi, 200)
y_sine = np.sin(x)
y_cosine = np.cos(x)

plt.plot(x, y_sine, label="sin(x)")
plt.plot(x, y_cosine, label="cos(x)")
plt.scatter(x[::20], y_sine[::20], marker="o", label="sampled sin(x)")
plt.xlabel("x")
plt.ylabel("function value")
plt.title("Basic Matplotlib example")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
