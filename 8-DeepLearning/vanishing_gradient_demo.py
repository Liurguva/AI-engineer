# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example demonstrates vanishing and exploding gradients in deep chains.
"""


import matplotlib.pyplot as plt
import numpy as np

layer_count = np.arange(1, 101)
vanishing = 0.7**layer_count
stable = 1.0**layer_count
exploding = 1.2**layer_count

print(f"Gradient after 100 layers with factor 0.7: {vanishing[-1]:.3e}")
print(f"Gradient after 100 layers with factor 1.2: {exploding[-1]:.3e}")

plt.semilogy(layer_count, vanishing, label="factor 0.7")
plt.semilogy(layer_count, stable, label="factor 1.0")
plt.semilogy(layer_count, exploding, label="factor 1.2")
plt.xlabel("number of multiplied derivatives")
plt.ylabel("gradient magnitude")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
