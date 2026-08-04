# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example demonstrates NumPy convolution and pooling in a LeNet-style forward pass.
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

image = load_digits().images[0] / 16.0
kernels = np.array(
    [
        [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]],
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
    ]
)

feature_maps = []
for kernel in kernels:
    output = np.zeros((image.shape[0] - 2, image.shape[1] - 2))
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            patch = image[row : row + 3, column : column + 3]
            output[row, column] = np.sum(patch * kernel)
    feature_maps.append(np.maximum(output, 0.0))
feature_maps = np.array(feature_maps)

pooled_maps = np.zeros((feature_maps.shape[0], feature_maps.shape[1] // 2, feature_maps.shape[2] // 2))
for channel in range(feature_maps.shape[0]):
    for row in range(pooled_maps.shape[1]):
        for column in range(pooled_maps.shape[2]):
            patch = feature_maps[channel, 2 * row : 2 * row + 2, 2 * column : 2 * column + 2]
            pooled_maps[channel, row, column] = patch.mean()

flattened_features = pooled_maps.ravel()
print("Input shape:", image.shape)
print("Convolution output shape:", feature_maps.shape)
print("Pooled feature vector length:", flattened_features.size)

plt.imshow(image, cmap="gray")
plt.title("Example digit used for the LeNet-style forward pass")
plt.axis("off")
plt.tight_layout()
plt.show()
