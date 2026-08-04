# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements categorical ID3, C4.5, and CART decision trees.
"""


from collections import Counter
from math import log2
from pathlib import Path
import time

import numpy as np

METHOD = "ID3"  # Select "ID3", "C4_5", or "CART".
folder = Path(__file__).resolve().parent
data = np.loadtxt(folder / "DT_watermelon_data.txt", dtype=str).tolist()
features = np.loadtxt(folder / "DT_watermelon_feature.txt", dtype=str).tolist()


def entropy_and_gini(dataset):
    counts = Counter(row[-1] for row in dataset)
    probabilities = np.array(list(counts.values()), dtype=float) / len(dataset)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    gini = 1.0 - np.sum(probabilities**2)
    return float(entropy), float(gini)


def split_dataset(dataset, feature_index, feature_value):
    subset = []
    for row in dataset:
        if row[feature_index] == feature_value:
            subset.append(row[:feature_index] + row[feature_index + 1 :])
    return subset


def best_feature(dataset, method):
    base_entropy, _ = entropy_and_gini(dataset)
    candidates = []
    for feature_index in range(len(dataset[0]) - 1):
        values = sorted({row[feature_index] for row in dataset})
        weighted_entropy = 0.0
        weighted_gini = 0.0
        intrinsic_value = 0.0
        for value in values:
            subset = split_dataset(dataset, feature_index, value)
            proportion = len(subset) / len(dataset)
            subset_entropy, subset_gini = entropy_and_gini(subset)
            weighted_entropy += proportion * subset_entropy
            weighted_gini += proportion * subset_gini
            intrinsic_value -= proportion * log2(proportion)
        information_gain = base_entropy - weighted_entropy
        gain_ratio = information_gain / intrinsic_value if intrinsic_value > 0 else 0.0
        score = {
            "ID3": information_gain,
            "C4_5": gain_ratio,
            "CART": -weighted_gini,
        }[method]
        candidates.append(score)
    return int(np.argmax(candidates))


def majority_label(labels):
    return Counter(labels).most_common(1)[0][0]


def create_tree(dataset, available_features, method):
    labels = [row[-1] for row in dataset]
    if len(set(labels)) == 1:
        return labels[0]
    if not available_features or len({tuple(row[:-1]) for row in dataset}) == 1:
        return majority_label(labels)

    feature_index = best_feature(dataset, method)
    feature_name = available_features[feature_index]
    tree = {feature_name: {}}
    remaining_features = available_features[:feature_index] + available_features[feature_index + 1 :]
    for value in sorted({row[feature_index] for row in dataset}):
        subset = split_dataset(dataset, feature_index, value)
        tree[feature_name][value] = create_tree(subset, remaining_features.copy(), method)
    return tree


start = time.perf_counter()
decision_tree = create_tree(data, features.copy(), METHOD)
elapsed = time.perf_counter() - start
print(f"Method: {METHOD}")
print("Tree:", decision_tree)
print(f"Construction time: {elapsed:.6f} s")

import tree_plot_function

tree_plot_function.main(decision_tree)
