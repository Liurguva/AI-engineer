# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This helper plots a nested-dictionary decision tree.
"""


import matplotlib.pyplot as plt


def _leaf_count(tree):
    if not isinstance(tree, dict):
        return 1
    root = next(iter(tree))
    return sum(_leaf_count(child) for child in tree[root].values())


def _depth(tree):
    if not isinstance(tree, dict):
        return 1
    root = next(iter(tree))
    return 1 + max(_depth(child) for child in tree[root].values())


def _plot_node(axis, text, center, parent, style):
    axis.annotate(
        text,
        xy=parent,
        xycoords="axes fraction",
        xytext=center,
        textcoords="axes fraction",
        ha="center",
        va="center",
        bbox=style,
        arrowprops={"arrowstyle": "<-"},
    )


def _plot_tree(axis, tree, parent, edge_text, state):
    root = next(iter(tree))
    leaves = _leaf_count(tree)
    center = (
        state["x_offset"] + (1.0 + leaves) / (2.0 * state["total_leaves"]),
        state["y_offset"],
    )
    axis.text((parent[0] + center[0]) / 2.0, (parent[1] + center[1]) / 2.0, edge_text, ha="center")
    _plot_node(axis, root, center, parent, {"boxstyle": "round", "fc": "0.9"})
    state["y_offset"] -= 1.0 / state["total_depth"]

    for value, child in tree[root].items():
        if isinstance(child, dict):
            _plot_tree(axis, child, center, str(value), state)
        else:
            state["x_offset"] += 1.0 / state["total_leaves"]
            leaf_center = (state["x_offset"], state["y_offset"])
            axis.text((center[0] + leaf_center[0]) / 2.0, (center[1] + leaf_center[1]) / 2.0, str(value), ha="center")
            _plot_node(axis, str(child), leaf_center, center, {"boxstyle": "round4", "fc": "1.0"})
    state["y_offset"] += 1.0 / state["total_depth"]


def main(tree):
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.set_axis_off()
    state = {
        "total_leaves": _leaf_count(tree),
        "total_depth": _depth(tree),
        "x_offset": -0.5 / _leaf_count(tree),
        "y_offset": 1.0,
    }
    _plot_tree(axis, tree, (0.5, 1.0), "", state)
    figure.tight_layout()
    plt.show()
