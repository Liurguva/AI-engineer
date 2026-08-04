# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements FP-growth frequent-itemset mining.
"""

from collections import Counter

transactions = [
    {"bread", "milk"},
    {"bread", "diapers", "beer", "eggs"},
    {"milk", "diapers", "beer", "cola"},
    {"bread", "milk", "diapers", "beer"},
    {"bread", "milk", "diapers", "cola"},
]
minimum_support_count = 2


class Node:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None

    def increment(self, count):
        self.count += count


def append_to_header(node, first_node):
    current = first_node
    while current.link is not None:
        current = current.link
    current.link = node


def update_tree(ordered_items, tree, header, count):
    first_item = ordered_items[0]
    if first_item in tree.children:
        tree.children[first_item].increment(count)
    else:
        tree.children[first_item] = Node(first_item, count, tree)
        if header[first_item][1] is None:
            header[first_item][1] = tree.children[first_item]
        else:
            append_to_header(tree.children[first_item], header[first_item][1])
    if len(ordered_items) > 1:
        update_tree(ordered_items[1:], tree.children[first_item], header, count)


def build_tree(weighted_transactions, minimum_support):
    item_counts = Counter()
    for transaction, count in weighted_transactions:
        for item in transaction:
            item_counts[item] += count
    frequent_items = {item for item, count in item_counts.items() if count >= minimum_support}
    if not frequent_items:
        return None, None

    header = {item: [item_counts[item], None] for item in frequent_items}
    root = Node(None, 0, None)
    for transaction, count in weighted_transactions:
        filtered = [item for item in transaction if item in frequent_items]
        ordered = sorted(filtered, key=lambda item: (-item_counts[item], item))
        if ordered:
            update_tree(ordered, root, header, count)
    return root, header


def prefix_path(node):
    path = []
    parent = node.parent
    while parent is not None and parent.item is not None:
        path.append(parent.item)
        parent = parent.parent
    path.reverse()
    return path


def conditional_pattern_base(item, header):
    patterns = Counter()
    node = header[item][1]
    while node is not None:
        path = prefix_path(node)
        if path:
            patterns[tuple(path)] += node.count
        node = node.link
    return [(set(path), count) for path, count in patterns.items()]


def mine_tree(header, minimum_support, prefix, results):
    ordered_items = sorted(header, key=lambda item: (header[item][0], item))
    for item in ordered_items:
        new_itemset = prefix | {item}
        results[frozenset(new_itemset)] = header[item][0]
        conditional_transactions = conditional_pattern_base(item, header)
        _, conditional_header = build_tree(conditional_transactions, minimum_support)
        if conditional_header is not None:
            mine_tree(conditional_header, minimum_support, new_itemset, results)


weighted_transactions = [(transaction, 1) for transaction in transactions]
_, header = build_tree(weighted_transactions, minimum_support_count)
frequent_itemsets = {}
if header is not None:
    mine_tree(header, minimum_support_count, set(), frequent_itemsets)

for itemset, count in sorted(frequent_itemsets.items(), key=lambda value: (len(value[0]), sorted(value[0]))):
    print(sorted(itemset), "support count =", count)
