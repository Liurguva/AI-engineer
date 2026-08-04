# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements Apriori frequent-itemset mining.
"""


from itertools import combinations

transactions = [
    {"bread", "milk"},
    {"bread", "diapers", "beer", "eggs"},
    {"milk", "diapers", "beer", "cola"},
    {"bread", "milk", "diapers", "beer"},
    {"bread", "milk", "diapers", "cola"},
]
minimum_support_count = 2
items = sorted(set().union(*transactions))
frequent_itemsets = {}
current = {frozenset([item]) for item in items}
k = 1

while current:
    counts = {candidate: sum(candidate.issubset(transaction) for transaction in transactions) for candidate in current}
    frequent = {candidate for candidate, count in counts.items() if count >= minimum_support_count}
    for candidate in frequent:
        frequent_itemsets[candidate] = counts[candidate]
    k += 1
    joined = set()
    frequent_list = sorted(frequent, key=lambda itemset: tuple(sorted(itemset)))
    for left, right in combinations(frequent_list, 2):
        candidate = left | right
        if len(candidate) == k and all(frozenset(subset) in frequent for subset in combinations(candidate, k - 1)):
            joined.add(candidate)
    current = joined

for itemset, count in sorted(frequent_itemsets.items(), key=lambda item: (len(item[0]), sorted(item[0]))):
    print(sorted(itemset), "support count =", count)
