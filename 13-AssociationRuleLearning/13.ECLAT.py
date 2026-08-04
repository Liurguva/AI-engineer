# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements ECLAT frequent-itemset mining with vertical transaction sets.
"""


transactions = [
    {"bread", "milk"},
    {"bread", "diapers", "beer", "eggs"},
    {"milk", "diapers", "beer", "cola"},
    {"bread", "milk", "diapers", "beer"},
    {"bread", "milk", "diapers", "cola"},
]
minimum_support_count = 2
vertical = {}
for transaction_id, transaction in enumerate(transactions):
    for item in transaction:
        vertical.setdefault(item, set()).add(transaction_id)


def eclat(prefix, candidates):
    results = []
    for index, (item, tidset) in enumerate(candidates):
        new_prefix = prefix + (item,)
        results.append((new_prefix, len(tidset)))
        suffix = []
        for other_item, other_tidset in candidates[index + 1 :]:
            intersection = tidset & other_tidset
            if len(intersection) >= minimum_support_count:
                suffix.append((other_item, intersection))
        results.extend(eclat(new_prefix, suffix))
    return results

initial = sorted((item, tidset) for item, tidset in vertical.items() if len(tidset) >= minimum_support_count)
for itemset, count in eclat(tuple(), initial):
    print(list(itemset), "support count =", count)
