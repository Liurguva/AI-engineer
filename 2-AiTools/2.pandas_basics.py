# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example introduces basic Pandas data operations.
"""


import pandas as pd

records = pd.DataFrame(
    {
        "material": ["steel", "concrete", "steel", "timber"],
        "strength_mpa": [350.0, 40.0, 420.0, 30.0],
        "cost_index": [1.4, 1.0, 1.6, 0.9],
    }
)

strong_materials = records.loc[records["strength_mpa"] >= 100.0]
summary = records.groupby("material", as_index=False).agg(
    mean_strength_mpa=("strength_mpa", "mean"),
    mean_cost_index=("cost_index", "mean"),
)

print("Strong materials:\n", strong_materials)
print("\nGrouped summary:\n", summary)
