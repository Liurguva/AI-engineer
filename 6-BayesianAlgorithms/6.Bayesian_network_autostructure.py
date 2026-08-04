# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example learns a discrete Bayesian-network structure from data.
"""


import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch

rng = np.random.default_rng(14)
n_samples = 3000
rain = rng.binomial(1, 0.25, n_samples)
sprinkler = rng.binomial(1, np.where(rain == 1, 0.10, 0.45))
wet_grass = rng.binomial(1, np.where((rain == 1) | (sprinkler == 1), 0.90, 0.05))
slippery = rng.binomial(1, np.where(wet_grass == 1, 0.80, 0.05))
data = pd.DataFrame({"Rain": rain, "Sprinkler": sprinkler, "WetGrass": wet_grass, "Slippery": slippery})

estimated_dag = HillClimbSearch(data).estimate(scoring_method="bic-d", show_progress=False)
print("Estimated edges:", sorted(estimated_dag.edges()))
