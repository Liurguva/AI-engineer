# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example estimates Bayesian-network CPDs for a fixed graph structure.
"""


import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork

rng = np.random.default_rng(13)
n_samples = 2000
rain = rng.binomial(1, 0.25, n_samples)
sprinkler_probability = np.where(rain == 1, 0.10, 0.45)
sprinkler = rng.binomial(1, sprinkler_probability)
wet_probability = np.where((rain == 1) | (sprinkler == 1), 0.90, 0.05)
wet_grass = rng.binomial(1, wet_probability)
data = pd.DataFrame({"Rain": rain, "Sprinkler": sprinkler, "WetGrass": wet_grass})

model = DiscreteBayesianNetwork([("Rain", "WetGrass"), ("Sprinkler", "WetGrass")])
model = model.fit(data, estimator=MaximumLikelihoodEstimator)
for cpd in model.get_cpds():
    print(cpd)
