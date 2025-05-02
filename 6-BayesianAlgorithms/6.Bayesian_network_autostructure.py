# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at www.AI-engineer.org.
This script file is for implementing a Bayesian network with automatic structure learning.
"""

import pandas as pd
import numpy as np
from pgmpy.estimators.StructureScore import BDeuScore, K2Score, BicScore
from pgmpy.models import BayesianNetwork  # BayesianModel

# Generate synthetic data using random numbers:
# There are 3 variables, where Z depends on X and Y
data = pd.DataFrame(np.random.randint(0, 4, size=(5000, 2)), columns=list('XY'))
data['Z'] = data['X'] + data['Y']

# Scoring methods for structure learning
bdeu = BDeuScore(data, equivalent_sample_size=5)
k2 = K2Score(data)
bic = BicScore(data)

# Define two different model structures
model1 = BayesianNetwork([('X', 'Z'), ('Y', 'Z')])  # X -> Z <- Y
model2 = BayesianNetwork([('X', 'Z'), ('X', 'Y')])  # Y <- X -> Z

print("----- Scores for model1 -----")
print(bdeu.score(model1))
print(k2.score(model1))
print(bic.score(model1))

print("----- Scores for model2 -----")
print(bdeu.score(model2))
print(k2.score(model2))
print(bic.score(model2))

# Local scoring: compute score of Z given different parent sets
print("----- Local Scores -----")
print(bdeu.local_score('Z', parents=[]))
print(bdeu.local_score('Z', parents=['X']))
print(bdeu.local_score('Z', parents=['X', 'Y']))

# Automatic Structure Search: 1. Exhaustive Search, 2. Hill Climbing (a greedy method)

# Method 1: Exhaustive Search
from pgmpy.estimators import ExhaustiveSearch

es = ExhaustiveSearch(data, scoring_method=bic)
best_model = es.estimate()
print(best_model.edges())

print("\nAll DAGs ranked by score:")
for score, dag in reversed(es.all_scores()):
    print(score, dag.edges())

# Method 2: Hill Climbing Search
from pgmpy.estimators import HillClimbSearch

# You can uncomment and use custom synthetic data if needed
# data = pd.DataFrame(np.random.randint(0, 3, size=(2500, 8)), columns=list('ABCDEFGH'))
# data['A'] += data['B'] + data['C']
# data['H'] = data['G'] - data['A']

hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BicScore(data))
print(best_model.edges())

print("\nAll DAGs ranked by score:")
for score, dag in reversed(hc.all_scores()):
    print(score, dag.edges())
