# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at www.AI-engineer.org.
This script file implements a Bayesian network with automatically learned coefficients.
"""

from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
from matplotlib import pyplot as plt
from pgmpy.estimators.MLE import MaximumLikelihoodEstimator

import numpy as np
import pandas as pd

# Generate random binary data (1000 samples, 5 variables)
raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
data = pd.DataFrame(raw_data, columns=["D", "I", "G", "L", "S"])
data.head()  # Display the first few rows of the data

# Define the structure of the Bayesian network using edges
model = BayesianNetwork([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

# ---------------------- Generate CPDs from data -------------------------------
# Learn the CPDs using maximum likelihood estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)
# ------------------------------------------------------------------------------

# Display all learned CPDs
for cpd in model.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)

# Check if the structure and CPDs are valid (all probabilities sum to 1)
model.check_model()

# Visualize the Bayesian network and overlay CPDs
nx.draw(model,
        with_labels=True,
        node_size=1000,
        font_weight='bold',
        node_color='y',
        pos={"L": [4, 3], "G": [4, 5], "S": [8, 5], "D": [2, 7], "I": [6, 7]})
plt.text(2, 7, model.get_cpds("D"), fontsize=10, color='b')
plt.text(5, 6, model.get_cpds("I"), fontsize=10, color='b')
plt.text(1, 4, model.get_cpds("G"), fontsize=10, color='b')
plt.text(4.2, 2, model.get_cpds("L"), fontsize=10, color='b')
plt.text(7, 3.4, model.get_cpds("S"), fontsize=10, color='b')
plt.title('test')
plt.show()

# Get the CPD (Conditional Probability Distribution) for node G
print(model.get_cpds('G'))
# Get the cardinality (number of states) of node G
print(model.get_cardinality('G'))
# Get the local independencies in the Bayesian network
print(model.local_independencies(['D', 'I', 'S', 'G', 'L']))
