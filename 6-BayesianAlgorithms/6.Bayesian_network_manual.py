# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbols, and procedures are explained in the book. Major info can be found at www.AI-engineer.org.
This script implements a Bayesian Network manually.
"""

from pgmpy.models import BayesianNetwork  # BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# Define the structure of the Bayesian Network using directed edges
# BayesianModel has been renamed to BayesianNetwork. Please use BayesianNetwork, as BayesianModel will be removed in the future.
model = BayesianNetwork([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

# ---------------------- Manually enter probabilities -------------------------
# Define CPDs (Conditional Probability Distributions)
cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6], [0.4]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.7], [0.3]])

# variable: node
# variable_card: number of possible values (cardinality)
# values: probability values
# evidence: parent nodes (conditioning variables)
cpd_g = TabularCPD(variable='G', variable_card=3,
                   values=[[0.3, 0.05, 0.9,  0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7,  0.02, 0.2]],
                   evidence=['I', 'D'],
                   evidence_card=[2, 2])

cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.1, 0.4, 0.99],
                           [0.9, 0.6, 0.01]],
                   evidence=['G'],
                   evidence_card=[3])

cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence=['I'],
                   evidence_card=[2])

# Add CPDs to the model
model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
# ------------------------------------------------------------------------------

# Display all CPDs
for cpd in model.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)

# Check if the network structure and CPDs are valid (i.e., all probabilities sum to 1)
model.check_model()

# Visualize the Bayesian network and display CPDs on the plot
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

# Retrieve the CPD for node G
print(model.get_cpds('G'))
# Retrieve the cardinality (number of states) of node G
print(model.get_cardinality('G'))
# Get the local independencies in the Bayesian network
print(model.local_independencies(['D', 'I', 'S', 'G', 'L']))
# Perform Bayesian inference using variable elimination
infer = VariableElimination(model)
print(infer.query(['G']), ['G'])
