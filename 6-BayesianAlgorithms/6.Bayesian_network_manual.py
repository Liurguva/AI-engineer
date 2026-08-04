# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example creates a discrete Bayesian network manually and performs inference.
"""


from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

model = DiscreteBayesianNetwork([("Rain", "WetGrass"), ("Sprinkler", "WetGrass")])
cpd_rain = TabularCPD("Rain", 2, [[0.8], [0.2]], state_names={"Rain": ["No", "Yes"]})
cpd_sprinkler = TabularCPD("Sprinkler", 2, [[0.6], [0.4]], state_names={"Sprinkler": ["Off", "On"]})
cpd_wet_grass = TabularCPD(
    "WetGrass",
    2,
    [[0.99, 0.20, 0.10, 0.01], [0.01, 0.80, 0.90, 0.99]],
    evidence=["Rain", "Sprinkler"],
    evidence_card=[2, 2],
    state_names={
        "WetGrass": ["Dry", "Wet"],
        "Rain": ["No", "Yes"],
        "Sprinkler": ["Off", "On"],
    },
)
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet_grass)
if not model.check_model():
    raise RuntimeError("The Bayesian network is inconsistent.")

inference = VariableElimination(model)
posterior = inference.query(variables=["Rain"], evidence={"WetGrass": "Wet"})
print(posterior)
