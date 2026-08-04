# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example implements categorical Naive Bayes from scratch.
"""


from collections import Counter, defaultdict
import math

# Outlook, Temperature, Humidity, Wind, Play
training_data = [
    ("Sunny", "Hot", "High", "Weak", "No"),
    ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"),
    ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Strong", "No"),
]
query = ("Sunny", "Cool", "High", "Strong")
classes = sorted({row[-1] for row in training_data})
class_counts = Counter(row[-1] for row in training_data)
feature_values = [sorted({row[j] for row in training_data}) for j in range(len(query))]

log_scores = defaultdict(float)
for class_name in classes:
    log_score = math.log(class_counts[class_name] / len(training_data))
    class_rows = [row for row in training_data if row[-1] == class_name]
    for j, value in enumerate(query):
        count = sum(row[j] == value for row in class_rows)
        probability = (count + 1) / (len(class_rows) + len(feature_values[j]))
        log_score += math.log(probability)
    log_scores[class_name] = log_score

prediction = max(log_scores, key=log_scores.get)
print("Query:", query)
print("Log posterior scores:", dict(log_scores))
print("Predicted class:", prediction)
