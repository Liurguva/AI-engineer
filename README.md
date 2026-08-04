# AI-engineer

This repository contains concise Python examples for the book **Artificial Intelligence for Engineers** by **Zhen (Leo) Liu**. The examples emphasize the mathematical steps behind commonly used artificial-intelligence and machine-learning methods, followed by compact implementations with established libraries where useful.

## Design principles

- Keep the connection between equations and code visible.
- Prefer small, reproducible examples over production-scale abstractions.
- Implement core algorithms from scratch when that improves understanding.
- Use deterministic random seeds and local, script-relative data paths.
- Separate optional deep-learning, Bayesian-network, and reinforcement-learning dependencies from the core installation.

## Repository contents

| Chapter | Folder | Main topics |
|---|---|---|
| 1 | `1-AiBasics` | Physics-based, numerical, and data-driven modeling |
| 2 | `2-AiTools` | NumPy, Matplotlib, Pandas, TensorFlow/Keras, and PyTorch |
| 3 | `3-LinearModels` | Linear regression, logistic regression, kernels, and Lasso |
| 4 | `4-DecisionTrees` | ID3, C4.5, CART, plotting, and scikit-learn |
| 5 | `5-SVMs` | Primal SVM, dual SVM, and scikit-learn |
| 6 | `6-BayesianAlgorithms` | Naive Bayes, Bayesian networks, and Gaussian processes |
| 7 | `7-ANN` | Neurons, multilayer perceptrons, and backpropagation |
| 8 | `8-DeepLearning` | CNNs, RNNs, LSTMs, TensorFlow/Keras, and PyTorch |
| 9 | `9-EnsembleLearning` | Bagging, voting, AdaBoost, gradient boosting, stacking, and XGBoost |
| 10 | `10-Clustering` | K-nearest neighbors, K-means, DBSCAN, and Gaussian mixtures |
| 11 | `11-DimensionReduction` | Principal component analysis and linear discriminant analysis |
| 12 | `12-AnomalyDetection` | Statistical rules, boxplots, isolation forests, LOF, and autoencoders |
| 13 | `13-AssociationRuleLearning` | Apriori, ECLAT, and FP-growth |
| 14 | `14-Value-based-RL` | Q-learning, SARSA, Monte Carlo control, and Gymnasium examples |
| 15 | `15-Policy-based-RL` | REINFORCE and policy-gradient methods |
| 16 | `16-Appendices` | Evaluation metrics |

Each chapter folder contains its own `README.md` with the purpose and dependencies of every retained script.

## Installation

Create a virtual environment and install the core packages:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the examples that need TensorFlow, pgmpy, or Gymnasium:

```bash
python -m pip install -r requirements-optional.txt
```

Install PyTorch and torchvision using the platform-specific command provided by the official PyTorch installer.

## Running an example

Run scripts from the repository root so their relative imports are predictable:

```bash
python 3-LinearModels/3.linearmodel_solution.py
python 4-DecisionTrees/4.tree_leo_3.0.py
python 14-Value-based-RL/14.q_learning.py
```

Most examples display a plot and print one or more diagnostic values. Dataset downloads used by TensorFlow, Keras, PyTorch, or Gymnasium may require internet access on the first run.

## Validation

The repository includes syntax checks and smoke tests for examples that use the core dependency set:

```bash
python tests/smoke_test.py
```

Optional-dependency scripts are syntax-checked but are intentionally not executed by the default smoke test because several download datasets or run longer training jobs.

## Source and scope

The repository was reconstructed from earlier teaching files. Backup versions, cached bytecode, generated model files, scripts with missing private datasets, obsolete TensorFlow 1.x code, and duplicate experimental variants were removed or replaced. See `CLEANING_REPORT.md` for details.

## License

No open-source license has been selected in this prepared version. Before making the repository public, the owner should add the license that matches the intended permissions for reuse, modification, and redistribution.
