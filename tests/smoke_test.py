# -*- coding: utf-8 -*-
"""Syntax-check all examples and execute the core-dependency examples."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ast
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_RUNTIME_MARKERS = (
    "Lab2-Tensorflow.py",
    "Lab2-Keras.py",
    "2.pytorch_basics.py",
    "Bayesian_network_",
    "Lab7-CNN-MNIST.py",
    "Lab7-LeNet-MNIST.py",
    "Lab7-AlexNet-MNIST.py",
    "8.pytorch_cnn_mnist.py",
    "12.autoencoder_DNN.py",
    "12.autoencoder_CNN.py",
    "gymnasium.py",
    "15.REINFORCE_cartpole.py",
    "9.xgboost.py",
)

python_files = sorted(
    path
    for path in ROOT.glob("[0-9]*-*/*.py")
)
syntax_failures = []
for path in python_files:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        syntax_failures.append(f"{path.relative_to(ROOT)}: {error}")

if syntax_failures:
    print("Syntax failures:")
    print("\n".join(syntax_failures))
    raise SystemExit(1)

runtime_files = [
    path
    for path in python_files
    if not any(marker in path.name for marker in OPTIONAL_RUNTIME_MARKERS)
    and path.name != "tree_plot_function.py"
]


def run_example(path):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONWARNINGS"] = "ignore"
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return path, 124, "Timed out after 90 seconds."
    details = (result.stdout + "\n" + result.stderr).strip()
    return path, result.returncode, details

with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
    results = list(executor.map(run_example, runtime_files))

failures = [(path, code, details) for path, code, details in results if code != 0]
print(f"Syntax checked: {len(python_files)} files")
print(f"Runtime checked: {len(runtime_files)} core examples")
print(f"Optional runtime examples skipped: {len(python_files) - len(runtime_files)}")

if failures:
    print("\nRuntime failures:")
    for path, code, details in failures:
        print(f"--- {path.relative_to(ROOT)} (exit {code})")
        print(details[-2000:])
    raise SystemExit(1)

print("All core smoke tests passed.")
