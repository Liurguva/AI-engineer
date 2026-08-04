# Cleaning report

## What was retained

The repository keeps one clear instructional implementation for each topic described in the root README. Where useful, a second implementation using a standard library is included for comparison.

## What was corrected

- Replaced obsolete TensorFlow 1.x sessions and compatibility imports with TensorFlow 2 eager/Keras APIs.
- Replaced the removed `load_boston` dataset with maintained scikit-learn datasets.
- Corrected linear-regression coefficient and intercept extraction.
- Corrected the primal SVM objective and plotted decision boundary.
- Corrected dual-SVM sign conventions and support-vector recovery.
- Removed the target column from the decision-tree feature-name file.
- Prevented recursive decision-tree branches from sharing and mutating the same feature list.
- Corrected CART feature selection so lower weighted Gini impurity is preferred.
- Replaced chained Pandas assignment in reinforcement-learning examples.
- Replaced legacy `gym` calls with the current Gymnasium reset/step interface.
- Replaced hard-coded local file paths and missing private datasets with generated or built-in datasets.
- Removed Chinese annotations, author-machine timestamps, and unexplained copied snippets.

## What was removed

- `__pycache__` and compiled `.pyc` files.
- Backup files and numbered experimental variants when a single cleaned version was sufficient.
- The generated `autoencoder.h5` model artifact.
- Scripts that depended on unavailable files such as `engineInputs.csv`, `engineTargets.csv`, or private image folders.
- Broken drafts with syntax or indentation errors.
- Duplicate Decision Tree and LeNet drafts outside their chapter folders.

## Validation approach

- All Python files are parsed and byte-compiled.
- Core examples are executed with a non-interactive Matplotlib backend.
- Optional examples use maintained public APIs and are syntax-checked, but the default smoke test does not execute examples that require TensorFlow, pgmpy, Gymnasium, torchvision dataset downloads, or XGBoost.

## Final test result

- 72 Python files passed parsing and byte-compilation.
- 55 core-dependency examples passed runtime smoke tests.
- 17 optional or download-dependent examples passed syntax review and are excluded from the default runtime test.
