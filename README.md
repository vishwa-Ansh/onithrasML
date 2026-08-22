onithrasML

onithrasML is an open-source machine learning library for Python, designed to provide simple and efficient implementations of common machine learning utilities and algorithms.

The library combines a Python interface with a C++ backend for performance-oriented computation.

Features

* Data imputation utilities
* Feature scaling utilities
* Linear Regression
* Logistic Regression
* Decision Tree
* Model selection utilities
* C++ backend for computationally intensive operations
* Simple Python API
* Easy integration into machine learning projects

Installation

Install the latest version from PyPI:

pip install onithrasml

Quick Start

import onithrasML

You can then use the modules provided by the library according to your machine learning workflow.

Project Structure

onithrasML/
├── src/
│   └── onithrasML/
│       ├── __init__.py
│       ├── _backend/
│       ├── imputer/
│       ├── linear_model/
│       └── model_selection/
│
├── cpp/
│   ├── bindings.cpp
│   ├── decision_tree.cpp
│   ├── linear_regression.cpp
│   └── logistic_regression.cpp
│
├── tests/
├── benchmarks/
├── README.md
├── LICENSE
└── pyproject.toml

Machine Learning Models

Linear Regression

from onithrasML.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

Logistic Regression

from onithrasML.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

Imputation

The imputer module provides utilities for handling missing values in datasets.

from onithrasML.imputer import ...

Model Selection

The model_selection module contains utilities useful for selecting and evaluating machine learning models.

from onithrasML.model_selection import ...

C++ Backend

Performance-critical components are implemented in C++ and exposed to Python through bindings.

The C++ backend currently includes implementations related to:

* Linear Regression
* Logistic Regression
* Decision Tree

This allows computationally intensive operations to be executed using native compiled code while maintaining a Python-friendly API.

Development

Clone the repository:

git clone https://github.com/vishwa-Ansh/onithrasML.git
cd onithrasML

Create a virtual environment:

python -m venv .venv
source .venv/bin/activate

Install the development dependencies:

pip install -r requirements.txt

Build the package:

python -m build

Check the distributions:

python -m twine check dist/*

Testing

Run the test suite with:

pytest

Benchmarks

Benchmark programs are available in the benchmarks/ directory and can be used to evaluate the performance of the implemented algorithms.

License

This project is open source. See the LICENSE file for details.

Contributing

Contributions, bug reports, feature requests, and improvements are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Add or update tests.
5. Commit your changes.
6. Open a pull request.

Author

Ansh Vishwakarma

GitHub: https://github.com/vishwa-Ansh

Project

onithrasML — Machine Learning library for Python with a C++ backend.