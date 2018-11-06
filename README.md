# Geepers
[![PyPI version](https://badge.fury.io/py/geepers-pkg.svg)](https://badge.fury.io/py/geepers-pkg)

This is a small toolkit that allows the user to create classifiers via genetic programming, which can then be used
by a hyper-heuristic (in this case a genetic algorithm) in order to improve results.

The current implementations are geared towards solving the Network Intrusion Detection problem, but users can easily
adapt this to any other problem by inheriting from the respective classes in `geepers/gp.py` and
`geepers/hyper_heuristics.py` and overwriting the relevant methods and attributes.
