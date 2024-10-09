#!/bin/bash
rm -f .coverage *.xml coverage.xml fret21_indicators.xlsx .nfs*
rm -rf **.egg-info .pytest_cache build **/*.spec .lprof doc/_build **/*.log ./doc/auto_examples .ruff_cache .venv
find . -type d -name "__pycache__" -exec rm -rf {} +
