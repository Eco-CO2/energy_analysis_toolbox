#!/bin/bash

# Function to clean project files
clean_project() {
    echo "Cleaning project files..."
    rm -f .coverage *.xml coverage.xml fret21_indicators.xlsx .nfs*
    rm -rf **.egg-info .pytest_cache build **/*.spec .lprof doc/_build **/*.log ./doc/auto_examples .ruff_cache .venv .mypy_cache
    find . -type d -name "__pycache__" -exec rm -rf {} +
    echo "Clean complete."
}

# Function to generate documentation
generate_docs() {
    echo "Generating documentation..."
    pip install .[doc]
    cd doc || exit
    sphinx-build -b html . _build/html
    cd ../
    echo "Documentation generation complete. Open _build/html/index.html to view."
}

# Function to run tests
run_tests() {
    echo "Running tests..."
    pip install --upgrade --force-reinstall .
    python3 -c "from energy_analysis_toolbox.tests.main import run; run()"
    echo "Tests complete."
}

# Main script execution

echo "Select the action you want to perform:"
echo "1) Clean project"
echo "2) Generate documentation"
echo "3) Run tests"
read -r action

case $action in
    1)
        clean_project
        ;;
    2)
        generate_docs
        ;;
    3)
        run_tests
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac
