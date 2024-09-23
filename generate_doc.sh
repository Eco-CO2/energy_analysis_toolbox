# Run this script to build the doc locally
# Open _build/html/index.html to see the homepage
pip install .[doc]
cd doc
sphinx-build -b html . _build/html
cd ../
