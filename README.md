[![tests](https://github.com/Eco-CO2/energy_analysis_toolbox/actions/workflows/test.yml/badge.svg)](https://github.com/Eco-CO2/energy_analysis_toolbox/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Eco-CO2/energy_analysis_toolbox/graph/badge.svg?token=XWWPB8E4SD)](https://codecov.io/gh/Eco-CO2/energy_analysis_toolbox)
[![PyPI latest release](https://img.shields.io/pypi/v/energy_analysis_toolbox.svg)](https://pypi.org/project/energy-analysis-toolbox/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/energy_analysis_toolbox)
![Ruff Lint](https://img.shields.io/badge/linting-Ruff-blue)
![Pandas Versions](https://img.shields.io/badge/pandas-2.1.2%20|%202.1.3%20|%202.1.4%20|%202.2-blue)
[![License - MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Eco-CO2/energy_analysis_toolbox/blob/main/LICENSE)

<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="doc/_static/energy_analysis_toolbox_logo_horizontal_white.svg">
  <img alt="Pandas Logo" src="doc/_static/energy_analysis_toolbox_logo_horizontal.svg">
&nbsp;
</picture>

``energy_analysis_toolbox`` is a Python library designed to analyze and model power and energy time series. It provides a wide range of tools for processing time series, generating synthetic datasets, analyzing weather and thermosensitivity.

## Features

- **Time Series Resampling and Feature Extraction**: Includes utilities for manipulating and resampling energy-related time series data, making it easy to work with data from different sources.
- **Weather and Thermosensitivity Analysis**: Tools for calculating degree days and assessing thermosensitivity.
- **Power Consumption Analysis**: Detection of unusual power consumption patterns, and analyze load profiles.
- **Synthetic Data**: Generation of synthetic time series energy datasets.
- **Integration with Pandas**: Many utilities integrate seamlessly with `pandas` DataFrames, allowing to extend `pandas` with energy and power-specific tools through an accessor.

## Installation

``energy_analysis_toolbox`` can be installed from the repository or via `pip`. Ensure that you use Python 3.10 or above.

```sh
pip install energy-analysis-toolbox
```

Alternatively, you can clone this repository and install the dependencies directly:

```sh
git clone https://github.com/Eco-CO2/energy_analysis_toolbox
cd energy_analysis_toolbox
pip install .
```

## Usage

The toolbox is structured into several modules, each handling a different aspect of energy data processing. Below is a quick example to get started.

### Example: Resampling a time series and computing energy

```python
import numpy as np
import pandas as pd
import energy_analysis_toolbox.pandas
import matplotlib.pyplot as plt

power = pd.Series(
    data=5*np.sin(np.linspace(0, 6, 100)) + np.random.randn(100) + 7,
    index=pd.date_range(start='2023-01-01', periods=100, freq='d'),
)
energy_resampled = power.eat.to_energy().eat.to_freq("1W")/3600000

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
power.plot(ax=axes[0], ylabel="Power (W)", title="Power")
energy_resampled.plot(ax=axes[1], ylabel="Energy (kWh)", title="Weekly resampled energy")
```

![demo_energy_resampling](doc/_static/demo_energy_resampling.png)

## Documentation

The complete documentation, including detailed guides for each module, API reference, and tutorials, can be found [here](https://energy_analysis_toolbox.readthedocs.io).

To generate the documentation locally, you can run:

```sh
./generate_doc.sh
```

## Tests

The Energy Analysis Toolbox comes with a comprehensive test suite that ensures code reliability and robustness. The tests cover a variety of scenarios including time series resampling, power overconsumption analysis, and synthetic data generation.

To run the tests, execute the following command:

```sh
./run_tests.sh
```

We use `pytest` for unit testing, which helps ensure that our code is reliable and that any modifications do not break existing functionality.

## Contributing

This toolbox was originally developed by the R&D team of Eco CO2. It is now open to anyone who wants to contribute: whether it is a bug fix, a new feature, or improving documentation, we appreciate your help! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to your branch (`git push origin feature-branch`)
5. Create a pull request

Please make sure to add appropriate unit tests for any new code and verify all tests pass.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, feature requests, or to report issues, please open an issue on the [GitHub issue tracker](https://github.com/username/energy_analysis_toolbox/issues).
