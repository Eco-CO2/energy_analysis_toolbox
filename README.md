[![Tests](https://github.com/Eco-CO2/energy_analysis_toolbox/actions/workflows/test.yml/badge.svg)](https://github.com/Eco-CO2/energy_analysis_toolbox/actions/workflows/test.yml)
[![Coverage](https://codecov.io/github/pandas-dev/pandas/coverage.svg?branch=main)](https://codecov.io/gh/pandas-dev/pandas)
[![PyPI Latest Release](https://img.shields.io/pypi/v/energy_analysis_toolbox.svg)](https://pypi.org/project/energy-analysis-toolbox/)
[![License - MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Eco-CO2/energy_analysis_toolbox/blob/main/LICENSE)

# Energy Analysis Toolbox

**energy_analysis_toolbox** is a Python library designed to facilitate the analysis and modeling of energy data. It provides a wide range of tools for processing time series, generating synthetic datasets, analyzing weather and thermosensitivity, and assessing power consumption. The toolbox aims to make energy analytics straightforward and reproducible for researchers, engineers, and data scientists working with energy data.

<div align="center">
  <a href="https://github.com/Eco-CO2/energy_analysis_toolbox">
    <img height="150px"
         src="doc/_static/logo.png"
         align="center">
  </a>
</div>
<br>

## Features

- **Time Series Resampling and Feature Extraction**: Includes utilities for manipulating and resampling energy-related time series data, making it easy to work with data from different sources.
- **Weather and Thermosensitivity Analysis**: Tools for calculating degree days and assessing thermosensitivity of energy consumption.
- **Power Consumption Analysis**: Detect power overconsumption, analyze load profiles, and evaluate overconsumption risks.
- **Synthetic Data Generation**: Simulate synthetic energy datasets for experimentation and modeling purposes.
- **Integration with Pandas**: Many utilities integrate seamlessly with `pandas` DataFrames, allowing for easy data manipulation.

## Installation

You can install the Energy Analysis Toolbox from the repository or via `pip`. Ensure that Python 3.6 or above is installed.

```sh
pip install energy-analysis-toolbox
```

Alternatively, you can clone this repository and install the dependencies directly:

```sh
git clone https://github.com/username/energy_analysis_toolbox.git
cd energy_analysis_toolbox
pip install -r requirements.txt
```

## Usage

The toolbox is structured into several modules, each handling a different aspect of energy data processing. Below is a quick example to get started.

### Example: Resampling Time Series

```python
import pandas as pd
from energy_analysis_toolbox.timeseries import resample_conservative

# Load some sample energy data
data = {
    'timestamp': pd.date_range(start='2023-01-01', periods=24, freq='H'),
    'energy': [5.2, 5.5, 5.0, 4.8, 6.1, 6.5, 6.0, 6.8, 5.9, 6.2, 7.1, 7.3,
               8.0, 7.5, 7.8, 7.9, 8.2, 8.3, 8.1, 7.6, 6.9, 6.5, 6.0, 5.8]
}

df = pd.DataFrame(data)
resampled_df = resample_conservative(df, freq='2H')
print(resampled_df)
```

### Example: Degree Days Calculation

```python
from energy_analysis_toolbox.weather import calculate_degree_days

degree_days = calculate_degree_days(temperatures=[15, 18, 12, 20], base_temperature=18)
print(f"Degree Days: {degree_days}")
```

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

Contributions are welcome! Whether it is a bug fix, a new feature, or improving documentation, we appreciate your help. To contribute:

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

## Acknowledgements

This toolbox is developed and maintained by a dedicated group of energy data scientists and engineers. We would like to thank everyone who contributed, directly or indirectly, to make this project possible.

---

With these tools at your disposal, you should be able to tackle a variety of energy analysis challenges efficiently and accurately. If you have questions or suggestions, we are always eager to collaborate and improve the toolbox together.

