r"""Process the thermosensitivity data.

# Available classes

## ThermoSensitivity

Class to compute the thermosensitivity of a building.
Needs a time series of energy consumption and outdoor temperature.

## CategoricalThermoSensitivity

Class to compute the thermosensitivity of a building with labeled periods.
Needs a time series of energy consumption, outdoor temperature, and labeled periods.

The labeled periods are resampled to the same frequency as the energy and temperature
data by taking the most common category in the period.

Currently, the class only calibrates one base temperature for all the categories
aggregated.

# Implementation details

## Resampling frequency

The energy and temperature data are resampled at a given frequency.
The degree days are computed at the same frequency.

## Thermo Sensitivity

The thermo-sensitivity is modelled as a linear regression between the energy
consumption and the degree days.

    .. math::

        E ~ E0 + TS \times DegreeDays

The degree days are computed from the temperature data and the base temperature.

    .. math::

        DegreeDays = \\int max(0, BaseTemperature - T(t)) dt

Different methods are available to compute the degree days:

- Integral: sum the difference between the base temperature and the temperature.
    .. math::
        DegreeDays = \\sum_{t=0}^N max(0, BaseTemperature - T(t))

- Mean: sum the difference between the base temperature and the mean temperature.
    .. math::
        DegreeDays = max(0, BaseTemperature - \\bar{T} )

- MinMax: sum the difference between the base temperature and the mean temperature
  computed as the mean of the minimum and maximum temperature.

    .. math::
        DegreeDays = max(0, BaseTemperature - \\frac{T_{min} + T_{max}}{2} )

See the `dd_compute` function in the `energy_analysis_toolbox.weather.degree_days`
module.

Over a long period, the data can present a thermosensitivity with different types of
degree days:

- Heating: the energy consumption increases when the temperature decreases. Usually,
  the base temperature is around 18°C.
- Cooling: the energy consumption increases when the temperature increases. Usually,
  the base temperature is around 24°C.

## Auto-calibration

Two aspects of the thermosensitivity can be automatically detected:

- The degree days type: heating, cooling, or both.
- The base temperature.

### Degree days type

Each building, depending of the installed systems, can have different thermosensitivity
types :
- use of heating systems (heating degree days, during the winter)
- use of cooling systems (cooling degree days, during the summer)
- use of both systems (heating and cooling degree days)

The degree days type can be automatically detected by setting the `degree_days_type`
parameter to ``"auto"``. The method will compute the Spearman correlation between the
energy and the temperature for the periods with the mean temperature below and above
the intersaison mean temperature (default is 20°C).

### Base temperature

The base temperature can be calibrated by minimizing the mean squared error between the
data and the model.

Each degree days type has a specific base temperature that is determined by analyzing
the data over the corresponding periods. The heating (resp. cooling) base temperature
is calibrated by minimizing the mean squared error between the energy and the heating
(resp. cooling) degree days for the periods with the mean temperature below (resp.
above) the intersaison mean temperature.

The optimization is done with the `scipy.optimize.minimize_scalar` function with the
`bounded` method.

"""

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import numpy as np
import pandas as pd
import statsmodels
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from statsmodels.api import OLS

from energy_analysis_toolbox.energy.resample import to_freq as energy_to_freq
from energy_analysis_toolbox.logger import init_logging
from energy_analysis_toolbox.weather.degree_days import (
    dd_compute,
    dd_types,
    literal_computation_dd_types,
    literal_dd_types,
    literal_valid_dd_types,
)

if TYPE_CHECKING:
    from pandas.core.resample import Resampler


ThermosensitivityInstance = TypeVar(
    "ThermosensitivityInstance",
    bound="ThermoSensitivity",
)


class ThermoSensitivity:
    """Class to compute the thermosensitivity of a building.

    Examples
    --------
    >>> from energy_analysis_toolbox.thermosensitivity import ThermoSensitivity
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> temperature_data = 15 + 2*pd.Series(np.random.rand(366),
    ... index=pd.date_range("2020-01-01", "2020-12-31", freq="D"))
    >>> energy_data = 10 + (16 - temperature_data).clip(0) * 5 + np.random.rand(366)
    >>> ts = ThermoSensitivity(energy_data, temperature_data, degree_days_type="auto")
    >>> ts.fit()
    >>> ts
    ThermoSensitivity(frequency=1D,
            degree_days_type=heating,
            degree_days_base_temperature={"heating": 15.98},
            degree_days_computation_method="integral",
            interseason_mean_temperature=20)
    <BLANKLINE>
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 energy   R-squared:                       0.969
    Model:                            OLS   Adj. R-squared:                  0.969
    No. Observations:                 366   F-statistic:                 1.137e+04
    Covariance Type:            nonrobust   Prob (F-statistic):          1.25e-276
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025   \
                                  0.975]
    ---------------------------------------------------------------------------------------
    heating_degree_days     5.1177      0.048    106.638      0.000       5.023    \
        5.212
    Intercept              10.5120      0.019    539.733      0.000      10.474    \
        10.550
    =======================================================================================
    <BLANKLINE>
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly
    specified.

    As you can see from the example above:

    - the type of thermosensitivity is automatically detected (heating in this case)
    - the base temperature is calibrated to 15.98 (true value is 16)
    - the model is fitted with a R-squared of 0.969
    - the heating degree days coefficient is 5.1177 (true value is 5)
    - the intercept is 10.5120 (true value is 10)

    """

    target_name = "energy"
    temperature_name = "temperature"

    def __init__(
        self,
        energy_data: pd.Series,
        temperature_data: pd.Series,
        frequency: str = "1D",
        degree_days_type: literal_dd_types = "heating",
        degree_days_base_temperature: dict | None = None,
        degree_days_computation_method: literal_computation_dd_types = "integral",
        interseason_mean_temperature: float = 20,
        base_logger_name: str | None = None,
        min_logger_level_stdout: int | str = logging.ERROR,
    ) -> None:
        """Initialize a ``ThermoSensitivity`` instance.

        Parameters
        ----------
        energy_data : pd.Series
            Time series of energy consumption data for the building.
        temperature_data : pd.Series
            Time series of outdoor temperature data.
        frequency : str, optional
            Frequency for resampling the data (default is "1D").
            Options include:
            - "1D": daily resampling
            - "7D": weekly resampling
            - "1W-MON": weekly resampling starting on Monday.
        degree_days_type : str, optional
            Type of degree days to compute (default is "heating").
            Options are:
            - "heating": heating degree days.
            - "cooling": cooling degree days.
            - "both": both heating and cooling degree days.
            - "auto": automatically detect the degree days type.
        degree_days_base_temperature : dict, optional
            Base temperature(s) for degree day calculations (default is an empty dict).
            Should include keys "heating" and/or "cooling".
        degree_days_computation_method : str, optional
            Method to compute degree days (default is "integral").
            Options are:
            - "integral": integral calculation.
            - "mean": mean temperature calculation.
            - "min_max": min-max temperature calculation.
            - "pro": energy-professionals calculation.
        interseason_mean_temperature : float, optional
            Mean temperature to differentiate heating and cooling periods (default 20).
        base_logger_name : str, optional
            Name of the logger. By default, it is the class name. All following
            following instances receive a unique identifier, based on the first
            one, with the pattern:

            - ``<base_logger_name>``
            - ``<base_logger_name>_1``
            - ...

        min_logger_level_stdout: str, int, optional
            Minimum logger level below which no message is transferred to stdout
            (i.e. not printed). Default is ``"ERROR"``.

        Raises
        ------
        ValueError
            If the degree days type is not valid or if required parameters are missing.

        Example
        -------
        >>> ts = ThermoSensitivity(energy_data, temperature_data,
            degree_days_type="auto")

        """
        self._init_logger(
            base_logger_name=base_logger_name,
            min_logger_level_stdout=min_logger_level_stdout,
        )
        self._energy_data = energy_data
        self._temperature_data = temperature_data
        self._frequency = frequency
        self._aggregated_data: pd.DataFrame | None = None
        self.degree_days_type = degree_days_type
        self.degree_days_base_temperature = degree_days_base_temperature or {}
        self.degree_days_computation_method = degree_days_computation_method
        self.interseason_mean_temperature = interseason_mean_temperature
        self.predictors: list[str] = []
        self._model = None
        self._validate_data()
        self._post_init()

    def _init_logger(
        self,
        base_logger_name: str | None,
        min_logger_level_stdout: str | int = logging.ERROR,
    ) -> None:
        """Initialize the logger for the current instance of the class.

        This method configures a logger specific to the instance of the class. If a
        `base_logger_name` is provided, it will be used as the logger's name; otherwise,
        the class name will be used. The logger is set up to manage output levels,
        allowing only messages with a severity level above the specified minimum to be
        printed to standard output.

        Parameters
        ----------
        base_logger_name : str, optional
            The name of the logger. If not provided, the logger will default to the
            name of the class. Subsequent instances will receive unique logger names
            by appending an incrementing counter to the base name.

        min_logger_level_stdout : str or int, optional
            The minimum severity level for messages to be printed to standard output.
            Messages below this level will not be displayed. The default is
            ``"ERROR"``, meaning only errors and critical messages will
            be printed. Acceptable values can be specified as strings (e.g.,
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") or as corresponding
            integer constants from the `logging` module. If you want to have details
            on what's going on in the code, set this to ``"INFO"`` or ``"WARNING"``.

        Returns
        -------
        None
            This method does not return a value; it initializes the logger instance
            for use in logging messages throughout the class.

        Example
        -------
        >>> ts = ThermoSensitivity(energy_data, temperature_data)
        >>> ts._init_logger(base_logger_name="my_custom_logger",
        ... min_logger_level_stdout=logging.ERROR)

        Notes
        -----
        - Ensuring unique logger names prevents conflicts when multiple instances of
          the class are used simultaneously.
        - Log messages can be directed to different outputs (e.g., console, file) based
          on logger configuration.

        """
        if not base_logger_name:
            base_logger_name = self.__class__.__name__
        self.logger = init_logging(
            base_logger_name=base_logger_name,
            min_logger_level_stdout=min_logger_level_stdout,
        )

    @property
    def frequency(
        self,
    ) -> str:
        """The frequency of the resampled data.

        The property is unmutable. To change the frequency, create a new object.
        """
        return self._frequency

    @property
    def energy_data(
        self,
    ) -> pd.Series:
        """The energy data of the building.

        The property is unmutable. To change the energy data, create a new object.
        """
        return self._energy_data

    @property
    def temperature_data(
        self,
    ) -> pd.Series:
        """The outdoor temperature data.

        The property is unmutable. To change the temperature data, create a new object.
        """
        return self._temperature_data

    @cached_property
    def resampled_energy(
        self,
    ) -> pd.Series:
        """The energy data resampled at the given frequency.

        Uses the `to_freq` function from the `energy_analysis_toolbox.energy.resample`
        module to convert the energy data to the desired frequency.

        This property is cached to avoid recomputing it multiple times.
        """
        energy = self.energy_data.copy()
        last_period = energy.index[-1] - energy.index[-2]
        energy[energy.index[-1] + last_period] = 0
        return energy_to_freq(
            energy,
            self.frequency,
        ).rename(self.target_name)

    @cached_property
    def resampled_temperature(
        self,
    ) -> pd.Series:
        """The temperature data resampled at the given frequency.

        Average the temperature data over the given frequency.

        This property is cached to avoid recomputing it multiple times.
        """
        return (
            self.temperature_data.resample(self.frequency)
            .mean()
            .rename(self.temperature_name)
        )

    @cached_property
    def resampled_energy_temperature(
        self,
    ) -> pd.DataFrame:
        """The resampled energy and temperature data.

        The DataFrame contains the resampled energy and temperature data.
        Periods with missing values are removed.
        """
        return pd.concat(
            [self.resampled_energy, self.resampled_temperature],
            axis=1,
        ).dropna(how="any", axis=0)

    @property
    def model(
        self,
    ) -> statsmodels.regression.linear_model.RegressionResults:
        """The thermosensitivity model.

        A `statsmodels.regression.linear_model.RegressionResults` object.

        Raises
        ------
        ValueError
            If the model is not fitted. Use the `fit` method to train the model.

        """
        if self._model is None:
            err = "Model not fitted. Please run the `fit` method."
            raise ValueError(err)
        return self._model

    @property
    def aggregated_data(
        self,
    ) -> pd.DataFrame:
        """The aggregated data used to fit the model.

        The Data is a DataFrame resampled at the provided Frequency with the following
        columns:
        - "energy": the total energy at the frequency.
        - "temperature": the mean temperature at the frequency.
        - (Optional) "heating_degree_days": the heating degree days at the frequency.
        - (Optional) "cooling_degree_days": the cooling degree days at the frequency.

        Raises
        ------
        ValueError
            If the data is not aggregated. Use the `fit` method to aggregate the data.

        """
        if self._aggregated_data is None:
            err = "Data not aggregated. Please run the `fit` method."
            raise ValueError(err)
        return self._aggregated_data

    @aggregated_data.setter
    def aggregated_data(
        self,
        value: pd.DataFrame,
    ) -> None:
        """Set the aggregated data, and reset the model."""
        self._aggregated_data = value
        self._model = None

    def _validate_data(
        self,
    ) -> None:
        """Check the validity of the parameters.

        Raises
        ------
        ValueError
            If the degree days type is not valid.
        ValueError
            If the base temperature is not specified for the heating or cooling degree
            days. While not empty.

        """
        if self.degree_days_type not in dd_types:
            err = f"Invalid degree days type. Must be one of {dd_types}."
            raise ValueError(err)
        if self.degree_days_base_temperature != {}:
            if self.degree_days_type in ["heating", "both"]:
                try:
                    self.degree_days_base_temperature["heating"]
                except KeyError:
                    err = (
                        "Base temperature for heating degree days must be specified."
                        "\n Example: degree_days_base_temperature={'heating': 18, "
                        "'cooling': 24}"
                    )
                    raise ValueError(err) from None
            elif self.degree_days_type in ["cooling", "both"]:
                try:
                    self.degree_days_base_temperature["cooling"]
                except KeyError:
                    err = (
                        "Base temperature for cooling degree days must be specified."
                        "\n Example: degree_days_base_temperature={'heating': 18, "
                        "'cooling': 24}"
                    )
                    raise ValueError(err) from None

    def _post_init(
        self,
    ) -> None:
        """End the initialization process.

        If the degree days type is set to ``"auto"``, the method will detect the degree
        days type. See :meth:`_detect_degree_days_type`. After the detection, the
        predictors will be set.
        """
        self._detect_degree_days_type()
        if self.degree_days_type == "both":
            self.predictors = ["heating_degree_days", "cooling_degree_days"]
        elif self.degree_days_type in ["heating", "cooling"]:
            self.predictors = [f"{self.degree_days_type}_degree_days"]

    def _aggregate_data(
        self,
        degree_days_base_temperature: dict,
    ) -> None:
        """Compute the degree days and aggregate the data.

        Store the aggregated data in the `aggregated_data` property.
        """
        degree_days = self._calculate_degree_days(degree_days_base_temperature)
        self.aggregated_data = pd.concat(
            [
                self.resampled_energy_temperature,
                degree_days,
            ],
            axis=1,
        )

    def _detect_degree_days_type(
        self,
        significance_level: float = 0.05,
    ) -> None:
        """Estimate the degree days type if it is set to ``"auto"``.

        It will compute the Spearman correlation (with the alternative hypothesis)
        between the energy and the temperature. If the p-value is below the
        significance level, the degree days type will be set to ``"heating"`` or
        ``"cooling"`` or ``"both"``.

        If no significant correlation is found, the method will raise a ValueError.

        - **Heating**: The energy consumption is negatively correlated with the
          temperature for the periods with the mean temperature below the intersaison
          mean temperature.
        - **Cooling**: The energy consumption is positively correlated with the
          temperature for the periods with the mean temperature above the intersaison
          mean temperature.

        Note:
        ----
        The Spearman correlation is a non-parametric test that measures the strength
        and direction of the monotonic relationship between two variables. The relation
        is not necessarily linear, and the test does not assume that the data is
        normally distributed.

        .. warning::
            If the data contains multiple groups, the Simpson's paradox may occur. The
            Paradox states that the correlation observed in the aggregated data may not
            hold when the data is split into subgroups. See the
            `CategoricalThermoSensitivity` class for a solution.

            References: `wikipedia <https://en.wikipedia.org/wiki/Simpson%27s_paradox>`_

        """
        if self.degree_days_type == "auto":
            heating_mask = (
                self.resampled_energy_temperature[self.temperature_name]
                < self.interseason_mean_temperature
            )
            min_nb_points_heating = 10
            if sum(heating_mask) <= min_nb_points_heating:
                warn = (
                    "Not enough data for the heating period."
                    f"Number of data points: {sum(heating_mask)}"
                )
                self.logger.warning(warn)
                # too few point to do any test
                heating_sp = 1
            else:
                heating_sp = spearmanr(
                    self.resampled_energy_temperature.loc[
                        heating_mask,
                        [self.target_name, self.temperature_name],
                    ],
                    alternative="less",
                ).pvalue
            cooling_mask = (
                self.resampled_energy_temperature[self.temperature_name]
                > self.interseason_mean_temperature
            )
            min_nb_points_cooling = 10
            if sum(cooling_mask) <= min_nb_points_cooling:
                warn = (
                    "Not enough data for the cooling period."
                    f"Number of data points: {sum(heating_mask)}"
                )
                self.logger.warning(warn)
                # too few point to do any test
                cooling_sp = 1
            else:
                data_to_test = self.resampled_energy_temperature.loc[
                    cooling_mask,
                    [self.target_name, self.temperature_name],
                ]
                cooling_sp = spearmanr(data_to_test, alternative="greater").pvalue
            if heating_sp < significance_level and cooling_sp < significance_level:
                self.degree_days_type = "both"
            elif heating_sp < significance_level:
                self.degree_days_type = "heating"
            elif cooling_sp < significance_level:
                self.degree_days_type = "cooling"
            else:
                err = (
                    "Cannot detect the degree days type. Please specify it manually.\n"
                    f"{cooling_sp=} \n"
                    f"{heating_sp=} \n"
                    f"{self.resampled_energy_temperature=} \n"
                    f"{self.resampled_energy=} \n"
                    f"{self.resampled_temperature=} \n"
                )
                raise ValueError(err)

    def _calculate_degree_days(
        self,
        degree_days_base_temperature: dict,
    ) -> pd.DataFrame:
        """Compute the degree days.

        Parameters
        ----------
        degree_days_base_temperature : dict
            Base temperature for the computation of the degree days.
            Must be a dictionary with the keys ``"heating"`` and/or ``"cooling"``.
            Example: degree_days_base_temperature={'heating': 18, 'cooling': 24}

        Returns
        -------
        pd.DataFrame
            DataFrame with the heating and/or cooling degree days
            sampled at the given frequency.

        """
        for dd_type in degree_days_base_temperature:
            if self.degree_days_type in [dd_type, "both"]:
                degree_days = [
                    dd_compute(
                        self.temperature_data,
                        degree_days_base_temperature[dd_type],
                        dd_type=dd_type,
                        method=self.degree_days_computation_method,
                    )
                    .resample(self.frequency)
                    .sum()
                    for dd_type in degree_days_base_temperature
                ]
        return pd.concat(degree_days, axis=1)

    def calibrate_base_temperature(
        self,
        dd_type: literal_dd_types = "heating",
        t0: float | None = None,
        xatol: float = 1e-1,
    ) -> float:
        """Calibrate the base temperature for the specified degree days type.

        This method optimizes the base temperature used to compute the degree days
        by minimizing the mean squared error between the energy consumption data and
        the degree days model. The optimization is done using the
        `scipy.optimize.minimize_scalar` function with a bounded method.

        Parameters
        ----------
        dd_type : str, optional
            The type of degree days to calibrate, must be one of the following:
            - "heating": to calibrate the base temperature for heating degree days.
            - "cooling": to calibrate the base temperature for cooling degree days.
            The default is "heating".

        t0 : float, optional
            The initial guess for the base temperature.
            If not provided, the default initial guess is 16°C for heating or 24°C for
            cooling.

        xatol : float, optional
            The absolute error tolerance for the optimization.
            This controls how precise the optimized base temperature needs to be.
            Default is 1e-1 (0.1°C).

        Returns
        -------
        float
            The optimized base temperature for the specified degree days type.

        Raises
        ------
        ValueError
            If the `dd_type` is invalid (not one of "heating" or "cooling").

        Example
        -------
        >>> ts.calibrate_base_temperature(dd_type="heating", t0=15, xatol=0.05)

        """
        if dd_type not in ["heating", "cooling"]:
            err = "Invalid degree days type. Must be one of 'heating' or 'cooling'."
            raise ValueError(err)
        if t0 is None:
            t0 = 16 if dd_type == "heating" else 24
        if dd_type == "heating":
            mask = (
                self.resampled_energy_temperature[self.temperature_name]
                < self.interseason_mean_temperature
            )
            bounds = (10.0, self.interseason_mean_temperature)
        elif dd_type == "cooling":
            mask = (
                self.resampled_energy_temperature[self.temperature_name]
                > self.interseason_mean_temperature
            )
            bounds = (self.interseason_mean_temperature, 30.0)
        else:
            err = "Invalid degree days type. Must be one of 'heating' or 'cooling'."
            raise ValueError(err)
        res = minimize_scalar(
            self.loss_function,
            args=(
                dd_type,
                self.resampled_energy_temperature[self.target_name],
                self.temperature_data,
                self.frequency,
                mask,
                self.degree_days_computation_method,
            ),
            bounds=bounds,
            method="bounded",
            options={
                "xatol": xatol,
            },
        )
        return res.x

    def calibrate_base_temperatures(
        self,
        t0_heating: float | None = None,
        t0_cooling: float | None = None,
        xatol: float = 1e-1,
    ) -> None:
        """Calibrate the base temperatures for both heating and cooling degree days.

        This method optimizes the base temperatures for heating and/or cooling degree
        days by minimizing the mean squared error between the energy consumption data
        and the degree days model. The method will calibrate the base temperatures based
        on the detected or specified `degree_days_type`.

        If the `degree_days_type` is "heating", only the heating base temperature is
        calibrated. If it is "cooling", only the cooling base temperature is calibrated.
        If it is "both", both base temperatures are calibrated.

        Parameters
        ----------
        t0_heating : float, optional
            The initial guess for the heating base temperature. If not provided, the
            default
            initial guess is 16°C.

        t0_cooling : float, optional
            The initial guess for the cooling base temperature. If not provided, the
            default initial guess is 24°C.

        xatol : float, optional
            The absolute error tolerance for the optimization.
            This controls how precise the optimized base temperatures need to be.
            Default is 1e-1 (0.1°C).

        Returns
        -------
        None
            The method updates the `degree_days_base_temperature` attribute with the
            optimized base temperatures for heating and/or cooling.

        Raises
        ------
        ValueError
            If the `degree_days_type` is invalid.

        Example
        -------
        >>> ts.calibrate_base_temperatures(t0_heating=15, t0_cooling=25, xatol=0.05)

        """
        types_to_calibrate = []
        if self.degree_days_type in ["heating", "both"]:
            types_to_calibrate.append("heating")
        if self.degree_days_type in ["cooling", "both"]:
            types_to_calibrate.append("cooling")
        for dd_type in types_to_calibrate:
            if dd_type == "heating":
                t0 = t0_heating
            elif dd_type == "cooling":
                t0 = t0_cooling
            topt = self.calibrate_base_temperature(
                dd_type=cast(Literal["heating", "cooling"], dd_type),
                t0=t0,
                xatol=xatol,
            )
            self.degree_days_base_temperature[dd_type] = topt

    def _fit_thermosensitivity(
        self,
    ) -> None:
        """Fit the thermosensitivity model with interactions for categories.

        This method performs the following steps:

        1. Drops rows with missing values from the aggregated data.
        2. Creates a regression model where the dependent variable is energy consumption
          and the independent variables are the degree days (heating, cooling, or both)
          and category interactions.
        3. Encodes the category labels using one-hot encoding.
        4. Constructs interaction terms between the degree days and the one-hot encoded
          categories.
        5. Fits the Ordinary Least Squares (OLS) model to the data with the degree days
          and interaction terms.

        The fitted model is stored in the ``_model`` attribute.

        """
        data = self.aggregated_data.dropna(how="any", axis=0)
        y = data[self.target_name].copy()
        x = data[self.predictors].copy()
        x["Intercept"] = 1  # add constant
        self._model = OLS(y, x).fit()

    def fit(
        self: ThermosensitivityInstance,
    ) -> ThermosensitivityInstance:
        """Train the model.

        This method will:

        1. Calibrate the base temperature if it is not set.
           See :meth:`calibrate_base_temperature`.
        2. Aggregate the data. This consists of resampling the energy and temperature
           data and the computation of the degree days. See :meth:`_aggregate_data`.
        3. Fit the thermosensitivity model.
           See :meth:`_fit_thermosensitivity`.

        """
        self.calibrate_base_temperatures()
        self._aggregate_data(self.degree_days_base_temperature)
        self._fit_thermosensitivity()
        return self

    def __repr__(
        self,
    ) -> str:
        """Return the representation of the object."""
        class_name = self.__class__.__name__
        header = f"""{class_name}(frequency={self.frequency},
        degree_days_type={self.degree_days_type},
        degree_days_base_temperature={ {k:round(v, 2) for k,v
            in self.degree_days_base_temperature.items()} },
        degree_days_computation_method={self.degree_days_computation_method},
        interseason_mean_temperature={self.interseason_mean_temperature})"""
        if self._model is not None:
            message = f"{header}\n\n{self.model.summary(slim=True)}"
        else:
            message = header
        return message

    def loss_function(
        self,
        t0: float,
        dd_type: literal_valid_dd_types,
        resampled_energy: pd.Series,
        raw_temperature: pd.Series,
        frequency: str,
        mask: pd.Series | None = None,
        degree_days_computation_method: literal_computation_dd_types = "integral",
    ) -> float:
        """Loss function for the optimization of the base temperature.

        Compute the mean squared error (MSE) between the observed energy data and
        the energy model based on degree days. This function is used for calibrating the
        base temperature in the thermosensitivity model by finding the base temperature
        that minimizes this error.

        The degree days are calculated based on the input temperature data, resampled
        to the desired frequency, and compared against the resampled energy data. The
        MSE is used as the objective function to optimize the base temperature.

        Parameters
        ----------
        t0 : float
            Base temperature used to compute degree days.
        dd_type : literal_dd_types
            Type of degree days to compute. Must be one of:
            - "heating": degree days when temperatures are below the base temperature.
            - "cooling": degree days when temperatures are above the base temperature.
        resampled_energy : pd.Series
            The resampled time series of energy data to be modeled.
        raw_temperature : pd.Series
            The original time series of temperature data, which is used to compute
            degree days based on the base temperature.
        frequency : str
            The frequency to resample the data (e.g., "1D" for daily, "7D" for weekly).
        mask : pd.Series or None, optional
            A boolean mask to filter the data before fitting the model. This allows for
            focusing on specific periods (e.g., only heating or cooling periods).
            Default is None.
        degree_days_computation_method : literal_computation_dd_types, optional
            The method used to compute degree days. Must be one of:
            - "integral": sum the difference between the base temperature and actual
            temperature.
            - "mean": use the mean temperature for degree day calculations.
            - "min_max": use the average of daily minimum and maximum temperatures.
            Default is "integral".

        Returns
        -------
        float
            The mean squared error (MSE) of the residuals between the observed
            energy data and the modeled energy data based on degree days.

        Example
        -------
        >>> loss = loss_function(
                t0=18.0,
                dd_type="heating",
                resampled_energy=energy_data,
                raw_temperature=temperature_data,
                frequency="1D",
                degree_days_computation_method="integral",
            )
        >>> print(f"Calculated loss: {loss}")

        """
        degree_days = dd_compute(
            temperature=raw_temperature,
            reference=t0,
            dd_type=dd_type,
            method=degree_days_computation_method,
        )
        degree_days_resampled = (
            degree_days.resample(frequency).sum().rename("degree_days")
        )
        data = pd.concat([resampled_energy, degree_days_resampled], axis=1).dropna(
            how="any",
            axis=0,
        )
        data["Intercept"] = 1
        if mask is not None:
            data = data[mask]
        model = OLS(
            data[resampled_energy.name],
            data[["degree_days", "Intercept"]],
        ).fit()
        log_message = f"{t0=:.4f}, {model.mse_resid:.2f}, {model.mse_total:.2f}"
        self.logger.info(log_message)
        return model.mse_resid


class CategoricalThermoSensitivity(
    ThermoSensitivity,
):
    """Class to compute the thermosensitivity of a building with labeled periods.

    Based on the `ThermoSensitivity` class.

    """

    categories_name = "category"

    def __init__(
        self,
        energy_data: pd.Series,
        temperature_data: pd.Series,
        categories: pd.Series,
        frequency: str = "1D",
        degree_days_type: literal_dd_types = "heating",
        degree_days_base_temperature: dict | None = None,
        degree_days_computation_method: literal_computation_dd_types = "integral",
        interseason_mean_temperature: float = 20,
        base_logger_name: str | None = None,
        min_logger_level_stdout: int | str = logging.ERROR,
    ) -> None:
        """Return a ``CategoricalThermoSensitivity`` instance.

        Parameters
        ----------
        energy_data : pd.Series
            Energy data of the building.
        temperature_data : pd.Series
            Outdoor temperature data.
        categories : pd.Series
            Labels for the periods.
        frequency : str, optional
            Frequency for the analysis the data, by default "1D".
            If "1D", the data will be resampled daily.
            If "7D", the data will be resampled weekly.
            If "1W-MON", the data will be resampled weekly starting on Monday.
        degree_days_type : str, optional
            Type of degree days to compute. Must be one of the following:

            - "heating": compute only heating degree days (temperature below a
              threshold).
            - "cooling": compute only cooling degree days (temperature above a
              threshold).
            - "both": compute both heating and cooling degree days.
            - "auto": automatically detect the degree days type. See the
              `detect_degree_days_type` method.

        degree_days_base_temperature : dict, optional
            Base temperature for the computation of the degree days, by default {}.
            Must be a dictionary with the keys ``"heating"`` and/or ``"cooling"``.
            Example: degree_days_base_temperature={'heating': 18, 'cooling': 24}
        degree_days_computation_method : str, optional
            Method to compute the degree days, by default ``"integral"``. Possibilities
            are:

            - "integral": integral the degree days above or below the base temperature.
            - "mean": sum the difference between the base temperature and the mean
              temperature.
            - "min_max": sum the difference between the base temperature and the mean
              temperature computed as the mean of the minimum and maximum temperature.

        interseason_mean_temperature : float, optional
            Mean temperature to detect the heating and cooling periods, by default 20.
            Used only:

            - to detect the degree days type automatically. See the
              `detect_degree_days_type` method.
            - to estimate the base temperature. See the `calibrate_base_temperature`
              method.

        base_logger_name : str, optional
            Name of the logger. By default, it is the class name. All following
            following instances receive a unique identifier, based on the first
            one, with the pattern:

            - ``<base_logger_name>``
            - ``<base_logger_name>_1``
            - ...

        min_logger_level_stdout : str or int, optional
            The minimum severity level for messages to be printed to standard output.
            Messages below this level will not be displayed. The default is
            ``"ERROR"``, meaning only errors and critical messages will
            be printed. Acceptable values can be specified as strings (e.g.,
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") or as corresponding
            integer constants from the `logging` module. If you want to have details
            on what's going on in the code, set this to ``"INFO"`` or ``"WARNING"``.

        """
        degree_days_base_temperature = {} or degree_days_base_temperature
        self._categories = categories
        super().__init__(
            energy_data=energy_data,
            temperature_data=temperature_data,
            frequency=frequency,
            degree_days_type=degree_days_type,
            degree_days_base_temperature=degree_days_base_temperature,
            degree_days_computation_method=degree_days_computation_method,
            interseason_mean_temperature=interseason_mean_temperature,
            base_logger_name=base_logger_name,
            min_logger_level_stdout=min_logger_level_stdout,
        )

    @cached_property
    def resampled_categories(
        self,
    ) -> pd.Series:
        """Resample categories at the specified frequency.

        This method resamples the categorical data (self.categories) at the given
        frequency (self.frequency) and computes the most common category within each
        resampled period. If a period contains no data, it returns None for that
        period. The method uses an efficient aggregation approach by leveraging `agg()`
        and `value_counts()` to find the most frequent category.

        The result is a `pd.Series` with the same frequency as the resampling,
        containing the most common category for each resampled time window.

        This property is cached to avoid recomputing it multiple times, improving
        performance in case of repeated access.

        Returns
        -------
        pd.Series
            A Series where each value corresponds to the most common category in the
            resampled period, indexed by the resampled time index.

        Example
        -------
        >>> self.frequency = '1D'  # Daily resampling
        >>> self.categories = pd.Series(['A', 'B', 'A', 'A', 'B'],
        index=pd.date_range('2023-01-01', periods=5))
        >>> resampled = self.resampled_categories
        >>> print(resampled)
        2023-01-01    A
        2023-01-02    A
        2023-01-03    A
        2023-01-04    B
        Freq: D, dtype: object

        .. note::
            In case there are multiple categories for one resampled period, the
            category assigned to the resampled period is the most common one.

        """
        categories: pd.Series = self.categories
        resampler: Resampler = categories.resample(self.frequency)
        resampled_categories: pd.Series = resampler.agg(
            lambda x: x.value_counts().idxmax() if not x.empty else None,
        )
        return resampled_categories.rename(self.categories_name)

    @property
    def categories(
        self,
    ) -> pd.Series:
        """The categories of the periods."""
        return self._categories

    @categories.setter
    def categories(
        self,
        value: pd.Series,
    ) -> None:
        """Set the categories attribute.

        Parameters
        ----------
        value : pd.Series
            A pandas Series to assign as the categories.

        """
        self._categories = value

    @cached_property
    def resampled_energy_temperature_category(
        self,
    ) -> pd.DataFrame:
        """The resampled energy, temperature and category data.

        The DataFrame contains the resampled energy and temperature data.
        Periods with missing values are removed.
        """
        return pd.concat(
            [
                self.resampled_energy,
                self.resampled_temperature,
                self.resampled_categories,
            ],
            axis=1,
        ).dropna(how="any", axis=0)

    def _aggregate_data(
        self,
        degree_days_base_temperature: dict,
    ) -> None:
        """Compute the degree days and aggregate the data.

        Store the aggregated data in the `aggregated_data` property.
        """
        degree_days = self._calculate_degree_days(degree_days_base_temperature)
        self.aggregated_data = pd.concat(
            [
                self.resampled_energy_temperature_category,
                degree_days,
            ],
            axis=1,
        )

    def _detect_degree_days_type(
        self,
        significance_level: float = 0.05,
    ) -> None:
        """Detect the degree days type (heating, cooling, or both).

        Detection of the degree-days types is based on correlations between
        energy consumption and temperature, across different categories.

        This method automatically detects the type of degree days (heating, cooling, or
        both) if the `degree_days_type` is set to "auto". It manages potential
        Simpson's paradox by calculating the Spearman correlation for each category in
        the data, rather than relying on aggregated data. The detection is performed by
        evaluating the correlation between energy consumption and temperature in each
        resampled category.

        If the Spearman correlation between energy consumption and temperature is
        significant for any category:

        - A negative correlation for periods below the interseason mean temperature
          indicates "heating" degree days.
        - A positive correlation for periods above the interseason mean temperature
          indicates "cooling" degree days.
        - If both correlations are significant, the system is assumed to have both
          heating and cooling sensitivity.

        Parameters
        ----------
        significance_level : float, optional
            The threshold for the p-value to consider the correlation as statistically
            significant. The default is 0.05 (5% significance level).

        Returns
        -------
        None
            The method updates the `degree_days_type` attribute of the class, setting
            it to either "heating", "cooling", or "both" based on the correlation
            results. If no significant correlation is found, a `ValueError` is raised.

        Raises
        ------
        ValueError
            If no significant correlation can be detected for either heating or cooling
            degree days, the method raises a `ValueError` and suggests manual
            specification of the `degree_days_type`.

        Notes
        -----
        - **Spearman Correlation**: This non-parametric correlation test is used to
          determine monotonic relationships between energy consumption and temperature.
          It is robust to non-linear relationships and does not assume normality.
        - **Simpson's Paradox**: By evaluating correlations within each category, the
          method avoids potential misleading conclusions that might arise from
          aggregated data. Simpson's Paradox occurs when trends that appear in
          different groups of data disappear or reverse when the data is combined.

        See Also
        --------
        :py:func:`ThermoSensitivity._detect_degree_days_type` : Base implementation for
        non-categorical data.

        Example
        -------
        >>> self._detect_degree_days_type(significance_level=0.05)
        Detected degree days type: heating

        """
        if self.degree_days_type == "auto":
            heating_sp = 1
            cooling_sp = 1
            for cat in list(self.resampled_categories.unique()):
                log_message = f"{cat=}"
                self.logger.info(log_message)
                heating_mask = (
                    self.resampled_energy_temperature_category[self.temperature_name]
                    < self.interseason_mean_temperature
                )
                cat_mask = (
                    self.resampled_energy_temperature_category[self.categories_name]
                    == cat
                )
                tmp_heating_sp = spearmanr(
                    self.resampled_energy_temperature_category.loc[
                        heating_mask & cat_mask,
                        [self.target_name, self.temperature_name],
                    ],
                    alternative="less",
                )
                possib_heat_sp_no_nan = [
                    el for el in [tmp_heating_sp.pvalue, heating_sp] if ~np.isnan(el)
                ]
                heating_sp = min(possib_heat_sp_no_nan)
                cooling_mask = (
                    self.resampled_temperature > self.interseason_mean_temperature
                )
                tmp_cooling_sp = spearmanr(
                    self.resampled_energy_temperature_category.loc[
                        cooling_mask & cat_mask,
                        [self.target_name, self.temperature_name],
                    ],
                    alternative="greater",
                )
                log_message = f"{tmp_cooling_sp}"
                self.logger.info(log_message)
                possib_cool_sp_no_nan = [
                    el for el in [tmp_cooling_sp.pvalue, cooling_sp] if ~np.isnan(el)
                ]
                cooling_sp = min(possib_cool_sp_no_nan)
            if heating_sp < significance_level and cooling_sp < significance_level:
                self.degree_days_type = "both"
            elif heating_sp < significance_level:
                self.degree_days_type = "heating"
            elif cooling_sp < significance_level:
                self.degree_days_type = "cooling"
            else:
                err = (
                    "Cannot detect the degree days type. Please specify it manually. "
                    f"For info, {cooling_sp=}, {heating_sp=}"
                )
                raise ValueError(err)

    def _fit_thermosensitivity(
        self,
    ) -> None:
        """Fit the thermosensitivity model with interactions for categories.

        This method performs the following steps:

        1. Drops rows with missing values from the aggregated data.
        2. Creates a regression model where the dependent variable is energy consumption
          and the independent variables are the degree days (heating, cooling, or both)
          and category interactions.
        3. Encodes the category labels using one-hot encoding.
        4. Constructs interaction terms between the degree days and the one-hot encoded
          categories.
        5. Fits the Ordinary Least Squares (OLS) model to the data with the degree days
          and interaction terms.

        The fitted model is stored in the ``_model`` attribute.

        """
        data = self.aggregated_data.dropna(how="any", axis=0)
        y = data[self.target_name].copy()
        x = data[self.predictors].copy()
        x["Intercept"] = 1  # add constant
        one_hot_encoding = pd.get_dummies(data["category"], dtype=int)
        # create interaction terms
        interactions = None
        for col in x.columns:
            for cat in one_hot_encoding.columns:
                tmp_int = x[col] * one_hot_encoding[cat]
                tmp_int.name = f"{col}:{cat}"
                if interactions is None:
                    interactions = tmp_int
                else:
                    interactions = pd.concat([interactions, tmp_int], axis=1)
        self._model = OLS(y, interactions).fit()
