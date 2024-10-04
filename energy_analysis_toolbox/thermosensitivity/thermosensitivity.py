"""Process the thermosensitivity data.

# Available classes

## ThermoSensitivity

Class to compute the thermosensitivity of a building.
Needs a time series of energy consumption and outdoor temperature.

## CategoricalThermoSensitivity

Class to compute the thermosensitivity of a building with labeled periods.
Needs a time series of energy consumption, outdoor temperature, and labeled periods.

The labeled periods are resampled to the same frequency as the energy and temperature data by
taking the most common category in the period.

Currently, the class only calibrates one base temperature for all the categories aggregated.

# Implementation details

## Resampling frequency

The energy and temperature data are resampled at a given frequency.
The degree days are computed at the same frequency.

## Thermo Sensitivity

The thermo-sensitivity is modelled as a linear regression between the energy consumption and the degree days.

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
- MinMax: sum the difference between the base temperature and the mean temperature computed as the mean of the minimum and maximum temperature.
    .. math::
        DegreeDays = max(0, BaseTemperature - \\frac{T_{min} + T_{max}}{2} )

See the `dd_compute` function in the `energy_analysis_toolbox.weather.degree_days` module.

Over a long period, the data can present a thermosensitivity with different types of degree days:

- Heating: the energy consumption increases when the temperature decreases. Usually, the base temperature is around 18°C.
- Cooling: the energy consumption increases when the temperature increases. Usually, the base temperature is around 24°C.

## Auto-calibration

Two aspects of the thermosensitivity can be automatically detected:

- The degree days type: heating, cooling, or both.
- The base temperature.

### Degree days type

Each building, depending of the installed systems, can have different thermosensitivity types :
- use of heating systems (heating degree days, during the winter)
- use of cooling systems (cooling degree days, during the summer)
- use of both systems (heating and cooling degree days)

The degree days type can be automatically detected by setting the `degree_days_type` parameter to ``"auto"``.
The method will compute the Spearman correlation between the energy and the temperature for the periods
with the mean temperature below and above the intersaison mean temperature (default is 20°C).

### Base temperature

The base temperature can be calibrated by minimizing the mean squared error between the data and the model.

Each degree days type has a specific base temperature that is determined by analyzing the data over the corresponding periods:
The heating (resp. cooling) base temperature is calibrated by minimizing the mean squared error between the energy and the heating (resp. cooling) degree days for the periods with the mean temperature below (resp. above) the intersaison mean temperature.

The optimization is done with the `scipy.optimize.minimize_scalar` function with the `bounded` method.

"""

import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from energy_analysis_toolbox.weather.degree_days import dd_compute
from statsmodels.api import OLS
from functools import cached_property
from energy_analysis_toolbox.energy.resample import to_freq as energy_to_freq


class ThermoSensitivity:
    """Class to compute the thermosensitivity of a building.

    Parameters
    ----------
    energy_data : pd.Series
        Energy data of the building.
    temperature_data : pd.Series
        Outdoor temperature data.
    frequency : str, optional
        Frequency for the analysis the data, by default ``"1D"``.
        - If ``"1D"``, the data will be resampled daily.
        - If ``"7D"``, the data will be resampled weekly.
        - If ``"1W-MON"``, the data will be resampled weekly starting on Monday.

    degree_days_type : str, optional
        Type of degree days to compute. Must be one of the following:
        - ``"heating"``: compute only heating degree days (temperature below a threshold).
        - ``"cooling"``: compute only cooling degree days (temperature above a threshold).
        - ``"both"``: compute both heating and cooling degree days.
        - ``"auto"``: automatically detect the degree days type. See the `detect_degree_days_type` method.

    degree_days_base_temperature : dict, optional
        Base temperature for the computation of the degree days, by default {}.
        Must be a dictionary with the keys ``"heating"`` and/or ``"cooling"``.
        If not specified, the base temperature will be calibrated. See the `calibrate_base_temperature` method.
        If specified, the base temperature will be used directly. It must match the degree_days_type.
        Example: degree_days_base_temperature={'heating': 18, 'cooling': 24}
    degree_days_computation_method : str, optional
        Method to compute the degree days, by default ``"integral"``.
        Availables are:

        - "integral": integral the degree days above or below the base temperature.
        - "mean": sum the difference between the base temperature and the mean temperature.
        - "min_max": sum the difference between the base temperature and the mean temperature
          computed as the mean of the minimum and maximum temperature.

    interseason_mean_temperature : float, optional
        Mean temperature to separate the heating and cooling periods, by default 20.
        Used only:
        - to detect the degree days type automatically. See the `detect_degree_days_type` method.
        - to estimate the base temperature. See the `calibrate_base_temperature` method.

    Examples
    --------
    >>> from energy_analysis_toolbox.thermosensitivity import ThermoSensitivity
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> temperature_data = 15 + 2*pd.Series(np.random.rand(366), index=pd.date_range("2020-01-01", "2020-12-31", freq="D"))
    >>> energy_data = 10 + (16 - temperature_data).clip(0) * 5 + np.random.rand(366)
    >>> ts = ThermoSensitivity(energy_data, temperature_data, degree_days_type="auto")
    >>> ts.fit()
    >>> ts
    ThermoSensitivity(frequency=1D,
            degree_days_type=heating,
            degree_days_base_temperature={'heating': 15.98},
            degree_days_computation_method=integral,
            interseason_mean_temperature=20)
    <BLANKLINE>
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 energy   R-squared:                       0.969
    Model:                            OLS   Adj. R-squared:                  0.969
    No. Observations:                 366   F-statistic:                 1.137e+04
    Covariance Type:            nonrobust   Prob (F-statistic):          1.25e-276
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    heating_degree_days     5.1177      0.048    106.638      0.000       5.023       5.212
    Intercept              10.5120      0.019    539.733      0.000      10.474      10.550
    =======================================================================================
    <BLANKLINE>
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

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
        frequency="1D",
        degree_days_type="heating",
        degree_days_base_temperature: dict = {},
        degree_days_computation_method="integral",
        interseason_mean_temperature=20,
    ):
        self._energy_data = energy_data
        self._temperature_data = temperature_data
        self._frequency = frequency
        self._aggregated_data = None
        self.degree_days_type = degree_days_type
        self.degree_days_base_temperature = degree_days_base_temperature
        self.degree_days_computation_method = degree_days_computation_method
        self.interseason_mean_temperature = interseason_mean_temperature
        self.predictors = []
        self._model = None
        self._validate_data()
        self._post_init()

    @property
    def frequency(
        self,
    ):
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
    ):
        """The outdoor temperature data.

        The property is unmutable. To change the temperature data, create a new object.
        """
        return self._temperature_data

    @cached_property
    def resampled_energy(
        self,
    ) -> pd.Series:
        """The energy data resampled at the given frequency.

        Uses the `to_freq` function from the `energy_analysis_toolbox.energy.resample` module
        to convert the energy data to the desired frequency.

        This property is cached to avoid recomputing it multiple times.
        """
        energy = self.energy_data.copy()
        last_period = energy.index[-1] - energy.index[-2]
        energy[energy.index[-1] + last_period] = 0
        new_energy = energy_to_freq(
            energy,
            self.frequency,
            # origin=energy.index[0].floor(self.frequency),
        ).rename(self.target_name)
        return new_energy

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
    ):
        """The thermosensitivity model.
        A `statsmodels.regression.linear_model.RegressionResults` object.

        Raises
        ------
        ValueError
            If the model is not fitted. Use the `fit` method to train the model.
        """
        if self._model is None:
            raise ValueError("Model not fitted. Please run the `fit` method.")
        return self._model

    @property
    def aggregated_data(
        self,
    ) -> pd.DataFrame:
        """The aggregated data used to fit the model.

        The Data is a DataFrame resampled at the provided Frequency with the following columns:
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
            raise ValueError(
                "Data not aggregated. Please run the `fit` method."
            )
        return self._aggregated_data

    @aggregated_data.setter
    def aggregated_data(
        self,
        value: pd.DataFrame,
    ):
        """Set the aggregated data, and reset the model."""
        self._aggregated_data = value
        self._model = None

    def _validate_data(
        self,
    ):
        """Check the validity of the parameters.

        Raises
        ------
        ValueError
            If the degree days type is not valid.
        ValueError
            If the base temperature is not specified for the heating or cooling degree days.
            While not empty.
        """
        if self.degree_days_type not in ["heating", "cooling", "both", "auto"]:
            raise ValueError(
                "Invalid degree days type. Must be one of 'heating', 'cooling', 'both' or 'auto'."
            )
        if self.degree_days_base_temperature != {}:
            if self.degree_days_type in ["heating", "both"]:
                try:
                    self.degree_days_base_temperature["heating"]
                except KeyError:
                    raise ValueError(
                        "Base temperature for heating degree days must be specified.\n Example: degree_days_base_temperature={'heating': 18, 'cooling': 24}"
                    )
            elif self.degree_days_type in ["cooling", "both"]:
                try:
                    self.degree_days_base_temperature["cooling"]
                except KeyError:
                    raise ValueError(
                        "Base temperature for cooling degree days must be specified.\n Example: degree_days_base_temperature={'heating': 18, 'cooling': 24}"
                    )

    def _post_init(
        self,
    ):
        """End the initialization process.

        If the degree days type is set to ``"auto"``, the method will detect the degree days type.
        See :meth:`_detect_degree_days_type`.
        After the detection, the predictors will be set.
        """
        self._detect_degree_days_type()
        if self.degree_days_type == "both":
            self.predictors = ["heating_degree_days", "cooling_degree_days"]
        elif self.degree_days_type in ["heating", "cooling"]:
            self.predictors = [f"{self.degree_days_type}_degree_days"]

    def _aggregate_data(
        self,
        degree_days_base_temperature: dict | None = None,
    ):
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
        significance_level=0.05,
    ):
        """Estimate the degree days type if it is set to ``"auto"``.

        It will compute the Spearman correlation (with the alternative hypothesis) between the energy and the temperature.
        If the p-value is below the significance level, the degree days type will be set to ``"heating"`` or ``"cooling"`` or ``"both"``.

        If no significant correlation is found, the method will raise a ValueError.

        - **Heating**: The energy consumption is negatively correlated with the temperature for the periods with the mean temperature below the intersaison mean temperature.
        - **Cooling**: The energy consumption is positively correlated with the temperature for the periods with the mean temperature above the intersaison mean temperature.

        Note
        ----
        The Spearman correlation is a non-parametric test that measures the strength and direction of the monotonic relationship between two variables.
        The relation is not necessarily linear, and the test does not assume that the data is normally distributed.

        .. warning::
            If the data contains multiple groups, the Simpson's paradox may occur. The Paradox states that
            the correlation observed in the aggregated data may not hold when the data is split into subgroups.
            See the `CategoricalThermoSensitivity` class for a solution.

            References: `wikipedia <https://fr.wikipedia.org/wiki/Paradoxe_de_Simpson>`_
        """
        if self.degree_days_type == "auto":
            heating_mask = (
                self.resampled_energy_temperature[self.temperature_name]
                < self.interseason_mean_temperature
            )
            if sum(heating_mask) <= 10:
                print("Not enough data for the heating period.")
                print(f"Number of data points: {sum(heating_mask)}")
                # too few point to do any test
                heating_sp = 1
            else:
                heating_sp = spearmanr(
                    self.resampled_energy_temperature.loc[
                        heating_mask, [self.target_name, self.temperature_name]
                    ],
                    alternative="less",
                ).pvalue
            cooling_mask = (
                self.resampled_energy_temperature[self.temperature_name]
                > self.interseason_mean_temperature
            )
            if sum(cooling_mask) <= 10:
                print("Not enough data for the heating period.")
                print(f"Number of data points: {sum(heating_mask)}")
                # too few point to do any test
                cooling_sp = 1
            else:
                data_to_test = self.resampled_energy_temperature.loc[
                    cooling_mask, [self.target_name, self.temperature_name]
                ]
                cooling_sp = spearmanr(
                    data_to_test, alternative="greater"
                ).pvalue
            if (
                heating_sp < significance_level
                and cooling_sp < significance_level
            ):
                self.degree_days_type = "both"
            elif heating_sp < significance_level:
                self.degree_days_type = "heating"
            elif cooling_sp < significance_level:
                self.degree_days_type = "cooling"
            else:
                print(cooling_sp, heating_sp)
                print(self.resampled_energy_temperature)
                print(self.resampled_energy)
                print(self.resampled_temperature)
                raise ValueError(
                    "Cannot detect the degree days type. Please specify it manually."
                )

    def _calculate_degree_days(
        self,
        degree_days_base_temperature: dict,
    ) -> pd.Series:
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
        degree_days = []
        for dd_type in degree_days_base_temperature.keys():
            if self.degree_days_type in [dd_type, "both"]:
                degree_days.append(
                    dd_compute(
                        self.temperature_data,
                        degree_days_base_temperature[dd_type],
                        type=dd_type,
                        method=self.degree_days_computation_method,
                    )
                    .resample(self.frequency)
                    .sum()
                )
        return pd.concat(degree_days, axis=1)

    def calibrate_base_temperature(
        self,
        dd_type="heating",
        t0: float | None = None,
        xatol: float = 1e-1,
        disp: bool = True,
    ):
        """Fit the base temperature to the data."""
        if dd_type not in ["heating", "cooling"]:
            raise ValueError(
                "Invalid degree days type. Must be one of 'heating' or 'cooling'."
            )
        if t0 is None:
            t0 = 16 if dd_type == "heating" else 24
        if dd_type == "heating":
            mask = (
                self.resampled_energy_temperature[self.temperature_name]
                < self.interseason_mean_temperature
            )
            bounds = (10, self.interseason_mean_temperature)
        elif dd_type == "cooling":
            mask = (
                self.resampled_energy_temperature[self.temperature_name]
                > self.interseason_mean_temperature
            )
            bounds = (self.interseason_mean_temperature, 30)
        else:
            raise ValueError(
                "Invalid degree days type. Must be one of 'heating' or 'cooling'."
            )
        res = minimize_scalar(
            loss_function,
            args=(
                dd_type,
                self.resampled_energy_temperature[self.target_name],
                self.temperature_data,
                self.frequency,
                mask,
                self.degree_days_computation_method,
                disp,
            ),
            bounds=bounds,
            method="bounded",
            options={
                "xatol": xatol,
                "disp": disp,
            },
        )
        return res.x

    def calibrate_base_temperatures(
        self,
        t0_heating: float | None = None,
        t0_cooling: float | None = None,
        xatol: float = 1e-1,
        disp: bool = True,
    ):
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
            topt = self.calibrate_base_temperature(dd_type, t0, xatol, disp)
            self.degree_days_base_temperature[dd_type] = topt

    def _fit_thermosensitivity(
        self,
    ):
        data = self.aggregated_data.dropna(how="any", axis=0)
        Y = data[self.target_name].copy()
        X = data[self.predictors].copy()
        X["Intercept"] = 1  # add constant
        self._model = OLS(Y, X).fit()

    def fit(
        self,
    ):
        """Train the model.

        This method will:

        1. Calibrate the base temperature if it is not set.
           See :meth:`calibrate_base_temperature`.
        2. Aggregate the data. This consists of resampling the energy and temperature data
           and the computation of the degree days. See :meth:`_aggregate_data`.
        3. Fit the thermosensitivity model.
           See :meth:`_fit_thermosensitivity`.

        """
        self.calibrate_base_temperatures(disp=False)
        self._aggregate_data(self.degree_days_base_temperature)
        self._fit_thermosensitivity()
        return self

    def __repr__(
        self,
    ):
        """Return the representation of the object."""
        class_name = self.__class__.__name__
        header = f"""{class_name}(frequency={self.frequency},
        degree_days_type={self.degree_days_type},
        degree_days_base_temperature={ {k:round(v, 2) for k,v in self.degree_days_base_temperature.items()} },
        degree_days_computation_method={self.degree_days_computation_method},
        interseason_mean_temperature={self.interseason_mean_temperature})"""
        if self._model is not None:
            message = f"{header}\n\n{self.model.summary(slim=True)}"
        else:
            message = header
        return message


def loss_function(
    t0: float,
    dd_type: str,
    resampled_energy: pd.Series,
    raw_temperature: pd.Series,
    frequency: str,
    mask: pd.Series = None,
    dd_computation_method="integral",
    verbose=False,
) -> float:
    """Compute the error between the data and the model.
    Used to calibrate the base temperature.

    Parameters
    ----------
    t0 : float
        Base temperature.
    dd_type : str
        Degree days type. Must be one of ``"heating"`` or ``"cooling"``.
    verbose : bool, optional
        Print the results, by default False.

    Returns
    -------
    float
        Mean squared error of the residuals.

    """
    degree_days = dd_compute(
        raw_temperature,
        t0,
        type=dd_type,
        method=dd_computation_method,
    )
    degree_days_resampled = (
        degree_days.resample(frequency).sum().rename("degree_days")
    )
    data = pd.concat([resampled_energy, degree_days_resampled], axis=1).dropna(
        how="any", axis=0
    )
    data["Intercept"] = 1
    if mask is not None:
        data = data[mask]
    model = OLS(
        data[resampled_energy.name], data[["degree_days", "Intercept"]]
    ).fit()
    if verbose:
        print(f"{t0=:.4f}, {model.mse_resid:.2f}, {model.mse_total:.2f}")
    return model.mse_resid


class CategoricalThermoSensitivity(
    ThermoSensitivity,
):
    """Class to compute the thermosensitivity of a building with labeled periods.
    Based on the `ThermoSensitivity` class.

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

        - "heating": compute only heating degree days (temperature below a threshold).
        - "cooling": compute only cooling degree days (temperature above a threshold).
        - "both": compute both heating and cooling degree days.
        - "auto": automatically detect the degree days type. See the `detect_degree_days_type` method.

    degree_days_base_temperature : dict, optional
        Base temperature for the computation of the degree days, by default {}.
        Must be a dictionary with the keys ``"heating"`` and/or ``"cooling"``.
        Example: degree_days_base_temperature={'heating': 18, 'cooling': 24}
    degree_days_computation_method : str, optional
        Method to compute the degree days, by default ``"integral"``. Possibilities are:

        - "integral": integral the degree days above or below the base temperature.
        - "mean": sum the difference between the base temperature and the mean temperature.
        - "min_max": sum the difference between the base temperature and the mean temperature
          computed as the mean of the minimum and maximum temperature.

    interseason_mean_temperature : float, optional
        Mean temperature to detect the heating and cooling periods, by default 20.
        Used only:

        - to detect the degree days type automatically. See the `detect_degree_days_type` method.
        - to estimate the base temperature. See the `calibrate_base_temperature` method.

    """

    categories_name = "category"

    def __init__(
        self,
        energy_data: pd.Series,
        temperature_data: pd.Series,
        categories: pd.Series,
        frequency="1D",
        degree_days_type="heating",
        degree_days_base_temperature: dict = {},
        degree_days_computation_method="integral",
        interseason_mean_temperature=20,
    ):
        self._categories = categories
        super().__init__(
            energy_data=energy_data,
            temperature_data=temperature_data,
            frequency=frequency,
            degree_days_type=degree_days_type,
            degree_days_base_temperature=degree_days_base_temperature,
            degree_days_computation_method=degree_days_computation_method,
            interseason_mean_temperature=interseason_mean_temperature,
        )

    @cached_property
    def resampled_categories(
        self,
    ) -> pd.Series:
        """Resampled categories at the given frequency.

        Return the most common category in the period.

        This property is cached to avoid recomputing it multiple times.
        """

        def category_resampler(series: pd.Series):
            """Return the most common category in the series"""
            try:
                return series.value_counts().idxmax()
            except ValueError:
                return None

        return (
            self.categories.resample(self.frequency)
            .apply(category_resampler)
            .rename(self.categories_name)
        )

    @property
    def categories(
        self,
    ):
        """The categories of the periods."""
        return self._categories

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
        degree_days_base_temperature: dict | None = None,
    ):
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
        significance_level=0.05,
        verbose=False,
    ):
        """Detect the degree days type if it is set to ``"auto"``.

        Manage the Simpson's paradox by computing the Spearman correlation for each category.
        If any category has a significant correlation, the degree days type will be set to ``"heating"`` or ``"cooling"`` or ``"both"``.

        See also
        --------
        :py:meth:`ThermoSensitivity._detect_degree_days_type`
        """
        if self.degree_days_type == "auto":
            heating_sp = 1
            cooling_sp = 1
            for cat in list(self.resampled_categories.unique()):
                if verbose:
                    print("f{cat=}")
                heating_mask = (
                    self.resampled_energy_temperature_category[
                        self.temperature_name
                    ]
                    < self.interseason_mean_temperature
                )
                cat_mask = (
                    self.resampled_energy_temperature_category[
                        self.categories_name
                    ]
                    == cat
                )
                tmp_heating_sp = spearmanr(
                    self.resampled_energy_temperature_category.loc[
                        heating_mask & cat_mask,
                        [self.target_name, self.temperature_name],
                    ],
                    alternative="less",
                )

                if tmp_heating_sp.pvalue < heating_sp:
                    heating_sp = tmp_heating_sp.pvalue
                cooling_mask = (
                    self.resampled_temperature
                    > self.interseason_mean_temperature
                )
                tmp_cooling_sp = spearmanr(
                    self.resampled_energy_temperature_category.loc[
                        cooling_mask & cat_mask,
                        [self.target_name, self.temperature_name],
                    ],
                    alternative="greater",
                )
                if verbose:
                    print(f"{tmp_cooling_sp}")
                if tmp_cooling_sp.pvalue < cooling_sp:
                    cooling_sp = tmp_cooling_sp.pvalue
            if (
                heating_sp < significance_level
                and cooling_sp < significance_level
            ):
                self.degree_days_type = "both"
            elif heating_sp < significance_level:
                self.degree_days_type = "heating"
            elif cooling_sp < significance_level:
                self.degree_days_type = "cooling"
            else:
                if verbose:
                    print(f"{cooling_sp=}, {heating_sp=}")
                raise ValueError(
                    "Cannot detect the degree days type. Please specify it manually."
                )

    def _fit_thermosensitivity(
        self,
    ):
        data = self.aggregated_data.dropna(how="any", axis=0)
        Y = data[self.target_name]
        X = data[self.predictors].copy()
        X["Intercept"] = 1  # add constant
        one_hot_encoding = pd.get_dummies(data["category"], dtype=int)
        # create interaction terms
        interactions = None
        for col in X.columns:
            for cat in one_hot_encoding.columns:
                tmp_int = X[col] * one_hot_encoding[cat]
                tmp_int.name = f"{col}:{cat}"
                if interactions is None:
                    interactions = tmp_int
                else:
                    interactions = pd.concat([interactions, tmp_int], axis=1)
        self._model = OLS(Y, interactions).fit()
