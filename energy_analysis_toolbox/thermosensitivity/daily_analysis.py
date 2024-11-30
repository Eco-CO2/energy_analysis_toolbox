"""Analyze daily-sampled thermosensitivity data."""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from energy_analysis_toolbox.weather.degree_days import (
    literal_computation_dd_types,
    literal_dd_types,
)

from .thermosensitivity import CategoricalThermoSensitivity


class DailyCategoricalThermoSensitivity(
    CategoricalThermoSensitivity,
):
    """Class for daily analysis of thermosensitivity data.

    Based on CategoricalThermoSensitivity, it is made to categorize the days.

    Example:
    --------
    See :py:class:`DayOfWeekCategoricalThermoSensitivity`

    """

    def __init__(
        self,
        energy_data: pd.Series,
        temperature_data: pd.Series,
        categories_func: Callable[[pd.DatetimeIndex], pd.Series],
        degree_days_type: literal_dd_types = "heating",
        degree_days_base_temperature: dict | None = None,
        degree_days_computation_method: literal_computation_dd_types = "integral",
        interseason_mean_temperature: float = 20,
        base_logger_name: str | None = None,
        min_logger_level_stdout: int | str = logging.ERROR,
    ) -> None:
        """Initialize a ``DailyCategoricalThermoSensitivity`` instance.

        Parameters
        ----------
        energy_data : pd.Series
            Time series of energy consumption data for the building.
        temperature_data : pd.Series
            Time series of outdoor temperature data.
        categories_func : Callable[[pd.DatetimeIndex], pd.Series]
            A function that takes a ``pd.DatetimeIndex`` (representing days in the
            analysis period) and returns a ``pd.Series`` that assigns a category to
            each day. This allows the analysis to group energy data by custom-defined
            categories (e.g., weekdays vs. weekends, seasons).
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

        """
        frequency = "1D"
        degree_days_base_temperature = degree_days_base_temperature or {}
        start_ts = min(energy_data.index.min(), temperature_data.index.min())
        end_ts = max(energy_data.index.max(), temperature_data.index.max())
        days = pd.date_range(
            start=start_ts,
            end=end_ts,
            freq=frequency,
            inclusive="both",
        )
        categories = categories_func(days)
        super().__init__(
            energy_data=energy_data,
            temperature_data=temperature_data,
            categories=categories,
            frequency=frequency,
            degree_days_type=degree_days_type,
            degree_days_base_temperature=degree_days_base_temperature,
            degree_days_computation_method=degree_days_computation_method,
            interseason_mean_temperature=interseason_mean_temperature,
            base_logger_name=base_logger_name,
            min_logger_level_stdout=min_logger_level_stdout,
        )


class DayOfWeekCategoricalThermoSensitivity(
    DailyCategoricalThermoSensitivity,
):
    """Models independently the 7 days of the week.

    Based on :py:class:`DailyCategoricalThermoSensitivity`.
    """

    def __init__(
        self,
        energy_data: pd.Series,
        temperature_data: pd.Series,
        degree_days_type: literal_dd_types = "heating",
        degree_days_base_temperature: dict | None = None,
        degree_days_computation_method: literal_computation_dd_types = "integral",
        interseason_mean_temperature: float = 20,
        base_logger_name: str | None = None,
        min_logger_level_stdout: int | str = logging.ERROR,
    ) -> None:
        """Initialize a ``DayOfWeekCategoricalThermoSensitivity`` instance.

        Parameters
        ----------
        energy_data : pd.Series
            Time series of energy consumption data for the building.
        temperature_data : pd.Series
            Time series of outdoor temperature data.
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

        """
        degree_days_base_temperature = degree_days_base_temperature or {}

        def day_of_week_categoriser(
            index: pd.DatetimeIndex,
        ) -> pd.Series:
            """Return a series of categories based on the day of the week of the index.

            Parameters
            ----------
            index : pd.DatetimeIndex
                A datetime index representing the dates to categorize.

            Returns
            -------
            pd.Series
                A pandas Series where each entry is the name of the day of the week
                corresponding to the respective index value.

            Examples
            --------
            >>> index = pd.date_range(start="2023-01-01", periods=7, freq="D")
            >>> day_of_week_categoriser(index)
            2023-01-01       Sunday
            2023-01-02       Monday
            2023-01-03      Tuesday
            2023-01-04    Wednesday
            2023-01-05     Thursday
            2023-01-06       Friday
            2023-01-07     Saturday
            Freq: D, dtype: object

            """
            return pd.Series(index=index, data=index.day_name())

        super().__init__(
            energy_data=energy_data,
            temperature_data=temperature_data,
            categories_func=day_of_week_categoriser,
            degree_days_type=degree_days_type,
            degree_days_base_temperature=degree_days_base_temperature,
            degree_days_computation_method=degree_days_computation_method,
            interseason_mean_temperature=interseason_mean_temperature,
            base_logger_name=base_logger_name,
            min_logger_level_stdout=min_logger_level_stdout,
        )


class AutoCategoricalThermoSensitivity(
    DayOfWeekCategoricalThermoSensitivity,
):
    """Automatically categorizes thermosensitivity data based on predefined criteria.

    Based on :py:class:`DayOfWeekCategoricalThermoSensitivity`.
    """

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
        """Set the categories and reset associated cached data.

        This method assigns new category labels to the ``categories`` attribute and
        resets the cached data that depends on these categories, ensuring that all
        computations are up to date with the new categorization.

        Parameters
        ----------
        value : pd.Series
            A pandas Series representing the new categories to assign.

        .. note::
            Setting new categories automatically resets the following internal
            attributes:

            - ``resampled_energy_temperature_category``: Cached energy-temperature data
            that is resampled by category.
            - ``resampled_categories``: Cached resampled category values.
            - ``_aggregated_data``: Cached aggregated data, if any.

            These attributes are recalculated upon the next request, ensuring
            consistency with the newly assigned categories.

        """
        self._categories = value
        self.__dict__.pop("resampled_energy_temperature_category", None)
        self.__dict__.pop("resampled_categories", None)
        self._aggregated_data = None

    def new_categories(
        self,
        significant_level: float = 0.1,
    ) -> dict:
        """Return new category mappings based on interaction term significance.

        This method identifies significant differences between the thermosensitivity
        of each category and provides new mappings that merge similar categories.
        Categories are grouped based on the significance of interaction terms
        in the model, which helps reduce complexity while retaining meaningful
        distinctions.

        Parameters
        ----------
        significant_level : float, optional
            The significance level for the Wald test (a p-value below this level
            is considered significant). Must be between 0 and 1. The higher the
            value, the more categories will be kept separate. Lower values will
            merge categories that are not significantly different.

        Returns
        -------
        dict
            A dictionary mapping old categories to new merged categories.
            The new labels are concatenated with a "-" separator to indicate
            merged groups.

        Notes
        -----
        - The new categories are based on the result of multiple Wald tests conducted
          between interaction terms for each category.
        - The returned dictionary allows for updating the category labels to reflect
          merged groupings that exhibit similar behavior.


        Example
        -------
        >>> auto = AutoCategoricalThermoSensitivity(...)
        >>> auto.fit()
        >>> auto.new_categories(significant_level=0.1)
        {'Monday': 'Monday-Wednesday-Sunday',
         'Tuesday': 'Tuesday',
         'Wednesday': 'Monday-Wednesday-Sunday',
         'Thursday': 'Thursday',
         'Friday': 'Friday',
         'Saturday': 'Monday-Wednesday-Sunday',
         'Sunday': 'Sunday'
        }

        """
        categories_sorted = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        categories = self.resampled_categories.unique()
        predictors = [*self.predictors, "Intercept"]
        new_categories_mapping = {str(term): [str(term)] for term in categories}
        for i, cat_term1 in enumerate(categories):
            for _, cat_term2 in enumerate(categories[i + 1 :]):
                is_same_group = True
                for pred in predictors:
                    interaction_term1 = pred + ":" + cat_term1
                    interaction_term2 = pred + ":" + cat_term2
                    contrast_matrix = np.zeros((1, len(self.model.params)))
                    contrast_matrix[
                        0,
                        self.model.params.index.get_loc(interaction_term1),
                    ] = 1
                    contrast_matrix[
                        0,
                        self.model.params.index.get_loc(interaction_term2),
                    ] = -1
                    wald_test = self.model.wald_test(contrast_matrix, scalar=True)
                    if wald_test.pvalue < significant_level:
                        is_same_group &= False
                if is_same_group:
                    new_categories_mapping[cat_term1].append(cat_term2)
                    new_categories_mapping[cat_term2] = new_categories_mapping[
                        cat_term1
                    ]
        reduced_mapping = {
            k: sorted(set(v), key=lambda d: categories_sorted.index(d))
            for k, v in new_categories_mapping.items()
        }
        return {k: "-".join(v) for k, v in reduced_mapping.items()}

    def merge_and_fit(
        self,
        significant_level: float = 0.1,
    ) -> None:
        """Merge similar categories and fit the model with updated categories.

        This method merges categories that exhibit similar thermosensitivity based
        on their interaction term significance and then refits the model using
        the updated categorization. This helps reduce model complexity by grouping
        categories with similar behavior.

        Parameters
        ----------
        significant_level : float, optional
            The significance level for the Wald test (a p-value below this level
            is considered significant). Must be between 0 and 1. The higher the value,
            the more categories will be kept separate. Lower values will merge
            categories that are not significantly different.

        Returns
        -------
        None

        Notes
        -----
        - The method first calculates new categories using the ``new_categories``
          method, then assigns these new categories and fits the model to reflect the
          updated categorization.
        - This process is especially useful for reducing overfitting by combining
          similar days (e.g., merging weekdays that do not show significant
          thermosensitivity differences).

        """
        new_cats_maps = self.new_categories(significant_level=significant_level)
        self.categories = self.categories.map(new_cats_maps)
        self.fit()
