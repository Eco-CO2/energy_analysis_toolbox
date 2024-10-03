"""This module contains functions for analysis of daily-sampled thermosensitivity data."""

from typing import Callable
import pandas as pd
import numpy as np
from .thermosensitivity import CategoricalThermoSensitivity


class DailyCategoricalThermoSensitivity(
    CategoricalThermoSensitivity,
):
    """Class for daily analysis of thermosensitivity data.

    Based on CategoricalThermoSensitivity, it is made to
    categories the days with a function.

    Example
    -------
    See :py:class:`DayOfWeekCategoricalThermoSensitivity`

    """

    def __init__(
        self,
        energy_data: pd.Series,
        temperature_data: pd.Series,
        categories_func: Callable[[pd.DatetimeIndex], pd.Series],
        degree_days_type="heating",
        degree_days_base_temperature: dict = {},
        degree_days_computation_method="integral",
        interseason_mean_temperature=20,
    ):
        frequency = "1D"
        start_ts = min(energy_data.index.min(), temperature_data.index.min())
        end_ts = max(energy_data.index.max(), temperature_data.index.max())
        days = pd.date_range(
            start=start_ts, end=end_ts, freq=frequency, inclusive="both"
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
        )


class DayOfWeekCategoricalThermoSensitivity(
    DailyCategoricalThermoSensitivity,
):
    """Models independantly the 7 days of the week.

    Based on :py:class:`DailyCategoricalThermoSensitivity`.
    """

    def __init__(
        self,
        energy_data: pd.Series,
        temperature_data: pd.Series,
        degree_days_type="heating",
        degree_days_base_temperature: dict = {},
        degree_days_computation_method="integral",
        interseason_mean_temperature=20,
    ):
        def day_of_week_categoriser(
            series: pd.DatetimeIndex,
        ):
            """Return a series of categories based on the day of the week of the index"""
            return pd.Series(
                data=[
                    "Monday"
                    if timestamp.weekday() == 0
                    else "Tuesday"
                    if timestamp.weekday() == 1
                    else "Wednesday"
                    if timestamp.weekday() == 2
                    else "Thursday"
                    if timestamp.weekday() == 3
                    else "Friday"
                    if timestamp.weekday() == 4
                    else "Saturday"
                    if timestamp.weekday() == 5
                    else "Sunday"
                    for timestamp in series
                ],
                index=series,
            )

        super().__init__(
            energy_data=energy_data,
            temperature_data=temperature_data,
            categories_func=day_of_week_categoriser,
            degree_days_type=degree_days_type,
            degree_days_base_temperature=degree_days_base_temperature,
            degree_days_computation_method=degree_days_computation_method,
            interseason_mean_temperature=interseason_mean_temperature,
        )


class AutoCategoricalThermoSensitivity(
    DayOfWeekCategoricalThermoSensitivity,
):
    @DayOfWeekCategoricalThermoSensitivity.categories.setter
    def categories(
        self,
        value,
    ):
        """Set the categories and reset ``resampled_energy_temperature_category``."""
        self._categories = value
        self.__dict__.pop("resampled_energy_temperature_category", None)
        self.__dict__.pop("resampled_categories", None)
        self._aggregated_data = None

    def new_categories(
        self,
        signicant_level=0.1,
    ):
        """Return a new categories mapping based on the significance of the interaction terms.

        The new categories labels are concatenated with a "-" separator.

        Parameters
        ----------
        signicant_level : float
            The significance level for the Wald test (a p-value below this level is considered significant).
            Must be between 0 and 1.
            The higher the value, the more categories will be kept separate.
            Lower values will merge categories that are not significantly different.

        Returns
        -------
        dict
            A mapping of the new categories.

        Example
        -------
        >>> auto = AutoCategoricalThermoSensitivity(...)
        >>> auto.fit()
        >>> auto.new_categories(signicant_level=0.1)
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
        predictors = self.predictors + ["Intercept"]
        new_categories_mapping = {str(term): [str(term)] for term in categories}
        for i, cat_term1 in enumerate(categories):
            for j, cat_term2 in enumerate(categories[i + 1 :]):
                is_same_group = True
                for pred in predictors:
                    interaction_term1 = pred + ":" + cat_term1
                    interaction_term2 = pred + ":" + cat_term2
                    contrast_matrix = np.zeros((1, len(self.model.params)))
                    contrast_matrix[
                        0, self.model.params.index.get_loc(interaction_term1)
                    ] = 1
                    contrast_matrix[
                        0, self.model.params.index.get_loc(interaction_term2)
                    ] = -1
                    wald_test = self.model.wald_test(
                        contrast_matrix, scalar=True
                    )
                    if wald_test.pvalue < signicant_level:
                        is_same_group &= False
                if is_same_group:
                    new_categories_mapping[cat_term1].append(cat_term2)
                    new_categories_mapping[cat_term2] = new_categories_mapping[
                        cat_term1
                    ]
        reduced_mapping = {
            k: sorted(list(set(v)), key=lambda d: categories_sorted.index(d))
            for k, v in new_categories_mapping.items()
        }
        return {k: "-".join(v) for k, v in reduced_mapping.items()}

    def merge_and_fit(
        self,
        signicant_level=0.1,
    ):
        new_cats_maps = self.new_categories(signicant_level=signicant_level)
        self.categories = self.categories.map(new_cats_maps)
        self.fit()
