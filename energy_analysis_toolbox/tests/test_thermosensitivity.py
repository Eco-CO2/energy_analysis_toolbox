"""Test the Thermosensibility module."""

import pandas as pd
import numpy as np
import pytest
from energy_analysis_toolbox.thermosensitivity import (
    ThermoSensitivity,
    CategoricalThermoSensitivity,
    DailyCategoricalThermoSensitivity,
    DayOfWeekCategoricalThermoSensitivity,
    AutoCategoricalThermoSensitivity,
)
from energy_analysis_toolbox.synthetic.thermosensitive_consumption import (
    DateSynthTSConsumption,
    CategorySynthTSConsumption,
)


class TestThermoSensitivity:
    """ThermoSensitivity class tests.

    The tests are based on synthetic data generated with the DateSynthTSConsumption class.

    The tests asserts that, when the noise is low, we can recover the parameters of the model
    as expected. Includes

    - Calibration of the base temperature (heating, cooling, both)
    - Fitting the model (heating, cooling, both)
    - determining the degree days (heating, cooling, both) when set to auto


    """

    base_energy = 100
    ts_heating = 100
    ts_cooling = 100
    tref_heating = 16.5
    tref_cooling = 23
    noise_std = 0.0001
    interseason_mean_temperature = 20
    frequency = "1D"
    data_generated_size = 200
    expected_intercept_factor = 1

    def setup_method(self):
        self.synth_heating_only = DateSynthTSConsumption(
            base_energy=self.base_energy,
            t_ref_heat=self.tref_heating,
            ts_heat=self.ts_heating,
            ts_cool=0,
            noise_std=self.noise_std,
        )
        self.synth_cooling_only = DateSynthTSConsumption(
            base_energy=self.base_energy,
            t_ref_cool=self.tref_cooling,
            ts_heat=0,
            ts_cool=self.ts_cooling,
            noise_std=self.noise_std,
        )
        self.synth_both = DateSynthTSConsumption(
            base_energy=self.base_energy,
            t_ref_heat=self.tref_heating,
            t_ref_cool=self.tref_cooling,
            ts_heat=self.ts_heating,
            ts_cool=self.ts_cooling,
            noise_std=self.noise_std,
        )
        self.expected_intercept = (
            self.base_energy * self.expected_intercept_factor
        )

    def test_fit_only_heating_no_noise(self):
        """Test the ThermoSensitivity class."""
        data = self.synth_heating_only.random_consumption(
            size=self.data_generated_size
        )

        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="heating",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )

        ts.fit()
        assert ts.model.params["heating_degree_days"] == pytest.approx(
            self.ts_heating, rel=1e-1
        )
        assert ts.model.params["Intercept"] == pytest.approx(
            self.expected_intercept, rel=1e-1
        )
        with pytest.raises(KeyError):
            ts.model.params["cooling_degree_days"]

    def test_fit_only_cooling_no_noise(self):
        """Test the ThermoSensitivity class."""
        data = self.synth_cooling_only.random_consumption(
            size=self.data_generated_size, start="2022-07-01"
        )

        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={"cooling": self.tref_cooling},
            degree_days_computation_method="mean",
            degree_days_type="cooling",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )

        ts.fit()
        assert ts.model.params["cooling_degree_days"] == pytest.approx(
            self.ts_cooling, rel=1e-1
        )
        assert ts.model.params["Intercept"] == pytest.approx(
            self.expected_intercept, rel=1e-1
        )
        with pytest.raises(KeyError):
            ts.model.params["heating_degree_days"]

    def test_fit_heating_and_cooling_no_noise(self):
        """Test the ThermoSensitivity class."""
        data = self.synth_both.random_consumption(size=self.data_generated_size)

        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={
                "heating": self.tref_heating,
                "cooling": self.tref_cooling,
            },
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )

        ts.fit()
        assert ts.model.params["heating_degree_days"] == pytest.approx(
            self.ts_heating, rel=1e-1
        )
        assert ts.model.params["cooling_degree_days"] == pytest.approx(
            self.ts_cooling, rel=1e-1
        )

    def test_only_heating_no_noise_tref_calibration(self):
        """Test the ThermoSensitivity class."""
        data = self.synth_heating_only.random_consumption(
            size=self.data_generated_size
        )

        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="heating",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )

        ts.calibrate_base_temperatures()
        assert ts.degree_days_base_temperature["heating"] == pytest.approx(
            self.tref_heating, rel=1e-1
        )
        with pytest.raises(KeyError):
            ts.degree_days_base_temperature["cooling"]

    def test_only_cooling_no_noise_tref_calibration(self):
        """Test the ThermoSensitivity class."""
        data = self.synth_cooling_only.random_consumption(
            size=self.data_generated_size, start="2022-07-01"
        )

        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="cooling",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )

        ts.calibrate_base_temperatures()
        assert ts.degree_days_base_temperature["cooling"] == pytest.approx(
            self.tref_cooling, rel=1e-1
        )
        with pytest.raises(KeyError):
            ts.degree_days_base_temperature["heating"]

    def test_heating_and_cooling_no_noise_tref_calibration(self):
        """Test the ThermoSensitivity class."""
        data = self.synth_both.random_consumption(size=self.data_generated_size)
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )

        ts.calibrate_base_temperatures()
        assert ts.degree_days_base_temperature["heating"] == pytest.approx(
            self.tref_heating, rel=1e-1
        )
        assert ts.degree_days_base_temperature["cooling"] == pytest.approx(
            self.tref_cooling, rel=1e-1
        )

    def test_aggregted_data(self):
        data = self.synth_both.random_consumption(size=self.data_generated_size)
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )

        with pytest.raises(ValueError):
            ts.aggregated_data

        ts.degree_days_type = "heating"
        ts._aggregate_data({"heating": self.tref_heating})
        agg_data = ts.aggregated_data
        assert set(agg_data.columns) == {
            "energy",
            "temperature",
            "heating_degree_days",
        }

        ts.degree_days_type = "cooling"
        ts._aggregate_data({"cooling": self.tref_cooling})
        agg_data = ts.aggregated_data
        assert set(agg_data.columns) == {
            "energy",
            "temperature",
            "cooling_degree_days",
        }

        ts.degree_days_type = "both"
        ts._aggregate_data(
            {"heating": self.tref_heating, "cooling": self.tref_cooling}
        )
        agg_data = ts.aggregated_data
        assert set(agg_data.columns) == {
            "energy",
            "temperature",
            "heating_degree_days",
            "cooling_degree_days",
        }

    def test_model(self):
        data = self.synth_heating_only.random_consumption(
            size=self.data_generated_size
        )
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="heating",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )
        with pytest.raises(ValueError):
            ts.model
        ts.fit()
        assert ts.model is not None

    def test_init(self):
        data = self.synth_heating_only.random_consumption(
            size=self.data_generated_size
        )

        with pytest.raises(ValueError):
            ThermoSensitivity(
                energy_data=data["energy"],
                temperature_data=data["T"],
                degree_days_base_temperature={"heating": self.tref_heating},
                degree_days_computation_method="mean",
                degree_days_type="NOT_A_TYPE",
                interseason_mean_temperature=self.interseason_mean_temperature,
                frequency="1H",
            )

        with pytest.raises(ValueError):
            ThermoSensitivity(
                energy_data=data["energy"],
                temperature_data=data["T"],
                degree_days_base_temperature={"heating": self.tref_heating},
                degree_days_computation_method="mean",
                degree_days_type="cooling",
                interseason_mean_temperature=self.interseason_mean_temperature,
                frequency="1H",
            )

        with pytest.raises(ValueError):
            ThermoSensitivity(
                energy_data=data["energy"],
                temperature_data=data["T"],
                degree_days_base_temperature={"cooling": self.tref_cooling},
                degree_days_computation_method="mean",
                degree_days_type="heating",
                interseason_mean_temperature=self.interseason_mean_temperature,
                frequency="1H",
            )

    def test_post_init(self):
        data = self.synth_heating_only.random_consumption(
            size=self.data_generated_size
        )
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="auto",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )
        assert ts.degree_days_type == "heating"
        assert ts.predictors == ["heating_degree_days"]

        data = self.synth_cooling_only.random_consumption(
            size=self.data_generated_size, start="2022-07-01"
        )
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="auto",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )
        assert ts.degree_days_type == "cooling"
        assert ts.predictors == ["cooling_degree_days"]

        data = self.synth_both.random_consumption(size=self.data_generated_size)
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="auto",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )
        assert ts.degree_days_type == "both"
        assert ts.predictors == ["heating_degree_days", "cooling_degree_days"]

    def test_calculate_degree_days(self):
        data = self.synth_both.random_consumption(size=self.data_generated_size)
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )
        degree_days = ts._calculate_degree_days(
            {"heating": self.tref_heating, "cooling": self.tref_cooling}
        )
        assert isinstance(degree_days, pd.DataFrame)
        assert degree_days.index.equals(ts.resampled_temperature.index)
        assert set(degree_days.columns) == {
            "heating_degree_days",
            "cooling_degree_days",
        }
        degree_days = ts._calculate_degree_days({"heating": self.tref_heating})
        assert isinstance(degree_days, pd.DataFrame)
        assert degree_days.index.equals(ts.resampled_temperature.index)
        assert set(degree_days.columns) == {"heating_degree_days"}
        degree_days = ts._calculate_degree_days({"cooling": self.tref_cooling})
        assert isinstance(degree_days, pd.DataFrame)
        assert set(degree_days.columns) == {"cooling_degree_days"}
        assert degree_days.index.equals(ts.resampled_temperature.index)

    def test_calibrate_base_temperature(self):
        data = self.synth_both.random_consumption(size=self.data_generated_size)
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )
        tref = ts.calibrate_base_temperature(
            dd_type="heating",
            t0=12,
            xatol=1e-3,
        )
        assert tref == pytest.approx(self.tref_heating, rel=1e-1)
        tref = ts.calibrate_base_temperature(
            dd_type="cooling",
            t0=30,
            xatol=1e-3,
        )
        assert tref == pytest.approx(self.tref_cooling, rel=1e-1)
        with pytest.raises(ValueError):
            ts.calibrate_base_temperature(
                dd_type="both",
                t0=30,
                xatol=1e-3,
            )

    def test_calibrate_base_temperatures(self):
        data = self.synth_both.random_consumption(size=self.data_generated_size)
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )
        ts.calibrate_base_temperatures(
            t0_cooling=30,
            t0_heating=12,
            xatol=1e-3,
        )
        assert ts.degree_days_base_temperature["heating"] == pytest.approx(
            self.tref_heating, rel=1e-1
        )
        assert ts.degree_days_base_temperature["cooling"] == pytest.approx(
            self.tref_cooling, rel=1e-1
        )

        ts.degree_days_base_temperature = {}
        ts.degree_days_type = "heating"
        ts.calibrate_base_temperatures(
            t0_cooling=30,
            t0_heating=12,
            xatol=1e-3,
        )
        assert ts.degree_days_base_temperature["heating"] == pytest.approx(
            self.tref_heating, rel=1e-1
        )
        assert "cooling" not in ts.degree_days_base_temperature

        ts.degree_days_base_temperature = {}
        ts.degree_days_type = "cooling"
        ts.calibrate_base_temperatures(
            t0_cooling=30,
            t0_heating=12,
            xatol=1e-3,
        )
        assert ts.degree_days_base_temperature["cooling"] == pytest.approx(
            self.tref_cooling, rel=1e-1
        )
        assert "heating" not in ts.degree_days_base_temperature

    def test_repr(self):
        data = self.synth_both.random_consumption(size=self.data_generated_size)
        ts = ThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=self.interseason_mean_temperature,
            frequency=self.frequency,
        )
        assert isinstance(repr(ts), str)
        assert "ThermoSensitivity" in repr(ts)
        assert len(repr(ts).splitlines()) == 5
        ts.fit()
        assert len(repr(ts).splitlines()) == 22
        assert "heating_degree_days" in repr(ts)
        assert "cooling_degree_days" in repr(ts)
        assert "Intercept" in repr(ts)


class TestThermoSensitivityWeeks(TestThermoSensitivity):
    frequency = "7D"
    data_generated_size = 7 * 100
    expected_intercept_factor = 7


class TestCategoricalThermoSensitivity:
    parameters = [
        {"base_energy": 100, "ts_heat": 2, "ts_cool": 1, "noise_std": 0.0001},
        {"base_energy": 100, "ts_heat": 1, "ts_cool": 2, "noise_std": 0.0001},
    ]

    tref_heating = 16.5
    tref_cooling = 23

    @staticmethod
    def category_func(t_samples):
        """Categorise the samples in weekday or weekend.

        Parameters
        ----------
        t_samples : pd.Series
            Any time series

        Returns
        -------
        np.array
            The categories
        """
        return np.where(t_samples.index.dayofweek < 5, "weekday", "weekend")

    def setup_method(self):
        """Setup the test class."""
        self.synth = CategorySynthTSConsumption(
            category_func=self.category_func,
            parameters=self.parameters,
            t_ref_cool=self.tref_cooling,
            t_ref_heat=self.tref_heating,
            list_categories=["weekday", "weekend"],
        )

    def test_init(self):
        """Test that the class can be initialised."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        data["category"] = self.category_func(data["T"])
        CategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            categories=data["category"],
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="heating",
            interseason_mean_temperature=20,
        )

    def test_post_init(self):
        """Test the post init method.
        In particular, the capability to detect the degree days type when set to auto."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        data["category"] = self.category_func(data["T"])

        ts = CategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            categories=data["category"],
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="auto",
            interseason_mean_temperature=20,
        )
        assert ts.degree_days_type == "both"
        assert ts.predictors == ["heating_degree_days", "cooling_degree_days"]

    def test_fit(self):
        """Test the fit method."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        data["category"] = self.category_func(data["T"])

        ts = CategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            categories=data["category"],
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=20,
        )

        ts.fit()

        model = ts.model
        np.testing.assert_allclose(
            model.params["heating_degree_days:weekend"],
            self.parameters[1]["ts_heat"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["heating_degree_days:weekday"],
            self.parameters[0]["ts_heat"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["cooling_degree_days:weekend"],
            self.parameters[1]["ts_cool"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["cooling_degree_days:weekday"],
            self.parameters[0]["ts_cool"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["Intercept:weekday"],
            self.parameters[0]["base_energy"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["Intercept:weekend"],
            self.parameters[1]["base_energy"],
            rtol=1e-1,
        )


class TestDailyCategoricalThermoSensitivity:
    parameters = [
        {"base_energy": 100, "ts_heat": 2, "ts_cool": 1, "noise_std": 0.0001},
        {"base_energy": 100, "ts_heat": 1, "ts_cool": 2, "noise_std": 0.0001},
    ]

    tref_heating = 16.5
    tref_cooling = 23

    @staticmethod
    def category_func(t_samples):
        """Categorise the samples in weekday or weekend.

        Parameters
        ----------
        t_samples : pd.Series
            Any time series

        Returns
        -------
        np.array
            The categories
        """
        return np.where(t_samples.index.dayofweek < 5, "weekday", "weekend")

    @staticmethod
    def category_func_series(index):
        """Categorise the samples in weekday or weekend.

        Parameters
        ----------
        t_samples : pd.Series
            Any time series

        Returns
        -------
        np.array
            The categories
        """
        return pd.Series(
            np.where(index.dayofweek < 5, "weekday", "weekend"), index=index
        )

    def setup_method(self):
        """Setup the test class."""
        self.synth = CategorySynthTSConsumption(
            category_func=self.category_func,
            parameters=self.parameters,
            t_ref_cool=self.tref_cooling,
            t_ref_heat=self.tref_heating,
            list_categories=["weekday", "weekend"],
        )

    def test_init(self):
        """Test that the class can be initialised."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        DailyCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            categories_func=self.category_func_series,
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="heating",
            interseason_mean_temperature=20,
        )

    def test_post_init(self):
        """Test the post init method.
        In particular, the capability to detect the degree days type when set to auto."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        ts = DailyCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            categories_func=self.category_func_series,
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="auto",
            interseason_mean_temperature=20,
        )
        assert ts.degree_days_type == "both"
        assert ts.predictors == ["heating_degree_days", "cooling_degree_days"]

    def test_fit(self):
        """Test the fit method."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        ts = DailyCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            categories_func=self.category_func_series,
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=20,
        )

        ts.fit()

        model = ts.model
        np.testing.assert_allclose(
            model.params["heating_degree_days:weekend"],
            self.parameters[1]["ts_heat"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["heating_degree_days:weekday"],
            self.parameters[0]["ts_heat"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["cooling_degree_days:weekend"],
            self.parameters[1]["ts_cool"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["cooling_degree_days:weekday"],
            self.parameters[0]["ts_cool"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["Intercept:weekday"],
            self.parameters[0]["base_energy"],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            model.params["Intercept:weekend"],
            self.parameters[1]["base_energy"],
            rtol=1e-1,
        )


class TestDayOfWeekCategoricalThermoSensitivity:
    parameters = [
        {"base_energy": 100, "ts_heat": 1, "ts_cool": 2, "noise_std": 0.0001},
        {"base_energy": 100, "ts_heat": 2, "ts_cool": 4, "noise_std": 0.0001},
        {"base_energy": 100, "ts_heat": 3, "ts_cool": 6, "noise_std": 0.0001},
        {"base_energy": 100, "ts_heat": 4, "ts_cool": 8, "noise_std": 0.0001},
        {"base_energy": 100, "ts_heat": 5, "ts_cool": 10, "noise_std": 0.0001},
        {"base_energy": 100, "ts_heat": 6, "ts_cool": 12, "noise_std": 0.0001},
        {"base_energy": 100, "ts_heat": 7, "ts_cool": 14, "noise_std": 0.0001},
    ]

    tref_heating = 16.5
    tref_cooling = 23

    @staticmethod
    def category_func(t_samples):
        """Categorise the samples in weekday or weekend.

        Parameters
        ----------
        t_samples : pd.Series
            Any time series

        Returns
        -------
        np.array
            The categories
        """
        mapping = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
        return t_samples.index.dayofweek.map(mapping)

    def setup_method(self):
        """Setup the test class."""
        self.synth = CategorySynthTSConsumption(
            category_func=self.category_func,
            parameters=self.parameters,
            t_ref_cool=self.tref_cooling,
            t_ref_heat=self.tref_heating,
            list_categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )

    def test_init(self):
        """Test that the class can be initialised."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        DayOfWeekCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="heating",
            interseason_mean_temperature=20,
        )

    def test_post_init(self):
        """Test the post init method.
        In particular, the capability to detect the degree days type when set to auto."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        ts = DayOfWeekCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="auto",
            interseason_mean_temperature=20,
        )
        assert ts.degree_days_type == "both"
        assert ts.predictors == ["heating_degree_days", "cooling_degree_days"]

    def test_fit(self):
        """Test the fit method."""
        data: pd.DataFrame = self.synth.random_consumption(size=400)
        ts = DayOfWeekCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=20,
        )

        ts.fit()

        model = ts.model
        for idex, dayname in enumerate(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        ):
            np.testing.assert_allclose(
                model.params[f"heating_degree_days:{dayname}"],
                self.parameters[idex]["ts_heat"],
                rtol=1e-1,
            )
            np.testing.assert_allclose(
                model.params[f"cooling_degree_days:{dayname}"],
                self.parameters[idex]["ts_cool"],
                rtol=1e-1,
            )
            np.testing.assert_allclose(
                model.params[f"Intercept:{dayname}"],
                self.parameters[idex]["base_energy"],
                rtol=1e-1,
            )


class TestAutoCategoricalThermoSensitivity:
    parameters = [
        {
            "base_energy": 100,
            "ts_heat": 1,
            "ts_cool": 2,
            "noise_std": 0.00001,
        },  # class 1
        {
            "base_energy": 100,
            "ts_heat": 2,
            "ts_cool": 4,
            "noise_std": 0.00001,
        },  # class 2
        {
            "base_energy": 100,
            "ts_heat": 1,
            "ts_cool": 2,
            "noise_std": 0.00001,
        },  # class 1
        {
            "base_energy": 100,
            "ts_heat": 4,
            "ts_cool": 8,
            "noise_std": 0.00001,
        },  # class 3
        {
            "base_energy": 100,
            "ts_heat": 4,
            "ts_cool": 8,
            "noise_std": 0.00001,
        },  # class 3
        {
            "base_energy": 100,
            "ts_heat": 2,
            "ts_cool": 4,
            "noise_std": 0.00001,
        },  # class 2
        {
            "base_energy": 100,
            "ts_heat": 1,
            "ts_cool": 2,
            "noise_std": 0.00001,
        },  # class 1
    ]

    tref_heating = 16.5
    tref_cooling = 23

    @staticmethod
    def category_func(t_samples):
        """Categorise the samples in weekday or weekend.

        Parameters
        ----------
        t_samples : pd.Series
            Any time series

        Returns
        -------
        np.array
            The categories
        """
        mapping = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
        return t_samples.index.dayofweek.map(mapping)

    def setup_method(self):
        """Setup the test class."""
        self.synth = CategorySynthTSConsumption(
            category_func=self.category_func,
            parameters=self.parameters,
            t_ref_cool=self.tref_cooling,
            t_ref_heat=self.tref_heating,
            list_categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )

    def test_init(self):
        """Test that the class can be initialised."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        AutoCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="heating",
            interseason_mean_temperature=20,
        )

    def test_post_init(self):
        """Test the post init method.
        In particular, the capability to detect the degree days type when set to auto."""
        data: pd.DataFrame = self.synth.random_consumption(size=200)
        ts = AutoCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={"heating": self.tref_heating},
            degree_days_computation_method="mean",
            degree_days_type="auto",
            interseason_mean_temperature=20,
        )
        assert ts.degree_days_type == "both"
        assert ts.predictors == ["heating_degree_days", "cooling_degree_days"]

    def test_fit(self):
        """Test the fit method."""
        data: pd.DataFrame = self.synth.random_consumption(size=600)
        ts = AutoCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={},
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=20,
        )

        ts.fit()

        model = ts.model
        for idex, dayname in enumerate(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        ):
            np.testing.assert_allclose(
                model.params[f"heating_degree_days:{dayname}"],
                self.parameters[idex]["ts_heat"],
                rtol=2e-1,
            )
            np.testing.assert_allclose(
                model.params[f"cooling_degree_days:{dayname}"],
                self.parameters[idex]["ts_cool"],
                rtol=2e-1,
            )
            np.testing.assert_allclose(
                model.params[f"Intercept:{dayname}"],
                self.parameters[idex]["base_energy"],
                rtol=2e-1,
            )

    def test_merge_and_fit(self):
        """Test the fit method."""
        data: pd.DataFrame = self.synth.random_consumption(size=800)
        ts = AutoCategoricalThermoSensitivity(
            energy_data=data["energy"],
            temperature_data=data["T"],
            degree_days_base_temperature={
                "heating": self.tref_heating,
                "cooling": self.tref_cooling,
            },
            degree_days_computation_method="mean",
            degree_days_type="both",
            interseason_mean_temperature=20,
        )

        ts.fit()
        ts.merge_and_fit(signicant_level=0.01)

        model = ts.model
        daynames = [
            "Monday-Wednesday-Sunday",
            "Tuesday-Saturday",
            "Thursday-Friday",
        ]
        indexes = [0, 1, 3]
        print(model.params)
        for idex, dayname in zip(indexes, daynames):
            np.testing.assert_allclose(
                model.params[f"heating_degree_days:{dayname}"],
                self.parameters[idex]["ts_heat"],
                rtol=2e-1,
            )
            np.testing.assert_allclose(
                model.params[f"cooling_degree_days:{dayname}"],
                self.parameters[idex]["ts_cool"],
                rtol=2e-1,
            )
            np.testing.assert_allclose(
                model.params[f"Intercept:{dayname}"],
                self.parameters[idex]["base_energy"],
                rtol=2e-1,
            )
