"""Test the synthtic data generators"""
import pytest
import pandas as pd
import numpy as np
from  ..synthetic import (
    SynthDJUConsumption,
    SynthTSConsumption,
    DateSynthTSConsumption,
)


class TestSynthDJUConsumption:

    base_energry=1000
    ts_slope=10
    noise_std=5
    noise_seed=42
    expected_columns = ["DJU", "energy", "thermosensitive", "base", "residual"]

    @pytest.fixture
    def setup(self) -> SynthDJUConsumption:
        # Initialize your SynthDJUConsumption object here
        synth_dju_consumption = SynthDJUConsumption(
            base_energy = self.base_energry,
            ts_slope = self.ts_slope,
            noise_std = self.noise_std
        )
        return synth_dju_consumption

    def reset_seed(self, synth):
        # resetting the seed
        synth._rng = np.random.default_rng(seed=self.noise_seed)

    def test_init(self):
        """Test the initialisation"""
        synth_dju_consumption = SynthDJUConsumption(
            base_energy = self.base_energry,
            ts_slope = self.ts_slope,
            noise_std = self.noise_std
        )

    def test_random_djus(self, setup: SynthDJUConsumption):
        synth_dju_consumption = setup
        size = 45
        start = "2024-04-18"
        djus = synth_dju_consumption.random_djus(size=size,
                                            start=start)
        assert isinstance(djus, pd.Series)
        assert len(djus) == size
        assert djus.index[0] == pd.Timestamp(start)
        assert djus.index.freq == "D"

        # test that start can be a pd.Timestamp
        djus = synth_dju_consumption.random_djus(start=pd.Timestamp(start))
        assert djus.index[0] == pd.Timestamp(start)
        # test that start can be a Date
        djus = synth_dju_consumption.random_djus(start= pd.Timestamp(start).date())
        assert djus.index[0] == pd.Timestamp(start)

    def test_random_consumption(self, setup: SynthDJUConsumption):
        synth_dju_consumption = setup
        size = 45
        start = "2024-04-18"
        consumption = synth_dju_consumption.random_consumption(size=size,
                                            start=start)
        assert isinstance(consumption, pd.DataFrame)
        assert len(consumption) == size

        for column in self.expected_columns:
            assert column in consumption.columns
        assert consumption.index[0] == pd.Timestamp(start)
        assert consumption.index.freq == "D"
        assert all(consumption["base"] == self.base_energry)
        # generate a large sample to test the noise std
        consumption = synth_dju_consumption.random_consumption(size=10_000)
        assert abs(consumption["residual"].std() - self.noise_std) < 0.05 # 1% error


    def test_measures(self, setup: SynthDJUConsumption):
        synth_dju_consumption = setup
        measures = synth_dju_consumption.measures()
        self.reset_seed(synth_dju_consumption)
        consumption = synth_dju_consumption.random_consumption()
        pd.testing.assert_frame_equal(measures, consumption)


class TestSynthTSConsumption(TestSynthDJUConsumption):
    """Test the synthtic data generators"""
    base_energry=1000
    ts_heat=10
    ts_cool=5
    t_ref_heat=15
    t_ref_cool=25
    noise_std=5
    noise_seed=42
    expected_columns = ["DJU_heating",
                        "DJU_cooling",
                        "energy",
                        "thermosensitive",
                        "base",
                        "residual"]


    @pytest.fixture
    def setup(self) -> SynthTSConsumption:
        # Initialize your SynthTSConsumption object here
        synth_ts_consumption = SynthTSConsumption(
            base_energy = self.base_energry,
            ts_heat = self.ts_heat,
            ts_cool = self.ts_cool,
            t_ref_heat = self.t_ref_heat,
            t_ref_cool = self.t_ref_cool,
            noise_std = self.noise_std
        )
        return synth_ts_consumption

    def reset_seed(self, synth):
        # resetting the seed
        synth._rng = np.random.default_rng(seed=self.noise_seed)
        synth.heating._rng = np.random.default_rng(seed=self.noise_seed)
        synth.cooling._rng = np.random.default_rng(seed=self.noise_seed)

    def test_random_djus(self, setup: SynthTSConsumption):
        synth_dju_consumption = setup
        size = 45
        start = "2024-04-18"
        djus = synth_dju_consumption.random_djus(size=size,
                                            start=start)
        assert isinstance(djus, pd.DataFrame)
        assert len(djus) == size
        assert djus.index[0] == pd.Timestamp(start)
        assert djus.index.freq == "D"

        # test that start can be a pd.Timestamp
        djus = synth_dju_consumption.random_djus(start=pd.Timestamp(start))
        assert djus.index[0] == pd.Timestamp(start)
        # test that start can be a Date
        djus = synth_dju_consumption.random_djus(start= pd.Timestamp(start).date())
        assert djus.index[0] == pd.Timestamp(start)

class TestDateTimeSynthTSConsumption(TestSynthTSConsumption):

    @pytest.fixture
    def setup(self) -> DateSynthTSConsumption:
        # Initialize your SynthTSConsumption object here
        synth_ts_consumption = DateSynthTSConsumption(
            base_energy = self.base_energry,
            ts_heat = self.ts_heat,
            ts_cool = self.ts_cool,
            t_ref_heat = self.t_ref_heat,
            t_ref_cool = self.t_ref_cool,
            noise_std = self.noise_std
        )
        return synth_ts_consumption
