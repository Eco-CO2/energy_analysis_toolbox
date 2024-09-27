"""Provides classes to generate fake energy consumption data following
a thermo-sensitive model."""

from typing import (
    Callable,
    TypedDict,
)
import numpy as np
import pandas as pd

# change the default display options for the doctest
pd.options.display.max_columns = 10
pd.options.display.width = 256


class SynthDDConsumption:
    """This class can be used to generate synthetic thermosensitive energy consumption.

    The generated consumption is split into a base, a thermosensitive and a residual
    contributions.

    .. note::

        Only one DD can be used at a time to generate the energy consumption.
        For multiple DDs, use the :py:class:`FakeTSConsumption` class.


    Example
    -------
    >>> synth_consumption = SynthDDConsumption(base_energy=100, ts_slope=0.1, noise_std=5)
    >>> synth_consumption.random_consumption(size=5)
                     DD  base  thermosensitive  residual      energy
    2022-11-01  7.212626   100         0.721263 -6.510898   94.210365
    2022-11-02  7.008569   100         0.700857  0.639202  101.340059
    2022-11-03  7.154283   100         0.715428 -1.581213   99.134215
    2022-11-04  0.839383   100         0.083938 -0.084006   99.999932
    2022-11-05  0.259312   100         0.025931 -4.265220   95.760712

    """

    def __init__(
        self,
        base_energy,
        ts_slope,
        noise_std,
        noise_seed=42,
        exp_scale_dd=3.0,
        clip_negative=True,
    ):
        """Return a ``FakeDDConsumption`` instance.

        Parameters
        ----------
        base_energy : float
            The value of the averaged non thermosensitive energy consumption.
        ts_slope : float
            The value of the consumption thermosensitivity given in unit of
            energy (same as base) per degree-day.
        noise_std : float
            The standard deviation of the gaussian noise added to the base
            consumption.
        noise_seed : float, optional
            Seed value for the random number generator bound to ``self``.
        exp_scale_dd : float, optional
            Scale (mean) parameter of the exponential distribution used to generate
            fake DD samples. Default is 5°C.
        clip_negative : bool, default : True
            If True, the energy is clipped so that it cannot be below 0.
        """
        self.base_energy = base_energy
        self.ts_slope = ts_slope
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed=noise_seed)
        self.exp_scale_dd = exp_scale_dd
        self.clip_negative = clip_negative

    def random_dds(
        self,
        size=100,
        start="2022-11-01",
    ):
        """Return realistic DD samples.

        Parameters
        ----------
        size : int, default : 100
            Number of days in the sample.
        start : pd.Timestamp or alike, optional
            First day in the generated sample.

        Returns
        -------
        pd.Series :
            A time-series with 1 day period containing randomly
            generated DD values.

        .. warning::

            The date provided in the sample is not consistent
            with the DD values.
            Still, the distribution of the DDs remain correct
            when using large enough number of samples.


        Notes
        -----
        The samples are drawn from an exponential law.

        Example
        -------
        >>> synth_consumption = SynthDDConsumption(base_energy=100, ts_slope=0.1, noise_std=5)
        >>> synth_consumption.random_dds(size=5)
        2022-11-01    7.212626
        2022-11-02    7.008569
        2022-11-03    7.154283
        2022-11-04    0.839383
        2022-11-05    0.259312
        Freq: D, Name: DD, dtype: float64

        """
        fake_dds = pd.Series(
            self._rng.exponential(
                scale=self.exp_scale_dd,
                size=size,
            ),
            index=pd.date_range(start=start, periods=size, freq="1D"),
            name="DD",
        )
        return fake_dds

    def random_consumption(
        self,
        size=100,
        start="2022-11-01",
    ):
        """Return a random decomposed energy consumption.

        Parameters
        ----------
        size : int, default : 100
            Number of days in the sample.
        start : pd.Timestamp or alike, optional
            First day in the generated sample.

        Returns
        -------
        pd.DataFrame :
            The table of fake consumption, as described in :py:meth:`.fake_energy`.


        .. warning::

            The date provided in the sample is not consistent
            with the DD values.
            Still, the distribution of the DDs remain correct
            when using large enough number of samples.

        Example
        -------
        >>> synth_consumption = SynthDDConsumption(base_energy=100, ts_slope=2, noise_std=5)
        >>> synth_consumption.random_consumption(size=5)
                         DD  base  thermosensitive  residual      energy
        2022-11-01  7.212626   100        14.425252 -6.510898  107.914354
        2022-11-02  7.008569   100        14.017138  0.639202  114.656340
        2022-11-03  7.154283   100        14.308566 -1.581213  112.727353
        2022-11-04  0.839383   100         1.678766 -0.084006  101.594760
        2022-11-05  0.259312   100         0.518624 -4.265220   96.253405

        """
        return self.fake_energy(self.random_dds(size=size, start=start))

    def fake_energy(
        self,
        dd_samples,
    ):
        """Return a fake energy consumption for each day in the DD samples.

        Parameters
        ----------
        dd_samples : pd.Series
            A timeseries of DDs to be used to infer the energy consumption.

        Returns
        -------
        pd.DataFrame :
            A table with rows labeled by ``dd_samples`` index and the
            following columns :

            - ``DD``: the ``dd_samples`` series.
            - ``energy``: the energy consumption for each raw in the table.
            - ``thermosensitive``: the value of the thermosensitive energy
              consumption for each period.
            - ``base``: the value of the averaged non-thermosensitive consumption.
              This value is constant across the table, equal to ``self.base_energy``.
            - ``residual``: the energy-noise, i.e. the residual between the
              affine model and the actual energy for each period.

        Notes
        -----
        The energy is generated assuming that the energy consumption satisfies
        the following equation:

        .. math::

            E = \\Theta. DD + E_{base} + \\epsilon

        Where :math:`\\epsilon` is a gaussian, centered, random variable, which
        standard deviation is ``self.noise_std``.

        Example
        -------
        >>> synth_consumption = SynthDDConsumption(base_energy=100, ts_slope=2, noise_std=5)
        >>> dds = pd.Series(data=[0,2,12])
        >>> synth_consumption.fake_energy(dds)
           DD  base  thermosensitive  residual      energy
        0    0   100                0  1.523585  101.523585
        1    2   100                4 -5.199921   98.800079
        2   12   100               24  3.752256  127.752256

        """
        fake_data = pd.DataFrame(
            np.empty((dd_samples.shape[0], 5)),
            index=dd_samples.index,
            columns=["DD", "base", "thermosensitive", "residual", "energy"],
        )
        fake_data["DD"] = dd_samples
        fake_data["thermosensitive"] = dd_samples * self.ts_slope
        fake_data["base"] = self.base_energy
        fake_data["residual"] = self._rng.normal(
            loc=0.0,
            scale=self.noise_std,
            size=dd_samples.size,
        )
        assembled_energy = fake_data.loc[
            :, ["base", "thermosensitive", "residual"]
        ].sum(axis=1)
        if self.clip_negative:
            assembled_energy = assembled_energy.clip(lower=0.0)
            fake_data["energy"] = assembled_energy
            fake_data["residual"] = fake_data["energy"] - fake_data.loc[
                :, ["base", "thermosensitive"]
            ].sum(axis=1)
        else:
            fake_data["energy"] = assembled_energy
        return fake_data

    def measures(
        self,
        *args,
        **kwargs,
    ):
        """Return a fake energy consumption  decomposition VS temperature.

        This method is a wrapper around :py:meth:`.random_consumption` to keep the same
        signature as the Handlers.

        See :py:meth:`random_consumption`

        """
        return self.random_consumption(*args, **kwargs)


class SynthTSConsumption:
    """A class to generate fake energy consumptions as a function of the
    temperature. Based on :py:class:`FakeDDConsumption`, including both heating
    and cooling domains.

    The generation relies on the assumption of linear DD dependencies in the
    heating and cooling domains.

    Example
    -------

    >>> synth_consumption = SynthTSConsumption(base_energy=100, ts_heat=2, ts_cool=3, noise_std=5)
    >>> synth_consumption.random_consumption(size=5, t_mean=20, t_std=20)
                base  thermosensitive  residual      energy    heating    cooling          T  DD_heating  DD_cooling
    2022-11-01   100        18.283025 -6.510898  111.772127   0.000000  18.283025  26.094342     0.000000     6.094342
    2022-11-02   100        35.599364  0.639202  136.238566  35.599364   0.000000  -0.799682    17.799682     0.000000
    2022-11-03   100        45.027072 -1.581213  143.445859   0.000000  45.027072  35.009024     0.000000    15.009024
    2022-11-04   100        56.433883 -0.084006  156.349877   0.000000  56.433883  38.811294     0.000000    18.811294
    2022-11-05   100        72.041408 -4.265220  167.776188  72.041408   0.000000 -19.020704    36.020704     0.000000

    """

    def __init__(
        self,
        base_energy,
        ts_heat,
        ts_cool,
        t_ref_heat=17,
        t_ref_cool=20,
        noise_std=0,
        noise_seed=42,
    ):
        """

        Parameters
        ----------
        base_energy : float
            The value of the averaged non-thermosensitive consumption. Its unit
            can be anything consistent with which of ``ts_heat`` and ``ts_cool``
            when describing an energy.
        ts_heat : float
            The thermosensitivity of the consumption in the heating domain, i.e.
            under ``self.t_ref_heat``. Its has the dimension of one unit of
            ``base_energy`` per degree day.
        ts_cool : float
            The thermosensitivity of the consumption in the cooling domain, i.e.
            over ``self.t_ref_cool``. Same unit as ``ts_heat``.
        t_ref_heat : float, default : 17
            The reference temperature of the heating domain, i.e. the outdoor
            temperature under which the heating is assumed to start.
        t_ref_cool : float, default : 20
            The reference temperature of the cooling domain, i.e. the outdoor
            temperature over which the cooling is assumed to start.
        noise_std : float, default : 0.
            The standard deviation of the gaussian noise added to the affine per
            part model used to generate the energy consumption from the temperature.
        noise_seed : float, default : 42
            A seed for the random noise generator bound to ``self``.


        """
        self.heating = SynthDDConsumption(
            base_energy=0,
            ts_slope=ts_heat,
            noise_std=0,
            clip_negative=False,
            noise_seed=noise_seed,
        )
        self.cooling = SynthDDConsumption(
            base_energy=0,
            ts_slope=ts_cool,
            noise_std=0,
            clip_negative=False,
            noise_seed=noise_seed,
        )
        self.base_energy = base_energy
        self.noise_std = noise_std
        self.t_ref_heat = t_ref_heat
        self.t_ref_cool = t_ref_cool
        self._rng = np.random.default_rng(seed=noise_seed)

    def fake_energy(
        self,
        dd_heating,
        dd_cooling,
        t_samples,
    ):
        """Return a fake energy consumption depending on the daily temperatures
        passed in input.

        Parameters
        ----------
        dd_heating : pd.Series
            A series of DD. Usually daily aggregates, depending on the
            scale chosen for the thermosensitivity and base consumption values.
        dd_cooling : pd.Series
            A series of DD. Usually daily aggregates, depending on the
            scale chosen for the thermosensitivity and base consumption values.
        t_samples : pd.Series

        Returns
        -------
        pd.DataFrame :
            A table with rows labeled by ``dd_samples`` index and the
            following columns :

            - ``T`` : the ``t_samples`` series.
            - ``energy`` : the energy consumption for each raw in the table.
            - ``thermosensitive`` : the value of the thermosensitive energy
              consumption for each period.
            - ``base`` : the value of the averaged non-thermosensitive consumption.
              This value is constant across the table, equal to ``self.base_energy``.
            - ``residual`` : the energy-noise, i.e. the residual between the
              affine model and the actual energy for each period.

        Notes
        -----
        The fake energy generation relies on two instances of :py:class:`SynthDDConsumption`,
        each one being associated with one of the heating and cooling temperature
        domains which bounds are defined by ``self.t_ref_heat`` and ``self.t_ref_cool``,
        leading to an energy which is assumed to be affine per part depending on
        the temperature:

        - with slope ``-self.ts_heat`` in the heating domain (colder than
          ``self.t_ref_heat``.);
        - with slope ``self.ts_cool`` in the cool domain (warmer than
          ``self.t_ref_cool``.);
        - with slope 0 and constant value ``self.base_energy`` between
          ``self.t_ref_heat`` and ``self.t_ref_cool``.

        Example
        -------
        >>> synth_consumption = SynthTSConsumption(base_energy=100, ts_heat=2, ts_cool=0.2)
        >>> dds_cool = pd.Series(data=[0,2,12])
        >>> dds_heat = pd.Series(data=[52,1,0])
        >>> t_samples = pd.Series(data=[10, 15, 20])
        >>> synth_consumption.fake_energy(dds_heat, dds_cool, t_samples)
           base  thermosensitive  residual  energy  heating  cooling   T  DD_heating  DD_cooling
        0   100            104.0       0.0   204.0      104      0.0  10           52            0
        1   100              2.4       0.0   102.4        2      0.4  15            1            2
        2   100              2.4       0.0   102.4        0      2.4  20            0           12
        """
        heating = self.heating.fake_energy(dd_heating)
        cooling = self.cooling.fake_energy(dd_cooling)
        dd_heating = heating.pop("DD")
        dd_cooling = cooling.pop("DD")
        fake_data = heating + cooling
        fake_data["heating"] = heating["thermosensitive"]
        fake_data["cooling"] = cooling["thermosensitive"]
        fake_data["residual"] = self._rng.normal(
            loc=0.0,
            scale=self.noise_std,
            size=dd_cooling.size,
        )
        fake_data["energy"] += fake_data["residual"] + self.base_energy
        fake_data["base"] = self.base_energy
        fake_data["T"] = t_samples
        fake_data["DD_heating"] = dd_heating
        fake_data["DD_cooling"] = dd_cooling
        return fake_data

    def random_dds(
        self,
        t_mean=15,
        t_std=5,
        size=100,
        start="2022-11-01",
        end=None,
    ):
        """Return realistic DD samples.

        Parameters
        ----------
        size : int, optional
            The number of samples to generate. Default is 100.
        start : pd.Timestamp or alike, optional
            The first date of the generated time-series.
            Default is "2022-11-01".
        end : pd.Timestamp or alike, optional
            The last date of the generated time-series.
            Default is None.

        Returns
        -------
        pd.Series :
            A time-series with 1 day period containing randomly
            generated DD values.

        .. warning::

            The date provided in the sample is not consistent
            with the DD values (think winter in July!)
            Still, the distribution of the DDs remain correct
            when using large enough number of samples.


        Notes
        -----
        The daily mean temperature is drawn from a Normal distribution.
        The DD uses the mean method to compute the DD values.

        Example
        -------
        >>> synth_consumption = SynthTSConsumption(base_energy=100, ts_heat=2, ts_cool=0.2)
        >>> synth_consumption.random_dds(size=5, t_mean=20, t_std=20)
                    DD_heating  DD_cooling          T
        2022-11-01     0.000000     6.094342  26.094342
        2022-11-02    17.799682     0.000000  -0.799682
        2022-11-03     0.000000    15.009024  35.009024
        2022-11-04     0.000000    18.811294  38.811294
        2022-11-05    36.020704     0.000000 -19.020704

        """
        index = pd.date_range(start=start, end=end, periods=size, freq="1D")
        size = len(index)
        t_samples = pd.Series(
            self._rng.normal(loc=t_mean, scale=t_std, size=size),
            name="T(°C)",
            index=index,
        )
        fake_dds = pd.concat(
            [
                (self.t_ref_heat - t_samples).clip(lower=0),
                (t_samples - self.t_ref_cool).clip(lower=0),
                t_samples,
            ],
            axis=1,
        )
        fake_dds.columns = ["DD_heating", "DD_cooling", "T"]
        return fake_dds

    def random_consumption(
        self,
        size=100,
        t_mean=15,
        t_std=5,
        start="2022-11-01",
        end=None,
    ):
        """Return a fake energy consumption  decomposition VS temperature.

        The input temperatures are generated using a gaussian distribution.

        Parameters
        ----------
        size : int, optional
            The number of samples to generate. Default is 100.
        t_mean : float, default 15
            The average of the gaussian temperature distribution.
        t_std : float, default 5
            The std if the gaussian temperature distribution.
        start : pd.Timestamp or alike, optional
            The first date of the generated time-series.
            Default is "2022-11-01".
        end : pd.Timestamp or alike, optional
            The last date of the generated time-series.
            Default is None.

        Returns
        -------
        pd.DataFrame :
            A table of randomly generated energy consumption as a function of the
            temperature. See :py:meth:`.fake_energy`.

        """
        dd_samples = self.random_dds(
            size=size, t_mean=t_mean, t_std=t_std, start=start, end=end
        )
        return self.fake_energy(
            dd_samples["DD_heating"], dd_samples["DD_cooling"], dd_samples["T"]
        )

    def measures(
        self,
        *args,
        **kwargs,
    ):
        """Return a fake energy consumption  decomposition VS temperature.

        This method is a wrapper around :py:meth:`.random_consumption` to keep the same
        signature as the Handlers.

        """
        return self.random_consumption(*args, **kwargs)


class DateSynthTSConsumption(SynthTSConsumption):
    """A class to generate fake energy consumptions as a function of the
    temperature. Based on :py:class:`SynthTSConsumption`, including both heating
    and cooling domains.

    The generation relies on the assumption of linear DD dependencies in the
    heating and cooling domains.

    This class extends the :py:class:`SynthTSConsumption` to generate a realistic temperature
    as function of the date.
    """

    def __init__(
        self,
        base_energy,
        ts_heat,
        ts_cool,
        t_ref_heat=17,
        t_ref_cool=20,
        noise_std=0,
        noise_seed=42,
        temperature_amplitude_year=9.36,
        temperature_mean_year=14,
        temperature_period_year=2 * np.pi / 364.991,
        temperature_phase_year=13,
    ):
        """
        Parameters
        ----------
        temperature_amplitude_year : float, optional
            The temperature amplitude over the year, by default 9.36.
            Think of it as the difference between the hottest and coldest days.
        temperature_mean_year : int, optional
            The average daily temperature, by default 14
        temperature_period_year : float, optional
            The period of the year, by default 2*np.pi/364.991
            Should be close to 2*np.pi/365.
        temperature_phase_year : int, optional
            The phase, in days, by default 13
        Other Parameters
            See :py:class:`SynthTSConsumption` for the other parameters.

        """
        super().__init__(
            base_energy=base_energy,
            ts_heat=ts_heat,
            ts_cool=ts_cool,
            t_ref_heat=t_ref_heat,
            t_ref_cool=t_ref_cool,
            noise_std=noise_std,
            noise_seed=noise_seed,
        )
        self.temperature_amplitude_year = temperature_amplitude_year
        self.temperature_period_year = temperature_period_year
        self.temperature_phase_year = temperature_phase_year
        self.temperature_mean_year = temperature_mean_year

    def synthetic_temperature(
        self,
        t_std=5,
        size=100,
        start="2022-11-01",
        end=None,
        *args,
        **kwargs,
    ):
        """Return a synthetic temperature time-series.

        .. note::

            The temperature is generated using a sinusoidal model with a gaussian
            noise added.
            The parameters of the sinusoidal model are the class attributes.

            The noise model should be improved in the future, as the temperature
            fluctuations present actually several frequencies.

        Parameters
        ----------
        t_std : float
            The standard deviation of the gaussian noise added to the temperature
            base.
        size : int, optional
            The number of samples to generate. Default is 100.
        start : pd.Timestamp or alike, optional
            The first date of the generated time-series.
            Default is "2022-11-01".
        end : pd.Timestamp or alike, optional
            The last date of the generated time-series.
            Default is None.

        Returns
        -------
        pd.Series :
            A time-series of synthetic temperatures.

        Example
        -------
        >>> synth_consumption = DateTimeSynthTSConsumption(base_energy=100, ts_heat=2, ts_cool=0.2)
        >>> synth_consumption.synthetic_temperature(size=5)
        2022-11-01    13.187131
        2022-11-02     6.307951
        2022-11-03    15.105192
        2022-11-04    15.901608
        2022-11-05     1.290287
        Freq: D, Name: T(°C), dtype: float64

        Notes
        -----
        From the 3 parameter `size`, `start` and `end`, only two must be
        given.

        """
        index = pd.date_range(
            start=start,
            periods=size,
            end=end,
            freq="1D",
        )
        julian_date = index.to_julian_date()
        temperature_base = (
            self.temperature_mean_year
            + self.temperature_amplitude_year
            * np.sin(
                self.temperature_period_year
                * (julian_date - self.temperature_phase_year)
            )
        )
        t_samples = pd.Series(
            self._rng.normal(
                loc=temperature_base, scale=t_std, size=len(index)
            ),
            index=index,
            name="T(°C)",
        )
        return t_samples

    def random_dds(
        self,
        t_std=5,
        size=100,
        start="2022-11-01",
        end=None,
        *args,
        **kwargs,
    ):
        """Return realistic DD samples.

        Parameters
        ----------
        t_std : float, optional
            The standard deviation of the gaussian noise added to the temperature
            base. Default is 5.
        size : int, optional
            The number of samples to generate. Default is 100.
        start : pd.Timestamp or alike, optional
            The first date of the generated time-series.
            Default is "2022-11-01".
        end : pd.Timestamp or alike, optional
            The last date of the generated time-series.
            Default is None.

        Returns
        -------
        pd.DataFrame :
            A DataFrame with the following columns :
            - ``DD_heating`` : the heating DD values.
            - ``DD_cooling`` : the cooling DD values.
            - ``T`` : the temperature.

        Notes
        -----
        The DDs are computed using the mean temperature of the day as

        .. math::

                DD = \\max(0, T_{mean} - T_{ref})

        Hence, it is really just the same as the mean temperature of the day.

        From the 3 parameter `size`, `start` and `end`, only two must be
        given.

        Example
        -------
        >>> synth_consumption = DateTimeSynthTSConsumption(base_energy=100, ts_heat=2, ts_cool=0.2)
        >>> synth_consumption.random_dds(size=5, t_std=25)
                    DD_heating  DD_cooling          T
        2022-11-01     0.000000     0.000000  19.281473
        2022-11-02    31.491731     0.000000 -14.491731
        2022-11-03     0.000000    10.114216  30.114216
        2022-11-04     0.000000    14.712902  34.712902
        2022-11-05    54.730417     0.000000 -37.730417

        """
        t_samples = self.synthetic_temperature(t_std, size, start, end=end)
        dd_heating = (self.t_ref_heat - t_samples).clip(lower=0)
        dd_cooling = (t_samples - self.t_ref_cool).clip(lower=0)
        data = pd.concat([dd_heating, dd_cooling, t_samples], axis=1)
        data.columns = ["DD_heating", "DD_cooling", "T"]
        return data


class TSParameters(TypedDict):
    base_energy: float
    ts_heat: float
    ts_cool: float
    noise_std: float


class CategorySynthTSConsumption(DateSynthTSConsumption):
    """A class to generate fake energy consumptions as a function of the
    temperature. Based on :py:class:`DateSynthTSConsumption`, including both heating
    and cooling domains.

    Add the possibility to generate different categories of energy consumption
    with a function that categorize the periods.

    .. note::

        The categories are labeled by a function. The base temperatures
        (``t_ref_heat`` and ``t_ref_cool``) are the same for all categories.


    For an example, see the :py:class:`WeekEndSynthTSConsumption` class
    that implement the concept of week-end and week days.
    """

    def __init__(
        self,
        parameters: list[TSParameters],
        t_ref_heat: float,
        t_ref_cool: float,
        noise_seed=42,
        temperature_amplitude_year=9.36,
        temperature_mean_year=14,
        temperature_period_year=2 * np.pi / 364.991,
        temperature_phase_year=13,
        list_categories: list = None,
        category_func: Callable = None,
    ):
        """
        Parameters
        ----------
        parameters : list[TSParameters]
            A list of dictionaries containing the parameters for each category.
        list_categories : list
            The list of the different categories labels.
        category_func : Callable
            A function that takes a pd.Series with DateTimeIndex as input and
            return a pd.Series with the categories labels.
        Other Parameters
            See :py:class:`SynthTSConsumption` for the other parameters.

        Notes
        -----
        The number of categories in ``list_categories`` must match the number of ``parameters``.
        In assition, the order of the categories must match the order of the parameters.

        """
        self.list_of_synths = [
            DateSynthTSConsumption(
                **param,
                t_ref_heat=t_ref_heat,
                t_ref_cool=t_ref_cool,
                noise_seed=noise_seed,
                temperature_amplitude_year=temperature_amplitude_year,
                temperature_mean_year=temperature_mean_year,
                temperature_period_year=temperature_period_year,
                temperature_phase_year=temperature_phase_year,
            )
            for param in parameters
        ]
        self.t_ref_cool = t_ref_cool
        self.t_ref_heat = t_ref_heat
        self.temperature_amplitude_year = temperature_amplitude_year
        self.temperature_period_year = temperature_period_year
        self.temperature_phase_year = temperature_phase_year
        self.temperature_mean_year = temperature_mean_year
        self._rng = np.random.default_rng(seed=noise_seed)
        self.list_categories = list_categories
        self.category_func = category_func
        if len(self.list_of_synths) != len(self.list_categories):
            raise ValueError(
                "The number of categories must match the number of synthetizers"
            )

    def fake_energy(
        self,
        dd_heating,
        dd_cooling,
        t_samples,
    ):
        """Return a fake energy consumption depending on the daily temperatures
        passed in input.

        Parameters
        ----------
        dd_heating : pd.Series
            A series of DD. Usually daily aggregates, depending on the
            scale chosen for the thermosensitivity and base consumption values.
        dd_cooling : pd.Series
            A series of DD. Usually daily aggregates, depending on the
            scale chosen for the thermosensitivity and base consumption values.
        t_samples : pd.Series

        Returns
        -------
        pd.DataFrame :
            A table with rows labeled by ``dd_samples`` index and the
            following columns :

            - ``T`` : the ``t_samples`` series.
            - ``energy`` : the energy consumption for each row in the table.
            - ``thermosensitive`` : the value of the thermosensitive energy
              consumption for each period.
            - ``base`` : the value of the averaged non-thermosensitive consumption.
              This value is constant across the table, equal to ``self.base_energy``.
            - ``residual`` : the energy-noise, i.e. the residual between the
              affine model and the actual energy for each period.

        Notes
        -----
        The fake energy generation relies on two instances of
        :py:class:`SynthDDConsumption`, each one being associated with one of
        the heating and cooling temperature domains which bounds are defined by
        ``self.t_ref_heat`` and ``self.t_ref_cool``, leading to an energy which
        is assumed to be affine per part depending on the temperature:

        - with slope ``-self.ts_heat`` in the heating domain (colder than
          ``self.t_ref_heat``);
        - with slope ``self.ts_cool`` in the cool domain (warmer than
          ``self.t_ref_cool``);
        - with slope 0 and constant value ``self.base_energy`` between
          ``self.t_ref_heat`` and ``self.t_ref_cool``.

        See :py:class:`SynthTSConsumption` for more details.

        Example
        -------
        >>> synth_consumption = SynthTSConsumption(base_energy=100, ts_heat=2, ts_cool=0.2)
        >>> dds_cool = pd.Series(data=[0,2,12])
        >>> dds_heat = pd.Series(data=[52,1,0])
        >>> t_samples = pd.Series(data=[10, 15, 20])
        >>> synth_consumption.fake_energy(dds_heat, dds_cool, t_samples)
        base  thermosensitive  residual  energy  heating  cooling   T  DD_heating  DD_cooling
        0   100            104.0       0.0   204.0      104      0.0  10           52            0
        1   100              2.4       0.0   102.4        2      0.4  15            1            2
        2   100              2.4       0.0   102.4        0      2.4  20            0           12

        """
        categories_series = self.category_func(t_samples)
        list_of_fake_data = []
        for category, synth in zip(self.list_categories, self.list_of_synths):
            mask = categories_series == category
            fake_data = synth.fake_energy(
                dd_heating[mask],
                dd_cooling[mask],
                t_samples[mask],
            )
            fake_data["category"] = category
            list_of_fake_data.append(fake_data)
        return pd.concat(list_of_fake_data, axis=0).sort_index()


class WeekEndSynthTSConsumption(CategorySynthTSConsumption):
    """A class to generate fake energy consumptions as a function of the
    temperature with two categories: week days and weekends.

    Based on :py:class:`CategorySynthTSConsumption`.
    """

    def __init__(
        self,
        parameters: list[TSParameters],
        t_ref_heat: float,
        t_ref_cool: float,
        noise_seed=42,
        temperature_amplitude_year=9.36,
        temperature_mean_year=14,
        temperature_period_year=2 * np.pi / 364.991,
        temperature_phase_year=13,
    ):
        """
        Parameters
        ----------
        parameters: list[TSParameters]
            A list of dictionaries containing the parameters for the two categories:

            1. The first dictionary is for the week days.
            2. The second dictionary is for the weekends.

        Other Parameters
            See :py:class:`CategorySynthTSConsumption` for the other parameters.

        Example
        -------
        >>> parameters = [
        ...     {"base_energy": 100, "ts_heat": 2, "ts_cool": 0.2, "noise_std": 5},
        ...     {"base_energy": 100, "ts_heat": 1, "ts_cool": 0.1, "noise_std": 5},
        ... ]
        >>> synth_consumption = WeekEndSynthTSConsumption(parameters, t_ref_heat=17, t_ref_cool=20)
        """
        list_categories = ["weekend", "weekday"]

        def category_func(t_samples):
            return np.where(t_samples.index.dayofweek < 5, "weekday", "weekend")

        super().__init__(
            parameters=parameters,
            t_ref_heat=t_ref_heat,
            t_ref_cool=t_ref_cool,
            noise_seed=noise_seed,
            temperature_amplitude_year=temperature_amplitude_year,
            temperature_mean_year=temperature_mean_year,
            temperature_period_year=temperature_period_year,
            temperature_phase_year=temperature_phase_year,
            list_categories=list_categories,
            category_func=category_func,
        )
