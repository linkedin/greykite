# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Reza Hosseini
"""Helper functions for
`~greykite.algo.forecast.silverkite.forecast_silverkite.py.`
"""

import math
import warnings
from typing import List
from typing import Optional

import pandas as pd

from greykite.common.constants import TimeFeaturesEnum
from greykite.common.enums import SimpleTimeFrequencyEnum
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.features.timeseries_features import get_default_origin_for_time_vars


def get_similar_lag(freq_in_days):
    """For a given frequency, it returns a lag which is likely to be most correlated
    to the observation at current time.

    For daily data, this will return 7 and for hourly data it will return 24*7.
    In general for sub-weekly frequencies, it returns the lag which corresponds to
    the same time in the last week.
    For data which is weekly or with frequencies larger than a week, it returns None.

    Parameters
    ----------
    freq_in_days : `float`
        The time frequency of the timeseries given in day units.

    Returns
    -------
    similar_lag : `int` or None
        The returned lag or None.
    """
    similar_lag = None
    # Get the number of observations per week
    obs_num_per_week = 7 / freq_in_days

    if obs_num_per_week > 1:
        similar_lag = math.ceil(obs_num_per_week)

    return similar_lag


def get_default_changepoints_dict(
        changepoints_method,
        num_days,
        forecast_horizon_in_days):
    """Get a changepoint dictionary based on the number of days in the observed
    timeseries and forecast horizon length in days to be provided as input to
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
    For the "uniform" method, we place the change points at a distance of
    ``max(28, forecast_horizon)``.
    For the "auto" method, we have used some defaults which seem to work for general
    applications::

        changepoints_dict = {
            "method": "auto",
            "yearly_seasonality_order": 10,
            "resample_freq": "7D",
            "regularization_strength": 0.8,
            "actual_changepoint_min_distance": "14D",
            "potential_changepoint_distance": "7D",
            "no_changepoint_distance_from_end": "14D"}

    If the length of data is smaller than ``2*max(28, forecast_horizon)``,
    the function will return None for all methods.

    Parameters
    ----------
    changepoints_method : `str`
        The method to locate changepoints.
        Valid options:

            - "uniform". Places changepoints evenly spaced changepoints to allow
            growth to change. The distance between the uniform change points is
            set to be ``max(28, forecast_horizon)``
            - "auto". Automatically detects change points.
            For configuration, see
            `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_trend_changepoints`

        For more details for both methods, also check the documentation for
        `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
    num_days : `int`
        Number of days appearing in the observed timeseries.
    forecast_horizon_in_days : `float`
        The length of the forecast horizon in days.

    Returns
    -------
    changepoints_dict : `dict` or None
        A dictionary with change points information to be used as input to
        `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
        See that function's documentation for more details.
    """

    changepoints_dict = None
    # A reasonable distance defined based on ``forecast_horizon``
    # Here the minimum is set at 28 days
    uniform_distance = max(28, forecast_horizon_in_days)
    # Number of change points for "uniform"
    # Also if this number is zero both methods will return `None`
    changepoint_num = num_days // uniform_distance - 1

    if changepoint_num > 0:
        if changepoints_method == "uniform":
            changepoints_dict = {
                "method": "uniform",
                "n_changepoints": changepoint_num,
                "continuous_time_col": TimeFeaturesEnum.ct1.value,
                "growth_func": lambda x: x}

        elif changepoints_method == "auto":
            changepoints_dict = {
                "method": "auto",
                "yearly_seasonality_order": 10,
                "resample_freq": "7D",
                "regularization_strength": 0.8,
                "actual_changepoint_min_distance": "14D",
                "potential_changepoint_distance": "7D",
                "no_changepoint_distance_from_end": "14D"}

    return changepoints_dict


def get_silverkite_uncertainty_dict(
        uncertainty,
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        coverage=None):
    """Returns an uncertainty_dict for
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`
    input parameter: uncertainty_dict.

    The logic is as follows:

        - If ``uncertainty`` is passed as dict:
            - If ``quantiles`` are not passed through ``uncertainty`` we fill them
              using `coverage`.
            - If ``coverage`` also missing or quantiles calculated
              in two ways (via ``uncertainty["params"]["quantiles"]`` and ``coverage``)
              do not match, we throw Exceptions

        - If ``uncertainty=="auto"``:
            - We provide defaults based on time frequency of data.
            - Specify ``uncertainty["params"]["quantiles"]`` based on
              ``coverage`` if provided, otherwise the default coverage is 0.95.

    Parameters
    ----------
    uncertainty : `str` or `dict` or None
        It specifies what method should be used for uncertainty.
        If a dict is passed then it is directly returned to be passed to
        `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast` as `uncertainty_dict`.

        If "auto", it builds a generic dict depending on frequency.
            - For frequencies less than or equal to one day it sets
              `conditional_cols` to be ["dow_hr"].
            - Otherwise it sets the conditional_cols to be `None`

        If None and `coverage` is None, the upper/lower predictions are not returned
    simple_freq : `str`, optional
        SimpleTimeFrequencyEnum member that best matches the input data frequency
        according to `get_simple_time_frequency_from_period`
    coverage : `float` or None, optional
        Intended coverage of the prediction bands (0.0 to 1.0)
        If None and `uncertainty` is None, the upper/lower predictions are not returned

    Returns
    -------
    uncertainty : `dict` or None
        An uncertainty dict to be used as input to
        `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
        See that function's docstring for more details.
    """
    frequency = SimpleTimeFrequencyEnum[simple_freq].value

    # boolean to determine if freq is longer than one day
    freq_is_longer_than_day = (
            frequency.seconds_per_observation
            > SimpleTimeFrequencyEnum.DAY.value.seconds_per_observation)

    uncertainty_dict = None

    # if both `uncertainty` and `coverage` are None, we return None
    if uncertainty is None and coverage is None:
        return None

    # checking if coverage input is sensible
    if coverage is not None and (coverage < 0 or coverage > 1):
        raise ValueError("coverage must be between 0 and 1")

    # if only coverage is provided, consider uncertainty to be "auto"
    if coverage is not None and uncertainty is None:
        uncertainty = "auto"

    # The case where `uncertainty` is input as a dict
    # We check if quantiles are passed through `uncertainty`
    # If not, we use `coverage` to fill them in
    # If quantiles are passed in `uncertainty` and inferrable from `coverage`:
    # and they are inconsistent, we throw an Exception
    if isinstance(uncertainty, dict):
        uncertainty_dict = uncertainty
        # boolean to check if quantiles are passed through uncertainty
        try:
            quantiles_specified = (uncertainty["params"]["quantiles"] is not None)
        except KeyError:
            quantiles_specified = False
        if "params" not in uncertainty_dict:
            uncertainty_dict["params"] = {}

        if quantiles_specified:
            quantiles = uncertainty["params"]["quantiles"]
            # If quantiles are specified, we do some sanity checks on their values:
            # We give warnings if more than two quantiles were passed
            # or if they are not symmetric i.e. first quantiles distance to zero
            # is not the same as last quantile distance to 1
            # We throw exceptions if quantiles are not increasing
            # or if `coverage` is also passed and inconsistent with `quantiles`
            if len(quantiles) > 2:
                warnings.warn(
                    "More than two quantiles are passed in `uncertainty`."
                    " Confidence intervals will be based on"
                    " the first (lower limit) and last (upper limit) quantile",
                    Warning)
            coverage_via_uncertainty = quantiles[-1] - quantiles[0]
            if coverage_via_uncertainty <= 0:
                raise ValueError(
                    "`quantiles` is expected to be an increasing sequence"
                    " of at least two elements."
                    f"These quantiles were passed: quantiles = {quantiles}")
            if round(quantiles[-1], 3) != round(1 - quantiles[0], 3):
                warnings.warn(
                    "1 - (quantiles upper limit) is not equal to (quantiles lower limit)"
                    " (lack of symmetry)."
                    f" Asymmetric quantiles: {quantiles} were used.",
                    Warning)
            if coverage is not None:
                # The case where quantiles are both provided through `uncertainty`
                # and inferrable using `coverage`
                # We check for conflict in coverage specification
                if round(coverage_via_uncertainty, 3) != round(coverage, 3):
                    raise ValueError(
                        "Coverage is specified/inferred both via `coverage` and via `uncertainty` input"
                        " and values do not match."
                        f" Coverage specified via `coverage`: {round(coverage, 3)}."
                        f" Coverage inferred via `uncertainty`: {round(coverage_via_uncertainty, 2)}.")
        if not quantiles_specified:
            if coverage is None:
                raise ValueError(
                    "`quantiles` are not specified in `uncertainty`"
                    " and `coverage` is not provided to infer them")
            else:
                # The case where quantiles is not provided through `uncertainty`
                # but coverage is passed
                q1 = (1 - coverage)/2
                q2 = 1 - q1
                uncertainty_dict["params"]["quantiles"] = [q1, q2]

    # The case where `uncertainty` is passed as "auto"
    # The auto case conditions data on `dow_hr` which represents day of week and hour
    # for data with frequency less than or equal to a day (e.g. hourly, daily)
    # note that for daily case this works too as dow_hr will only depend on dow
    if uncertainty == "auto":
        if not freq_is_longer_than_day:
            uncertainty_dict = {
                "uncertainty_method": "simple_conditional_residuals",
                "params": {
                    "conditional_cols": [TimeFeaturesEnum.dow_hr.value],
                    "quantiles": [0.025, 0.975],
                    "quantile_estimation_method": "normal_fit",
                    "sample_size_thresh": 5,
                    "small_sample_size_method": "std_quantiles",
                    "small_sample_size_quantile": 0.98}}
        else:
            uncertainty_dict = {
                "uncertainty_method": "simple_conditional_residuals",
                "params": {
                    "conditional_cols": None,
                    "quantiles": [0.025, 0.975],
                    "quantile_estimation_method": "normal_fit",
                    "sample_size_thresh": 5,
                    "small_sample_size_method": "std_quantiles",
                    "small_sample_size_quantile": 0.98}}
        # if coverage is provided the quantiles are overridden in auto
        # we do not give warnings as it is the auto case and
        # user expects using the coverage provided
        if coverage is not None:
            q1 = (1 - coverage)/2
            q2 = 1 - q1
            uncertainty_dict["params"]["quantiles"] = [q1, q2]

    return uncertainty_dict


def get_fourier_feature_col_names(
        df: pd.DataFrame,
        time_col: str,
        fs_func: callable,
        conti_year_origin: Optional[int] = None) -> List[str]:
    """Gets the Fourier feature column names.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input data.
    time_col : `str`
        The column name for timestamps in ``df``.
    fs_func : callable
        The function to generate Fourier features.
    conti_year_origin : `int` or None, default None
        The continuous year origin.
        If None, will be inferred from time column.
        The names do not depend on this parameter though.

    Returns
    -------
    col_names : `list` [`str`]
        The list of Fourier feature column names.
    """
    if conti_year_origin is None:
        conti_year_origin = get_default_origin_for_time_vars(
            df=df,
            time_col=time_col
        )
    time_features_example_df = build_time_features_df(
        df[time_col][:2],
        conti_year_origin=conti_year_origin)
    fs = fs_func(time_features_example_df)
    return fs["cols"]
