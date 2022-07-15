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
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Kaixu Yang
"""Automatically populates parameters based on input time series."""

from datetime import timedelta
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from greykite.algo.changepoint.adalasso.auto_changepoint_params import generate_trend_changepoint_detection_params
from greykite.algo.common.holiday_inferrer import HolidayInferrer
from greykite.algo.common.seasonality_inferrer import SeasonalityInferConfig
from greykite.algo.common.seasonality_inferrer import SeasonalityInferrer
from greykite.algo.common.seasonality_inferrer import TrendAdjustMethodEnum
from greykite.common.constants import GrowthColEnum
from greykite.common.constants import TimeFeaturesEnum


def get_auto_seasonality(
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        yearly_seasonality: bool = True,
        quarterly_seasonality: bool = True,
        monthly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True):
    """Automatically infers the following seasonality Fourier series orders:

        - yearly seasonality
        - quarterly seasonality
        - monthly seasonality
        - weekly seasonality
        - daily seasonality

    The inferring is done with `~greykite.algo.common.seasonality_inferrer.SeasonalityInferrer`.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input time series.
    time_col : `str`
        The column name for timestamps in ``df``.
    value_col : `str`
        The column name for values in ``df``.
    yearly_seasonality : `bool`, default True
        If False, the yearly seasonality order will be forced to be zero
        regardless of the inferring result.
    quarterly_seasonality : `bool`, default True
        If False, the quarterly seasonality order will be forced to be zero
        regardless of the inferring result.
    monthly_seasonality : `bool`, default True
        If False, the monthly seasonality order will be forced to be zero
        regardless of the inferring result.
    weekly_seasonality : `bool`, default True
        If False, the weekly seasonality order will be forced to be zero
        regardless of the inferring result.
    daily_seasonality : `bool`, default True
        If False, the daily seasonality order will be forced to be zero
        regardless of the inferring result.

    Returns
    -------
    result : `dict`
        A dictionary with the keys being:

            - "yearly_seasonality"
            - "quarterly_seasonality"
            - "monthly_seasonality"
            - "weekly_seasonality"
            - "daily_seasonality"

        and values being the corresponding inferred Fourier series orders.
    """
    result = dict()
    seasonalities = ["yearly_seasonality", "quarterly_seasonality", "monthly_seasonality",
                     "weekly_seasonality", "daily_seasonality"]
    seasonality_defaults = [yearly_seasonality, quarterly_seasonality, monthly_seasonality,
                            weekly_seasonality, daily_seasonality]
    # Iterate through the five seasonality frequencies.
    # Frequencies that are less than or equal to the data frequency will have order 0.
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    min_increment = min((df[time_col] - df[time_col].shift(1)).dropna())
    # Yearly seasonality/quarterly seasonality is activated when data is at most weekly.
    # For monthly data, it will not return yearly seasonality but will try to use "C(month)".
    configs = []
    if min_increment <= timedelta(days=7):
        yearly_config = SeasonalityInferConfig(
            seas_name="yearly",
            col_name=TimeFeaturesEnum.toy.name,
            period=1.0,
            max_order=30,
            adjust_trend_param=dict(
                trend_average_col=TimeFeaturesEnum.year.name
            ),
            aggregation_period="W"
        )
        quarterly_config = SeasonalityInferConfig(
            seas_name="quarterly",
            col_name=TimeFeaturesEnum.toq.name,
            period=1.0,
            max_order=10,
            adjust_trend_param=dict(
                trend_average_col=TimeFeaturesEnum.year_quarter.name
            ),
            aggregation_period="W"
        )
        configs += [yearly_config, quarterly_config]
    # Monthly/weekly seasonality is activated when data is at most daily.
    if min_increment <= timedelta(days=1):
        monthly_config = SeasonalityInferConfig(
            seas_name="monthly",
            col_name=TimeFeaturesEnum.tom.name,
            period=1.0,
            max_order=10,
            adjust_trend_param=dict(
                trend_average_col=TimeFeaturesEnum.year_month.name
            ),
            aggregation_period="D"
        )
        weekly_config = SeasonalityInferConfig(
            seas_name="weekly",
            col_name=TimeFeaturesEnum.tow.name,
            period=7.0,
            max_order=5,
            adjust_trend_param=dict(
                trend_average_col=TimeFeaturesEnum.year_woy_iso.name
            ),
            aggregation_period=None
        )
        configs += [monthly_config, weekly_config]
    # Daily seasonality is activated when data is at most hourly.
    if min_increment <= timedelta(hours=1):
        daily_config = SeasonalityInferConfig(
            seas_name="daily",
            col_name=TimeFeaturesEnum.tod.name,
            period=24.0,
            max_order=15,
            adjust_trend_param=dict(
                trend_average_col=TimeFeaturesEnum.year_woy_dow_iso.name
            ),
            aggregation_period=None
        )
        configs += [daily_config]

    # Infers seasonality orders.
    seasonality_inferrer = SeasonalityInferrer()
    seasonality_result = seasonality_inferrer.infer_fourier_series_order(
        df=df,
        time_col=time_col,
        value_col=value_col,
        configs=configs,
        adjust_trend_method=TrendAdjustMethodEnum.seasonal_average.name,
        fit_algorithm="ridge",
        tolerance=0.0,
        plotting=False,
        offset=0,
        criterion="bic"
    )

    # Transforms seasonality infer results to dictionary.
    for seasonality, default in zip(seasonalities, seasonality_defaults):
        # Seasonalities are named like "yearly_seasonality" while the keys in the result are like "yearly".
        # If a seasonality is not found in the result, the default is 0.
        if default is not False:
            result[seasonality] = seasonality_result["best_orders"].get(seasonality.split("_")[0], 0)
        else:
            result[seasonality] = 0

    return result


def get_auto_growth(
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        forecast_horizon: int,
        changepoints_dict_override: Optional[dict] = None):
    """Automatically gets the parameters for growth.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input time series.
    time_col : `str`
        The column name for timestamps in ``df``.
    value_col : `str`
        The column name for values in ``df``.
    forecast_horizon : `int`
        The forecast horizon.
    changepoints_dict_override : `dict` or None, default None
        The changepoints configuration.
        If the following customized dates related keys are given,
        they will be copied to the final configuration:

            - "dates"
            - "combine_changepoint_min_distance"
            - "keep_detected"

    Returns
    -------
    result : `dict`
        A dictionary with the following keys:

            - "growth_term": `str`
              The growth term.
            - "changepoints_dict": `dict`
              The changepoints configuration used for automatic changepoint detection.

    """
    # Gets the growth and changepoints configurations.
    growth_term = GrowthColEnum.linear.name
    trend_changepoints_param = generate_trend_changepoint_detection_params(
        df=df,
        forecast_horizon=forecast_horizon,
        time_col=time_col,
        value_col=value_col
    )
    trend_changepoints_param["method"] = "auto"
    # Inherits the user specified dates.
    if changepoints_dict_override is not None and changepoints_dict_override.get("method", None) != "custom":
        for key in ["dates", "combine_changepoint_min_distance", "keep_detected"]:
            if key in changepoints_dict_override:
                trend_changepoints_param[key] = changepoints_dict_override[key]
    return dict(
        growth_term=growth_term,
        changepoints_dict=trend_changepoints_param
    )


def get_auto_holidays(
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        countries: List[str] = ("UnitedStates", "India", "UnitedKingdom"),
        forecast_horizon: Optional[int] = None,
        daily_event_dict_override: Optional[Dict[str, pd.DataFrame]] = None):
    """Automatically infers significant holidays and their neighboring days.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input time series.
    time_col : `str`
        The column name for timestamps in ``df``.
    value_col : `str`
        The column name for values in ``df``.
    countries : `list` [`str`], default ("UnitedStates", "India", "UnitedKingdom")
        A list of countries to look up holidays.
    forecast_horizon : `int` or None, default None
        The forecast horizon used to calculate the years needed to populate holidays.
    daily_event_dict_override : `dict` [`str`, `pandas.DataFrame`] or None, default None
        The daily event dict passed to the configuration.
        When auto holiday is activated,
        the entries in ``daily_event_dict`` will be added
        to the holidays' ``daily_event_dict``.

    Returns
    -------
    daily_event_dict : `dict` [`str`, `pandas.DataFrame`]
        A dictionary with the keys being the event names
        and values being the dataframes including 2 columns:

            - EVENT_DF_DATE_COL : the events' occurrence dates.
            - EVENT_DF_LABEL_COL : the events' names.

    """
    # Calculates the number of extra years needed to cover the forecast period.
    if forecast_horizon and forecast_horizon > 0:
        timestamps = pd.to_datetime(df[time_col])
        min_increment = min((timestamps - timestamps.shift(1)).dropna())
        min_increment_in_days = min_increment / timedelta(days=1)
        forecast_horizon_in_days = min_increment_in_days * forecast_horizon
        extra_years = int(np.ceil(forecast_horizon_in_days / 366)) + 2  # +2 in case of incomplete years
    else:
        extra_years = 3
    # Automatically infers holidays.
    hi = HolidayInferrer()
    result = hi.infer_holidays(
        df=df,
        time_col=time_col,
        value_col=value_col,
        countries=countries,
        pre_search_days=2,
        post_search_days=2,
        baseline_offsets=[-7, 7],
        plot=False,
        independent_holiday_thres=0.85,
        together_holiday_thres=0.95,
        extra_years=extra_years
    )
    if result is None:
        # This happens when data is super-daily.
        holiday_dict = {}
    else:
        holiday_dict = hi.generate_daily_event_dict()
    if daily_event_dict_override is None:
        daily_event_dict_override = {}
    # Updates the result with pre-specified daily events.
    holiday_dict.update(daily_event_dict_override)
    return holiday_dict
