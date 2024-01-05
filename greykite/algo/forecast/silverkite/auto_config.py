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

import inspect
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from greykite.algo.changepoint.adalasso.auto_changepoint_params import generate_trend_changepoint_detection_params
from greykite.algo.common.holiday_grouper import HolidayGrouper
from greykite.algo.common.seasonality_inferrer import SeasonalityInferConfig
from greykite.algo.common.seasonality_inferrer import SeasonalityInferrer
from greykite.algo.common.seasonality_inferrer import TrendAdjustMethodEnum
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import split_events_into_dictionaries
from greykite.common import constants as cst
from greykite.common.constants import GrowthColEnum
from greykite.common.constants import TimeFeaturesEnum
from greykite.common.features.timeseries_features import add_event_window_multi
from greykite.common.features.timeseries_features import get_holidays
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import update_dictionary
from greykite.common.time_properties import describe_timeseries


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
        start_year: int,
        end_year: int,
        pre_num: int = 2,
        post_num: int = 2,
        pre_post_num_dict: Optional[Dict[str, pd.DataFrame]] = None,
        holiday_lookup_countries: List[str] = ("UnitedStates", "India", "UnitedKingdom"),
        holidays_to_model_separately: Optional[List[str]] = None,
        daily_event_df_dict:  Optional[Dict] = None,
        auto_holiday_params: Optional[Dict] = None):
    """Automatically group holidays and their neighboring days based on estimated holiday impact.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The timeseries data used to infer holiday impact if no ``df`` is passed through ``auto_holiday_params``.
    time_col : `str`
        The column name for timestamps in ``df`` that will be used for holiday impact estimation in ``HolidayGrouper``.
        If ``time_col`` is passed in through ``auto_holiday_params``, this will be ignored.
    value_col : `str`
        The column name for values in ``df`` that will be used for holiday impact estimation in ``HolidayGrouper``.
        If ``value_col`` is passed in through ``auto_holiday_params``, this will be ignored.
    start_year : `int`
        Year of first training data point, used to generate holiday events based on ``holiday_lookup_countries``.
        This will not be used if `holiday_df` is passed in through ``auto_holiday_params``.
    end_year : `int`
        Year of last forecast data point, used to generate holiday events based on ``holiday_lookup_countries``.
        This will not be used if `holiday_df` is passed in through ``auto_holiday_params``.
    pre_num : `int`, default 2
        Model holiday effects for ``pre_num`` days before the holiday. This will be used as
        ``holiday_impact_pre_num_days`` when constructing ``HolidayGrouper`` if ``holiday_impact_pre_num_days``
        is not passed in though ``auto_holiday_params``.
    post_num : `int`, default 2
        Model holiday effects for ``post_num`` days before the holiday. This will be used as
        ``holiday_impact_post_num_days`` when constructing ``HolidayGrouper`` if ``holiday_impact_post_num_days``
        is not passed in though ``auto_holiday_params``.
    pre_post_num_dict : `dict` [`str`, (`int`, `int`)] or None, default None
        Overrides ``pre_num`` and ``post_num`` for each holiday in
        ``holidays_to_model_separately`` and in ``HolidayGrouper`` (as ``holiday_impact_dict``)
        if ``holiday_impact_dict`` is not passed in though ``auto_holiday_params``.
        For example, if ``holidays_to_model_separately`` contains "Thanksgiving" and "Labor Day",
        this parameter can be set to ``{"Thanksgiving": [1, 3], "Labor Day": [1, 2]}``,
        denoting that the "Thanksgiving" ``pre_num`` is 1 and ``post_num`` is 3, and "Labor Day"
        ``pre_num`` is 1 and ``post_num`` is 2.
        Holidays not specified use the default given by ``pre_num`` and ``post_num``.
    holiday_lookup_countries : `list` [`str`], default ("UnitedStates", "India", "UnitedKingdom")
        A list of countries to look up holidays. This will be used with `daily_event_df_dict` to
        generate `holiday_df` that contains holidays that will be modeled when ``holiday_df`` is
        not passed in through ``auto_holiday_params``. Otherwise, ``auto_holiday_params["holiday_df"]``
        will be used and this will be ignored.
    holidays_to_model_separately : `list` [`str`] or None
        Which holidays to include in the model by themselves. These holidays will not be passed into the
        ``HolidayGrouper``. The model creates a separate key, value for each item in ``holidays_to_model_separately``
        and their neighboring days. Generally, this is recommended to be kept as `None` unless some specific
        assumptions on holidays need to be applied.
    daily_event_df_dict : `dict` [`str`, `pandas.DataFrame`] or None, default None
        A dictionary of holidays to be used in ``HolidayGrouper``to generate `holiday_df` which contains holidays
        that will be modeled when ``holiday_df`` is not passed in through ``auto_holiday_params``.
        Each key presents a holiday name, and the values are data frames, with a date column that records all dates
        for the corresponding holiday.
        Otherwise, ``auto_holiday_params["holiday_df"]`` will be used and this will be ignored.
    auto_holiday_params: `dict` or None, default None
        This dictionary takes in parameters that can be passed in and used by holiday grouper when
        ``auto_holiday`` is set to `True`. It overwrites all configurations passed in or generated by
        other inputs.
        Examples of arguments that can be included here include:

            ``"df"`` : `str`
                Data Frame used by `HolidayGrouper` to infer holiday impact. If this exists,
                ``df`` will be ignored.
            ``"holiday_df"`` : `str`
                Input holiday dataframe that contains the dates and names of the holidays.
                If this exists, the following parameters used to generate holiday list will be ignored:

                    * "start_year"
                    * "end_year"
                    * "holiday_lookup_countries"
                    * "daily_event_df_dict"

            ``"holiday_date_col"`` : `str`
                This will be used as the date column when ``holiday_df`` is passed in through
                ``auto_holiday_params``
            ``"holiday_name_col"`` : `str`
                This will be used as the holiday name column when ``holiday_df`` is passed in through
                ``auto_holiday_params``

        Please refer to
        `~greykite.algo.common.holiday_grouper.HolidayGrouper` for more details.

    Returns
    -------
    daily_event_df_dict : `dict` [`str`, `pandas.DataFrame` [cst.EVENT_DF_DATE_COL, cst.EVENT_DF_LABEL_COL]]
        A dictionary with the keys being the holiday group names and values being the dataframes including 2 columns:

            - EVENT_DF_DATE_COL : the events' occurrence dates.
            - EVENT_DF_LABEL_COL : the events' names.

        Suitable for use as ``daily_event_df_dict`` parameter in `forecast_silverkite`.

    """

    # Initializes `group_holiday_params`, the parameters to pass in for function `group_holidays`.
    if auto_holiday_params is None:
        group_holiday_params = dict()
    else:
        group_holiday_params = auto_holiday_params.copy()

    # Initializes `grouper_init_params`, the parameters for `HolidayGrouper`. Anything passed through
    # `auto_holiday_params`, which are stored in `group_holiday_params`, will be used for initializations.
    # Uses `pop` to remove these parameters from `group_holiday_params` if existed at the same time.

    # Handles `group_holiday_params["df"]` separately to avoid an ambiguity error on using `or` to check if it is empty.
    if group_holiday_params.get("df") is None:
        group_holiday_params["df"] = df

    grouper_init_params = dict(
        df=group_holiday_params.pop("df"),  # This will be standardized and updated later.
        time_col=group_holiday_params.pop("time_col", None) or time_col,
        value_col=group_holiday_params.pop("value_col", None) or value_col,
        holiday_df=group_holiday_params.pop("holiday_df", None),  # This will be updated later.
        holiday_date_col=group_holiday_params.pop("holiday_date_col", None) or cst.EVENT_DF_DATE_COL,
        # This will be standardized and updated later.
        holiday_name_col=group_holiday_params.pop("holiday_name_col", None) or cst.EVENT_DF_LABEL_COL,
        # This will be standardized and updated later.
        holiday_impact_pre_num_days=group_holiday_params.pop("holiday_impact_pre_num_days", None) or pre_num,
        holiday_impact_post_num_days=group_holiday_params.pop("holiday_impact_post_num_days", None) or post_num,
        holiday_impact_dict=group_holiday_params.pop("holiday_impact_dict", None) or pre_post_num_dict)

    # Checks and updates if any other parameter for `HolidayGrouper` exists in `group_holiday_params` that has not
    # been initialized through `grouper_init_params`.
    # ``.copy()`` is used below to avoid altering the dictionary keys within iteration on same keys
    group_holiday_params_key_copy = group_holiday_params.copy().keys()
    for key in group_holiday_params_key_copy:
        if key in inspect.signature(HolidayGrouper).parameters:
            grouper_init_params[key] = group_holiday_params.pop(key)

    # constructs initial `holiday_df`. If `holiday_df` is passed through `auto_holiday_params`, use it. Otherwise,
    # constructs it based on input country list and/or user input holidays through `daily_event_df_dict`.
    if grouper_init_params["holiday_df"] is not None:
        holiday_df = grouper_init_params["holiday_df"].copy()
        holiday_date_col = grouper_init_params["holiday_date_col"]
        holiday_name_col = grouper_init_params["holiday_name_col"]
        if not {holiday_date_col, holiday_name_col}.issubset(holiday_df.columns):
            raise ValueError(f"Columns `{holiday_date_col}` and/or "
                             f"`{holiday_name_col}` not found in input `holiday_df`.")
        # Standardizes `holiday_name_col` and `holiday_name_col` in `holiday_df`.
        holiday_df = holiday_df.rename(columns={
            holiday_date_col: cst.EVENT_DF_DATE_COL,
            holiday_name_col: cst.EVENT_DF_LABEL_COL
        })
    else:
        # Constructs `holiday_df` based on input country list and/or user input holidays through `daily_event_df_dict`.
        # When `holiday_lookup_countries` is not empty, the corresponding `holiday_df_from_countries` is constructed.
        holiday_df_from_countries = None
        if len(holiday_lookup_countries) > 0:
            holiday_df_from_countries_dict = get_holidays(
                countries=holiday_lookup_countries,
                year_start=start_year - 1,
                year_end=end_year + 1)
            holiday_df_from_countries_list = [holidays for _, holidays in holiday_df_from_countries_dict.items()]
            holiday_df_from_countries = pd.concat(holiday_df_from_countries_list)
            # Removes the observed holidays and only keeps the original holidays. This assumes that the original
            # holidays are consistently present in the output of `get_holidays` when their observed counterparts are
            # included. This assumption has been verified for all available countries within the date range of
            # 2015 to 2030.
            cond_observed_holiday = holiday_df_from_countries[cst.EVENT_DF_LABEL_COL].apply(
                lambda x: True if any(i in x for i in ["(Observed)", "(observed)"]) else False)
            holiday_df_from_countries = holiday_df_from_countries.loc[~cond_observed_holiday, [cst.EVENT_DF_DATE_COL, cst.EVENT_DF_LABEL_COL]]

        # When `daily_event_df_dict` is not empty, its format gets converted to dataframe `holiday_df_from_dict`.
        holiday_df_from_dict = None
        if daily_event_df_dict:
            holiday_df_from_dict_list = []
            for holiday_name, holiday_dates in daily_event_df_dict.items():
                holiday_dates[cst.EVENT_DF_LABEL_COL] = holiday_name
                # Checks and finds the first column where values can be recognized by `pd.to_datetime`, uses it as
                # the date column and cast it as `cst.EVENT_DF_DATE_COL`.
                flag = False
                for col in holiday_dates.columns:
                    try:
                        holiday_dates[cst.EVENT_DF_DATE_COL] = pd.to_datetime(holiday_dates[col])
                        flag = True
                    except ValueError:
                        continue
                    # When a valid date column is found, breaks the loop.
                    if flag:
                        break
                if flag is False:
                    raise ValueError(f"No valid date column is found in data frames in `daily_event_df_dict` to use "
                                     f"as {cst.EVENT_DF_DATE_COL}.")
                holiday_df_from_dict_list.append(holiday_dates[[cst.EVENT_DF_DATE_COL, cst.EVENT_DF_LABEL_COL]])
            holiday_df_from_dict = pd.concat(holiday_df_from_dict_list)

        if (holiday_df_from_countries is None) & (holiday_df_from_dict is None):
            raise ValueError("Automatic holiday grouping is enabled. Holiday list needs to be specified through"
                             "`holiday_lookup_countries` or `daily_event_df_dict`. Currently, None is found.")
        holiday_df = (
            pd.concat([holiday_df_from_countries, holiday_df_from_dict])
            .drop_duplicates()
            .reset_index(drop=True)
        )
    # Makes sure the `holiday_date_col` and `holiday_name_col` in `grouper_init_params` are standardized values.
    grouper_init_params["holiday_date_col"] = cst.EVENT_DF_DATE_COL
    grouper_init_params["holiday_name_col"] = cst.EVENT_DF_LABEL_COL

    # Separates holidays specified in `holidays_to_model_separately` from `holiday_df`, so that they will not be passed
    # into `HolidayGrouper`. When `holidays_to_model_separately` is not empty, a dictionary is constructed with
    # each key presents one holiday or a specific neighboring day for all holidays in `holidays_to_model_separately`.
    if holidays_to_model_separately is None:
        holidays_to_model_separately = []
    elif holidays_to_model_separately == "auto":
        holidays_to_model_separately = []
        log_message("Automatic holiday grouping is enabled. The `holidays_to_model_separately` parameter should be "
                    "`None` or a list. Since the current input is 'auto', it is set to an empty list and no"
                    "holiday will be modeled separately.",
                    LoggingLevelEnum.WARNING)

    if not isinstance(holidays_to_model_separately, (list, tuple)):
        raise ValueError(
            f"Automatic holiday grouping is enabled. The `holidays_to_model_separately` parameter should be `None` or "
            f"a list, found {holidays_to_model_separately}")
    elif len(holidays_to_model_separately) == 0:
        holiday_df_exclude_separate = holiday_df.copy()
        holiday_df_dict_separate_with_effect = dict()
    else:
        holiday_to_separate_condition = holiday_df[cst.EVENT_DF_LABEL_COL].isin(holidays_to_model_separately)
        holiday_df_exclude_separate = holiday_df[~holiday_to_separate_condition]
        holiday_df_separate = holiday_df[holiday_to_separate_condition]

        # Initializes the holiday dictionary for holidays modeled separately. Each key corresponds to
        # a holiday.
        holiday_df_dict_separate_with_effect = split_events_into_dictionaries(
            events_df=holiday_df_separate,
            events=holidays_to_model_separately)

        # Removes "'" from keys in `pre_post_num_dict_processed` because they are
        # removed from holiday names by `split_events_into_dictionaries`.
        if grouper_init_params["holiday_impact_dict"]:
            pre_post_num_dict_processed = grouper_init_params["holiday_impact_dict"].copy()
            for key in pre_post_num_dict.keys():
                new_key = key.replace("'", "")
                pre_post_num_dict_processed[new_key] = pre_post_num_dict_processed.pop(key)
        else:
            pre_post_num_dict_processed = dict()

        # Adds shifted events.
        shifted_event_dict = add_event_window_multi(
            event_df_dict=holiday_df_dict_separate_with_effect,
            time_col=cst.EVENT_DF_DATE_COL,
            label_col=cst.EVENT_DF_LABEL_COL,
            time_delta="1D",
            pre_num=grouper_init_params["holiday_impact_pre_num_days"],
            post_num=grouper_init_params["holiday_impact_post_num_days"],
            pre_post_num_dict=pre_post_num_dict_processed)
        holiday_df_dict_separate_with_effect.update(shifted_event_dict)

        # Raises warning if `holiday_df_exclude_separate` becomes empty, as there will be no holidays for grouping.
        # Returns `holiday_df_dict_separate_with_effect` in this case.
        if holiday_df_exclude_separate.empty:
            log_message(f"All input holidays are modeled separately and no remaining holidays can be used by the "
                        f"holiday grouper. A dictionary of all holidays with effects modeled separately is returned.",
                        LoggingLevelEnum.WARNING)
            return holiday_df_dict_separate_with_effect

    # Reassigns `grouper_init_params["holiday_df"]` with `holiday_df_exclude_separate` that excludes holidays that are
    # modeled separately.
    grouper_init_params["holiday_df"] = holiday_df_exclude_separate.copy()

    # Checks if `df` in `grouper_init_params` is None or empty.
    if (grouper_init_params["df"] is None) or grouper_init_params["df"].empty:
        raise ValueError("Automatic holiday grouping is enabled. Dataframe cannot be `None` or empty.")
    # Checks if `df`, `time_col`, `value_col` in `grouper_init_params` are valid.
    # Reassigns the values as they can potentially be overriden by `auto_holiday_params`.
    df = grouper_init_params["df"].copy()
    time_col = grouper_init_params["time_col"]
    value_col = grouper_init_params["value_col"]
    if not {time_col, value_col}.issubset(df.columns):
        raise ValueError("Input `df` for holiday grouper does not contain `time_col` or `value_col`.")

    # First pre-processes `df` to determine if it is appropriate for holiday effects inference.
    # If the data is sub-daily, aggregates it to daily before grouping.
    # If the data is less granular than daily, raise ValueError.

    # Converts time column.
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    # First infers frequency, if more granular than daily, aggregates it to daily.
    # If less granular than daily, raise ValueError.
    time_stats = describe_timeseries(df=df, time_col=time_col)
    if time_stats["freq_in_days"] > 1.0:
        raise ValueError("Input holiday df for holiday grouper has frequency less than daily. "
                         "Holiday effect cannot be inferred.")
    elif time_stats["freq_in_days"] < 1.0:
        df_tmp = df.resample("D", on=time_col).agg({value_col: np.nanmean})
        df = (df_tmp.drop(columns=time_col).reset_index() if time_col in df_tmp.columns
              else df_tmp.reset_index())
    # Reassigns back processed `df`.
    grouper_init_params["df"] = df

    # Calls holiday grouper.
    hg = HolidayGrouper(
        **grouper_init_params)
    hg.group_holidays(
        **group_holiday_params)

    daily_event_df_dict_exclude_separate = hg.result_dict["daily_event_df_dict"]

    # Adds back holidays that are modeled separately with their neighboring days.
    daily_event_df_dict_final = update_dictionary(
        daily_event_df_dict_exclude_separate,
        overwrite_dict=holiday_df_dict_separate_with_effect)

    return daily_event_df_dict_final
