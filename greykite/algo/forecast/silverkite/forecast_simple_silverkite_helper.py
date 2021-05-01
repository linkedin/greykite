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
# original authors: Albert Chen, Reza Hosseini
"""Helper functions for
`~greykite.algo.forecast.silverkite.forecast_simple_silverkite.py.`
"""

import warnings

import pandas as pd

from greykite.common import constants as cst
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import EVENT_INDICATOR
from greykite.common.features.timeseries_features import add_event_window_multi
from greykite.common.features.timeseries_features import get_fourier_col_name
from greykite.common.features.timeseries_features import get_holidays


def cols_interact(
        static_col,
        fs_name,
        fs_order,
        fs_seas_name=None):
    """Returns all interactions between static_col and fourier series up to specified order

    :param static_col:
        column to interact with fourier series. can be an arbitrary patsy model term
        e.g. "ct1", "C(woy)", "is_weekend:Q('events_Christmas Day')"
    :param fs_name:
        column the fourier series is generated from, same as col_name in fourier_series_fcn
    :param fs_order: int
        generate interactions up to this order. must be <= order in fourier_series_fcn
    :param fs_seas_name: str
        same as seas_name in fourier_series_fcn
    :return: list[str]
        interaction terms to include in patsy model formula
    """
    interaction_columns = [None] * fs_order * 2
    for i in range(fs_order):
        k = i + 1
        sin_col_name = get_fourier_col_name(
            k,
            fs_name,
            function_name="sin",
            seas_name=fs_seas_name)
        cos_col_name = get_fourier_col_name(
            k,
            fs_name,
            function_name="cos",
            seas_name=fs_seas_name)
        interaction_columns[2*i] = f"{static_col}:{sin_col_name}"
        interaction_columns[2*i + 1] = f"{static_col}:{cos_col_name}"
    return interaction_columns


def dedup_holiday_dict(holidays_dict):
    """Removes duplicates from get_holidays output

    :param holidays_dict: dict(str, pd.DataFrame(EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL))
        dictionary from get_holidays
    :return:
        concatenates rows of all DataFrames in holiday_df
        drops duplicate holiday names
    """
    result = pd.DataFrame()
    for country, country_holiday_df in holidays_dict.items():
        result = pd.concat([result, country_holiday_df], axis=0)
    result.drop_duplicates(inplace=True)
    return result


def split_events_into_dictionaries(
        events_df,
        events,
        date_col=EVENT_DF_DATE_COL,
        name_col=EVENT_DF_LABEL_COL,
        default_category="Other"):
    """Splits pd.Dataframe(date, holiday) into separate dataframes, one per event

    Can be used to create the `daily_event_df_dict` parameter for `forecast_silverkite`.
    Each event specified in `events` gets its own effect in the model.
        Other events are grouped together and modeled with the same effect

    :param events_df: pd.DataFrame with date_col and name_col columns
        contains events
    :param events: list(str)
        names of events in events_df.name_col.unique() to split into separate
        dataframes
    :param date_col: str, default "date"
        column in event_df containing the date
    :param name_col: str, default EVENT_DF_LABEL_COL
        column in event_df containing the event name
    :param default_category: str
        name of default event

    :return: dict(label: pd.Dataframe(date_col, name_col))
        with keys = events + [default_category]
            name_col column has a constant value = EVENT_INDICATOR
    """
    result = {}
    # separates rows corresponding to each event into their own dataframe
    for event_name in events:
        event_df = events_df[events_df[name_col] == event_name].copy()
        if event_df.shape[0] > 0:
            event_df[name_col] = EVENT_INDICATOR  # All dates in this df are for the event
            event_key = event_name.replace("'", "")  # ensures patsy module can parse column name in formula
            result[event_key] = event_df.drop_duplicates().reset_index(drop=True)
        else:
            warnings.warn(
                f"Requested holiday '{event_name}' does not occur in the provided countries")

    # groups other events into the same bucket
    other_df = events_df[~events_df[name_col].isin(events)].copy()
    if other_df.shape[0] > 0:
        other_df[name_col] = EVENT_INDICATOR
        default_category = default_category.replace("'", "")
        result[default_category] = other_df.drop_duplicates().reset_index(drop=True)

    # there must be no duplicated dates in each DataFrame
    for k, df in result.items():
        assert not any(df[date_col].duplicated())

    return result


def generate_holiday_events(
        countries,
        holidays_to_model_separately,
        year_start,
        year_end,
        pre_num,
        post_num,
        pre_post_num_dict=None,
        default_category="Other"):
    """Returns holidays within the countries between ``year_start`` and ``year_end``.
    Creates a separate key, value for each item in ``holidays_to_model_separately``.
    The rest are grouped together.

    Useful when multiple countries share the same holiday (e.g. New Year's Day),
    to model a single effect for that holiday.

    Parameters
    ----------
    countries : `list` [`str`]
        Countries of interest.
    holidays_to_model_separately : `list` [`str`]
        Holidays to model.
    year_start: `int`
        Start year for holidays.
    year_end: `int`
        Ending year for holidays.
    pre_num: `int`
        Days to model a separate effect prior to each holiday
    post_num: `int`
        Days to model a separate effect after each holiday.
    pre_post_num_dict : `dict` [`str`, (`int`, `int`)] or None, default None
        Overrides ``pre_num`` and ``post_num`` for each holiday in
        ``holidays_to_model_separately``.
        For example, if ``holidays_to_model_separately`` contains "Thanksgiving" and "Labor Day",
        this parameter can be set to ``{"Thanksgiving": [1, 3], "Labor Day": [1, 2]}``,
        denoting that the "Thanksgiving" ``pre_num`` is 1 and ``post_num`` is 3, and "Labor Day"
        ``pre_num`` is 1 and ``post_num`` is 2.
        Holidays not specified use the default given by ``pre_num`` and ``post_num``.
    default_category: `str`
        Default category name, for holidays in countries not included
        in ``holidays_to_model_separately``.

    Returns
    -------
    daily_event_df_dict : `dict` [`str`, `pandas.DataFrame` (EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL)]
        suitable for use as ``daily_event_df_dict`` parameter in ``forecast_silverkite``
    """
    # retrieves separate DataFrame for each country, with list of holidays
    holidays_dict = get_holidays(
        countries,
        year_start=year_start,
        year_end=year_end)
    if len(holidays_dict) == 0:  # requested holidays are not found the countries
        daily_event_df_dict = None
    else:
        # merges country DataFrames, removes duplicate holidays
        holiday_df = dedup_holiday_dict(holidays_dict)
        # creates separate DataFrame for each holiday
        daily_event_df_dict = split_events_into_dictionaries(
            holiday_df,
            holidays_to_model_separately,
            default_category=default_category)

        # Removes "'" from keys in `pre_post_num_dict` because they are
        # removed from holiday names by ``split_events_into_dictionaries``.
        if pre_post_num_dict:
            # ``.copy()`` is used below to avoid altering the dictionary keys within iteration on same keys
            keys = pre_post_num_dict.copy().keys()
            for key in keys:
                new_key = key.replace("'", "")
                if key not in daily_event_df_dict:
                    pre_post_num_dict[new_key] = pre_post_num_dict.pop(key)
                if new_key not in daily_event_df_dict:
                    warnings.warn(
                        f"Requested holiday '{new_key}' is not valid. Valid holidays are: "
                        f"{list(daily_event_df_dict.keys())}", UserWarning)

        shifted_event_dict = add_event_window_multi(
            event_df_dict=daily_event_df_dict,
            time_col=EVENT_DF_DATE_COL,
            label_col=EVENT_DF_LABEL_COL,
            time_delta="1D",
            pre_num=pre_num,
            post_num=post_num,
            pre_post_num_dict=pre_post_num_dict)

        daily_event_df_dict.update(shifted_event_dict)
    return daily_event_df_dict


def patsy_categorical_term(
        term,
        levels=None,
        coding=None,
        quote=True):
    """Returns categorical term for patsy.
        Optionally specify levels, coding, and quote the term
    :param term: str
        name of the categorical variable
    :param levels: list(str) or None
        levels for the categorical variable
    :param coding: str
        A valid coding. E.g. Treatment, Sum, Diff, Poly
        https://patsy.readthedocs.io/en/latest/API-reference.html#handling-categorical-data
    :param quote: bool
        whether to quote the term. Useful if there is a space or "." in the term
    :return: str
        categorical factor for patsy model formula
    """
    if quote:
        term = f"Q('{term}')"
    # constructs the string for the patsy model term
    string = f"C({term}"
    if coding is not None:
        string += f", {coding}"
    if levels is not None:
        string += f", levels={levels}"
    string += ")"
    return string


def get_event_pred_cols(daily_event_df_dict):
    """Generates the names of internal predictor columns from
    the event dictionary passed to
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
    These can be passed via the ``extra_pred_cols`` parameter to model event effects.

    .. note::

        The returned strings are patsy model formula terms.
        Each provides full set of levels so that prediction works even if a
        level is not found in the training set.

        If a level does not appear in the training set, its coefficient may
        be unbounded in the "linear" fit_algorithm. A method with regularization
        avoids this issue (e.g. "ridge", "elastic_net").

    Parameters
    ----------
    daily_event_df_dict : `dict` or None, optional, default None
        A dictionary of data frames, each representing events data for the corresponding key.
        See `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

    Returns
    -------
    event_pred_cols : `list` [`str`]
        List of patsy model formula terms, one for each
        key of ``daily_event_df_dict``.
    """
    event_pred_cols = []
    if daily_event_df_dict is not None:
        for key in sorted(daily_event_df_dict.keys()):
            # `add_daily_events` creates a column with this name.
            term = f"{cst.EVENT_PREFIX}_{key}"
            # Its values are set to the event df label column. Dates that do not correspond
            # to the event are set to `cst.EVENT_DEFAULT`.
            event_levels = [cst.EVENT_DEFAULT]  # reference level for non-event days
            event_levels += list(daily_event_df_dict[key][cst.EVENT_DF_LABEL_COL].unique())  # this event's levels
            event_pred_cols += [patsy_categorical_term(term=term, levels=event_levels)]
    return event_pred_cols
