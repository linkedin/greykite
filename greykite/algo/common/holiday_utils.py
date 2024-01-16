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
# original author: Yi Su, Kaixu Yang
"""Constants and utility functions for `HolidayInferrer` and `HolidayGrouper`."""

import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from pandas.tseries.frequencies import to_offset

from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import get_event_pred_cols
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import EVENT_PREFIX
from greykite.common.constants import EVENT_SHIFTED_SUFFIX_AFTER
from greykite.common.constants import EVENT_SHIFTED_SUFFIX_BEFORE
from greykite.common.python_utils import split_offset_str


HOLIDAY_NAME_COL = "country_holiday"
"""Holiday name column used in `HolidayInferrer`.
This comes from the default output of `holidays_ext.get_holidays.get_holiday_df`.
"""

HOLIDAY_DATE_COL = "ts"
"""Holiday date column used in `HolidayInferrer`.
This comes from the default output of `holidays_ext.get_holidays.get_holiday_df`.
"""

HOLIDAY_IMPACT_DICT = {
    "All Saints Day": (1, 1),  # 11/1.
    "Ascension Day": (1, 4),  # Thursday.
    "Assumption of Mary": (4, 4),  # 8/15, duplicated with "India_Independence Day".
    "Chinese New Year": (4, 4),  # Varying.
    "Christmas Day": (4, 3),  # 12/25.
    "Diwali": (4, 4),  # Varying.
    "Easter Sunday": (6, 1),  # Sunday.
    "Eid al-Adha": (1, 1),  # Varying.
    "Eid al-Fitr": (1, 1),  # Varying.
    "Europe DST End": (0, 0),  # Sunday.
    "Europe DST Start": (0, 0),  # Sunday.
    "Halloween": (1, 1),  # 10/31.
    "New Year's Day": (3, 3),  # 1/1.
}
"""Number of pre/post days that a holiday has impact on.
For example, if Halloween has neighbor (1, 1), Halloween_minus_1
and Halloween_plus_1 will be generated as two additional events.
If not specified, (0, 0) will be used.
"""


def get_dow_grouped_suffix(date: datetime.datetime) -> str:
    """Utility function to generate a suffix given an input ``date``.

    Parameters
    ----------
    date : `datetime.datetime`
        Input timestamp.

    Returns
    -------
    suffix : `str`
        The suffix string starting with "_".
    """
    if date.day_name() == "Saturday":
        return "_Sat"
    elif date.day_name() == "Sunday":
        return "_Sun"
    elif date.day_name() in ["Friday", "Monday"]:
        return "_Mon/Fri"
    else:
        return "_WD"


def get_weekday_weekend_suffix(date: datetime.datetime) -> str:
    """Utility function to generate a suffix given an input ``date``.

    Parameters
    ----------
    date : `datetime.datetime`
        Input timestamp.

    Returns
    -------
    suffix : `str`
        The suffix string starting with "_".
    """
    if date.day_name() in ("Saturday", "Sunday"):
        return "_WE"
    else:
        return "_WD"


def get_autoreg_holiday_interactions(
        daily_event_df_dict: Dict[str, pd.DataFrame],
        lag_names: List[str]) -> List[str]:
    """Gets the interaction terms between holidays and autoregression terms or other lag terms.

    Parameters
    ----------
    daily_event_df_dict : `Dict` [`str`, `pandas.DataFrame`]
        The input event configuration.
        A dictionary with keys being the event names and values being a pandas DataFrame of dates.
        See `~greykite.algo.forecast.silverkite.forecast_silverkite` for details.
    lag_names : `List` [`str`]
        A list of lag names, e.g. ["y_lag1"].
        Each will be interacting with the holidays.

    Returns
    -------
    interactions : `List` [`str`]
        A list of interaction terms between holidays and lags, to be passed to ``extra_pred_cols``.
    """
    holiday_names = get_event_pred_cols(daily_event_df_dict)
    interactions = [f"{holiday}:{lag}" for lag in lag_names for holiday in holiday_names]
    return interactions


def add_shifted_events(
        daily_event_df_dict: Dict[str, pd.DataFrame],
        shifted_effect_lags: Optional[List[str]] = None,
        event_df_date_col: str = EVENT_DF_DATE_COL,
        event_df_label_col: str = EVENT_DF_LABEL_COL) -> Dict[str, Any]:
    """This function does two things:

        - (1) adds shifted events to ``daily_event_df_dict`` and returns the new event dictionary.
        - (2) returns a list of new column names to be added in the model.
          This is useful when we need to remove these main effects from the model.

    Parameters
    ----------
    daily_event_df_dict : `Dict` [`str`, `pandas.DataFrame`]
        The input event configuration.
        A dictionary with keys being the event names and values being a pandas DataFrame of dates.
        See `~greykite.algo.forecast.silverkite.forecast_silverkite` for details.
    shifted_effect_lags : `List` [`str`] or None, default None
        Additional neighbor events based on given events.
        For example, passing ["-1D", "7D"] will add extra daily events which are 1 day before
        and 7 days after the given events.
        Offset format is {d}{freq} with any integer plus a frequency string. Must be parsable by pandas ``to_offset``.
        The new events' names will be the current events' names with suffix "{offset}_before" or "{offset}_after".
        For example, if we have an event named "US_Christmas Day",
        a "7D" shift will have name "US_Christmas Day_7D_after".
        This is useful when you expect an offset of the current holidays also has impact on the
        time series, or you want to interact the lagged terms with autoregression.
    event_df_date_col : `str`, default ``EVENT_DF_DATE_COL``
        Date column of the dataframes in ``daily_event_df_dict``.
    event_df_label_col : `str`, default ``EVENT_DF_LABEL_COL``
        Label column of the dataframes in ``daily_event_df_dict``.

    Returns
    -------
    shifted_events_dict : `Dict` [`str`, `Any`]
        A dictionary of results:

            - "new_daily_event_df_dict": the new event dictionary that expands the input
                ``daily_event_df_dict`` by adding new shifted events.
                Note that this is intended to be used to manually add the events.
                One can also specify the ``events["daily_event_shifted_effect"]`` field to directly add them.
            - "shifted_events_cols": the column names of the newly added shifted event in the model.
                This is useful when we need to remove these main effects from the model.
                One can specify this in ``drop_pred_cols`` to achieve this.
    """
    if shifted_effect_lags is None:
        shifted_effect_lags = []

    new_daily_event_df_dict = {}
    shifted_events_cols = []
    drop_pred_cols = []
    for order in shifted_effect_lags:
        num, freq = split_offset_str(order)
        if num == 0:
            break
        num = int(num)
        lag_offset = to_offset(order)
        for name, event_df in daily_event_df_dict.items():
            new_event_df = event_df.copy()
            new_event_df[event_df_date_col] = pd.to_datetime(new_event_df[event_df_date_col])
            new_event_df[event_df_date_col] += lag_offset
            # Sets suffix of the new event.
            suffix = EVENT_SHIFTED_SUFFIX_BEFORE if num < 0 else EVENT_SHIFTED_SUFFIX_AFTER
            # Creates a new dataframe to add to `new_daily_event_df_dict`.
            new_name = f"{name}_{abs(num)}{freq}{suffix}"
            new_event_df[event_df_label_col] = new_name
            new_daily_event_df_dict[new_name] = new_event_df
            # Records the new column name in the model.
            new_col = f"{EVENT_PREFIX}_{new_name}"
            shifted_events_cols.append(new_col)

    if len(new_daily_event_df_dict) > 0:
        drop_pred_cols = get_event_pred_cols(new_daily_event_df_dict)

    new_daily_event_df_dict.update(daily_event_df_dict)

    return {
        "new_daily_event_df_dict": new_daily_event_df_dict,
        "shifted_events_cols": shifted_events_cols,
        "drop_pred_cols": drop_pred_cols
    }
