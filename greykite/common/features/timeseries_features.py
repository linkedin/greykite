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
# original author: Reza Hosseini, Albert Chen, Kaixu Yang, Sayan Patra
"""Functions to generate derived time features useful
in forecasting, such as growth, seasonality, holidays.
"""

import math
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz
from holidays_ext import get_holidays as get_hdays
from pandas.tseries.frequencies import to_offset
from scipy.special import expit

from greykite.common import constants as cst
from greykite.common.python_utils import split_offset_str


def convert_date_to_continuous_time(dt):
    """Converts date to continuous time. Each year is one unit.

    Parameters
    ----------
    dt : datetime object
        the date to convert

    Returns
    -------
    conti_date : `float`
        the date represented in years
    """
    year_length = datetime(dt.year, 12, 31).timetuple().tm_yday
    tt = dt.timetuple()

    return (dt.year +
            (tt.tm_yday - 1
             + dt.hour / 24
             + dt.minute / (24 * 60)
             + dt.second / (24 * 3600)) / float(year_length))


def get_default_origin_for_time_vars(df, time_col):
    """Sets default value for origin_for_time_vars

    Parameters
    ----------
    df : `pandas.DataFrame`
        Training data. A data frame which includes the timestamp and value columns
    time_col : `str`
        The column name in `df` representing time for the time series data.

    Returns
    -------
    dt_continuous_time : `float`
        The time origin used to create continuous variables for time
    """
    date = pd.to_datetime(df[time_col].iloc[0])
    return convert_date_to_continuous_time(date)


def pytz_is_dst_fcn(time_zone):
    """For a given timezone, it constructs a function which determines
    if a timestamp (`dt`) is inside the daylight saving period or not for
    a list of timestamps.

    This function, should work for regions in US / Canada and Europe.

    The returned function assumes that the timestamps are in the given
    ``time_zone``.
    Note that since daylight saving is the same for all of mainland US / Canada,
    one can pass any US time zone e.g. ``"US/Pacific"`` to construct a function
    which works for all of mainland US.
    Similarly for most of Europe, it is suffcient to pass any Europe time zone e.g.
    ``"Europe/London"``.

    Note: Since this function is slow, a faster version is available:
    `~greykite.common.features.timeseries_features.is_dst_fcn`.
    However, we expect the current function would be more accurate assuming
    the package `pytz` keeps up to date with potential changes in DST.


    Parameters
    ----------
    time_zone : `str`
        A string denoting the timestamp e.g. "US/Pacific", "Canada/Eastern",
        "Europe/London".

    Returns
    -------
    is_dst : callable
        A function which takes a list of datetime-like objects
        and returns a list of colleans to determine if each timestamp is in daylight saving.
    """
    timezone = pytz.timezone(time_zone)

    def is_dst(dt):
        """A function which takes a list of datetime-like objects
        and returns a list of booleans to determine if that timestamp
        is in daylight saving.

        Parameters
        ----------
        dt : `list` of datetime-like object

        Returns
        -------
        result : `list` [`bool`]
            A list of booleans:

            - If True, the input time is in daylight saving.
            - If False, the input time is NOT in daylight saving.

        """
        diff = []
        for dt0 in dt:
            timezone_date = timezone.localize(dt0, is_dst=False)
            diff.append(timezone_date.tzinfo._dst.seconds)
        return list(pd.Series(diff) != 0)

    return is_dst


def get_us_dst_start(year):
    """For each year, it returns the second Sunday in March,
    which is the start of the daylight saving (DST) in US/Canada.

    We assume DST starts on Second Sunday of March at 2 a.m.

    Parameters
    ----------
    year : `int`
        Year for which DST start date is desired.

    Returns
    -------
    result : `datetime.datetime`
        The timestamp of start of DST in US/Canada.
    """
    # Finds a date in third week of March.
    date_in_3rd_week = datetime(year, 3, 15, 2)
    # Finds out which week day it is:
    weekday = date_in_3rd_week.weekday()
    # Finds the Sunday before that by going back to Monday and does an extra -1.
    second_sunday_date = date_in_3rd_week.replace(day=(15 - weekday - 1))
    return second_sunday_date


def get_us_dst_end(year):
    """For each year, it returns the first Sunday in November,
    which is the end of the daylight saving (DST) in US/Canada.

    We assume DST ends on Second Sunday of Novemeber at 2 a.m.

    Parameters
    ----------
    year : `int`
        Year for which DST end date is desired.

    Returns
    -------
    result : `datetime.datetime`
        The timestamp of end of DST in US/Canada.
    """
    # Finds the first date in the second week.
    date_in_2nd_week = datetime(year, 11, 8, 2)
    # Finds out which week day it is:
    weekday = date_in_2nd_week.weekday()
    # Goes back to Monday of the second week and does an extra -1.
    first_sunday_date = date_in_2nd_week.replace(day=(8 - weekday - 1))
    return first_sunday_date


def get_eu_dst_start(year):
    """For each year, it returns the last Sunday in March,
    which is the start of the daylight saving (DST) in Europe.

    We assume Europe DST starts on last Sunday of March at 1 a.m.

    Parameters
    ----------
    year : `int`
        Year for which DST start date is desired.

    Returns
    -------
    result : `datetime.datetime`
        The timestamp of start of DST in Europe.
    """
    # March is 31 days.
    # Finds the last date in the month.
    date_in_last_week = datetime(year, 3, 31, 1)
    # Finds out which week day it is:
    weekday = date_in_last_week.weekday()
    # If above date is already a Sunday, returns it.
    if weekday == 6:
        return date_in_last_week
    # Otherwise, goes back to the Monday of that week and does an extra -1.
    last_sunday_date = date_in_last_week.replace(day=(31 - weekday - 1))
    return last_sunday_date


def get_eu_dst_end(year):
    """For each year, it returns the last Sunday in October,
    which is the end of the daylight saving (DST) in Europe.

    We assume Europe DST ends on last Sunday of October at 2 a.m.

    Parameters
    ----------
    year : `int`
        Year for which DST end date is desired.

    Returns
    -------
    result : `datetime.datetime`
        The timestamp of end of DST in Europe.
    """
    # October is 31 days.
    # Finds the last date in the month.
    date_in_last_week = datetime(year, 10, 31, 2)
    # Finds out which week day it is:
    weekday = date_in_last_week.weekday()
    # If above date is already a Sunday, returns it.
    if weekday == 6:
        return date_in_last_week
    # Otherwise, goes back to the Monday of that week and does an extra -1.
    last_sunday_date = date_in_last_week.replace(day=(31 - weekday - 1))
    return last_sunday_date


def is_dst_fcn(time_zone):
    """For a given timezone, it constructs a function which determines
    if a timestamp (`dt`) is inside the daylight saving period or not for
    a list of timestamps.

    This function, should work for regions in US / Canada and Europe.

    The returned function assumes that the timestamps are in the given
    ``time_zone``.
    Note that since daylight saving is the same for all of mainland US / Canada,
    one can pass any US time zone e.g. ``"US/Pacific"`` to construct a function
    which works for all of mainland US.
    Similarly for most of Europe, it is suffcient to pass any Europe time zone e.g.
    ``"Europe/London"``.

    Some references on when did DST start in modern era:

    - Europe: https://www.timeanddate.com/time/europe/daylight-saving-history.html
    - US: https://en.wikipedia.org/wiki/Daylight_saving_time_in_the_United_States

    Note: This function assumes the DST rules remain the same as what they
    are in the year 2022 (when this code was written).
    A potentially more accurate (but much slower) version is available:
    `~greykite.common.features.timeseries_features.pytz_is_dst_fcn`.
    However, we expect the current function would be much faster and it can be
    updated in case DST rules change.


    Parameters
    ----------
    time_zone : `str`
        A string denoting the timestamp e.g. "US/Pacific", "Canada/Eastern",
        "Europe/London".

    Returns
    -------
    is_dst : callable
        A function which takes a list of datetime-like objects
        and returns a list of colleans to determine if each timestamp is in daylight saving.
    """
    if "US" in time_zone or "Canada" in time_zone:
        get_dst_start = get_us_dst_start
        get_dst_end = get_us_dst_end
    elif "Europe" in time_zone:
        get_dst_start = get_eu_dst_start
        get_dst_end = get_eu_dst_end
    else:
        raise ValueError(
            f"`time_zone` string {time_zone} does not include "
            "either of: 'US'/'Canada'/'Europe'")

    # For US, the current convention seems to have started in 2007
    # See references in function docstring
    us_year_range = range(2007, 2080)
    us_starts = {year: get_dst_start(year) for year in us_year_range}
    us_ends = {year: get_dst_end(year) for year in us_year_range}

    # For Europe, the current convention seems to have started in 1996
    # See references in function docstring:
    # Quoting from the link: "In 1996, the European Union (EU) standardized the DST schedule"
    europe_year_range = range(1996, 2080)
    europe_starts = {year: get_dst_start(year) for year in europe_year_range}
    europe_ends = {year: get_dst_end(year) for year in europe_year_range}

    if "US" in time_zone or "Canada" in time_zone:
        year_range = us_year_range
        starts = us_starts
        ends = us_ends
    else:
        # Note that due to above if statements, we now that else maps to "Europe"
        # Otherwise a `ValueError` would have been raised.
        year_range = europe_year_range
        starts = europe_starts
        ends = europe_ends

    def is_dst(dt):
        """A function which takes a list of datetime-like objects
        and returns a list of booleans to determine if that timestamp
        is in daylight saving.

        Parameters
        ----------
        dt : `list` of datetime-like object

        Returns
        -------
        result : `list` [`bool`]
            A list of booleans:

            - If True, the input time is in daylight saving.
            - If False, the input time is NOT in daylight saving.

        """
        is_dst_bool = []
        for dt0 in dt:
            year = dt0.year
            if year in year_range:
                start, end = starts[year], ends[year]
                if dt0 >= start and dt0 <= end:
                    # This will be true at most for one year in the range
                    is_dst_bool.append(True)
                else:
                    is_dst_bool.append(False)
            else:
                # This is the rare case for which the timestamp is not within
                # the range of all years considered in `year_range = range(1950, 2080)`
                is_dst_bool.append(False)

        return is_dst_bool

    return is_dst


def build_time_features_df(
        dt,
        conti_year_origin,
        add_dst_info=True):
    """This function gets a datetime-like vector and creates new columns containing temporal
    features useful for time series analysis and forecasting e.g. year, week of year, etc.

    Parameters
    ----------
    dt : array-like (1-dimensional)
        A vector of datetime-like values
    conti_year_origin : `float`
        The origin used for creating continuous time which is in years unit.
    add_dst_info : `bool`, default True
        Determines if daylight saving columns for US and Europe should be added.
    Returns
    -------
    time_features_df : `pandas.DataFrame`
        Dataframe with the following time features.

            * "datetime": `datetime.datetime` object, a combination of date and a time
            * "date": `datetime.date` object, date with the format (year, month, day)
            * "year": integer, year of the date e.g. 2018
            * "year_length": integer, number of days in the year e.g. 365 or 366
            * "quarter": integer, quarter of the date, 1, 2, 3, 4
            * "quarter_start": `pandas.DatetimeIndex`, date of beginning of the current quarter
            * "quarter_length": integer, number of days in the quarter, 90/91 for Q1, 91 for Q2, 92 for Q3 and Q4
            * "month": integer, month of the year, January=1, February=2, ..., December=12
            * "month_length": integer, number of days in the month, 28/ 29/ 30/ 31
            * "woy": integer, ISO 8601 week of the year where a week starts from Monday, 1, 2, ..., 53
            * "doy": integer, ordinal day of the year, 1, 2, ..., year_length
            * "doq": integer, ordinal day of the quarter, 1, 2, ..., quarter_length
            * "dom": integer, ordinal day of the month, 1, 2, ..., month_length
            * "dow": integer, day of the week, Monday=1, Tuesday=2, ..., Sunday=7
            * "str_dow": string, day of the week as a string e.g. "1-Mon", "2-Tue", ..., "7-Sun"
            * "str_doy": string, day of the year e.g. "2020-03-20" for March 20, 2020
            * "hour": integer, discrete hours of the datetime, 0, 1, ..., 23
            * "minute": integer, minutes of the datetime, 0, 1, ..., 59
            * "second": integer, seconds of the datetime, 0, 1, ..., 3599
            * "year_month": string, (year, month) e.g. "2020-03" for March 2020
            * "year_woy": string, (year, week of year) e.g. "2020_42" for 42nd week of 2020
            * "month_dom": string, (month, day of month) e.g. "02/20" for February 20th
            * "year_woy_dow": string, (year, week of year, day of week) e.g. "2020_03_6" for Saturday of 3rd week in 2020
            * "woy_dow": string, (week of year, day of week) e.g. "03_6" for Saturday of 3rd week
            * "dow_hr": string, (day of week, hour) e.g. "4_09" for 9am on Thursday
            * "dow_hr_min": string, (day of week, hour, minute) e.g. "4_09_10" for 9:10am on Thursday
            * "tod": float, time of day, continuous, 0.0 to 24.0
            * "tow": float, time of week, continuous, 0.0 to 7.0
            * "tom": float, standardized time of month, continuous, 0.0 to 1.0
            * "toq": float, time of quarter, continuous, 0.0 to 1.0
            * "toy": float, standardized time of year, continuous, 0.0 to 1.0
            * "conti_year": float, year in continuous time, eg 2018.5 means middle of the year 2018
            * "is_weekend": boolean, weekend indicator, True for weekend, else False
            * "dow_grouped": string, Monday-Thursday=1234-MTuWTh, Friday=5-Fri, Saturday=6-Sat, Sunday=7-Sun
            * "ct1": float, linear growth based on conti_year_origin, -infinity to infinity
            * "ct2": float, signed quadratic growth, -infinity to infinity
            * "ct3": float, signed cubic growth, -infinity to infinity
            * "ct_sqrt": float, signed square root growth, -infinity to infinity
            * "ct_root3": float, signed cubic root growth, -infinity to infinity
            * "us_dst": bool, determines if the time inside the daylight saving time of US
                This column is only generated if ``add_dst_info=True``
            * "eu_dst": bool, determines if the time inside the daylight saving time of Europe. This column is only generated if ``add_dst_info=True``

    """
    dt = pd.DatetimeIndex(dt)
    if len(dt) == 0:
        raise ValueError("Length of dt cannot be zero.")

    # basic time features
    date = dt.date
    year = dt.year
    year_length = (365.0 + dt.is_leap_year)
    quarter = dt.quarter
    month = dt.month
    month_length = dt.days_in_month

    # finds first day of quarter
    quarter_start = pd.DatetimeIndex(
        dt.year.map(str) + "-" + (3 * quarter - 2).map(int).map(str) + "-01")
    next_quarter_start = dt + pd.tseries.offsets.QuarterBegin(startingMonth=1)
    quarter_length = (next_quarter_start - quarter_start).days
    # finds offset from first day of quarter (rounds down to nearest day)
    doq = ((dt - quarter_start) / pd.to_timedelta("1D") + 1).astype(int)

    # week of year, "woy", follows ISO 8601:
    #   - Week 01 is the week with the year's first Thursday in it.
    #   - A week begins with Monday and ends with Sunday.
    # So the week number of the week that overlaps both years, is 1, 52, or 53,
    # depending on whether it has more days in the previous year or new year.
    #   - e.g. Jan 1st, 2018 is Monday. woy of first 8 days = [1, 1, 1, 1, 1, 1, 1, 2]
    #   - e.g. Jan 1st, 2019 is Tuesday. woy of first 8 days = [1, 1, 1, 1, 1, 1, 2, 2]
    #   - e.g. Jan 1st, 2020 is Wednesday. woy of first 8 days = [1, 1, 1, 1, 1, 2, 2, 2]
    #   - e.g. Jan 1st, 2015 is Thursday. woy of first 8 days = [1, 1, 1, 1, 2, 2, 2, 2]
    #   - e.g. Jan 1st, 2021 is Friday. woy of first 8 days = [53, 53, 53, 1, 1, 1, 1, 1]
    #   - e.g. Jan 1st, 2022 is Saturday. woy of first 8 days = [52, 52, 1, 1, 1, 1, 1, 1]
    #   - e.g. Jan 1st, 2023 is Sunday. woy of first 8 days = [52, 1, 1, 1, 1, 1, 1, 1]
    woy = dt.strftime("%V").astype(int)
    doy = dt.dayofyear
    dom = dt.day
    dow = dt.strftime("%u").astype(int)
    str_dow = dt.strftime("%u-%a")  # e.g. 1-Mon, 2-Tue, ..., 7-Sun
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    # grouped time feature
    year_quarter = dt.strftime("%Y-") + quarter.astype(str)  # e.g. 2020-1 for March 2020
    str_doy = dt.strftime("%Y-%m-%d")       # e.g. 2020-03-20 for March 20, 2020
    year_month = dt.strftime("%Y-%m")       # e.g. 2020-03 for March 2020
    month_dom = dt.strftime("%m/%d")        # e.g. 02/20 for February 20th
    year_woy = dt.strftime("%Y_%V")         # e.g. 2020_42 for 42nd week of 2020
    year_woy_dow = dt.strftime("%Y_%V_%u")  # e.g. 2020_03_6 for Saturday of 3rd week in 2020
    woy_dow = dt.strftime("%W_%u")          # e.g. 03_6 for Saturday of 3rd week
    dow_hr = dt.strftime("%u_%H")           # e.g. 4_09 for 9am on Thursday
    dow_hr_min = dt.strftime("%u_%H_%M")    # e.g. 4_09_10 for 9:10am on Thursday

    # iso features https://en.wikipedia.org/wiki/ISO_week_date
    # Uses `pd.Index` to avoid overriding the indices in the output df.
    year_iso = pd.Index(dt.isocalendar()["year"])
    year_woy_iso = pd.Index(year_iso.astype(str) + "_" + dt.strftime("%V"))
    year_woy_dow_iso = pd.Index(year_woy_iso + "_" + dt.isocalendar()["day"].astype(str))

    # derived time features
    tod = hour + (minute / 60.0) + (second / 3600.0)
    tow = dow - 1 + (tod / 24.0)
    tom = (dom - 1 + (tod / 24.0)) / month_length
    toq = (doq - 1 + (tod / 24.0)) / quarter_length
    # time of year, continuous, 0.0 to 1.0. e.g. Jan 1, 12 am = 0/365, Jan 2, 12 am = 1/365, ...
    # To handle leap years, Feb 28 = 58/365 - 59/365, Feb 29 = 59/365, Mar 1 = 59/365 - 60/365
    # offset term is nonzero only in leap years
    # doy_offset reduces doy by 1 from from Mar 1st (doy > 60)
    doy_offset = (year_length == 366) * 1.0 * (doy > 60)
    # tod_offset sets tod to 0 on Feb 29th (doy == 60)
    tod_offset = 1 - (year_length == 366) * 1.0 * (doy == 60)
    toy = (doy - 1 - doy_offset + (tod / 24.0) * tod_offset) / 365.0

    # year of date in continuous time, eg 2018.5 means middle of year 2018
    # this is useful for modeling features that do not care about leap year e.g. environmental variables
    conti_year = year + (doy - 1 + (tod / 24.0)) / year_length
    is_weekend = pd.Series(dow).apply(lambda x: x in [6, 7]).values  # weekend indicator
    # categorical var with levels (Mon-Thu, Fri, Sat, Sun), could help when training data are sparse.
    dow_grouped = pd.Series(str_dow).apply(
        lambda x: "1234-MTuWTh" if (x in ["1-Mon", "2-Tue", "3-Wed", "4-Thu"]) else x).values

    # growth terms
    ct1 = conti_year - conti_year_origin
    ct2 = signed_pow(ct1, 2)
    ct3 = signed_pow(ct1, 3)
    ct_sqrt = signed_pow(ct1, 1/2)
    ct_root3 = signed_pow(ct1, 1/3)

    # All keys must be added to constants.
    features_dict = {
        cst.TimeFeaturesEnum.datetime.value: dt,
        cst.TimeFeaturesEnum.date.value: date,
        cst.TimeFeaturesEnum.year.value: year,
        cst.TimeFeaturesEnum.year_length.value: year_length,
        cst.TimeFeaturesEnum.quarter.value: quarter,
        cst.TimeFeaturesEnum.quarter_start.value: quarter_start,
        cst.TimeFeaturesEnum.quarter_length.value: quarter_length,
        cst.TimeFeaturesEnum.month.value: month,
        cst.TimeFeaturesEnum.month_length.value: month_length,
        cst.TimeFeaturesEnum.woy.value: woy,
        cst.TimeFeaturesEnum.doy.value: doy,
        cst.TimeFeaturesEnum.doq.value: doq,
        cst.TimeFeaturesEnum.dom.value: dom,
        cst.TimeFeaturesEnum.dow.value: dow,
        cst.TimeFeaturesEnum.str_dow.value: str_dow,
        cst.TimeFeaturesEnum.str_doy.value: str_doy,
        cst.TimeFeaturesEnum.hour.value: hour,
        cst.TimeFeaturesEnum.minute.value: minute,
        cst.TimeFeaturesEnum.second.value: second,
        cst.TimeFeaturesEnum.year_quarter.value: year_quarter,
        cst.TimeFeaturesEnum.year_month.value: year_month,
        cst.TimeFeaturesEnum.year_woy.value: year_woy,
        cst.TimeFeaturesEnum.month_dom.value: month_dom,
        cst.TimeFeaturesEnum.year_woy_dow.value: year_woy_dow,
        cst.TimeFeaturesEnum.woy_dow.value: woy_dow,
        cst.TimeFeaturesEnum.dow_hr.value: dow_hr,
        cst.TimeFeaturesEnum.dow_hr_min.value: dow_hr_min,
        cst.TimeFeaturesEnum.year_iso.value: year_iso,
        cst.TimeFeaturesEnum.year_woy_iso.value: year_woy_iso,
        cst.TimeFeaturesEnum.year_woy_dow_iso.value: year_woy_dow_iso,
        cst.TimeFeaturesEnum.tod.value: tod,
        cst.TimeFeaturesEnum.tow.value: tow,
        cst.TimeFeaturesEnum.tom.value: tom,
        cst.TimeFeaturesEnum.toq.value: toq,
        cst.TimeFeaturesEnum.toy.value: toy,
        cst.TimeFeaturesEnum.conti_year.value: conti_year,
        cst.TimeFeaturesEnum.is_weekend.value: is_weekend,
        cst.TimeFeaturesEnum.dow_grouped.value: dow_grouped,
        cst.TimeFeaturesEnum.ct1.value: ct1,
        cst.TimeFeaturesEnum.ct2.value: ct2,
        cst.TimeFeaturesEnum.ct3.value: ct3,
        cst.TimeFeaturesEnum.ct_sqrt.value: ct_sqrt,
        cst.TimeFeaturesEnum.ct_root3.value: ct_root3,
    }
    df = pd.DataFrame(features_dict)

    if add_dst_info:
        df[cst.TimeFeaturesEnum.us_dst.value] = is_dst_fcn("US/Pacific")(
            df[cst.TimeFeaturesEnum.datetime.value])

        df[cst.TimeFeaturesEnum.eu_dst.value] = is_dst_fcn("Europe/London")(
            df[cst.TimeFeaturesEnum.datetime.value])

    return df


def add_time_features_df(
        df,
        time_col,
        conti_year_origin,
        add_dst_info=True):
    """Adds a time feature data frame to a data frame by calling
    `~greykite.common.features.timeseries_features.build_time_features_df`.

    Parameters
    ----------
    df : `pandas.Dataframe`
        The input data frame
    time_col: `str`
        The name of the time column of interest
    conti_year_origin:
        The origin of time for the continuous time variable which is in years unit.
    add_dst_info : `bool`, default True
        Determines if daylight saving columns for US and Europe should be added.

    Returns
    -------
    result : `pandas.Dataframe`
        The same data frame (df) augmented with new columns generated by
        `~greykite.common.features.timeseries_features.build_time_features_df`
    """
    df = df.reset_index(drop=True)
    time_df = build_time_features_df(
        dt=df[time_col],
        conti_year_origin=conti_year_origin,
        add_dst_info=add_dst_info)
    time_df = time_df.reset_index(drop=True)
    return pd.concat([df, time_df], axis=1)


def get_holidays(countries, year_start, year_end):
    """This function extracts a holiday data frame for the period of interest
    [year_start to year_end] for the given countries.
    This is done using the holidays libraries in pypi:holidays-ext

    Parameters
    ----------
    countries : `list` [`str`]
        countries for which we need holidays
    year_start : `int`
        first year of interest, inclusive
    year_end : `int`
        last year of interest, inclusive

    Returns
    -------
    holiday_df_dict : `dict` [`str`, `pandas.DataFrame`]
        - key: country name
        - value: data frame with holidays for that country
          Each data frame has two columns: EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL
    """
    country_holiday_dict = {}
    year_list = list(range(year_start, year_end + 1))

    country_holidays = get_hdays.get_holiday(
        country_list=countries,
        years=year_list
    )

    for country, holidays in country_holidays.items():
        country_df = pd.DataFrame({
            cst.EVENT_DF_DATE_COL: list(holidays.keys()),
            cst.EVENT_DF_LABEL_COL: list(holidays.values())})
        # Replaces any occurrence of "/" with ", " in order to avoid saving / loading error in
        # `~greykite.framework.templates.pickle_utils` because a holiday name can be the key
        # of a dictionary that will be used as directory name.
        # For example, "Easter Monday [England/Wales/Northern Ireland]" will be casted to
        # "Easter Monday [England, Wales, Northern Ireland]".
        country_df[cst.EVENT_DF_LABEL_COL] = country_df[cst.EVENT_DF_LABEL_COL].str.replace("/", ", ")
        country_df[cst.EVENT_DF_DATE_COL] = pd.to_datetime(country_df[cst.EVENT_DF_DATE_COL])

        country_holiday_dict[country] = country_df

    return country_holiday_dict


def get_available_holiday_lookup_countries(countries=None):
    """Returns list of available countries for modeling holidays

    :param countries: List[str]
        only look for available countries in this set
    :return: List[str]
        list of available countries for modeling holidays
    """
    return get_hdays.get_available_holiday_lookup_countries(
        countries=countries
    )


def get_available_holidays_in_countries(
        countries,
        year_start,
        year_end):
    """Returns a dictionary mapping each country to its holidays
        between the years specified.

    :param countries: List[str]
        countries for which we need holidays
    :param year_start: int
        first year of interest
    :param year_end: int
        last year of interest
    :return: Dict[str, List[str]]
        key: country name
        value: list of holidays in that country between [year_start, year_end]
    """
    return get_hdays.get_available_holidays_in_countries(
        countries=countries,
        year_start=year_start,
        year_end=year_end
    )


def get_available_holidays_across_countries(
        countries,
        year_start,
        year_end):
    """Returns a list of holidays that occur any of the countries
    between the years specified.

    :param countries: List[str]
        countries for which we need holidays
    :param year_start: int
        first year of interest
    :param year_end: int
        last year of interest
    :return: List[str]
        names of holidays in any of the countries between [year_start, year_end]
    """
    return get_hdays.get_available_holidays_across_countries(
        countries=countries,
        year_start=year_start,
        year_end=year_end
    )


def add_daily_events(
        df,
        event_df_dict,
        date_col=cst.EVENT_DF_DATE_COL,
        regular_day_label=cst.EVENT_DEFAULT,
        neighbor_impact=None,
        shifted_effect=None):
    """For each key of event_df_dict, it adds a new column to a data frame (df)
    with a date column (date_col).
    Each new column will represent the events given for that key.
    This function also generates 3 binary event flags
    ``IS_EVENT_EXACT_COL``, ``IS_EVENT_ADJACENT_COL`` and ``IS_EVENT_COL``
    given the information in ``event_df_dict`` with the following logic:

        (1) If the key contains "_minus_" or "_plus_", that means the event
        was generated by the ``add_event_window`` function, and it is a
        neighboring day of some exact event day.
        In this case, ``IS_EVENT_ADJACENT_COL`` will be 1 for all days in this key.

        (2) Otherwise the key indicates that it is on the exact event day being modeled.
        In this case, ``IS_EVENT_EXACT_COL`` will be 1 for all days in this key.

        (3) If a date appears in both types of keys, both above columns will be 1.

        (4) ``IS_EVENT_COL`` is 1 for all dates in the provided ``event_df_dict``.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The data frame which has a date column.
    event_df_dict : `dict` [`str`, `pandas.DataFrame`]
        A dictionary of data frames, each representing events data
        for the corresponding key.
        Values are DataFrames with two columns:

            - The first column contains the date. Must be at the same
              frequency as ``df[date_col]`` for proper join. Must be in a
              format recognized by `pandas.to_datetime`.
            - The second column contains the event label for each date

    date_col : `str`
        Column name in ``df`` that contains the dates for joining against
        the events in ``event_df_dict``.
    regular_day_label : `str`
        The label used for regular days which are not "events".
    neighbor_impact : `int`, `list` [`int`], callable or None, default None
        The impact of neighboring timestamps of the events in ``event_df_dict``.
        This is for daily events so the units below are all in days.

        For example, if the data is weekly ("W-SUN") and an event is daily,
        it may not exactly fall on the weekly date.
        But you can specify for New Year's day on 1-1, it affects all dates
        in the week, e.g. 12-31, 1-1, ..., 1-6, then it will be mapped to the weekly date.
        In this case you may want to map a daily event's date to a few dates,
        and can specify
        ``neighbor_impact=lambda x: [x-timedelta(days=x.isocalendar()[2]-1) + timedelta(days=i) for i in range(7)]``.

        Another example is that the data is rolling 7 day daily data,
        thus a holiday may affect the t, t+1, ..., t+6 dates.
        You can specify ``neighbor_impact=7``.

        If input is `int`, the mapping is t, t+1, ..., t+neighbor_impact-1.
        If input is `list`, the mapping is [t+x for x in neighbor_impact].
        If input is a function, it maps each daily event's date to a list of dates.
    shifted_effect : `list` [`str`] or None, default None
        Additional neighbor events based on given events.
        For example, passing ["-1D", "7D"] will add extra daily events which are 1 day before
        and 7 days after the given events.
        Offset format is {d}{freq} with any integer plus a frequency string. Must be parsable by pandas ``to_offset``.
        The new events' names will be the current events' names with suffix "{offset}_before" or "{offset}_after".
        For example, if we have an event named "US_Christmas Day",
        a "7D" shift will have name "US_Christmas Day_7D_after".
        This is useful when you expect an offset of the current holidays also has impact on the
        time series, or you want to interact the lagged terms with autoregression.
        If ``neighbor_impact`` is also specified, this will be applied after adding neighboring days.

    Returns
    -------
    df_daily_events : `pandas.DataFrame`
        An augmented data frame version of df with new label columns --
        one for each key of ``event_df_dict``.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    get_neighbor_days_func = None
    new_event_cols = [cst.IS_EVENT_EXACT_COL, cst.IS_EVENT_ADJACENT_COL, cst.IS_EVENT_COL]
    event_flag_df_list = []
    if neighbor_impact is not None:
        if isinstance(neighbor_impact, int):
            neighbor_impact = sorted([neighbor_impact, 0])
            neighbor_impact[1] += 1

            def get_neighbor_days_func(date):
                return [date + timedelta(days=d) for d in range(*neighbor_impact)]
        elif isinstance(neighbor_impact, list):
            if 0 not in neighbor_impact:
                neighbor_impact = sorted(neighbor_impact + [0])

            def get_neighbor_days_func(date):
                return [date + timedelta(days=d) for d in neighbor_impact]
        else:
            get_neighbor_days_func = neighbor_impact
    for label, event_df in event_df_dict.items():
        event_df = event_df.copy().drop_duplicates()  # Makes a copy to avoid modifying input
        new_col = f"{cst.EVENT_PREFIX}_{label}"
        event_df.columns = [date_col, new_col]
        event_df[date_col] = pd.to_datetime(event_df[date_col])
        # Handles neighboring impact.
        if get_neighbor_days_func is not None:
            new_event_df = None
            for i in range(len(event_df)):
                mapped_dates = get_neighbor_days_func(event_df["date"].iloc[i])
                new_event_df = pd.concat([
                    new_event_df, event_df.iloc[[i] * len(mapped_dates)].assign(**{date_col: mapped_dates})],
                    axis=0
                )
            event_df = new_event_df.drop_duplicates().reset_index(drop=True)
        df = df.merge(event_df, on=date_col, how="left")
        df[new_col] = df[new_col].fillna(regular_day_label)
        # Adds neighbor events if requested.
        new_event_dfs = []
        if shifted_effect is not None:
            for lag in shifted_effect:
                num, freq = split_offset_str(lag)
                num = int(num)
                if num != 0:
                    lag_offset = to_offset(lag)
                    new_event_df = event_df.copy()
                    new_event_df[date_col] += lag_offset
                    suffix = cst.EVENT_SHIFTED_SUFFIX_BEFORE if num < 0 else cst.EVENT_SHIFTED_SUFFIX_AFTER
                    new_col = f"{cst.EVENT_PREFIX}_{label}_{abs(num)}{freq}{suffix}"
                    new_event_df.columns = [date_col, new_col]
                    new_event_df[new_col] += f"_{abs(num)}{freq}{suffix}"
                    df = df.merge(new_event_df, on=date_col, how="left")
                    df[new_col] = df[new_col].fillna(regular_day_label)
                    new_event_dfs.append(new_event_df)
        # Generates event indicators.
        # `augmented_event_df` contains `date_col` and three event indicator columns to be added to `df`.
        for event_df_temp in [event_df] + new_event_dfs:
            augmented_event_df = event_df_temp[[date_col]].drop_duplicates()
            is_event_adjacent = "_minus_" in label or "_plus_" in label
            augmented_event_df[cst.IS_EVENT_EXACT_COL] = 0 if is_event_adjacent else 1
            augmented_event_df[cst.IS_EVENT_ADJACENT_COL] = 1 if is_event_adjacent else 0
            augmented_event_df[cst.IS_EVENT_COL] = 1  # In either case, `IS_EVENT_COL` is 1.
            event_flag_df_list.append(augmented_event_df)

    event_flag_df = pd.concat(event_flag_df_list)
    # Sets a day as 1 if it is marked by any of the keys in `event_df_dict`.
    event_flag_df = event_flag_df.groupby(by=date_col)[new_event_cols].sum().reset_index(drop=False)
    event_flag_df[new_event_cols] = 1 * (event_flag_df[new_event_cols] > 0)
    # Joins the new event indicators to `df`.
    df = df.merge(event_flag_df, on=date_col, how="left")
    df[new_event_cols] = df[new_event_cols].fillna(0)

    return df


def add_event_window(
        df,
        time_col,
        label_col,
        time_delta="1D",
        pre_num=1,
        post_num=1,
        events_name=""):
    """For a data frame of events with a time_col and label_col
        it adds shifted events
        prior and after the given events
        For example if the event data frame includes the row
            '2019-12-25, Christmas'
        the function will produce dataframes with the events:
            '2019-12-24, Christmas' and '2019-12-26, Christmas'
        if pre_num and post_num are 1 or more.
    :param df: pd.DataFrame
        the events data frame with two columns 'time_col' and 'label_col'
    :param time_col: str
        The column with the timestamp of the events.
        This can be daily but does not have to
    :param label_col: str
        the column with labels for the events
    :param time_delta: str
        the amount of the shift for each unit specified by a string
        e.g. "1D" stands for one day delta
    :param pre_num: int
        the number of events to be added prior to the given event for each event in df
    :param post_num: int
        the number of events to be added after to the given event for each event in df
    :param events_name: str
        for each shift, we generate a new data frame
        and those data frames will be stored in a dictionary with appropriate keys.
        Each key starts with "events_name"
        and follow up with:
            "_minus_1", "_minus_2", "_plus_1", "_plus_2", ...
        depending on pre_num and post_num
    :return: dict[key: pd.Dataframe]
        A dictionary of dataframes for each needed shift.
        For example if pre_num=2 and post_num=3.
        2 + 3 = 5 data frames will be stored in the return dictionary.
        """

    df_dict = {}
    pd_time_delta = pd.to_timedelta(time_delta)
    for num in range(pre_num):
        df0 = pd.DataFrame()
        df0[time_col] = df[time_col] - (num + 1) * pd_time_delta
        df0[label_col] = df[label_col]
        df_dict[events_name + "_minus_" + f"{(num + 1):.0f}"] = df0

    for num in range(post_num):
        df0 = pd.DataFrame()
        df0[time_col] = df[time_col] + (num + 1) * pd_time_delta
        df0[label_col] = df[label_col]
        df_dict[events_name + "_plus_" + f"{(num + 1):.0f}"] = df0

    return df_dict


def get_evenly_spaced_changepoints_values(
        df,
        continuous_time_col=cst.TimeFeaturesEnum.ct1.value,
        n_changepoints=2):
    """Partitions interval into n_changepoints + 1 segments,
        placing a changepoint at left endpoint of each segment.
        The left most segment doesn't get a changepoint.
        Changepoints should be determined from training data.

    :param df: pd.DataFrame
        training dataset. contains continuous_time_col
    :param continuous_time_col: str
        name of continuous time column (e.g. conti_year, ct1)
    :param n_changepoints: int
        number of changepoints requested
    :return: np.array
        values of df[continuous_time_col] at the changepoints
    """
    if not n_changepoints > 0:
        raise ValueError("n_changepoints must be > 0")

    n = df.shape[0]
    n_steps = n_changepoints + 1
    step_size = n / n_steps
    indices = np.floor(np.arange(start=1, stop=n_steps) * step_size)
    return df[continuous_time_col][indices].values


def get_evenly_spaced_changepoints_dates(
        df,
        time_col,
        n_changepoints):
    """Partitions interval into n_changepoints + 1 segments,
        placing a changepoint at left endpoint of each segment.
        The left most segment doesn't get a changepoint.
        Changepoints should be determined from training data.

    :param df: pd.DataFrame
        training dataset. contains continuous_time_col
    :param time_col: str
        name of time column
    :param n_changepoints: int
        number of changepoints requested
    :return: pd.Series
        values of df[time_col] at the changepoints
    """
    if not n_changepoints >= 0:
        raise ValueError("n_changepoints must be >= 0")
    changepoint_indices = np.floor(np.arange(start=1, stop=n_changepoints + 1) * (df.shape[0] / (n_changepoints + 1)))
    changepoint_indices = df.index[np.concatenate([[0], changepoint_indices.astype(int)])]
    return df.loc[changepoint_indices, time_col]


def get_custom_changepoints_values(
        df,
        changepoint_dates,
        time_col=cst.TIME_COL,
        continuous_time_col=cst.TimeFeaturesEnum.ct1.value):
    """Returns the values of continuous_time_col at the
        requested changepoint_dates.

    :param df: pd.DataFrame
        training dataset. contains continuous_time_col and time_col
    :param changepoint_dates: Iterable[Union[int, float, str, datetime]]
        Changepoint dates, interpreted by pd.to_datetime.
        Changepoints are set at the closest time on or after these dates
        in the dataset
    :param time_col: str
        The column name in `df` representing time for the time series data
        The time column can be anything that can be parsed by pandas DatetimeIndex
    :param continuous_time_col: str
        name of continuous time column (e.g. conti_year, ct1)
    :return: np.array
        values of df[continuous_time_col] at the changepoints
    """
    ts = pd.to_datetime(df[time_col])
    changepoint_dates = pd.to_datetime(changepoint_dates, format='mixed')
    # maps each changepoint to first date >= changepoint in the dataframe
    # if there is no such date, the changepoint is dropped (it would not be useful anyway)
    changepoint_ts = [ts[ts >= date].min() for date in changepoint_dates if any(ts >= date)]
    indices = ts.isin(changepoint_ts)
    changepoints = df[indices][continuous_time_col].values
    if changepoints.shape[0] == 0:
        changepoints = None
    return changepoints


def get_changepoint_string(changepoint_dates):
    """Gets proper formatted strings for changepoint dates.

    The default format is "_%Y_%m_%d_%H". When necessary, it appends "_%M" or "_%M_%S".

    Parameters
    ----------
    changepoint_dates : `list`
        List of changepoint dates, parsable by `pandas.to_datetime`.

    Returns
    -------
    date_strings : `list[`str`]`
        List of string formatted changepoint dates.
    """
    changepoint_dates = list(pd.to_datetime(changepoint_dates, format='mixed'))
    time_format = "_%Y_%m_%d_%H"
    if any([stamp.second != 0 for stamp in changepoint_dates]):
        time_format += "_%M_%S"
    elif any([stamp.minute != 0 for stamp in changepoint_dates]):
        time_format += "_%M"
    date_strings = [date.strftime(time_format) for date in changepoint_dates]
    return date_strings


def get_changepoint_features(
        df,
        changepoint_values,
        continuous_time_col=cst.TimeFeaturesEnum.ct1.value,
        growth_func=None,
        changepoint_dates=None):
    """Returns features for growth terms with continuous time origins at
        the changepoint_values (locations) specified

    Generates a time series feature for each changepoint:
        Let t = continuous_time value, c = changepoint value
        Then the changepoint feature value at time point t is
            `growth_func(t - c) * I(t >= c)`, where I is the indicator function
        This represents growth as a function of time, where the time origin is
        the changepoint

    In the typical case where growth_func(0) = 0 (has origin at 0),
        the total effect of the changepoints is continuous in time.
        If `growth_func` is the identity function, and `continuous_time`
        represents the year in continuous time, these terms form the basis for a
        continuous, piecewise linear curve to the growth trend.
        Fitting these terms with linear model, the coefficents represent slope
        change at each changepoint

    Intended usage
    ----------

    To make predictions (on test set)
        Allow growth term as a function of time to change at these points.

    Parameters
    ----------
    :param df: pd.Dataframe
        The dataset to make predictions. Contains column continuous_time_col.
    :param changepoint_values: array-like
        List of changepoint values (on same scale as df[continuous_time_col]).
        Should be determined from training data
    :param continuous_time_col: Optional[str]
        Name of continuous time column in df
        growth_func is applied to this column to generate growth term
        If None, uses "ct1", linear growth
    :param growth_func: Optional[callable]
        Growth function for defining changepoints (scalar -> scalar).
        If None, uses identity function to use continuous_time_col directly
        as growth term
    :param changepoint_dates: Optional[list]
        List of change point dates, parsable by `pandas.to_datetime`.
    :return: pd.DataFrame, shape (df.shape[0], len(changepoints))
        Changepoint features, 0-indexed
    """
    if continuous_time_col is None:
        continuous_time_col = cst.TimeFeaturesEnum.ct1.value
    if growth_func is None:
        def growth_func(x):
            return x

    if changepoint_dates is not None:
        time_postfixes = get_changepoint_string(changepoint_dates)
    else:
        time_postfixes = [""] * len(changepoint_values)

    changepoint_df_list = []
    for i, changepoint in enumerate(changepoint_values):
        time_feature = np.array(df[continuous_time_col]) - changepoint  # shifted time column (t - c_i)
        growth_term = np.array([growth_func(max(x, 0)) for x in time_feature])  # growth as a function of time
        time_feature_ind = time_feature >= 0  # Indicator(t >= c_i), lets changepoint take effect starting at c_i
        new_col = pd.Series(
            data=growth_term * time_feature_ind,
            name=f"{cst.CHANGEPOINT_COL_PREFIX}{i}{time_postfixes[i]}"
        )
        changepoint_df_list.append(new_col)
    if len(changepoint_values) > 0:
        changepoint_df = pd.concat(objs=changepoint_df_list, axis=1, ignore_index=False)
    else:
        changepoint_df = pd.DataFrame()
    return changepoint_df


def get_changepoint_values_from_config(
        changepoints_dict,
        time_features_df,
        time_col=cst.TIME_COL):
    """Applies the changepoint method specified in `changepoints_dict` to return the changepoint values

    :param changepoints_dict: Optional[Dict[str, any]]
        Specifies the changepoint configuration.
        "method": str
            The method to locate changepoints. Valid options:
                "uniform". Places n_changepoints evenly spaced changepoints to allow growth to change.
                "custom". Places changepoints at the specified dates.
            Additional keys to provide parameters for each particular method are described below.
        "continuous_time_col": Optional[str]
            Column to apply `growth_func` to, to generate changepoint features
            Typically, this should match the growth term in the model
        "growth_func": Optional[func]
            Growth function (scalar -> scalar). Changepoint features are created
            by applying `growth_func` to "continuous_time_col" with offsets.
            If None, uses identity function to use `continuous_time_col` directly
            as growth term
        If changepoints_dict["method"] == "uniform", this other key is required:
            "n_changepoints": int
                number of changepoints to evenly space across training period
        If changepoints_dict["method"] == "custom", this other key is required:
            "dates": Iterable[Union[int, float, str, datetime]]
                Changepoint dates. Must be parsable by pd.to_datetime.
                Changepoints are set at the closest time on or after these dates
                in the dataset.
    :param time_features_df: pd.Dataframe
        training dataset. contains column "continuous_time_col"
    :param time_col: str
        The column name in `time_features_df` representing time for the time series data
        The time column can be anything that can be parsed by pandas DatetimeIndex
        Used only in the "custom" method.
    :return: np.array
        values of df[continuous_time_col] at the changepoints
    """
    changepoint_values = None
    if changepoints_dict is not None:
        valid_changepoint_methods = ["uniform", "custom"]
        changepoint_method = changepoints_dict.get("method")
        continuous_time_col = changepoints_dict.get("continuous_time_col")

        if changepoint_method is None:
            raise Exception("changepoint method must be specified")

        if changepoint_method not in valid_changepoint_methods:
            raise NotImplementedError(
                f"changepoint method {changepoint_method} not recognized. "
                f"Must be one of {valid_changepoint_methods}")

        if changepoint_method == "uniform":
            if changepoints_dict["n_changepoints"] > 0:
                params = {"continuous_time_col": continuous_time_col} if continuous_time_col is not None else {}
                changepoint_values = get_evenly_spaced_changepoints_values(
                    df=time_features_df,
                    n_changepoints=changepoints_dict["n_changepoints"],
                    **params)
        elif changepoint_method == "custom":
            params = {}
            if time_col is not None:
                params["time_col"] = time_col
            if continuous_time_col is not None:
                params["continuous_time_col"] = continuous_time_col
            changepoint_values = get_custom_changepoints_values(
                df=time_features_df,
                changepoint_dates=changepoints_dict["dates"],
                **params)

    return changepoint_values


def get_changepoint_features_and_values_from_config(
        df,
        time_col,
        changepoints_dict=None,
        origin_for_time_vars=None):
    """Extracts changepoints from changepoint configuration and input data

    :param df: pd.DataFrame
        Training data. A data frame which includes the timestamp and value columns
    :param time_col: str
        The column name in `df` representing time for the time series data
        The time column can be anything that can be parsed by pandas DatetimeIndex
    :param changepoints_dict: Optional[Dict[str, any]]
        Specifies the changepoint configuration.
        "method": str
            The method to locate changepoints. Valid options:
                "uniform". Places n_changepoints evenly spaced changepoints to allow growth to change.
                "custom". Places changepoints at the specified dates.
            Additional keys to provide parameters for each particular method are described below.
        "continuous_time_col": Optional[str]
            Column to apply `growth_func` to, to generate changepoint features
            Typically, this should match the growth term in the model
        "growth_func": Optional[func]
            Growth function (scalar -> scalar). Changepoint features are created
            by applying `growth_func` to "continuous_time_col" with offsets.
            If None, uses identity function to use `continuous_time_col` directly
            as growth term
        If changepoints_dict["method"] == "uniform", this other key is required:
            "n_changepoints": int
                number of changepoints to evenly space across training period
        If changepoints_dict["method"] == "custom", this other key is required:
            "dates": Iterable[Union[int, float, str, datetime]]
                Changepoint dates. Must be parsable by pd.to_datetime.
                Changepoints are set at the closest time on or after these dates
                in the dataset.
    :param origin_for_time_vars: Optional[float]
        The time origin used to create continuous variables for time
    :return: Dict[str, any]
        Dictionary with the requested changepoints and associated information
        changepoint_df: pd.DataFrame, shape (df.shape[0], len(changepoints))
            Changepoint features for modeling the training data
        changepoint_values: array-like
            List of changepoint values (on same scale as df[continuous_time_col])
            Can be used to generate changepoints for prediction.
        continuous_time_col: Optional[str]
            Name of continuous time column in df
            growth_func is applied to this column to generate growth term.
            If None, uses "ct1", linear growth
            Can be used to generate changepoints for prediction.
        growth_func: Optional[callable]
            Growth function for defining changepoints (scalar -> scalar).
            If None, uses identity function to use continuous_time_col directly
            as growth term.
            Can be used to generate changepoints for prediction.
        changepoint_cols: List[str]
            Names of the changepoint columns for modeling
    """
    # extracts changepoint values
    if changepoints_dict is None:
        changepoint_values = None
        continuous_time_col = None
        growth_func = None
    else:
        if origin_for_time_vars is None:
            origin_for_time_vars = get_default_origin_for_time_vars(df, time_col)
        time_features_df = build_time_features_df(
            df[time_col],
            conti_year_origin=origin_for_time_vars)

        changepoint_values = get_changepoint_values_from_config(
            changepoints_dict=changepoints_dict,
            time_features_df=time_features_df,
            time_col="datetime")  # datetime column generated by `build_time_features_df`
        continuous_time_col = changepoints_dict.get("continuous_time_col")
        growth_func = changepoints_dict.get("growth_func")

    # extracts changepoint column names
    if changepoint_values is None:
        changepoint_df = None
        changepoint_cols = []
    else:
        if changepoints_dict is None:
            changepoint_dates = None
        elif changepoints_dict["method"] == "custom":
            changepoint_dates = list(pd.to_datetime(changepoints_dict["dates"], format='mixed'))
        elif changepoints_dict["method"] == "uniform":
            changepoint_dates = get_evenly_spaced_changepoints_dates(
                df=df,
                time_col=time_col,
                n_changepoints=changepoints_dict["n_changepoints"]
            ).tolist()[1:]  # the changepoint features does not include the growth term
        else:
            changepoint_dates = None
        changepoint_df = get_changepoint_features(
            df=time_features_df,
            changepoint_values=changepoint_values,
            continuous_time_col=continuous_time_col,
            growth_func=growth_func,
            changepoint_dates=changepoint_dates)
        changepoint_cols = list(changepoint_df.columns)

    return {
        "changepoint_df": changepoint_df,
        "changepoint_values": changepoint_values,
        "continuous_time_col": continuous_time_col,
        "growth_func": growth_func,
        "changepoint_cols": changepoint_cols
    }


def get_changepoint_dates_from_changepoints_dict(
        changepoints_dict,
        df=None,
        time_col=None):
    """Gets the changepoint dates from ``changepoints_dict``

    Parameters
    ----------
    changepoints_dict : `dict` or `None`
        The ``changepoints_dict`` which is compatible with
        `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`
    df : `pandas.DataFrame` or `None`, default `None`
        The data df to put changepoints on.
    time_col : `str` or `None`, default `None`
        The column name of time column in ``df``.

    Returns
    -------
    changepoint_dates : `list`
        List of changepoint dates.
    """
    if (changepoints_dict is None
            or "method" not in changepoints_dict.keys()
            or changepoints_dict["method"] not in ["auto", "uniform", "custom"]):
        return None
    method = changepoints_dict["method"]
    if method == "custom":
        # changepoints_dict["dates"] is `Iterable`, converts to list
        changepoint_dates = list(changepoints_dict["dates"])
    elif method == "uniform":
        if df is None or time_col is None:
            raise ValueError("When the method of ``changepoints_dict`` is 'uniform', ``df`` and "
                             "``time_col`` must be provided.")
        changepoint_dates = get_evenly_spaced_changepoints_dates(
            df=df,
            time_col=time_col,
            n_changepoints=changepoints_dict["n_changepoints"]
        )
        # the output is `pandas.Series`, converts to list
        changepoint_dates = changepoint_dates.tolist()[1:]
    else:
        raise ValueError("The method of ``changepoints_dict`` can not be 'auto'. "
                         "Please specify or detect change points first.")
    return changepoint_dates


def add_event_window_multi(
        event_df_dict,
        time_col,
        label_col,
        time_delta="1D",
        pre_num=1,
        post_num=1,
        pre_post_num_dict=None):
    """For a given dictionary of events data frames with a time_col and label_col
    it adds shifted events prior and after the given events
    For example if the event data frame includes the row '2019-12-25, Christmas' as a row
    the function will produce dataframes with the events '2019-12-24, Christmas' and '2019-12-26, Christmas' if
    pre_num and post_num are 1 or more.

    Parameters
    ----------
    event_df_dict: `dict` [`str`, `pandas.DataFrame`]
        A dictionary of events data frames
        with each having two columns: ``time_col`` and ``label_col``.
    time_col: `str`
        The column with the timestamp of the events.
        This can be daily but does not have to be.
    label_col : `str`
        The column with labels for the events.
    time_delta : `str`, default "1D"
        The amount of the shift for each unit specified by a string
        e.g. '1D' stands for one day delta
    pre_num : `int`, default 1
        The number of events to be added prior to the given event for each event in df.
    post_num: `int`, default 1
        The number of events to be added after to the given event for each event in df.
    pre_post_num_dict : `dict` [`str`, (`int`, `int`)] or None, default None
        Optionally override ``pre_num`` and ``post_num`` for each key in ``event_df_dict``.
        For example, if ``event_df_dict`` has keys "US" and "India", this parameter
        can be set to ``pre_post_num_dict = {"US": [1, 3], "India": [1, 2]}``,
        denoting that the "US" ``pre_num`` is 1 and ``post_num`` is 3, and "India" ``pre_num`` is 1
        and ``post_num`` is 2. Keys not specified by ``pre_post_num_dict`` use the default given by
        ``pre_num`` and ``post_num``.

    Returns
    -------
    df : `dict` [`str`, `pandas.DataFrame`]
        A dictionary of dataframes for each needed shift. For example if pre_num=2 and post_num=3.
        2 + 3 = 5 data frames will be stored in the return dictionary.
    """
    if pre_post_num_dict is None:
        pre_post_num_dict = {}

    shifted_df_dict = {}

    for event_df_key, event_df in event_df_dict.items():
        if event_df_key in pre_post_num_dict.keys():
            pre_num0 = pre_post_num_dict[event_df_key][0]
            post_num0 = pre_post_num_dict[event_df_key][1]
        else:
            pre_num0 = pre_num
            post_num0 = post_num

        df_dict0 = add_event_window(
            df=event_df,
            time_col=time_col,
            label_col=label_col,
            time_delta=time_delta,
            pre_num=pre_num0,
            post_num=post_num0,
            events_name=event_df_key)

        shifted_df_dict.update(df_dict0)

    return shifted_df_dict


def get_fourier_col_name(k, col_name, function_name="sin", seas_name=None):
    """Returns column name corresponding to a particular fourier term, as returned by fourier_series_fcn

    :param k: int
        fourier term
    :param col_name: str
        column in the dataframe used to generate fourier series
    :param function_name: str
        sin or cos
    :param seas_name: strcols_interact
        appended to new column names added for fourier terms
    :return: str
        column name in DataFrame returned by fourier_series_fcn
    """
    # patsy doesn't allow "." in formula term. Replace "." with "_" rather than quoting "Q()" all fourier terms
    name = f"{function_name}{k:.0f}_{col_name}"
    if seas_name is not None:
        name = f"{name}_{seas_name}"
    return name


def fourier_series_fcn(col_name, period=1.0, order=1, seas_name=None):
    """Generates a function which creates fourier series matrix for a column of an input df
    :param col_name: str
        is the column name in the dataframe which is to be used for
        generating fourier series. It needs to be a continuous variable.
    :param period: float
        the period of the fourier series
    :param order: int
        the order of the fourier series
    :param seas_name: Optional[str]
        appended to new column names added for fourier terms.
        Useful to distinguish multiple fourier
        series on same col_name with different periods.
    :return: callable
        a function which can be applied to any data.frame df
        with a column name being equal to col_name
    """

    def fs_func(df):
        out_df_list = []
        out_cols = []

        if col_name not in df.columns:
            raise ValueError("The data frame does not have the column: " + col_name)
        x = df[col_name]
        x = np.array(x)

        for i in range(order):
            k = i + 1
            sin_col_name = get_fourier_col_name(
                k,
                col_name,
                function_name="sin",
                seas_name=seas_name)
            cos_col_name = get_fourier_col_name(
                k,
                col_name,
                function_name="cos",
                seas_name=seas_name)
            out_cols.append(sin_col_name)
            out_cols.append(cos_col_name)
            omega = 2 * math.pi / period
            u = omega * k * x
            out_df_list.append(pd.Series(data=np.sin(u), name=sin_col_name))
            out_df_list.append(pd.Series(data=np.cos(u), name=cos_col_name))
        if len(out_df_list) > 0:
            out_df = pd.concat(objs=out_df_list, axis=1, ignore_index=False)
        else:
            out_df = pd.DataFrame()
        return {"df": out_df, "cols": out_cols}

    return fs_func


def fourier_series_multi_fcn(
        col_names,
        periods=None,
        orders=None,
        seas_names=None):
    """Generates a func which adds multiple fourier series with multiple periods.

    Parameters
    ----------
    col_names : `list` [`str`]
        the column names which are to be used to generate Fourier series.
        Each column can have its own period and order.
    periods:  `list` [`float`] or None
        the periods corresponding to each column given in col_names
    orders : `list` [`int`] or None
        the orders for each of the Fourier series
    seas_names : `list` [`str`] or None
        Appended to the Fourier series name.
        If not provided (None) col_names will be used directly.
    """

    k = len(col_names)
    if periods is None:
        periods = [1.0] * k
    if orders is None:
        orders = [1] * k

    if len(periods) != len(orders):
        raise ValueError("periods and orders must have the same length.")

    def fs_multi_func(df):
        out_df = None
        out_cols = []

        for i in range(k):
            col_name = col_names[i]
            period = periods[i]
            order = orders[i]
            seas_name = None
            if seas_names is not None:
                seas_name = seas_names[i]

            func0 = fourier_series_fcn(
                col_name=col_name,
                period=period,
                order=order,
                seas_name=seas_name)

            res = func0(df)
            fs_df = res["df"]
            fs_cols = res["cols"]
            out_df = pd.concat([out_df, fs_df], axis=1)
            out_cols = out_cols + fs_cols

        return {"df": out_df, "cols": out_cols}

    return fs_multi_func


def signed_pow(x, y):
    """ Takes the absolute value of x and raises it to power of y.
    Then it multiplies the result by sign of x.
    This guarantees this function is non-decreasing.
    This is useful in many contexts e.g. statistical modeling.
    :param x: the base number which can be any real number
    :param y: the power which can be any real number
    :return: returns abs(x) to power of y multiplied by sign of x
    """
    return np.sign(x) * np.power(np.abs(x), y)


def signed_pow_fcn(y):
    return lambda x: signed_pow(x, y)


signed_sqrt = signed_pow_fcn(1 / 2)
signed_sq = signed_pow_fcn(2)


def logistic(
        x,
        growth_rate=1.0,
        capacity=1.0,
        floor=0.0,
        inflection_point=0.0):
    """Evaluates the logistic function at x with the specified growth rate,
        capacity, floor, and inflection point.

    :param x: value to evaluate the logistic function
    :type x: float
    :param growth_rate: growth rate
    :type growth_rate: float
    :param capacity: max value (carrying capacity)
    :type capacity: float
    :param floor: min value (lower bound)
    :type floor: float
    :param inflection_point: the t value of the inflection point
    :type inflection_point: float
    :return: value of the logistic function at t
    :rtype: float
    """
    return floor + capacity * expit(growth_rate * (x - inflection_point))


def get_logistic_func(
        growth_rate=1.0,
        capacity=1.0,
        floor=0.0,
        inflection_point=0.0):
    """Returns a function that evaluates the logistic function at t with the
        specified growth rate, capacity, floor, and inflection point.

        f(x) = floor + capacity / (1 + exp(-growth_rate * (x - inflection_point)))

    :param growth_rate: growth rate
    :type growth_rate: float
    :param capacity: max value (carrying capacity)
    :type capacity: float
    :param floor: min value (lower bound)
    :type floor: float
    :param inflection_point: the t value of the inflection point
    :type inflection_point: float
    :return: the logistic function with specified parameters
    :rtype: callable
    """
    return lambda t: logistic(t, growth_rate, capacity, floor, inflection_point)
