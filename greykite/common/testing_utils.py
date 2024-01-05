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
"""Utility functions for test cases."""

import datetime
import warnings

import numpy as np
import pandas as pd

from greykite.common.constants import ADJUSTMENT_DELTA_COL
from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import METRIC_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.constants import TimeFeaturesEnum
from greykite.common.features.timeseries_features import add_daily_events
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.features.timeseries_features import fourier_series_multi_fcn
from greykite.common.features.timeseries_features import get_default_origin_for_time_vars
from greykite.common.features.timeseries_features import get_fourier_col_name
from greykite.common.features.timeseries_features import get_holidays
from greykite.common.python_utils import assert_equal


def generate_df_for_tests(
        freq,
        periods,
        train_start_date=datetime.datetime(2018, 7, 1),
        train_end_date=None,
        train_frac=0.8,
        conti_year_origin=None,
        noise_std=2.0,
        remove_extra_cols=True,
        autoreg_coefs=None,
        fs_coefs=[-1, 3, 4],
        growth_coef=3.0,
        growth_pow=1.1,
        intercept=0.0,
        seed=123):
    """Generates dataset for unit tests.

    :param freq: str
        pd.date_range freq parameter, e.g. H or D
    :param periods: int
        number of periods to generate
    :param train_start_date: datetime.datetime
        train start date
    :param train_end_date: Optional[datetime.datetime]
        train end date
    :param train_frac: Optional[float]
        fraction of data to use for training
        only used if train_end_date isn't provided
    :param noise_std: float
        standard deviation of gaussian noise
    :param conti_year_origin: float
        the time origin for continuous time variables
    :param remove_extra_cols: bool
        whether to remove extra columns besides TIME_COL, VALUE_COL
    :param autoreg_coefs: Optional[List[int]]
        The coefficients for the autoregressive terms.
        If provided the generated series denoted mathematically by Y(t) will be
        converted as follows:
        Y(t) -> Y(t) + c1 Y(t-1) + c2 Y(t-2) + c3 Y(t-3) + ...
        where autoreg_coefs = [c1, c2, c3, ...]
        In this fashion, the obtained series will have autoregressive
        properties not explained by seasonality and growth.
    :param fs_coefs: List[float]
        The fourier series coefficients used.
    :param growth_coef: float
        Multiplier for growth
    :param growth_pow: float
        Power for growth, as function of continuous time
    :param intercept: float
        Constant term added to Y(t)
    :param seed: int
        seed for reproducible result

    :return: Dict[str, any]
        contains full dataframe, train dataframe, test dataframe,
        and nrows in test dataframe
    """
    if seed is not None:
        np.random.seed(seed)

    date_list = pd.date_range(
        start=train_start_date,
        periods=periods,
        freq=freq).tolist()

    df0 = pd.DataFrame({TIME_COL: date_list})
    if conti_year_origin is None:
        conti_year_origin = get_default_origin_for_time_vars(df0, TIME_COL)
    time_df = build_time_features_df(
        dt=df0[TIME_COL],
        conti_year_origin=conti_year_origin)
    df = pd.concat([df0, time_df], axis=1)
    df["growth"] = growth_coef * (df[TimeFeaturesEnum.ct1.value] ** growth_pow)

    func = fourier_series_multi_fcn(
        col_names=[
            TimeFeaturesEnum.toy.value,
            TimeFeaturesEnum.tow.value,
            TimeFeaturesEnum.tod.value],
        periods=[1.0, 7.0, 24.0],
        orders=[1, 1, 1],
        seas_names=None)

    res = func(df)
    df_seas = res["df"]
    df = pd.concat([df, df_seas], axis=1)

    df[VALUE_COL] = (
            intercept
            + df["growth"]
            + fs_coefs[0] * df[get_fourier_col_name(1, TimeFeaturesEnum.tod.value, function_name="sin")]
            + fs_coefs[1] * df[get_fourier_col_name(1, TimeFeaturesEnum.tow.value, function_name="sin")]
            + fs_coefs[2] * df[get_fourier_col_name(1, TimeFeaturesEnum.toy.value, function_name="sin")]
            + noise_std * np.random.normal(size=df.shape[0]))

    if autoreg_coefs is not None:
        df["temporary_new_value"] = df[VALUE_COL]
        k = len(autoreg_coefs)
        for i in range(k):
            df["temporary_new_value"] = (
                    df["temporary_new_value"] +
                    autoreg_coefs[i]*df[VALUE_COL].shift(-i)).bfill()
        df[VALUE_COL] = df["temporary_new_value"]
        del df["temporary_new_value"]

    if train_end_date is None:
        train_rows = np.floor(train_frac * df.shape[0]).astype(int)
        train_end_date = df[TIME_COL][train_rows]

    if remove_extra_cols:
        df = df[[TIME_COL, VALUE_COL]]
    train_df = df.loc[df[TIME_COL] <= train_end_date]
    test_df = df.loc[df[TIME_COL] > train_end_date]
    fut_time_num = test_df.shape[0]

    return {
        "df": df,
        "train_df": train_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "fut_time_num": fut_time_num,
    }


def generate_df_with_holidays(freq, periods):
    # generate data
    df = generate_df_for_tests(freq, periods, remove_extra_cols=False)["df"]

    # generate holidays
    countries = ["US", "India"]
    event_df_dict = get_holidays(countries, year_start=2015, year_end=2025)

    for country in countries:
        event_df_dict[country][EVENT_DF_LABEL_COL] = country + "_holiday"

    df = add_daily_events(
        df=df,
        event_df_dict=event_df_dict,
        date_col=TIME_COL,
        regular_day_label="")

    df[VALUE_COL] = (
            df[VALUE_COL]
            + 2 * (df["events_US"] == "US_holiday") * df[
                get_fourier_col_name(1, TimeFeaturesEnum.tod.value, function_name="sin")]
            + 3 * (df["events_US"] == "US_holiday") * df[
                get_fourier_col_name(1, TimeFeaturesEnum.tod.value, function_name="cos")]
            + 4 * (df["events_India"] == "India_holiday") * df[
                get_fourier_col_name(1, TimeFeaturesEnum.tod.value, function_name="cos")])

    df = df[[TIME_COL, VALUE_COL]]
    thresh = datetime.datetime(2019, 8, 1)
    train_df = df[df[TIME_COL] <= thresh]
    test_df = df[df[TIME_COL] > thresh]
    fut_time_num = test_df.shape[0]

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "fut_time_num": fut_time_num,
    }


def generate_df_with_reg_for_tests(
        freq,
        periods,
        train_start_date=datetime.datetime(2018, 7, 1),
        train_end_date=None,
        train_frac=0.8,
        conti_year_origin=None,
        noise_std=2.0,
        remove_extra_cols=True,
        mask_test_actuals=False,
        seed=123):
    """Generates dataset for unit tests that includes regressor columns
    :param freq: str
        pd.date_range freq parameter, e.g. H or D
    :param periods: int
        number of periods to generate
    :param train_start_date: datetime.datetime
        train start date
    :param train_end_date: Optional[datetime.datetime]
        train end date
    :param train_frac: Optional[float]
        fraction of data to use for training
        only used if train_end_date isn't provided
    :param noise_std: float
        standard deviation of gaussian noise
    :param conti_year_origin: float
        the time origin for continuous time variables
    :param remove_extra_cols: bool
        whether to remove extra columns besides TIME_COL, VALUE_COL
    :param mask_test_actuals: bool
        whether to set y values to np.NaN in the test set.
    :param seed: int
        seed for reproducible result
    :return: Dict with train dataframe, test dataframe, and nrows in test dataframe
    """
    if seed is not None:
        np.random.seed(seed)

    result_list = generate_df_for_tests(
        freq,
        periods,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        train_frac=train_frac,
        conti_year_origin=conti_year_origin,
        noise_std=noise_std,
        remove_extra_cols=False)

    df = result_list["df"]
    df["regressor1"] = (
            df["growth"]
            + 4 * df[get_fourier_col_name(1, TimeFeaturesEnum.tow.value, function_name="sin")]
            - 3 * df[get_fourier_col_name(1, TimeFeaturesEnum.tod.value, function_name="sin")]
            + 7 * df[get_fourier_col_name(1, TimeFeaturesEnum.toy.value, function_name="sin")]
            - noise_std * np.random.normal(size=df.shape[0]))

    df["regressor2"] = (
            df["growth"]
            + 1 * df[get_fourier_col_name(1, TimeFeaturesEnum.tow.value, function_name="sin")]
            - 2 * df[get_fourier_col_name(1, TimeFeaturesEnum.tod.value, function_name="sin")]
            + 3 * df[get_fourier_col_name(1, TimeFeaturesEnum.toy.value, function_name="sin")]
            + noise_std * np.random.normal(size=df.shape[0]))

    df["regressor3"] = (
            df["growth"]
            + 9 * df[get_fourier_col_name(1, TimeFeaturesEnum.tow.value, function_name="sin")]
            - 8 * df[get_fourier_col_name(1, TimeFeaturesEnum.tod.value, function_name="sin")]
            + 5 * df[get_fourier_col_name(1, TimeFeaturesEnum.toy.value, function_name="sin")]
            + noise_std * np.random.normal(size=df.shape[0]))

    df["regressor_bool"] = np.random.rand(df.shape[0]) > 0.3
    df["regressor_categ"] = np.random.choice(a=["c1", "c2", "c3"], size=df.shape[0], p=[0.1, 0.2, 0.7])

    if train_end_date is None:
        train_rows = np.floor(train_frac * df.shape[0]).astype(int)
        train_end_date = df[TIME_COL][train_rows]

    regressor_cols = [
        "regressor1",
        "regressor2",
        "regressor3",
        "regressor_bool",
        "regressor_categ"]
    if remove_extra_cols:
        df = df[[TIME_COL, VALUE_COL] + regressor_cols]

    if mask_test_actuals:
        # False positive warning:
        #   pandas.core.common.SettingWithCopyError:
        #   A value is trying to be set on a copy of a slice from a DataFrame.
        #   Try using .loc[row_indexer,col_indexer] = value instead
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df.loc[df[TIME_COL] > train_end_date, VALUE_COL] = np.NaN

    train_df = df.loc[df[TIME_COL] <= train_end_date]
    test_df = df.loc[df[TIME_COL] > train_end_date]
    fut_time_num = test_df.shape[0]

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "fut_time_num": fut_time_num,
        "regressor_cols": regressor_cols,
    }


def daily_data_reg():
    return generate_df_with_reg_for_tests(
        freq="D",
        periods=500,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)


def hourly_data_reg():
    """Generate 500 days of hourly data for tests"""
    return generate_df_with_reg_for_tests(
        freq="H",
        periods=24*500,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)


def generate_test_changepoint_df(
        freq="D",
        periods=200,
        n_changepoints=3,
        signal_strength=1/5,
        err_std=1.0,
        seed=123):
    """Generates df to test change points

    The generated df is simple zigzag shaped over time with noise,
    with user specified frequency, length and number of change points.

    Parameters
    ----------
    freq : `DateOffset`, `Timedelta` or `str`, default is "D"
        The data frequency.
    periods : `int`, default is 200
        How many observation in frequency ``freq`` to be generated.
    n_changepoints : `int`
        How many change points to be included.
    signal_strength : `float`
        Time series signal strength.
    err_std : `float`
        Standard deviation of error.
    seed : `int`, default is 123
        Seed for reporducible result.

    Returns
    -------
    df : `pandas.DataFrame`
        The generated df with columns:
            `"ts"` : time column.
            `"y"` : value column.
    """
    if seed is not None:
        np.random.seed(seed)
    start = "2020-01-01"
    y = np.array([0])
    nochange_period = np.ceil(periods / (n_changepoints + 1))
    for i in range(n_changepoints + 1):
        temp = np.arange(nochange_period + 1)
        if i % 2 == 1:
            temp = temp[::-1]
        temp = temp[1:]  # remove the repeated term
        y = np.concatenate([y, temp])
    y = y * signal_strength + np.random.randn(len(y)) * err_std
    df = pd.DataFrame(
        {
            "ts": pd.date_range(start=start, freq=freq, periods=periods),
            "y": y[:periods]
        }
    )
    return df


def generate_anomalous_data(periods=10):
    ts = pd.date_range(start="1/1/2018", periods=periods)
    df = pd.DataFrame({
        "ts": ts,
        "y": range(periods),
        "z": range(20, 20+periods)})
    df["y"] = df["y"].astype(float)
    df["z"] = df["z"].astype(float)

    anomaly_df = pd.DataFrame({
        METRIC_COL: ["y", "y", "z", "z", "z"],
        "dimension1": ["level_1", "level_1", "level_2", "level_2", "level_2"],
        "dimension2": ["level_1", "level_2", "level_1", "level_1", "level_1"],
        START_TIME_COL: ["1/1/2018", "1/4/2018", "1/8/2018", "1/10/2018", "1/1/2099"],
        END_TIME_COL: ["1/2/2018", "1/6/2018", "1/9/2018", "1/10/2018", "1/2/2099"],
        ADJUSTMENT_DELTA_COL: [np.nan, 3., -5., np.nan, np.nan]})

    for col in [START_TIME_COL, END_TIME_COL]:
        anomaly_df[col] = pd.to_datetime(anomaly_df[col])

    return {
        "df": df,
        "anomaly_df": anomaly_df}


def generic_test_adjust_anomalous_data(
        value_col,
        adj_df_info,
        adj_values):
    """Generic test for the results of any given scenario"""
    augmented_df = adj_df_info["augmented_df"]
    adjusted_df = adj_df_info["adjusted_df"]
    assert list(adjusted_df.columns) == ["ts", "y", "z"]
    assert list(augmented_df.columns) == ["ts", "y", "z", ANOMALY_COL, f"adjusted_{value_col}"]
    assert_equal(
        adjusted_df[value_col],
        augmented_df[f"adjusted_{value_col}"],
        check_names=False)
    assert_equal(
        augmented_df[:len(adj_values)][f"adjusted_{value_col}"],
        adj_values,
        check_names=False)


def gen_sliced_df(
        sample_size_dict={"a": 100, "b": 200, "c": 300, "d": 8, "e": 3},
        seed_dict={"a": 301, "b": 167, "c": 593, "d": 893, "e": 191, "z": 397},
        err_magnitude_coef=1.0):
    """generates a data frame which includes a response variable with column name "y", and an
    estimate of y with column name "y_hat". y is designed to be a function of a categorical
    variable given in "x" and a continuous variable given in "z".
    For various levels of y, the variance of y is different and the sample size is also different.
    The noises added are all deterministic so that this data set does not change at random"""

    y_a = np.linspace(start=90, stop=100, num=sample_size_dict["a"], endpoint=True)
    np.random.seed(seed=seed_dict["a"])
    y_a_hat = y_a + err_magnitude_coef * np.random.random(len(y_a))

    y_b = np.linspace(start=190, stop=200, num=sample_size_dict["b"], endpoint=True)
    np.random.seed(seed=seed_dict["b"])
    y_b_hat = y_b + err_magnitude_coef * 1.5 * np.random.random(len(y_b))

    y_c = np.linspace(start=290, stop=300, num=sample_size_dict["c"], endpoint=True)
    np.random.seed(seed=seed_dict["c"])
    y_c_hat = y_c + err_magnitude_coef * 2 * np.random.random(len(y_c))

    y_d = np.linspace(start=490, stop=550, num=sample_size_dict["d"], endpoint=True)
    np.random.seed(seed=seed_dict["d"])
    y_d_hat = y_d + err_magnitude_coef * np.random.random(len(y_d))

    y_e = np.linspace(start=-10, stop=0, num=sample_size_dict["e"], endpoint=True)
    np.random.seed(seed=seed_dict["e"])
    y_e_hat = y_e + err_magnitude_coef * 3 * np.random.random(len(y_e))

    y = np.concatenate((y_a, y_b, y_c, y_d, y_e), axis=0, out=None)
    y_hat = np.concatenate((y_a_hat, y_b_hat, y_c_hat, y_d_hat, y_e_hat), axis=0, out=None)

    # also create a continuous covariate which might be useful in some simulations
    np.random.seed(seed=seed_dict["z"])
    z = np.random.random(len(y))
    y = y + 2 * z
    y_hat = y_hat + 2.1 * z

    df = pd.DataFrame({
        "x": ["a"]*len(y_a) + ["b"]*len(y_b) + ["c"]*len(y_c) + ["d"]*len(y_d) + ["e"]*len(y_e),
        "z": z,
        "y": y,
        "y_hat": y_hat})
    df["residual"] = df["y"] - df["y_hat"]
    # adding another categorical column
    df["z_categ"] = df["z"].round().astype(str)

    return df


def assert_eval_function_equal(f1, f2):
    """Checks whether evaluation functions return the same output for a few inputs"""
    y_true = pd.Series([3, 1, 3])
    y_pred = pd.Series([1, 4, 2])
    assert f1(y_true, y_pred) == f2(y_true, y_pred)

    y_true = pd.Series([-20, 2, -10, 3.0])
    y_pred = pd.Series(np.arange(4))
    assert f1(y_true, y_pred) == f2(y_true, y_pred)

    y_true = pd.Series([0])
    y_pred = pd.Series([0])
    assert f1(y_true, y_pred) == f2(y_true, y_pred)

    np.random.seed(92)
    y_true = np.random.random(42)
    y_pred = np.random.random(42)
    assert_equal(f1(y_true, y_pred), f2(y_true, y_pred))


def generate_df_with_arbitrary_trends_and_shifts(
        start_date="2015-01-01",
        length=365,
        freq="D",
        seed=10,
        trend_slopes=None,
        trend_intervals=None,
        level_shifts=None,
        level_shift_magnitudes=None):
    """Generates a Pandas DataFrame that represents time series data with arbitrary trends and level shifts.
    Example Usage: Calling `generate_df_with_arbitrary_trends_and_shifts(trend_slopes=[-1., 1.], trend_intervals=[.3, 1.],
    level_shifts=[(.3, .6)], level_shift_magnitudes=[100.])` produces a time series with a slope of -1 for the first 30%
    of the interval. The next 70% ((1. - .3)*100 = 70) of the interval has a slope of 1. The time series across the interval
    .3 to .6, of the entire length, will have a positive level shift of magnitude 100.

    Parameters
    ----------
    start_date : `str`
        The start date of the time series data in "YYYY-MM-DD" format. Defaults to '2015-01-01'.
    length : `int`
        The number of data points in the time series. Defaults to 365.
    freq : `str`
        Frequency of the timestamps (based on pandas date_range frequencies). Defaults to "D" for daily.
    seed : `int`
        The seed for the random number generator. Defaults to 10.
    trend_slopes : `Optional[List[float]]`
        A list of slopes representing trends in the time series. Each slope applies to a corresponding
        interval from 'trend_intervals'. Defaults to None and  the series produced has a slope of 1.
    trend_intervals : `Optional[List[float]]`
        A list of values between 0 and 1 that represent the intervals of the corresponding trends in
        'trend_slopes'. The start of a new interval indicates the end of the previous one. The final
        interval must end at 1. Defaults to None.
    level_shifts : `Optional[List[Tuple[float, float]]]`
        A list of tuples where each tuple contains two float values between 0 and 1. Each tuple
        represents the start and end of a level shift in the time series. Defaults to None.
    level_shift_magnitudes : `Optional[List[float]]`
        A list of magnitudes for each level shift. Each magnitude corresponds to a tuple from
        'level_shifts'. Defaults to None.

    Returns
    -------
        df : `pandas.DataFrame`
            A DataFrame representing the generated time series with columns:
                `"timestamp"` : time column.
                `"y"` : value column.

    Raises
    ------
    ValueError: If the first trend interval does not have a positive value.
    ValueError: If the last trend interval does not end at 1.
    ValueError: If the trend intervals are not strictly monotonically increasing.
    ValueError: If each level shift does not have a start and end specified for its duration.
    ValueError: If the start and end points of a level shift are not valid fractions of the time series duration.
    ValueError: If the number of trend slopes is not equal to the number of trend intervals.
    ValueError: If the number of level shifts is not equal to the number of level shift magnitudes.
    """
    if len(trend_slopes) != len(trend_intervals):
        raise ValueError("Each trend needs an interval specified for its duration.")

    def _verify_intervals(intervals):
        if intervals[0] <= 0:
            raise ValueError("The beginning trend must have a positive value")
        if float(intervals[-1]) != 1.:
            raise ValueError("The final interval must end at 1.")
        for i in range(len(intervals) - 1):
            if intervals[i+1] <= intervals[i]:
                raise ValueError("The intervals must be strictly monotonically increasing.")

    def _verify_level_shifts(level_shift_starts, level_shift_ends):
        if len(level_shift_starts) != len(level_shift_ends):
            raise ValueError("Each level shift needs a start and end specified for its duration.")
        for ls_start, ls_end in zip(level_shift_starts, level_shift_ends):
            if ls_end <= ls_start or ls_start < 0 or ls_end > 1:
                raise ValueError("Level shift start and end points should be valid fractions of the time series duration, with end > start.")
    np.random.seed(seed)
    if trend_slopes and trend_intervals:
        _verify_intervals(trend_intervals)

    if level_shifts and level_shift_magnitudes:
        level_shift_starts, level_shift_ends = zip(*level_shifts)
        _verify_level_shifts(level_shift_starts, level_shift_ends)
        if len(level_shifts) != len(level_shift_magnitudes):
            raise ValueError("Each level shift needs a magnitude specified.")
    ts = pd.date_range(start_date, freq=freq, periods=length)
    trend_slopes = trend_slopes or [1.]
    trend_intervals = trend_intervals or [1.]
    level_shifts = level_shifts or []
    level_shift_magnitudes = level_shift_magnitudes or []
    hop_size = length / len(trend_slopes)
    trend_intervals_indices = [0] + [int(hop_size * (i + 1)) for i in range(len(trend_slopes) - 1)] + [length]
    base_value_with_trend_changes = np.array([])
    for i in range(len(trend_intervals_indices) - 1):
        cur_trend_magnitude = trend_slopes[i]
        begin, end = trend_intervals_indices[i], trend_intervals_indices[i+1] - 1
        interval_length = end - begin + 1
        prev_endpoint = base_value_with_trend_changes[-1] if len(base_value_with_trend_changes) > 0 else 0.
        base_value_with_trend_changes = np.concatenate(
            (
                base_value_with_trend_changes,
                cur_trend_magnitude * np.arange(interval_length, dtype=np.float64) + prev_endpoint
            )
        )
    for (ls_start, ls_end), ls_magnitude in zip(level_shifts, level_shift_magnitudes):
        ls_start_index = int(ls_start * length)
        ls_end_index = int(ls_end * length)
        base_value_with_trend_changes[ls_start_index:ls_end_index] += ls_magnitude
    df = pd.DataFrame({"timestamp": ts, "y": base_value_with_trend_changes})
    return df
