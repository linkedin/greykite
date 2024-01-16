import itertools

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import LOGGER_NAME
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import TimeFeaturesEnum
from greykite.common.testing_utils import assert_equal
from greykite.detection.detector.ad_utils import add_new_params_to_records
from greykite.detection.detector.ad_utils import get_anomaly_df
from greykite.detection.detector.ad_utils import get_anomaly_df_from_outliers
from greykite.detection.detector.ad_utils import get_canonical_anomaly_df
from greykite.detection.detector.ad_utils import get_timestamp_ceil
from greykite.detection.detector.ad_utils import get_timestamp_floor
from greykite.detection.detector.ad_utils import optimize_df_with_constraints
from greykite.detection.detector.ad_utils import partial_return
from greykite.detection.detector.ad_utils import validate_volatility_features
from greykite.detection.detector.ad_utils import vertical_concat_dfs


@pytest.fixture(scope="module")
def y_clean():
    """Constructs a clean vector of random numbers, used in outlier removal tests."""
    sampler = np.random.default_rng(1317)
    # Defines two clean vectors, one for `fit` and one for `detect`.
    y_clean = np.arange(0, 998)
    # Add small noise
    y_clean = y_clean + sampler.normal(loc=0.0, scale=1.0, size=len(y_clean))

    return y_clean


def test_partial_return():
    """Tests `partial_return`."""
    def func(x):
        return {"1": x, "2": -x, "3": x+100}

    v = partial_return(func, "1")(x=10)
    assert v == 10

    v1 = partial_return(func, "1")(10)
    assert v1 == 10

    # The case for lists.
    def func(x):
        return [x + 1, x + 2, x + 33]

    v = partial_return(func, 0)(x=10)
    assert v == 11

    v1 = partial_return(func, 1)(200)
    assert v1 == 202

    # The case for which index is out of bound.
    v2 = partial_return(func, 25)(200)
    assert v2 is None


def test_vertical_concat_dfs():
    """Tests `vertical_concat_dfs`."""
    df0 = pd.DataFrame({
        "ts": [0, 1, 2, 3, 4],
        "day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "y": [10, 20, 30, 40, 50]})

    df1 = pd.DataFrame({
        "ts": [0, 1, 2, 3, 4],
        "day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "y": [11, 21, 31, 41, 51]})

    df = vertical_concat_dfs(
        df_list=[df0, df1],
        join_cols=["ts"],
        common_value_cols=["day"],
        different_value_cols=["y"])

    expected_df = pd.DataFrame({
        "ts": [0, 1, 2, 3, 4],
        "day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "y0": [10, 20, 30, 40, 50],
        "y1": [11, 21, 31, 41, 51]})

    assert pd.DataFrame.equals(df, expected_df)


def test_add_new_params_to_records():
    """Tests `add_new_params_to_records`."""
    grid_seed_dict = {
        "a": [1, 2, 3],
        "cat": ["boz", "asb"]}

    var_names = list(grid_seed_dict.keys())
    combinations_list = list(
        itertools.product(*[grid_seed_dict[var] for var in var_names]))

    df = pd.DataFrame(combinations_list, columns=var_names)

    records = df.to_dict("records")
    expanded_param_list = add_new_params_to_records(
        new_params={"dog": [1, 3], "horse": [13, 17]},
        records=records)

    assert (len(expanded_param_list)) == len(records) * 2 * 2
    assert expanded_param_list[0] == {"a": 1, "cat": "boz", "dog": 1, "horse": 13}
    assert expanded_param_list[-1] == {"a": 3, "cat": "asb", "dog": 3, "horse": 17}


def test_get_anomaly_df():
    """Tests `get_anomaly_df`."""
    # All anomalies
    df = pd.DataFrame({
        TIME_COL: pd.date_range(start="2020-01-01", periods=10, freq="D"),
        ANOMALY_COL: [True, True, True, True, True, True, True, True, True, True]})
    anomaly_df = get_anomaly_df(
        df=df,
        time_col=TIME_COL,
        anomaly_col=ANOMALY_COL)
    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-01"]),
        END_TIME_COL: pd.to_datetime(["2020-01-10"])})
    assert_equal(anomaly_df, expected_anomaly_df)

    # No anomalies
    df = pd.DataFrame({
        TIME_COL: pd.date_range(start="2020-01-01", periods=10, freq="D"),
        ANOMALY_COL: [False, False, False, False, False, False, False, False, False, False]})
    anomaly_df = get_anomaly_df(
        df=df,
        time_col=TIME_COL,
        anomaly_col=ANOMALY_COL)
    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime([]),
        END_TIME_COL: pd.to_datetime([])})
    assert_equal(anomaly_df, expected_anomaly_df)

    # Distinct anomalies (single data point and multiple data points)
    df = pd.DataFrame({
        TIME_COL: pd.date_range(start="2020-01-01", periods=10, freq="D"),
        ANOMALY_COL: [False, True, True, True, False, True, False, False, False, False]})
    anomaly_df = get_anomaly_df(
        df=df,
        time_col=TIME_COL,
        anomaly_col=ANOMALY_COL)
    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-02", "2020-01-06"]),
        END_TIME_COL: pd.to_datetime(["2020-01-04", "2020-01-06"])})
    assert_equal(anomaly_df, expected_anomaly_df)


def test_get_canonical_anomaly_df():
    """Tests `get_canonical_anomaly_df`."""
    # Non-overlapping anomaly periods
    anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-01", "2020-02-01"]),
        END_TIME_COL: pd.to_datetime(["2020-01-02", "2020-02-05"])})
    canonical_anomaly_df = get_canonical_anomaly_df(
        anomaly_df=anomaly_df,
        freq="D")
    assert_equal(canonical_anomaly_df, anomaly_df)
    # Partially overlapping anomaly periods
    anomaly_df = pd.DataFrame({
        "begin": ["2020-01-02-05", "2020-01-02-10", "2020-01-02-03", "2020-01-02-05"],
        "end": ["2020-01-02-15", "2020-01-02-20", "2020-01-02-17", "2020-01-02-13"]})
    canonical_anomaly_df = get_canonical_anomaly_df(
        anomaly_df=anomaly_df,
        freq="H",
        start_time_col="begin",
        end_time_col="end")
    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-02-03"]),
        END_TIME_COL: pd.to_datetime(["2020-01-02-20"])})
    assert_equal(canonical_anomaly_df, expected_anomaly_df)
    # One anomaly period covers others
    anomaly_df = pd.DataFrame({
        "begin": ["2020-02-05", "2020-02-19", "2020-02-03", "2020-02-05"],
        "end": ["2020-02-15", "2020-02-20", "2020-02-17", "2020-02-13"]})
    canonical_anomaly_df = get_canonical_anomaly_df(
        anomaly_df=anomaly_df,
        freq="D",
        start_time_col="begin",
        end_time_col="end")
    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-02-03", "2020-02-19"]),
        END_TIME_COL: pd.to_datetime(["2020-02-17", "2020-02-20"])})
    assert_equal(canonical_anomaly_df, expected_anomaly_df)
    # Checks anomaly merging logic.
    anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"]),
        END_TIME_COL: pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-05"])})
    canonical_anomaly_df = get_canonical_anomaly_df(
        anomaly_df=anomaly_df,
        freq="D")
    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-01"]),
        END_TIME_COL: pd.to_datetime(["2020-01-05"])})
    assert_equal(canonical_anomaly_df, expected_anomaly_df)
    # Checks anomaly merging logic
    anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"]),
        END_TIME_COL: pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-05"])})
    canonical_anomaly_df = get_canonical_anomaly_df(
        anomaly_df=anomaly_df,
        freq="H")
    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-01", "2020-01-04"]),
        END_TIME_COL: pd.to_datetime(["2020-01-03", "2020-01-05"])})
    assert_equal(canonical_anomaly_df, expected_anomaly_df)


def test_optimize_df_with_constraints():
    """Tests `optimize_df_with_constraints`."""
    df = pd.DataFrame({
        "a": [20, 20, 5, 10, 10, 0, 5],
        "b": [1, 1, 2, 2, 3, 3, 4],
        "c": [1, 2, 3, 4, 5, 6, 7]})
    # Constraint is satisfied, unique optimal
    with LogCapture(LOGGER_NAME) as log_capture:
        optimal_dict = optimize_df_with_constraints(
            df=df,
            objective_col="a",
            constraint_col="b",
            constraint_value=4)
        expected_optimal_dict = {"a": 5, "b": 4, "c": 7}
        assert_equal(optimal_dict, expected_optimal_dict)
        log_capture.check(
            (LOGGER_NAME,
             "INFO",
             f"Values satisfying the constraint are found.\n"
             f"Solving the following optimization problem:\n"
             f"Maximize a subject to b >= 4."))
    # Constraint is satisfied. Multiple rows with the maximum value
    # in ``objective_col``, resolved by considering ``constraint_col``
    optimal_dict = optimize_df_with_constraints(
        df=df,
        objective_col="a",
        constraint_col="b",
        constraint_value=2)
    expected_optimal_dict = {"a": 10, "b": 3, "c": 5}
    assert_equal(optimal_dict, expected_optimal_dict)
    # Constraint is satisfied. Multiple rows with the maximum value
    # in both ``objective_col`` and ``constraint_col``, last entry is chosen
    optimal_dict = optimize_df_with_constraints(
        df=df,
        objective_col="a",
        constraint_col="b",
        constraint_value=1)
    expected_optimal_dict = {"a": 20, "b": 1, "c": 2}
    assert_equal(optimal_dict, expected_optimal_dict)

    # Constraint is NOT satisfied, unique optimal
    with LogCapture(LOGGER_NAME) as log_capture:
        optimal_dict = optimize_df_with_constraints(
            df=df,
            objective_col="a",
            constraint_col="b",
            constraint_value=4.5)
        expected_optimal_dict = {"a": 5, "b": 4, "c": 7}
        assert_equal(optimal_dict, expected_optimal_dict)
        log_capture.check(
            (LOGGER_NAME,
             "INFO",
             f"No values satisfy the constraint.\n"
             f"Maximizing ``constraint_col`` (b) so that it is as "
             f"close as possible to the ``constraint_value`` (4.5)."))
    # Constraint is NOT satisfied. Multiple rows with the maximum value
    # in ``objective_col``, resolved by considering ``constraint_col``
    df = df[:6]
    optimal_dict = optimize_df_with_constraints(
        df=df,
        objective_col="a",
        constraint_col="b",
        constraint_value=3.5)
    expected_optimal_dict = {"a": 10, "b": 3, "c": 5}
    assert_equal(optimal_dict, expected_optimal_dict)
    # Constraint is NOT satisfied. Multiple rows with the maximum value
    # in both ``objective_col`` and ``constraint_col``, last entry is chosen
    df = df[:2]
    optimal_dict = optimize_df_with_constraints(
        df=df,
        objective_col="a",
        constraint_col="b",
        constraint_value=1.5)
    expected_optimal_dict = {"a": 20, "b": 1, "c": 2}
    assert_equal(optimal_dict, expected_optimal_dict)


def test_validate_volatility_features():
    """Tests ``validate_volatility_features``."""
    # Default behaviour, no
    volatility_features_list = [["dow"], ["dow"], ["dow", "hour"], ["dow", "hour", "dow"], ["is_holiday"]]
    validated_features_list = validate_volatility_features(
        volatility_features_list=volatility_features_list)
    expected_volatility_features_list = [["dow"], ["dow", "hour"], ["is_holiday"]]
    assert_equal(validated_features_list, expected_volatility_features_list)

    # Error when input feature is not in ``valid_features``
    with pytest.raises(ValueError, match="Unknown feature\\(s\\) \\({'is_holiday'}\\) in `volatility_features_list`."):
        valid_features = TimeFeaturesEnum._member_names_
        validate_volatility_features(
            volatility_features_list=volatility_features_list,
            valid_features=valid_features)


def test_get_timestamp_ceil():
    """Tests `get_timestamp_ceil`."""
    assert get_timestamp_ceil("2023-01-31", "M") == pd.to_datetime("2023-01-31")
    assert get_timestamp_ceil("2023-01-19", "M") == pd.to_datetime("2023-01-31")
    assert get_timestamp_ceil("2023-01-18", "W-MON") == pd.to_datetime("2023-01-23")
    assert get_timestamp_ceil("2023-01-23", "W-MON") == pd.to_datetime("2023-01-23")
    assert get_timestamp_ceil("2023-01-23", "D") == pd.to_datetime("2023-01-23")
    assert get_timestamp_ceil("2023-01-22 20:15", "D") == pd.to_datetime("2023-01-23")
    assert get_timestamp_ceil("2023-01-23", "B") == pd.to_datetime("2023-01-23")
    assert get_timestamp_ceil("2023-01-21 10:15", "B") == pd.to_datetime("2023-01-23")
    assert get_timestamp_ceil("2023-01-23 10:00:00", "H") == pd.to_datetime("2023-01-23 10:00:00")
    assert get_timestamp_ceil("2023-01-23 09:15:40", "H") == pd.to_datetime("2023-01-23 10:00:00")
    assert get_timestamp_ceil("2023-01-23 10:15:00", "T") == pd.to_datetime("2023-01-23 10:15:00")
    assert get_timestamp_ceil("2023-01-23 10:14:40", "T") == pd.to_datetime("2023-01-23 10:15:00")


def test_get_timestamp_floor():
    """Tests `get_timestamp_floor`."""
    assert get_timestamp_floor("2023-01-31", "M") == pd.to_datetime("2023-01-31")
    assert get_timestamp_floor("2023-01-19", "M") == pd.to_datetime("2022-12-31")
    assert get_timestamp_floor("2023-01-18", "W-MON") == pd.to_datetime("2023-01-16")
    assert get_timestamp_floor("2023-01-23", "W-MON") == pd.to_datetime("2023-01-23")
    assert get_timestamp_floor("2023-01-23", "D") == pd.to_datetime("2023-01-23")
    assert get_timestamp_floor("2023-01-22 20:15", "D") == pd.to_datetime("2023-01-22")
    assert get_timestamp_floor("2023-01-23", "B") == pd.to_datetime("2023-01-23")
    assert get_timestamp_floor("2023-01-21 10:15", "B") == pd.to_datetime("2023-01-20")
    assert get_timestamp_floor("2023-01-23 10:00:00", "H") == pd.to_datetime("2023-01-23 10:00:00")
    assert get_timestamp_floor("2023-01-23 09:15:40", "H") == pd.to_datetime("2023-01-23 09:00:00")
    assert get_timestamp_floor("2023-01-23 10:15:00", "T") == pd.to_datetime("2023-01-23 10:15:00")
    assert get_timestamp_floor("2023-01-23 10:14:40", "T") == pd.to_datetime("2023-01-23 10:14:00")


def test_get_anomaly_df_from_outliers1():
    """Tests `get_anomaly_df_from_outliers` when no outlier exists."""
    value_col = "y"
    freq = "H"

    # No outlier in Series - returns empty `pd.DataFrame` with columns `START_TIME_COL`, `END_TIME_COL`.
    df = pd.DataFrame({
        TIME_COL: pd.date_range(start="2020-01-01", end="2020-02-11 16:00:00", freq=freq),
        value_col: [2]*1001})
    empty_anomaly_df = pd.DataFrame(columns=[START_TIME_COL, END_TIME_COL])
    anomaly_df = get_anomaly_df_from_outliers(
        df=df,
        time_col=TIME_COL,
        value_col=value_col,
        freq=freq)
    assert_equal(anomaly_df, empty_anomaly_df)


def test_get_anomaly_df_from_outliers2(y_clean):
    """Tests `get_anomaly_df_from_outliers` when some outliers exist."""
    value_col = "y"
    freq = "H"

    # Tests when outliers exist.
    # The outliers are to be identified and saved to `anomaly_df`.
    y = [2, 1e10] + list(y_clean)
    df = pd.DataFrame({
        TIME_COL: pd.date_range(start="2020-01-01", end="2020-02-11 15:00:00", freq=freq),
        value_col: y})
    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-01 01:00:00"]),
        END_TIME_COL: pd.to_datetime(["2020-01-01 01:00:00"])})
    anomaly_df = get_anomaly_df_from_outliers(
        df=df,
        time_col=TIME_COL,
        value_col=value_col,
        freq=freq)
    assert_equal(anomaly_df, expected_anomaly_df)


def test_get_anomaly_df_from_outliers_raise_error():
    """Tests `get_anomaly_df_from_outliers` error catching."""
    value_col = "y"
    freq = "H"
    # Tests when outliers exist.
    # The outliers are to be identified and saved to `anomaly_df`.
    y = [2, 1e10] + [2] * 998
    df = pd.DataFrame({
        TIME_COL: pd.date_range(start="2020-01-01", end="2020-02-11 15:00:00", freq=freq),
        value_col: y})
    # Captures Error when input `time_col` is not in `df`.
    non_exist_col = "column not exist"
    with pytest.raises(ValueError, match=f"`df` does not have `time_col` with name {non_exist_col}."):
        get_anomaly_df_from_outliers(
            df=df,
            time_col=non_exist_col,
            value_col=value_col,
            freq=freq)

    # Captures Error when input `value_col` is not in `df`.
    with pytest.raises(ValueError, match=f"`df` does not have `value_col` with name {non_exist_col}."):
        get_anomaly_df_from_outliers(
            df=df,
            time_col=TIME_COL,
            value_col=non_exist_col,
            freq=freq)


def test_get_anomaly_df_from_outliers_small_data_size():
    """Tests `get_anomaly_df_from_outliers`."""
    value_col = "y"
    freq = "D"
    # Tests when outliers exist.
    # The outliers are to be identified and saved to `anomaly_df`.
    # This example contains less data
    ts = pd.date_range(start="2020-01-01", end="2020-05-01", freq=freq)
    y = list(range(len(ts)))
    # Let's overwrite the third value to be an outlier
    y[2] = 1000

    df = pd.DataFrame({
        TIME_COL: ts,
        value_col: y})

    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-03"]),
        END_TIME_COL: pd.to_datetime(["2020-01-03"])})

    anomaly_df = get_anomaly_df_from_outliers(
        df=df,
        time_col=TIME_COL,
        value_col=value_col,
        freq=freq,
        trim_percent=1.0)
    assert_equal(anomaly_df, expected_anomaly_df)

    # Disables trimming by setting it to zero.
    # In this case, we do not expect an empty dataframe
    anomaly_df = get_anomaly_df_from_outliers(
        df=df,
        time_col=TIME_COL,
        value_col=value_col,
        freq=freq,
        trim_percent=0.0)
    expected_anomaly_df = pd.DataFrame(columns=[START_TIME_COL, END_TIME_COL])
    assert_equal(anomaly_df, expected_anomaly_df)


def test_get_anomaly_df_from_outliers_with_missing1():
    """Tests `get_anomaly_df_from_outliers` with outliers and missing values."""
    value_col = "y"
    freq = "H"
    # Tests when outliers exist and there is a missing value.
    # The outliers are to be identified and saved to `anomaly_df`.
    y = [2, 1e10] + list(range(998))
    df = pd.DataFrame({
        TIME_COL: pd.date_range(
            start="2020-01-01",
            end="2020-02-11 15:00:00",
            freq=freq),
        value_col: y})

    df[value_col][2] = None

    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-01 01:00:00"]),
        END_TIME_COL: pd.to_datetime(["2020-01-01 01:00:00"])})
    anomaly_df = get_anomaly_df_from_outliers(
        df=df,
        time_col=TIME_COL,
        value_col=value_col,
        freq=freq,
        trim_percent=0.0)
    assert_equal(anomaly_df, expected_anomaly_df)


def test_get_anomaly_df_from_outliers_with_missing2(y_clean):
    """Tests `get_anomaly_df_from_outliers` with outliers and missing values."""
    value_col = "y"
    freq = "H"

    # Tests when outliers exist and there is a missing value.
    # The outliers are to be identified and saved to `anomaly_df`.
    y = [0.0, 1e10] + list(y_clean)
    ts = pd.date_range(
        start="2020-01-01",
        end="2020-02-11 15:00:00",
        freq=freq)
    assert len(ts) == len(y)
    df = pd.DataFrame({
        TIME_COL: ts,
        value_col: y})

    # Setting the third element to be None.
    df[value_col][2] = None

    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-01 01:00:00"]),
        END_TIME_COL: pd.to_datetime(["2020-01-01 01:00:00"])})

    anomaly_df = get_anomaly_df_from_outliers(
        df=df,
        time_col=TIME_COL,
        value_col=value_col,
        freq=freq,
        trim_percent=1.0)

    assert_equal(anomaly_df, expected_anomaly_df)


def test_get_anomaly_df_from_outliers_with_missing_raise_error():
    """Tests `get_anomaly_df_from_outliers` with outliers and missing values."""
    value_col = "y"
    freq = "H"
    # Tests when outliers exist and there is a missing value.
    # The outliers are to be identified and saved to `anomaly_df`.
    df = pd.DataFrame({
        TIME_COL: pd.date_range(
            start="2020-01-01",
            end="2020-02-11 15:00:00",
            freq=freq),
        value_col: [2, 1e10] + list(range(998))})

    df[value_col][2] = None
    # Captures Error when y does not have at least two values after removing outliers.
    df0 = df[:3]
    df0[value_col] = [None, 3, None]
    with pytest.raises(
            ValueError,
            match=f"Length of y after removing NAs is less than 2."):
        get_anomaly_df_from_outliers(
            df=df0,
            time_col=TIME_COL,
            value_col=value_col,
            freq=freq)

    # Captures warning when y has at least two values after removing outliers.
    df0 = df[:3]
    # In this example, only one value remains after trimming.
    df0[value_col] = [-5*10000, 1, 5*10000]
    with pytest.warns(
            UserWarning,
            match=f"After trimming there were less than two values:"):
        get_anomaly_df_from_outliers(
            df=df0,
            time_col=TIME_COL,
            value_col=value_col,
            freq=freq,
            trim_percent=5.0)
