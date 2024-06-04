from datetime import datetime

import pandas as pd
import pytest

from greykite.algo.changepoint.shift_detection.shift_detector import ShiftDetection
from greykite.common.constants import LEVELSHIFT_COL_PREFIX_SHORT


# Test common time and value column names for Greykite and OLOF respectively.
@pytest.mark.parametrize("time_col, value_col", [("ts", "actual"), ("timestamp", "value")])
def test_detect_daily(time_col: str, value_col: str):
    # create input_df
    input_val_ls = [100] * 10 + [200] * 10 + [300] * 10
    input_ts_ls = pd.date_range(datetime(2020, 1, 1), freq="D", periods=30)
    input_df = pd.DataFrame({time_col: input_ts_ls, value_col: input_val_ls})

    # create expected_df
    expected_df = pd.DataFrame({
        time_col: pd.date_range(datetime(2020, 1, 1), freq="D", periods=35),
        value_col: [100] * 10 + [200] * 10 + [300] * 10 + [None] * 5,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_11_00_00": [0] * 10 + [1] * 25,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_21_00_00": [0] * 20 + [1] * 15
    })

    # call the function
    detector = ShiftDetection()
    output_regressor_col, output_df = detector.detect(
        input_df,
        time_col=time_col,
        value_col=value_col,
        forecast_horizon=5,
        freq="D",
        z_score_cutoff=3
    )

    # unit test
    pd.testing.assert_frame_equal(output_df, expected_df)
    assert output_regressor_col == [f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_11_00_00",
                                    f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_21_00_00"]


@pytest.mark.parametrize("time_col, value_col", [("ts", "actual"), ("timestamp", "value")])
def test_detect_weekly(time_col: str, value_col: str):
    # create input_df
    input_val_ls = [100] * 10 + [200] * 10 + [300] * 10
    input_ts_ls = pd.date_range(datetime(2020, 1, 1), freq="W", periods=30)
    input_df = pd.DataFrame({time_col: input_ts_ls, value_col: input_val_ls})

    # create expected_df
    expected_df = pd.DataFrame({
        time_col: pd.date_range(datetime(2020, 1, 1), freq="W", periods=35),
        value_col: [100] * 10 + [200] * 10 + [300] * 10 + [None] * 5,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_03_15_00_00": [0] * 10 + [1] * 25,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_05_24_00_00": [0] * 20 + [1] * 15
    })

    # call the function
    detector = ShiftDetection()
    output_regressor_col, output_df = detector.detect(
        input_df,
        time_col=time_col,
        value_col=value_col,
        forecast_horizon=5,
        freq="W",
        z_score_cutoff=3
    )

    # unit test
    pd.testing.assert_frame_equal(output_df, expected_df)
    assert output_regressor_col == [f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_03_15_00_00",
                                    f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_05_24_00_00"]


def test_invalid_freq(time_col="ts", value_col="actual"):
    # create input_df
    input_val_ls = [100] * 10 + [200] * 10 + [300] * 10
    input_ts_ls = pd.date_range(datetime(2020, 1, 1), freq="S", periods=30)
    input_df = pd.DataFrame({time_col: input_ts_ls, value_col: input_val_ls})

    # call the function
    detector = ShiftDetection()
    with pytest.raises(ValueError):
        output_regressor_col, output_df = detector.detect(
            input_df,
            time_col=time_col,
            value_col=value_col,
            forecast_horizon=5,
            freq="S",
            z_score_cutoff=3
        )


def test_find_shifts():
    # create input_df
    input_val_ls = [100] * 10 + [200] * 10 + [300] * 10
    input_ts_ls = pd.date_range(datetime(2020, 1, 1), freq="D", periods=30)
    input_df = pd.DataFrame({"ts": input_ts_ls, "actual": input_val_ls})

    # create expected results
    expected_shift_dates = [
        (datetime(2020, 1, 11, 0, 0, 0), datetime(2020, 1, 11, 0, 0, 0)),
        (datetime(2020, 1, 21, 0, 0, 0), datetime(2020, 1, 21, 0, 0, 0))]

    expected_df_find_shifts = pd.DataFrame({
        "ts": pd.date_range(datetime(2020, 1, 1), freq="D", periods=30),
        "actual": [100] * 10 + [200] * 10 + [300] * 10,
        "actual_diff": [None] + [0] * 9 + [100] + [0] * 9 + [100] + [0] * 9,
        "zscore": [None] + [-0.267432] * 9 + [3.610330] + [-0.267432] * 9 + [3.610330] + [-0.267432] * 9
    })

    # call the function
    detector = ShiftDetection()
    output_df_find_shifts, output_shift_dates = detector.find_shifts(
        input_df,
        time_col="ts",
        value_col="actual",
        z_score_cutoff=3
    )

    # unit test
    assert output_shift_dates == expected_shift_dates
    pd.testing.assert_frame_equal(output_df_find_shifts, expected_df_find_shifts)


def test_find_no_shift():
    # create input_df with no shift
    input_val_ls = list(range(30))
    input_ts_ls = pd.date_range(datetime(2020, 1, 1), freq="D", periods=30)
    input_df = pd.DataFrame({"ts": input_ts_ls, "actual": input_val_ls})

    # create expected results
    expected_shift_dates = []

    expected_df_find_shifts = pd.DataFrame({
        "ts": pd.date_range(datetime(2020, 1, 1), freq="D", periods=30),
        "actual": list(range(30)),
        "actual_diff": [None] + [1] * 29,
        "zscore": [float('nan')] * 30  # actual_diff's standard deviation is 0 so zscore is 0/0=NaN
    })

    # call the function
    detector = ShiftDetection()
    output_df_find_shifts, output_shift_dates = detector.find_shifts(
        input_df,
        time_col="ts",
        value_col="actual",
        z_score_cutoff=3
    )

    # unit test
    assert output_shift_dates == expected_shift_dates
    pd.testing.assert_frame_equal(output_df_find_shifts, expected_df_find_shifts)


def test_create_df_with_regressor():
    # create inputs
    input_shiftsm = [
        (datetime(2020, 1, 11, 0, 0, 0), datetime(2020, 1, 11, 0, 0, 0)),
        (datetime(2020, 1, 21, 0, 0, 0), datetime(2020, 1, 21, 0, 0, 0))]
    input_val_ls = [100] * 10 + [200] * 10 + [300] * 10
    input_ts_ls = pd.date_range(datetime(2020, 1, 1), freq="D", periods=30)
    input_df = pd.DataFrame({"ts": input_ts_ls, "actual": input_val_ls})

    # create expected outputs
    expected_df_regressor = pd.DataFrame({
        "ts": pd.date_range(datetime(2020, 1, 1), freq="D", periods=30),
        "actual": [100] * 10 + [200] * 10 + [300] * 10,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_11_00_00": [0] * 10 + [1] * 20,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_21_00_00": [0] * 20 + [1] * 10
    })

    # call the function
    detector = ShiftDetection()
    output_df_regressor = detector.create_df_with_regressor(input_df, "ts", input_shiftsm)

    # unit test
    pd.testing.assert_frame_equal(output_df_regressor, expected_df_regressor, check_dtype=False)


def test_create_regressor_for_future_dates():
    # create inputs
    input_df_regressor = pd.DataFrame({
        "ts": pd.date_range(datetime(2020, 1, 1), freq="D", periods=30),
        "actual": [100] * 10 + [200] * 10 + [300] * 10,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_11_00_00": [0] * 10 + [1] * 20,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_21_00_00": [0] * 20 + [1] * 10
    })

    # create expected outputs
    expected_df = pd.DataFrame({
        "ts": pd.date_range(datetime(2020, 1, 1), freq="D", periods=35),
        "actual": [100] * 10 + [200] * 10 + [300] * 10 + [None] * 5,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_11_00_00": [0] * 10 + [1] * 25,
        f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_21_00_00": [0] * 20 + [1] * 15
    })

    # call the function
    detector = ShiftDetection()
    output_regressor_col, output_df = detector.create_regressor_for_future_dates(
        input_df_regressor,
        time_col="ts",
        value_col="actual",
        forecast_horizon=5,
        freq="D",
    )

    # unit test
    pd.testing.assert_frame_equal(output_df, expected_df)
    assert output_regressor_col == [f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_11_00_00",
                                    f"{LEVELSHIFT_COL_PREFIX_SHORT}_2020_01_21_00_00"]
