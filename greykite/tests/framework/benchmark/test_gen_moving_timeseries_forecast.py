import numpy as np
import pandas as pd
import pytest

from greykite.common.testing_utils import generate_df_for_tests
from greykite.framework.benchmark.gen_moving_timeseries_forecast import gen_moving_timeseries_forecast


def test_gen_moving_timeseries_forecast():
    """Basic test for `gen_moving_timeseries_forecast`"""
    data = generate_df_for_tests(freq="1H", periods=1000)
    df = data["df"]

    # A simple train-forecast function which always uses last available
    # value as the forecasted value
    def train_forecast_func(
            df,
            value_col,
            time_col=None,
            forecast_horizon=1):
        # Simply get last observed value and offer as forecast
        value = df[value_col].values[df.shape[0] - 1]
        forecasted_values = np.repeat(a=value, repeats=forecast_horizon)
        fut_df = pd.DataFrame({value_col: forecasted_values})
        return {"fut_df": fut_df}

    compare_df = gen_moving_timeseries_forecast(
        df=df,
        time_col="ts",
        value_col="y",
        train_forecast_func=train_forecast_func,
        train_move_ahead=10,
        forecast_horizon=7,
        min_training_end_point=100)["compare_df"]

    assert list(compare_df.head(5)["y_hat"].round(1).values) == [-0.6] * 5

    # Input start and end times
    compare_df = gen_moving_timeseries_forecast(
        df=df,
        time_col="ts",
        value_col="y",
        train_forecast_func=train_forecast_func,
        train_move_ahead=10,
        forecast_horizon=7,
        min_training_end_point=None,
        min_training_end_timestamp="2018-07-15",
        max_forecast_end_point=None,
        max_forecast_end_timestamp="2018-08-01")["compare_df"]

    assert list(compare_df.head(5)["y_hat"].round(1).values) == [-3.5] * 5

    expected_match = "No reasonble train test period is found for validation"
    with pytest.raises(ValueError, match=expected_match):
        gen_moving_timeseries_forecast(
            df=df,
            time_col="ts",
            value_col="y",
            train_forecast_func=train_forecast_func,
            train_move_ahead=10,
            forecast_horizon=700,
            min_training_end_point=1000)["compare_df"]


def test_gen_moving_timeseries_forecast_with_regressors():
    """Test for `gen_moving_timeseries_forecast` with regressors"""
    data = generate_df_for_tests(freq="1H", periods=1000)
    df = data["df"]
    df["x"] = 40

    # A simple train-forecast function which always uses last available
    # value, then adds the regressor value -> forecasted value
    def train_forecast_func(
            df,
            value_col,
            time_col,
            forecast_horizon,
            new_external_regressor_df):
        # Simply get last observed value and offer as forecast
        value = df[value_col].values[-1]
        forecasted_values = np.repeat(a=value, repeats=forecast_horizon)
        fut_df = pd.DataFrame({value_col: forecasted_values})
        # Adds regerssor value
        fut_df[value_col] = fut_df[value_col] + new_external_regressor_df["x"]
        return {"fut_df": fut_df}

    compare_df = gen_moving_timeseries_forecast(
        df=df,
        time_col="ts",
        value_col="y",
        train_move_ahead=10,
        forecast_horizon=7,
        train_forecast_func=train_forecast_func,
        min_training_end_point=100,
        regressor_cols=["x"])["compare_df"]

    assert list(compare_df.head(5)["y_hat"].round(1).values) == [39.4] * 5


def test_gen_moving_timeseries_forecast_extra_columns():
    """Basic test for `gen_moving_timeseries_forecast` to ensure
    the function will keep the desired columns in resulting ``compare_df``"""
    data = generate_df_for_tests(freq="1H", periods=1000)
    df = data["df"]
    df["input_dummy"] = np.array(range(df.shape[0]))

    # A simple train-forecast function which always uses last available
    # value as the forecasted value
    def train_forecast_func(
            df,
            value_col,
            time_col=None,
            forecast_horizon=1):
        # Simply get last observed value and offer as forecast
        value = df[value_col].values[df.shape[0] - 1]
        forecasted_values = np.repeat(a=value, repeats=forecast_horizon)
        fut_df = pd.DataFrame({value_col: forecasted_values})
        fut_df["forecast_dummy"] = np.array(range(fut_df.shape[0]))
        return {"fut_df": fut_df}

    compare_df = gen_moving_timeseries_forecast(
        df=df,
        time_col="ts",
        value_col="y",
        train_forecast_func=train_forecast_func,
        train_move_ahead=10,
        forecast_horizon=7,
        min_training_end_point=100,
        keep_cols=["input_dummy"],
        forecast_keep_cols=["forecast_dummy"])["compare_df"]

    assert "forecast_dummy" in list(compare_df.columns)
    assert "input_dummy" in list(compare_df.columns)
