import pandas as pd

from greykite.algo.common.forecast_one_by_one import forecast_one_by_one_fcn
from greykite.common.python_utils import assert_equal


def test_forecast_one_by_one_fcn():

    ts = pd.date_range(start="1/1/2018", end="1/10/2018")
    df = pd.DataFrame({"ts": ts, "y": range(10)})

    # A simple train-forecast function only for testing
    def train_forecast_func(
            df,
            value_col,
            time_col=None,
            forecast_horizon=1):
        # Gets last value and adds forecast horizon.
        # This is not meant to be a good forecast.
        # Rather, it is intended for test values to be simple enough to be derived
        # manually.
        value = df[value_col].values[-1] + forecast_horizon
        fut_df = pd.DataFrame({value_col: [value]*forecast_horizon})
        summary = {"mock summary": forecast_horizon}
        trained_model = {"summary": summary}
        return {"fut_df": fut_df, "trained_model": trained_model}

    # Forecasts with the original function ``train_forecast_func``
    forecast1 = train_forecast_func(
        df=df,
        value_col="y",
        time_col="ts",
        forecast_horizon=7)

    fut_df1 = forecast1["fut_df"]
    assert list(fut_df1["y"].values) == [16]*7

    # Forecasts with the composed function: ``forecast_one_by_one_fcn(train_forecast_func)``
    forecast2 = forecast_one_by_one_fcn(train_forecast_func)(
        df=df,
        value_col="y",
        time_col="ts",
        forecast_horizon=7)

    # Checks if forecasted values are as expected
    fut_df2 = forecast2["fut_df"]
    assert list(fut_df2["y"].values) == list(range(10, 17))

    # Checks if the trained models are as expected
    trained_model = forecast2["trained_model"]
    assert trained_model == {"summary": {"mock summary": 7}}
    trained_models_per_horizon = forecast2["trained_models_per_horizon"]

    for k in range(1, 8):
        assert trained_models_per_horizon[k] == {"summary": {"mock summary": k}}

    # Tests `model_params`. Passes some model params directly to `forecast_one_by_one_fcn`
    forecast3 = forecast_one_by_one_fcn(
        train_forecast_func,
        df=df,
        time_col="ts",
        forecast_horizon=7)(value_col="y")
    assert_equal(forecast2, forecast3)
