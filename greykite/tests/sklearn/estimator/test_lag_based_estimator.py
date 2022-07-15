from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta
from testfixtures import LogCapture

from greykite.common.constants import ERR_STD_COL
from greykite.common.constants import LOGGER_NAME
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.sklearn.estimator.lag_based_estimator import LagBasedEstimator


@pytest.fixture
def df_daily():
    """Defines a dataframe with values from a sequence,
    so it's easy to track lagged values."""
    df = pd.DataFrame({
        "ts": pd.date_range(
            start="2021-01-01",
            periods=314,
            freq="D"
        ),
        "y": list(range(314))
    })
    df_train = df.iloc[:300].reset_index(drop=True)
    df_test = df.iloc[300:].reset_index(drop=True)
    return {
        "df_train": df_train,
        "df_test": df_test
    }


@pytest.fixture
def df_hourly():
    """Defines a dataframe with values from a sequence,
    so it's easy to track lagged values."""
    df = pd.DataFrame({
        "ts": pd.date_range(
            start="2021-01-01",
            periods=314 * 24,
            freq="H"
        ),
        "y": list(range(314 * 24))
    })
    df_train = df.iloc[:300 * 24].reset_index(drop=True)
    df_test = df.iloc[300 * 24:].reset_index(drop=True)
    return {
        "df_train": df_train,
        "df_test": df_test
    }


@pytest.fixture
def df_monthly():
    """Defines a dataframe with values from a sequence,
    so it's easy to track lagged values."""
    df = pd.DataFrame({
        "ts": pd.date_range(
            start="2021-01-01",
            periods=314,
            freq="MS"
        ),
        "y": list(range(314))
    })
    df_train = df.iloc[:300].reset_index(drop=True)
    df_test = df.iloc[300:].reset_index(drop=True)
    return {
        "df_train": df_train,
        "df_test": df_test
    }


@pytest.fixture
def df_yearly():
    """Defines a dataframe with values from a sequence,
    so it's easy to track lagged values."""
    df = pd.DataFrame({
        "ts": pd.date_range(
            start="2021-01-01",
            periods=34,
            freq="Y"
        ),
        "y": list(range(34))
    })
    df_train = df.iloc[:20].reset_index(drop=True)
    df_test = df.iloc[20:].reset_index(drop=True)
    return {
        "df_train": df_train,
        "df_test": df_test
    }


def test_instantiation():
    """Tests instantiation."""
    model = LagBasedEstimator(
        coverage=0.95,
        freq="D",
        lag_unit="day",
        lags=[1, 2, 3],
        agg_func="weighted_average",
        agg_func_params={
            "weights": [3, 2, 1]
        },
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "conditional_cols": ["dow_hr"],
            "value_col": "y"
        },
        past_df=pd.DataFrame({}),
        series_na_fill_func=lambda s: s.bfill()
    )
    assert model.coverage == 0.95
    assert model.freq == "D"
    assert model.lag_unit == "day"
    assert model.lags == [1, 2, 3]
    assert model.agg_func == "weighted_average"
    assert model.agg_func_params == {
        "weights": [3, 2, 1]
    }
    assert model.uncertainty_dict == {
        "uncertainty_method": "simple_conditional_residuals",
        "conditional_cols": ["dow_hr"],
        "value_col": "y"
    }
    assert model.past_df.equals(pd.DataFrame({}))
    assert model.df is None
    assert model.uncertainty_model is None
    assert model.max_lag_order is None
    assert model.min_lag_order is None
    assert model.train_start is None
    assert model.train_end is None


def test_daily_data(df_daily):
    """Tests functionality on daily data."""
    df_train = df_daily["df_train"]
    df_test = df_daily["df_test"]

    model = LagBasedEstimator(
        series_na_fill_func=lambda s: s.bfill().ffill()
    )
    model.fit(df_train)
    # Checks ``model.df``.
    # It should include the past 1 week data interpolated.
    # The interpolated values are zero with the default function.
    df_expanded = pd.concat([
        pd.DataFrame({
            "ts": pd.date_range(
                start=df_train["ts"].iloc[0] - timedelta(weeks=1),
                periods=7,
                freq="D"
            ),
            "y": 0.
        }),
        df_train
    ], axis=0).reset_index(drop=True)
    assert model.df.equals(df_expanded)
    assert model.uncertainty_model is None
    assert model.max_lag_order == 1
    assert model.min_lag_order == 1
    assert model.train_start == df_train["ts"].iloc[0]
    assert model.train_end == df_train["ts"].iloc[-1]

    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The predicted column is the last week.
    assert df_fit[PREDICTED_COL].iloc[-5:].reset_index(drop=True).equals(
        df_fit["y"].iloc[-12:-7].reset_index(drop=True)
    )
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL]

    # Checks prediction in the future.
    df_pred = model.predict(df_test)
    # The value column should be NANs.
    assert all(df_pred["y"].isna())
    # The predicted column is the last week.
    assert df_pred[PREDICTED_COL].iloc[:7].reset_index(drop=True).equals(
        df_fit["y"].iloc[-7:].reset_index(drop=True)
    )
    # The future predictions use the predicted values.
    assert df_pred[PREDICTED_COL].iloc[7:14].reset_index(drop=True).equals(
        df_pred[PREDICTED_COL].iloc[:7].reset_index(drop=True)
    )


def test_hourly_data(df_hourly):
    """Tests functionality on hourly data."""
    df_train = df_hourly["df_train"]
    df_test = df_hourly["df_test"]

    model = LagBasedEstimator(
        series_na_fill_func=lambda s: s.bfill().ffill()
    )
    model.fit(df_train)
    # Checks ``model.df``.
    # It should include the past 1 week data interpolated.
    # The interpolated values are zero with the default function.
    df_expanded = pd.concat([
        pd.DataFrame({
            "ts": pd.date_range(
                start=df_train["ts"].iloc[0] - timedelta(weeks=1),
                periods=7 * 24,
                freq="H"
            ),
            "y": 0.
        }),
        df_train
    ], axis=0).reset_index(drop=True)
    assert model.df.equals(df_expanded)
    assert model.uncertainty_model is None
    assert model.max_lag_order == 1
    assert model.min_lag_order == 1
    assert model.train_start == df_train["ts"].iloc[0]
    assert model.train_end == df_train["ts"].iloc[-1]

    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The predicted column is the last week.
    assert df_fit[PREDICTED_COL].iloc[-5:].reset_index(drop=True).equals(
        df_fit["y"].iloc[-(24 * 7 + 5):-(24 * 7)].reset_index(drop=True)
    )
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL]

    # Checks prediction in the future.
    df_pred = model.predict(df_test)
    # The value column should be NANs.
    assert all(df_pred["y"].isna())
    # The predicted column is the last week.
    assert df_pred[PREDICTED_COL].iloc[:(24 * 7)].reset_index(drop=True).equals(
        df_fit["y"].iloc[-(24 * 7):].reset_index(drop=True)
    )
    # The future predictions use the predicted values.
    assert df_pred[PREDICTED_COL].iloc[(24 * 7):(24 * 14)].reset_index(drop=True).equals(
        df_pred[PREDICTED_COL].iloc[:(24 * 7)].reset_index(drop=True)
    )


def test_multiple_lags_and_weights(df_daily):
    """Test multiple lags and weights."""
    df_train = df_daily["df_train"]
    df_test = df_daily["df_test"]

    model = LagBasedEstimator(
        lags=[1, 2],
        agg_func="weighted_average",
        agg_func_params={
            "weights": [0.4, 0.6]
        },
        series_na_fill_func=lambda s: s.bfill().ffill()
    )
    model.fit(df_train)
    # Checks ``model.df``.
    # It should include the past 2 weeks data interpolated.
    # The interpolated values are zero with the default function.
    df_expanded = pd.concat([
        pd.DataFrame({
            "ts": pd.date_range(
                start=df_train["ts"].iloc[0] - timedelta(weeks=2),
                periods=14,
                freq="D"
            ),
            "y": 0.
        }),
        df_train
    ], axis=0).reset_index(drop=True)
    assert model.df.equals(df_expanded)
    assert model.uncertainty_model is None
    assert model.max_lag_order == 2
    assert model.min_lag_order == 1
    assert model.train_start == df_train["ts"].iloc[0]
    assert model.train_end == df_train["ts"].iloc[-1]

    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The predicted column is the weighted average of last two weeks.
    assert df_fit[PREDICTED_COL].iloc[-5:].reset_index(drop=True).equals(
        df_fit["y"].iloc[-12:-7].reset_index(drop=True) * 0.4 + df_fit["y"].iloc[-19:-14].reset_index(drop=True) * 0.6
    )
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL]

    # Checks prediction in the future.
    df_pred = model.predict(df_test)
    # The value column should be NANs.
    assert all(df_pred["y"].isna())
    # The predicted column is the weighted average of last two weeks.
    assert df_pred[PREDICTED_COL].iloc[:7].reset_index(drop=True).equals(
        df_fit["y"].iloc[-7:].reset_index(drop=True) * 0.4 + df_fit["y"].iloc[-14:-7].reset_index(drop=True) * 0.6
    )
    # The future predictions use the predicted values.
    assert df_pred[PREDICTED_COL].iloc[7:14].reset_index(drop=True).equals(
        df_pred[PREDICTED_COL].iloc[:7].reset_index(drop=True) * 0.4
        + df_fit["y"].iloc[-7:].reset_index(drop=True) * 0.6
    )


def test_past_df(df_daily):
    """Test past df."""
    df_train = df_daily["df_train"]

    past_df = pd.DataFrame({
        "ts": pd.date_range(
            start=df_train["ts"].iloc[0] - timedelta(weeks=1),
            periods=7,
            freq="D"
        ),
        "y": -1.
    })

    model = LagBasedEstimator(
        past_df=past_df
    )
    model.fit(df_train)
    # Checks ``model.df``.
    # It should include the past 1 week data from ``past_df``.
    df_expanded = pd.concat([
        past_df,
        df_train
    ], axis=0).reset_index(drop=True)
    assert model.df.equals(df_expanded)

    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The first week's fitted values is the ``past_df``.
    assert df_fit[PREDICTED_COL].iloc[:7].reset_index(drop=True).equals(past_df["y"])
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL]


def test_no_fill_na(df_daily):
    """Test no ``series_na_fill_func``."""
    df_train = df_daily["df_train"]

    model = LagBasedEstimator(
        series_na_fill_func=None
    )
    model.fit(df_train)
    # Checks ``model.df``.
    # It should include the past 1 week NA data due to no NA fill function.
    df_expanded = pd.concat([
        pd.DataFrame({
            "ts": pd.date_range(
                start=df_train["ts"].iloc[0] - timedelta(weeks=1),
                periods=7,
                freq="D"
            ),
            "y": np.nan
        }),
        df_train
    ], axis=0).reset_index(drop=True)
    assert model.df.equals(df_expanded)

    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The first week's fitted values are all NANs due to no NA fill function.
    print(df_fit[PREDICTED_COL].iloc[:7].reset_index(drop=True))
    print(pd.Series({"y": [np.nan] * 7}))
    assert df_fit[PREDICTED_COL].iloc[:7].reset_index(drop=True).equals(
        pd.DataFrame({"y": [np.nan] * 7})["y"])
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL]


def test_uncertainty(df_daily):
    """Tests uncertainty."""
    df_train = df_daily["df_train"]
    df_test = df_daily["df_test"]

    model = LagBasedEstimator(
        coverage=0.95,
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "value_col": "y",
            "conditional_cols": ["dow"]
        }
    )
    model.fit(df_train)
    assert model.uncertainty_model is not None

    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The predicted column is the last week.
    assert df_fit[PREDICTED_COL].iloc[-5:].reset_index(drop=True).equals(
        df_fit["y"].iloc[-12:-7].reset_index(drop=True)
    )
    # Uncertainty columns are in the output.
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL, ERR_STD_COL]

    # Checks prediction in the future.
    df_pred = model.predict(df_test)
    # The value column should be NANs.
    assert all(df_pred["y"].isna())
    # The predicted column is the last week.
    assert df_pred[PREDICTED_COL].iloc[:7].reset_index(drop=True).equals(
        df_fit["y"].iloc[-7:].reset_index(drop=True)
    )
    # The future predictions use the predicted values.
    assert df_pred[PREDICTED_COL].iloc[7:14].reset_index(drop=True).equals(
        df_pred[PREDICTED_COL].iloc[:7].reset_index(drop=True)
    )
    # Uncertainty columns are in the output.
    assert list(df_pred.columns) == ["ts", "y", PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL, ERR_STD_COL]


def test_agg_func(df_daily):
    """Tests different aggregation functions."""
    df_train = df_daily["df_train"]
    df_test = df_daily["df_test"]

    # wo3w median
    model = LagBasedEstimator(
        lag_unit="week",
        lags=[1, 2, 3],
        agg_func="median"
    )
    model.fit(df_train)
    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The predicted column is the median of the past three weeks.
    assert df_fit[PREDICTED_COL].iloc[-5:].reset_index(drop=True).equals(
        df_fit["y"].iloc[-19:-14].reset_index(drop=True)
    )
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL]

    # Checks prediction in the future.
    df_pred = model.predict(df_test)
    # The value column should be NANs.
    assert all(df_pred["y"].isna())
    # The predicted column is the median of the past three weeks.
    assert df_pred[PREDICTED_COL].iloc[:7].reset_index(drop=True).equals(
        df_fit["y"].iloc[-14:-7].reset_index(drop=True)
    )
    # The future predictions use the predicted values.
    assert df_pred[PREDICTED_COL].iloc[7:14].reset_index(drop=True).equals(
        df_pred[PREDICTED_COL].iloc[-7:].reset_index(drop=True)
    )

    # wo3w minimum
    model = LagBasedEstimator(
        lag_unit="week",
        lags=[1, 2, 3],
        agg_func="minimum"
    )
    model.fit(df_train)
    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The predicted column is the minimum of the past three weeks.
    assert df_fit[PREDICTED_COL].iloc[-5:].reset_index(drop=True).equals(
        df_fit["y"].iloc[-26:-21].reset_index(drop=True)
    )
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL]

    # Checks prediction in the future.
    df_pred = model.predict(df_test)
    # The value column should be NANs.
    assert all(df_pred["y"].isna())
    # The predicted column is the minimum of the past three weeks.
    assert df_pred[PREDICTED_COL].iloc[:7].reset_index(drop=True).equals(
        df_fit["y"].iloc[-21:-14].reset_index(drop=True)
    )
    # The future predictions use the predicted values.
    assert df_pred[PREDICTED_COL].iloc[7:14].reset_index(drop=True).equals(
        df_pred[PREDICTED_COL].iloc[-7:].reset_index(drop=True)
    )


def test_monthly_data(df_monthly):
    """Tests functionality on monthly data."""
    df_train = df_monthly["df_train"]
    df_test = df_monthly["df_test"]

    model = LagBasedEstimator(
        lag_unit="month",
        lags=[2, 3],
        series_na_fill_func=lambda s: s.bfill().ffill()
    )
    model.fit(df_train)
    # Checks ``model.df``.
    # It should include the past 3 months data interpolated.
    # The interpolated values are zero with the default function.
    df_expanded = pd.concat([
        pd.DataFrame({
            "ts": pd.date_range(
                start=df_train["ts"].iloc[0] - relativedelta(months=3),
                periods=3,
                freq="MS"
            ),
            "y": 0.
        }),
        df_train
    ], axis=0).reset_index(drop=True)
    assert model.df.equals(df_expanded)
    assert model.uncertainty_model is None
    assert model.max_lag_order == 3
    assert model.min_lag_order == 2
    assert model.train_start == df_train["ts"].iloc[0]
    assert model.train_end == df_train["ts"].iloc[-1]

    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The predicted column is the average of the last 2 and 3 months.
    assert df_fit[PREDICTED_COL].iloc[-5:].reset_index(drop=True).equals(
        (df_fit["y"].iloc[-8:-3].reset_index(drop=True)
         + df_fit["y"].iloc[-7:-2].reset_index(drop=True)) / 2
    )
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL]

    # Checks prediction in the future.
    df_pred = model.predict(df_test)
    # The value column should be NANs.
    assert all(df_pred["y"].isna())
    # The predicted column is the average of the last 2 and 3 months.
    assert df_pred[PREDICTED_COL].iloc[:2].reset_index(drop=True).equals(
        (df_fit["y"].iloc[-2:].reset_index(drop=True)
         + df_fit["y"].iloc[-3:-1].reset_index(drop=True)) / 2
    )
    # The future predictions use the predicted values.
    assert df_pred[PREDICTED_COL].iloc[3:5].reset_index(drop=True).equals(
        (df_pred[PREDICTED_COL].iloc[:2].reset_index(drop=True)
         + df_pred[PREDICTED_COL].iloc[1:3].reset_index(drop=True)) / 2
    )


def test_yearly_data(df_yearly):
    """Tests functionality on yearly data."""
    df_train = df_yearly["df_train"]
    df_test = df_yearly["df_test"]

    model = LagBasedEstimator(
        lag_unit="year",
        lags=[5, 10],
        series_na_fill_func=lambda s: s.bfill().ffill()
    )
    model.fit(df_train)
    # Checks ``model.df``.
    # It should include the past 10 years data interpolated.
    # The interpolated values are zero with the default function.
    df_expanded = pd.concat([
        pd.DataFrame({
            "ts": pd.date_range(
                start=df_train["ts"].iloc[0] - relativedelta(years=10),
                periods=10,
                freq="Y"
            ),
            "y": 0.
        }),
        df_train
    ], axis=0).reset_index(drop=True)
    assert model.df.equals(df_expanded)
    assert model.uncertainty_model is None
    assert model.max_lag_order == 10
    assert model.min_lag_order == 5
    assert model.train_start == df_train["ts"].iloc[0]
    assert model.train_end == df_train["ts"].iloc[-1]

    # Checks prediction within training.
    df_fit = model.predict(df_train)
    # The value column should be the same as the input.
    assert df_fit["y"].equals(df_train["y"].astype(float))
    # The predicted column is the average of the last 5 and 10 years.
    assert df_fit[PREDICTED_COL].iloc[-5:].reset_index(drop=True).equals(
        (df_fit["y"].iloc[-10:-5].reset_index(drop=True)
         + df_fit["y"].iloc[-15:-10].reset_index(drop=True)) / 2
    )
    assert list(df_fit.columns) == ["ts", "y", PREDICTED_COL]

    # Checks prediction in the future.
    df_pred = model.predict(df_test)
    # The value column should be NANs.
    assert all(df_pred["y"].isna())
    # The predicted column is the average of the last 5 and 10 years.
    assert df_pred[PREDICTED_COL].iloc[:2].reset_index(drop=True).equals(
        (df_fit["y"].iloc[-5:-3].reset_index(drop=True)
         + df_fit["y"].iloc[-10:-8].reset_index(drop=True)) / 2
    )
    # The future predictions use the predicted values.
    assert df_pred[PREDICTED_COL].iloc[10:12].reset_index(drop=True).equals(
        (df_pred[PREDICTED_COL].iloc[:2].reset_index(drop=True)
         + df_pred[PREDICTED_COL].iloc[5:7].reset_index(drop=True)) / 2
    )


def test_log_message(df_daily):
    """Tests log messages."""
    df_train = df_daily["df_train"]
    df_test = df_daily["df_test"]
    # Lag not provided, using default 1.
    # Weights not provided, using default equal weights.
    with LogCapture(LOGGER_NAME) as log_capture:
        model = LagBasedEstimator()
        model.fit(df_train)
        assert (
            LOGGER_NAME,
            "DEBUG",
            "Lags not provided, setting lags = [1]."
        ) in log_capture.actual()

    # Inferred frequency is different.
    with LogCapture(LOGGER_NAME) as log_capture:
        model = LagBasedEstimator(
            freq="H"
        )
        model.fit(df_train)
        assert (
            LOGGER_NAME,
            "INFO",
            f"The inferred frequency 'D' is different from the provided 'H'. "
            f"Using the provided frequency."
        ) in log_capture.actual()
    # Prediction has irregular time increments.
    with LogCapture(LOGGER_NAME) as log_capture:
        model = LagBasedEstimator()
        model.fit(df_train)
        df_test_with_irregular = pd.concat([
            df_test,
            pd.DataFrame({
                "ts": [df_test["ts"].iloc[0] + timedelta(hours=6)],
                "y": [np.nan]
            })
        ])
        model.predict(df_test_with_irregular)
        assert (
            LOGGER_NAME,
            "WARNING",
            f"Some timestamps in the provided time periods for prediction do not match the "
            f"training frequency. Returning the matched timestamps."
        ) in log_capture.actual()
    # Warning on frequency "M".
    with LogCapture(LOGGER_NAME) as log_capture:
        df_month = pd.DataFrame({
            "ts": pd.date_range("2020-01-31", freq="M", periods=10),
            "y": list(range(10))
        })
        model = LagBasedEstimator(
            lag_unit="month"
        )
        model.fit(df_month)
        assert (
            LOGGER_NAME,
            "WARNING",
            "The data frequency is 'M' which may lead to unexpected behaviors. "
            "Please convert to 'MS' if applicable."
        ) in log_capture.actual()


def test_errors(df_daily):
    """Tests errors."""
    df_train = df_daily["df_train"]
    # Illegal lag unit.
    with pytest.raises(
            ValueError,
            match=f"The lag unit 'happy' is not recognized."):
        model = LagBasedEstimator(
            lag_unit="happy"
        )
        model.fit(df_train)
    # Illegal lags.
    lags = [1, "a"]
    with pytest.raises(
            ValueError,
            match=f"Not all lags in '\\[1, 'a'\\]' can be converted to integers."):
        model = LagBasedEstimator(
            lags=lags
        )
        model.fit(df_train)
    # Negative lags.
    lags = [1, -1]
    with pytest.raises(
            ValueError,
            match="All lags must be positive integers."):
        model = LagBasedEstimator(
            lags=lags
        )
        model.fit(df_train)
    # Lags is not a list.
    lags = 1
    with pytest.raises(
            ValueError,
            match="The lags must be a list of integers, found '1'."):
        model = LagBasedEstimator(
            lags=lags
        )
        model.fit(df_train)
    # Aggregation function not valid.
    with pytest.raises(
            ValueError,
            match=f"The aggregation function 'happy' is not recognized as a string. "
                  f"Please either pass a known string or a function."):
        model = LagBasedEstimator(
            agg_func="happy"
        )
        model.fit(df_train)
    # Empty df.
    with pytest.raises(
            ValueError,
            match="The input df is empty!"):
        model = LagBasedEstimator()
        model.fit(pd.DataFrame({}))
    # Frequency not provided and can not be inferred.
    df = df_train.iloc[[0, 1, 3]].reset_index(drop=True)
    with pytest.raises(
            ValueError,
            match="Frequency can not be inferred. Please provide frequency."):
        model = LagBasedEstimator()
        model.fit(df)
    # Lag unit is less than data frequency.
    with pytest.raises(
            ValueError,
            match=f"The lag unit 'minute' must be at least equal to the data frequency 'D'."):
        model = LagBasedEstimator(
            lag_unit="minute",
            freq="D"
        )
        model.fit(df_train)
    # Prediction before train start.
    with pytest.raises(
            ValueError,
            match="The lag based estimator does not support hindcasting."):
        model = LagBasedEstimator()
        model.fit(df_train)
        df = pd.concat([
            pd.DataFrame({
                "ts": [df_train["ts"].iloc[0] - timedelta(days=1)],
                "y": [0]
            }),
            df_train
        ], axis=0)
        model.predict(df)


def test_summary(df_daily):
    """Tests functionality on daily data."""
    df_train = df_daily["df_train"]

    model = LagBasedEstimator(
        lag_unit="day",
        lags=[2, 3],
        agg_func="median"
    )
    model.fit(df_train)
    summary = model.summary()
    assert (
        f"This is a lag based forecast model that uses lags '[2, 3]', "
        f"with unit 'day' and aggregation function"
    ) in summary
