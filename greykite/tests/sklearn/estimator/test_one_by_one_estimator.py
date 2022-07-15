import datetime
import sys

import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

import greykite.common.constants as cst
from greykite.common.testing_utils import generate_df_for_tests
from greykite.sklearn.estimator.one_by_one_estimator import OneByOneEstimator
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator


try:
    import prophet  # noqa
except ModuleNotFoundError:
    pass


@pytest.fixture
def daily_data():
    return generate_df_for_tests(
        freq="D",
        periods=730,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)


def test_basic_functionality(daily_data):
    """Tests __init__ and attributes set during fit"""
    coverage = 0.90
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=True,
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        estimator_params={
            "autoreg_dict": "auto",
            "yearly_seasonality": 2,
            "feature_sets_enabled": False
        })

    # Initialization.
    assert model.score_func == mean_squared_error
    assert model.coverage == coverage
    assert model.null_model_params is None
    assert model.estimator == "SimpleSilverkiteEstimator"
    assert model.forecast_horizon == 3
    assert model.estimator_map is True
    assert model.estimator_params == {
        "autoreg_dict": "auto",
        "yearly_seasonality": 2,
        "feature_sets_enabled": False
    }

    assert model.estimators is None
    assert model.estimator_map_list is None
    assert model.estimator_param_names is None
    assert model.estimator_class is None
    assert model.model_results is None
    assert model.pred_indices is None
    assert model.train_end_date is None

    # Attributes are set during fit.
    train_df = daily_data.get("train_df").copy()
    model.fit(train_df)
    assert model.estimator_map_list == [1, 1, 1]
    assert model.estimator_class == SimpleSilverkiteEstimator
    assert len(model.estimators) == 3
    assert len(model.model_results) == 3
    assert model.pred_indices == [0, 1, 2, 3]
    assert model.train_end_date == pd.to_datetime(train_df["ts"]).max()

    assert model.estimators[0].autoreg_dict == "auto"
    assert model.estimators[1].yearly_seasonality == 2
    assert model.estimators[2].feature_sets_enabled is False

    # Coverage provided, prediction intervals should present.
    test_df = daily_data.get("test_df").copy()
    predict = model.predict(test_df.iloc[:3])
    assert cst.PREDICTED_LOWER_COL in predict.columns
    assert cst.PREDICTED_UPPER_COL in predict.columns
    assert model.estimators[0].forecast.shape[0] == 1
    assert model.estimators[1].forecast.shape[0] == 1
    assert model.estimators[2].forecast.shape[0] == 1
    assert model.forecast.shape[0] == 3

    # Prediction on both training and testing.
    model.predict(pd.concat([train_df, test_df.iloc[:3]], axis=0).reset_index(drop=True))
    assert model.forecast.shape[0] == train_df.shape[0] + 3


def test_no_coverage(daily_data):
    """Tests no coverage is provided."""
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=[1, 2],
        coverage=None,
        estimator_params={
            "autoreg_dict": "auto",
            "yearly_seasonality": 2,
            "feature_sets_enabled": False
        })
    train_df = daily_data["train_df"]
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    test_df = daily_data["test_df"]
    predict = model.predict(test_df.iloc[:3])
    assert cst.PREDICTED_LOWER_COL not in predict.columns
    assert cst.PREDICTED_UPPER_COL not in predict.columns


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet(daily_data):
    """Tests prophet estimator."""
    model = OneByOneEstimator(
        estimator="ProphetEstimator",
        forecast_horizon=3,
        estimator_map=[1, 2],
        coverage=None)
    train_df = daily_data["train_df"]
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)


def test_forecast_one_by_one_not_activated(daily_data):
    """Tests forecast one-by-one is not activated when no parameter
    depends on forecast horizon.
    """
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=[1, 2],
        estimator_params={
            "autoreg_dict": None,
            "yearly_seasonality": 2,
            "feature_sets_enabled": False
        })
    train_df = daily_data["train_df"]
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert len(model.estimators) == 1
    assert model.estimator_map_list == [3]
    assert model.pred_indices is None


def test_set_params():
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=1,
        estimator_params={
            "autoreg_dict": "auto",
            "yearly_seasonality": 2,
            "feature_sets_enabled": False
        })
    assert model.estimator_map == 1
    assert model.estimator_params["yearly_seasonality"] == 2
    assert model.estimator_params.get("daily_seasonality") is None

    model.set_params(**{
        "estimator_map": 2,
        "yearly_seasonality": 4,
        "daily_seasonality": 2
    })
    assert model.estimator_map == 2
    assert model.estimator_params["yearly_seasonality"] == 4
    assert model.estimator_params["daily_seasonality"] == 2

    assert set(model.estimator_param_names) == set(SimpleSilverkiteEstimator().get_params().keys())

    with pytest.raises(
            ValueError,
            match=r"Invalid parameter some_param for estimator OneByOneEstimator. "
                  r"Check the list of available parameters with "
                  r"`estimator.get\_params\(\).keys\(\)`."):
        model.set_params(**{
            "some_param": 5
        })


def test_errors(daily_data):
    """Tests errors."""
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=[1, 1],
        estimator_params={
            "autoreg_dict": "auto",
            "yearly_seasonality": 2,
            "feature_sets_enabled": False
        })
    train_df = daily_data["train_df"]
    with pytest.raises(
            ValueError,
            match="Sum of forecast one by one estimator map must equal to forecast horizon."):
        model.fit(
            train_df,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL)

    model = OneByOneEstimator(
        estimator="SomeEstimator",
        forecast_horizon=3,
        estimator_map=[1, 1],
        estimator_params={
            "autoreg_dict": "auto",
            "yearly_seasonality": 2,
            "feature_sets_enabled": False
        })
    train_df = daily_data["train_df"]
    with pytest.raises(
            ValueError,
            match="Estimator SomeEstimator does not support forecast one-by-one."):
        model.fit(
            train_df,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL)


def test_summary(daily_data):
    """Checks summary function."""
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=2,
        estimator_params={
            "autoreg_dict": "auto",
            "yearly_seasonality": 2,
            "feature_sets_enabled": False
        })
    train_df = daily_data["train_df"].iloc[:365]
    model.fit(train_df)
    summary = model.summary()
    assert len(summary) == 2


def test_plot_components(daily_data):
    """Tests plot_components."""
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=2,
        estimator_params={
            "autoreg_dict": "auto",
            "yearly_seasonality": 2,
            "feature_sets_enabled": False
        })
    train_df = daily_data["train_df"].iloc[:365]
    model.fit(train_df)
    figs = model.plot_components()
    assert len(figs) == 2
