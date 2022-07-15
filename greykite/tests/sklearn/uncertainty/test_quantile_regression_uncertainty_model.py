"""Tests the quantile regression based uncertainty model."""

import pandas as pd
import pytest

from greykite.common import constants as cst
from greykite.common.data_loader import DataLoader
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator
from greykite.sklearn.uncertainty.exceptions import UncertaintyError
from greykite.sklearn.uncertainty.quantile_regression_uncertainty_model import QuantileRegressionUncertaintyModel


@pytest.fixture
def forecast_result():
    """The forecast model results."""
    df = DataLoader().load_peyton_manning().iloc[-365:].reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"])
    model = SimpleSilverkiteEstimator()
    model.fit(df)
    x_mat = model.model_dict["x_mat"]
    fit_df = model.predict(df)
    fit_df["y"] = df["y"]
    fut_df = pd.DataFrame({
        "ts": pd.date_range(df["ts"].max(), freq="D", periods=15)
    }).iloc[1:].reset_index(drop=True)
    predict_df = model.predict(fut_df)
    fut_x_mat = model.forecast_x_mat
    return {
        "df": df,
        "fit_df": fit_df,
        "x_mat": x_mat,
        "predict_df": predict_df,
        "fut_x_mat": fut_x_mat
    }


def test_init():
    """Tests instantiation."""
    model = QuantileRegressionUncertaintyModel(
        uncertainty_dict={}
    )
    assert model.coverage is None
    assert model.time_col is None
    assert model.uncertainty_dict == {}
    assert model.is_residual_based is None
    assert model.x_mat is None
    assert model.build_x_mat is None
    assert model.value_col is None
    assert model.residual_col is None
    assert model.quantiles is None
    assert model.models is None
    assert model.distribution_col is None
    assert model.offset_col is None


def test_check_input(forecast_result):
    """Tests processing inputs."""
    # Method name wrong.
    with pytest.raises(
            UncertaintyError,
            match=f"The uncertainty method uncertainty_method is not as expected quantile_regression."):
        model = QuantileRegressionUncertaintyModel(
            uncertainty_dict={
                "uncertainty_method": "uncertainty_method",
            }
        )
        model.train_df = forecast_result["df"]
        model.x_mat = forecast_result["x_mat"]
        model._check_input()

    # Value column not found.
    with pytest.raises(
            UncertaintyError,
            match=f"The parameter value_col is required but not found in"):
        model = QuantileRegressionUncertaintyModel(
            uncertainty_dict={
                "uncertainty_method": "quantile_regression",
            }
        )
        model.train_df = forecast_result["df"][["ts"]]
        model.x_mat = forecast_result["x_mat"]
        model._check_input()

    # Value column not found in df.
    with pytest.raises(
            UncertaintyError,
            match=f"`value_col` col not found in `train_df`."):
        model = QuantileRegressionUncertaintyModel(
            uncertainty_dict={
                "uncertainty_method": "quantile_regression",
                "params": {
                    "value_col": "col"
                }
            }
        )
        model.train_df = forecast_result["df"]
        model.x_mat = forecast_result["x_mat"]
        model._check_input()

    # No time column or ``x_mat``.
    with pytest.raises(
            UncertaintyError,
            match=f"Time column must be provided when `x_mat` is not given."):
        model = QuantileRegressionUncertaintyModel(
            uncertainty_dict={
                "uncertainty_method": "quantile_regression",
                "params": {
                }
            }
        )
        model.train_df = forecast_result["df"][["y"]]
        model._check_input()


def test_is_residual_based_x_mat(forecast_result):
    """Tests predicting residual based interval with ``x_mat``."""
    model = QuantileRegressionUncertaintyModel(
        uncertainty_dict={
            "uncertainty_method": "quantile_regression",
            "params": {
                "value_col": "y",
                "is_residual_based": True
            }
        },
        coverage=0.9
    )
    model.fit(
        train_df=forecast_result["fit_df"],
        x_mat=forecast_result["x_mat"]
    )
    pred = model.predict(
        fut_df=forecast_result["fit_df"],
        x_mat=forecast_result["x_mat"]
    )
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred
    pred = model.predict(
        fut_df=forecast_result["predict_df"],
        x_mat=forecast_result["fut_x_mat"]
    )
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred


def test_is_residual_based_no_x_mat(forecast_result):
    """Tests predicting residual based interval without ``x_mat``."""
    model = QuantileRegressionUncertaintyModel(
        uncertainty_dict={
            "uncertainty_method": "quantile_regression",
            "params": {
                "value_col": "y",
                "is_residual_based": True
            }
        },
        coverage=0.9
    )
    model.fit(
        train_df=forecast_result["fit_df"],
        x_mat=None
    )
    pred = model.predict(
        fut_df=forecast_result["fit_df"]
    )
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred
    pred = model.predict(
        fut_df=forecast_result["predict_df"]
    )
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred


def test_non_is_residual_based_x_mat(forecast_result):
    """Tests predicting non-residual based interval with ``x_mat``."""
    model = QuantileRegressionUncertaintyModel(
        uncertainty_dict={
            "uncertainty_method": "quantile_regression",
            "params": {
                "value_col": "y",
                "is_residual_based": False
            }
        },
        coverage=0.9
    )
    model.fit(
        train_df=forecast_result["fit_df"],
        x_mat=forecast_result["x_mat"]
    )
    pred = model.predict(
        fut_df=forecast_result["fit_df"],
        x_mat=forecast_result["x_mat"]
    )
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred
    pred = model.predict(
        fut_df=forecast_result["predict_df"],
        x_mat=forecast_result["fut_x_mat"]
    )
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred


def test_non_is_residual_based_no_x_mat(forecast_result):
    """Tests predicting non-residual based interval without ``x_mat``."""
    model = QuantileRegressionUncertaintyModel(
        uncertainty_dict={
            "uncertainty_method": "quantile_regression",
            "params": {
                "value_col": "y",
                "is_residual_based": False
            }
        },
        coverage=0.9
    )
    model.fit(
        train_df=forecast_result["fit_df"],
        x_mat=None
    )
    pred = model.predict(
        fut_df=forecast_result["fit_df"]
    )
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred
    pred = model.predict(
        fut_df=forecast_result["predict_df"]
    )
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred
