import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from testfixtures import LogCapture

from greykite.common.constants import LOGGER_NAME
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import r2_null_model_score
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator
from greykite.sklearn.uncertainty.uncertainty_methods import UncertaintyMethodEnum


@pytest.fixture
def params():
    return {
        "score_func": mean_absolute_error,
        "coverage": 0.9,
        "null_model_params": {
            "strategy": "quantile",
            "constant": None,
            "quantile": 0.8
        }
    }


@pytest.fixture
def X_custom():
    return pd.DataFrame({
        # The weekday of "2018-01-02" is 1
        "time_name": pd.date_range("2018-01-02", periods=11, freq="D"),
        "value_name": np.arange(11)
    })


@pytest.fixture
def X():
    return pd.DataFrame({
        # The weekday of "2018-01-02" is 1
        TIME_COL: pd.date_range("2018-01-02", periods=11, freq="D"),
        VALUE_COL: np.arange(11)
    })


@pytest.fixture
def df():
    length = 100
    df = pd.DataFrame({
        TIME_COL: pd.date_range("2020-01-01", freq="D", periods=length),
        VALUE_COL: np.arange(length),
        PREDICTED_COL: np.arange(length) + np.random.randn(length) * length / 10
    })
    return df


class ConstantBaseForecastEstimator(BaseForecastEstimator):
    """Simple estimator that predicts a constant (the weekday of the first date in
    ``X`` passed to ``predict``).
    Used to test BaseForecastEstimator methods, since abstract classes can't be instantiated.
    """

    def __init__(self, score_func=mean_squared_error, coverage=0.95, null_model_params=None):
        super().__init__(score_func=score_func, coverage=coverage, null_model_params=null_model_params)

    def fit(self, X, y=None, time_col=TIME_COL, value_col=VALUE_COL, **fit_params):
        super().fit(X, y=y, time_col=time_col, value_col=value_col)

    def predict(self, X, y=None):
        cached_predictions = super().predict(X=X)
        if cached_predictions is not None:
            return cached_predictions

        # For testing purposes, always predicts the weekday of the first date in X
        cst_pred = float(X[self.time_col_].iloc[0].weekday())
        predictions = pd.DataFrame({
            TIME_COL: X[self.time_col_],
            PREDICTED_COL: np.repeat(cst_pred, X.shape[0])
        })
        self.cached_predictions_ = predictions
        return predictions


def test_init(params):
    """Tests initialization"""
    model = ConstantBaseForecastEstimator(**params)
    assert model.score_func == mean_absolute_error
    assert model.coverage == 0.9
    assert model.null_model_params == params["null_model_params"]
    assert model.null_model is None
    assert model.time_col_ is None
    assert model.value_col_ is None

    # set_params must be able to replicate the init
    model2 = ConstantBaseForecastEstimator()
    model2.set_params(**params)
    assert model2.__dict__ == model.__dict__

    model = ConstantBaseForecastEstimator()
    assert model.score_func == mean_squared_error
    assert model.coverage == 0.95
    assert model.null_model_params is None
    assert model.null_model is None
    assert model.time_col_ is None
    assert model.value_col_ is None


def test_fit_predict(params, X_custom):
    """Tests model fit and predict with custom column names"""
    model = ConstantBaseForecastEstimator(**params)
    assert model.last_predicted_X_ is None
    assert model.cached_predictions_ is None
    model.fit(X_custom, time_col="time_name", value_col="value_name")
    assert model.time_col_ == "time_name"
    assert model.value_col_ == "value_name"

    # Checks model predictions
    with LogCapture(LOGGER_NAME) as log_capture:
        predicted = model.predict(X_custom)
        cst_pred = float(X_custom[model.time_col_].iloc[0].weekday())
        expected = pd.DataFrame({
            TIME_COL: X_custom["time_name"],
            PREDICTED_COL: np.repeat(cst_pred, X_custom.shape[0])
        })
        assert_frame_equal(predicted, expected)
        assert model.last_predicted_X_ is not None
        assert model.cached_predictions_ is not None
        assert_frame_equal(model.last_predicted_X_, X_custom)
        assert_frame_equal(model.cached_predictions_, expected)
        log_capture.check()  # no log messages (not using cached predictions)

    # Uses cached predictions
    with LogCapture(LOGGER_NAME) as log_capture:
        model.predict(X_custom)
        log_capture.check(
            (LOGGER_NAME, "DEBUG", "Returning cached predictions.")
        )

    # Predicts on a different dataset
    with LogCapture(LOGGER_NAME) as log_capture:
        X_new = X_custom.iloc[1:].copy()
        predicted = model.predict(X_new)
        log_capture.check()  # no log messages (not using cached predictions)
        cst_pred = float(X_new[model.time_col_].iloc[0].weekday())
        expected = pd.DataFrame({
            TIME_COL: X_new["time_name"],
            PREDICTED_COL: np.repeat(cst_pred, X_new.shape[0])
        })
        assert_frame_equal(predicted, expected)
        assert_frame_equal(model.last_predicted_X_, X_new)
        assert_frame_equal(model.cached_predictions_, expected)

    # .fit() clears the cached result
    model.fit(X_custom, time_col="time_name", value_col="value_name")
    assert model.last_predicted_X_ is None
    assert model.cached_predictions_ is None

    # Checks null model predictions
    null_predicted = model.null_model.predict(X_custom)
    null_expected = pd.DataFrame({
        TIME_COL: X_custom["time_name"],
        PREDICTED_COL: np.repeat(8.0, X_custom.shape[0])
    })
    assert null_predicted.equals(null_expected)


def test_score_default(X):
    """Tests default score function with no null model"""
    model = ConstantBaseForecastEstimator()
    model.fit(X)
    y = np.repeat(2, X.shape[0])
    score = model.score(X, y=y)
    # ConstantBaseForecastEstimator always returns 1.0
    y_pred = np.repeat(1.0, X.shape[0])
    assert score == mean_squared_error(y, y_pred)


def test_score_custom(X):
    """Tests custom score function with no null model"""
    model = ConstantBaseForecastEstimator(score_func=mean_absolute_error)
    model.fit(X)
    y = np.repeat(2, X.shape[0])
    score = model.score(X, y=y)
    assert score == mean_absolute_error(y, np.repeat(1.0, X.shape[0]))


def test_score_null_default(X):
    """Tests default score function with null model"""
    null_model_params = {
        "strategy": "quantile",
        "constant": None,
        "quantile": 0.8
    }
    model = ConstantBaseForecastEstimator(null_model_params=null_model_params)
    model.fit(X)
    y = np.repeat(2, X.shape[0])
    score = model.score(X, y=y)

    y_pred = np.repeat(1.0, X.shape[0])
    y_pred_null = np.repeat(8.0, X.shape[0])  # quantile 0.8 null model

    assert score == r2_null_model_score(y, y_pred, y_pred_null=y_pred_null, loss_func=mean_squared_error)


def test_score_null_custom(X):
    """Tests custom score function with null model"""
    null_model_params = {
        "strategy": "mean"
    }
    model = ConstantBaseForecastEstimator(score_func=mean_absolute_error, null_model_params=null_model_params)
    model.fit(X)
    y = np.repeat(2, X.shape[0])
    score = model.score(X, y=y)

    y_pred = np.repeat(1.0, X.shape[0])
    y_pred_null = np.repeat(5.0, X.shape[0])  # mean null model

    assert score == r2_null_model_score(y, y_pred, y_pred_null=y_pred_null, loss_func=mean_absolute_error)


def test_populate_uncertainty_params():
    model = ConstantBaseForecastEstimator()
    model.time_col_ = TIME_COL
    model.value_col_ = VALUE_COL

    # No ``value_col`` or ``residual_col``.
    uncertainty_dict = dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params=dict()
    )
    assert model._populate_uncertainty_params(
        uncertainty_dict=uncertainty_dict
    ) == dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params=dict(
            value_col=VALUE_COL,
            residual_col="residual_col"
        )
    )

    # No ``residual_col``.
    uncertainty_dict = dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params=dict(
            value_col="some_col"
        )
    )
    assert model._populate_uncertainty_params(
        uncertainty_dict=uncertainty_dict
    ) == dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params=dict(
            value_col="some_col",
            residual_col="residual_col"
        )
    )

    # No ``value_col``.
    uncertainty_dict = dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params=dict(
            residual_col="some_col"
        )
    )
    assert model._populate_uncertainty_params(
        uncertainty_dict=uncertainty_dict
    ) == dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params=dict(
            value_col=VALUE_COL,
            residual_col="some_col"
        )
    )


def test_uncertainty(df):
    model = ConstantBaseForecastEstimator()
    model.time_col_ = TIME_COL
    model.value_col_ = VALUE_COL

    # Only coverage is given.
    # The default uncertainty method is used.
    model.fit_uncertainty(
        df=df,
        uncertainty_dict={}
    )

    assert model.uncertainty_model is not None
    assert model.uncertainty_model.UNCERTAINTY_METHOD == UncertaintyMethodEnum.simple_conditional_residuals.name

    df_pred = model.predict_uncertainty(
        df=df
    )
    assert PREDICTED_LOWER_COL in df_pred.columns
    assert PREDICTED_UPPER_COL in df_pred.columns
    assert (df_pred[PREDICTED_LOWER_COL] + df_pred[PREDICTED_UPPER_COL]).round(2).equals(
        (df_pred[PREDICTED_COL] * 2).round(2))

    # Both coverage and uncertainty dict are given.
    model.fit_uncertainty(
        df=df,
        uncertainty_dict=dict(
            uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
            params=dict(
                value_col=VALUE_COL,
                is_residual_based=True
            )
        )
    )

    assert model.uncertainty_model is not None
    assert model.uncertainty_model.UNCERTAINTY_METHOD == UncertaintyMethodEnum.simple_conditional_residuals.name

    df_pred = model.predict_uncertainty(
        df=df
    )
    assert PREDICTED_LOWER_COL in df_pred.columns
    assert PREDICTED_UPPER_COL in df_pred.columns
    assert (df_pred[PREDICTED_LOWER_COL] + df_pred[PREDICTED_UPPER_COL]).round(2).equals(
        (df_pred[PREDICTED_COL] * 2).round(2))


def test_uncertainty_errors(df):
    # Uncertainty method not recognized.
    with LogCapture(LOGGER_NAME) as log_capture:
        model = ConstantBaseForecastEstimator()
        model.time_col_ = TIME_COL
        model.value_col_ = VALUE_COL

        # Only coverage is given.
        # The default uncertainty method is used.
        res = model.fit_uncertainty(
            df=df,
            uncertainty_dict=dict(
                uncertainty_method="some_method"
            )
        )
        assert (
                   (LOGGER_NAME,
                    "WARNING",
                    f"Uncertainty method some_method is not found in `UncertaintyMethodEnum`, "
                    f"uncertainty fitting is skipped. Valid methods are "
                    f"{', '.join(UncertaintyMethodEnum.__dict__['_member_names_'])}.")
               ) in log_capture.actual()
        assert res is None
        assert model.uncertainty_model is None

    # Uncertainty model not trained.
    with LogCapture(LOGGER_NAME) as log_capture:
        model = ConstantBaseForecastEstimator()
        model.time_col_ = TIME_COL
        model.value_col_ = VALUE_COL
        model.predict_uncertainty(
            df=df
        )
        assert (
                   (LOGGER_NAME,
                    "WARNING",
                    f"The uncertainty model is not trained.")
               ) in log_capture.actual()

    # Prediction failed.
    with LogCapture(LOGGER_NAME) as log_capture:
        model = ConstantBaseForecastEstimator()
        model.time_col_ = TIME_COL
        model.value_col_ = VALUE_COL
        model.fit_uncertainty(
            df=df,
            uncertainty_dict={
                "uncertainty_method": "simple_conditional_residuals",
                "params": {
                    "is_residual_based": True,
                    "predicted_col": PREDICTED_COL
                }
            }
        )
        model.predict_uncertainty(
            df=df[[TIME_COL]]
        )
        assert (
                   (LOGGER_NAME,
                    "WARNING",
                    f"The following errors occurred during predicting the uncertainty model, "
                    f"the uncertainty model is skipped."
                    f" The offset column {PREDICTED_COL} is not found in `fut_df`.")
               ) in log_capture.actual()
