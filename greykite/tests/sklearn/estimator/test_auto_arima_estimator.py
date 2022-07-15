import numpy as np
import pandas as pd
import pytest
from pmdarima.arima import AutoARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from testfixtures import LogCapture

from greykite.common.constants import LOGGER_NAME
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import calc_pred_err
from greykite.common.logging import LoggingLevelEnum
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.sklearn.estimator.auto_arima_estimator import AutoArimaEstimator


@pytest.fixture
def params():
    return dict(
        # Additional parameters
        regressor_cols=None,
        # pmdarima fit parameters
        start_p=1,
        d=1,
        start_q=1,
        max_p=1,
        max_d=1,
        max_q=1,
        start_P=1,
        D=1,
        start_Q=1,
        max_P=1,
        max_D=1,
        max_Q=1,
        max_order=1,
        m=1,
        seasonal=True,
        stationary=False,
        information_criterion='bic',
        alpha=0.05,
        test='kpss',
        seasonal_test='ocsb',
        stepwise=True,
        n_jobs=1,
        start_params=None,
        trend=None,
        method='lbfgs',
        maxiter=50,
        offset_test_args=None,
        seasonal_test_args=None,
        suppress_warnings=True,
        error_action='trace',
        trace=False,
        random=False,
        random_state=None,
        n_fits=10,
        out_of_sample_size=0,
        scoring='mse',
        scoring_args=None,
        with_intercept="auto",
        # pmdarima predict parameters
        return_conf_int=True,
        dynamic=False
    )


@pytest.fixture
def daily_data():
    return generate_df_for_tests(
        freq="D",
        periods=500,
        conti_year_origin=2018)


@pytest.fixture
def monthly_data():
    return generate_df_for_tests(
        freq="MS",
        periods=50,
        conti_year_origin=2018)


@pytest.fixture
def X():
    return pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=11, freq="D"),
        VALUE_COL: np.arange(1, 12)
    })


def test_arima_setup(params, X):
    """Checks if parameters are passed to Auto-Arima correctly"""
    coverage = 0.99
    model = AutoArimaEstimator(
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params)

    # set_params must be able to replicate the init
    model2 = AutoArimaEstimator()
    model2.set_params(**dict(
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params))
    assert model2.__dict__ == model.__dict__

    model.fit(X)
    direct_model = AutoARIMA(**params)

    model_params = model.model.__dict__
    direct_model_params = direct_model.__dict__

    assert model_params["start_p"] == direct_model_params["start_p"]
    assert model_params["d"] == direct_model_params["d"]
    assert model_params["start_q"] == direct_model_params["start_q"]
    assert model_params["max_p"] == direct_model_params["max_p"]
    assert model_params["max_d"] == direct_model_params["max_d"]
    assert model_params["max_q"] == direct_model_params["max_q"]
    assert model_params["start_P"] == direct_model_params["start_P"]
    assert model_params["D"] == direct_model_params["D"]
    assert model_params["start_Q"] == direct_model_params["start_Q"]
    assert model_params["max_P"] == direct_model_params["max_P"]
    assert model_params["max_D"] == direct_model_params["max_D"]
    assert model_params["max_Q"] == direct_model_params["max_Q"]
    assert model_params["max_order"] == direct_model_params["max_order"]
    assert model_params["m"] == direct_model_params["m"]
    assert model_params["seasonal"] == direct_model_params["seasonal"]
    assert model_params["stationary"] == direct_model_params["stationary"]
    assert model_params["information_criterion"] == direct_model_params["information_criterion"]
    assert model_params["alpha"] == direct_model_params["alpha"]
    assert model_params["test"] == direct_model_params["test"]
    assert model_params["seasonal_test"] == direct_model_params["seasonal_test"]
    assert model_params["stepwise"] == direct_model_params["stepwise"]
    assert model_params["n_jobs"] == direct_model_params["n_jobs"]
    assert model_params["start_params"] == direct_model_params["start_params"]
    assert model_params["trend"] == direct_model_params["trend"]
    assert model_params["method"] == direct_model_params["method"]
    assert model_params["maxiter"] == direct_model_params["maxiter"]
    assert model_params["offset_test_args"] == direct_model_params["offset_test_args"]
    assert model_params["seasonal_test_args"] == direct_model_params["seasonal_test_args"]
    assert model_params["suppress_warnings"] == direct_model_params["suppress_warnings"]
    assert model_params["error_action"] == direct_model_params["error_action"]
    assert model_params["trace"] == direct_model_params["trace"]
    assert model_params["random"] == direct_model_params["random"]
    assert model_params["random_state"] == direct_model_params["random_state"]
    assert model_params["n_fits"] == direct_model_params["n_fits"]
    assert model_params["out_of_sample_size"] == direct_model_params["out_of_sample_size"]
    assert model_params["scoring"] == direct_model_params["scoring"]
    assert model_params["scoring_args"] == direct_model_params["scoring_args"]
    assert model_params["with_intercept"] == direct_model_params["with_intercept"]
    assert model_params["kwargs"] == direct_model_params["kwargs"]


def test_null_model(X):
    """Checks null model"""
    model = AutoArimaEstimator(null_model_params={
        "strategy": "quantile",
        "constant": None,
        "quantile": 0.8})
    model.fit(X)
    y = np.repeat(2.0, X.shape[0])
    null_score = model.null_model.score(X, y=y)
    assert null_score == mean_squared_error(y, np.repeat(9.0, X.shape[0]))

    # tests if different score function gets propagated to null model
    model = AutoArimaEstimator(score_func=mean_absolute_error,
                               null_model_params={"strategy": "quantile",
                                                  "constant": None,
                                                  "quantile": 0.8})
    model.fit(X)
    y = np.repeat(2.0, X.shape[0])
    null_score = model.null_model.score(X, y=y)
    assert null_score == mean_absolute_error(y, np.repeat(9.0, X.shape[0]))


def test_score_function(daily_data):
    """Checks score function accuracy"""
    # with null model
    model = AutoArimaEstimator(null_model_params={"strategy": "mean"})
    train_df = daily_data["train_df"]
    value_col = "y"
    time_col = "ts"
    model.fit(train_df, time_col=time_col, value_col=value_col)
    score = model.score(daily_data["test_df"], daily_data["test_df"][value_col])
    assert score < 0.40

    # without null model
    model = AutoArimaEstimator()
    train_df = daily_data["train_df"]
    value_col = "y"
    time_col = "ts"
    model.fit(train_df, time_col=time_col, value_col=value_col)
    score = model.score(daily_data["test_df"], daily_data["test_df"][value_col])
    assert score < 8.0


def test_summary(daily_data):
    """Checks summary function output without error"""
    model = AutoArimaEstimator()
    train_df = daily_data["train_df"]
    value_col = "y"
    time_col = "ts"
    model.fit(train_df, time_col=time_col, value_col=value_col)
    model.summary()


def test_fit_predict(daily_data, monthly_data):
    """Tests fit and predict."""
    for data in daily_data, monthly_data:
        model = AutoArimaEstimator()
        train_df = daily_data["train_df"]
        test_df = daily_data["test_df"]
        assert model.last_predicted_X_ is None
        assert model.cached_predictions_ is None

        model.fit(train_df, time_col=TIME_COL, value_col=VALUE_COL)
        assert model.last_predicted_X_ is None
        assert model.cached_predictions_ is None
        with LogCapture(LOGGER_NAME) as log_capture:
            predicted = model.predict(test_df)
            assert list(predicted.columns) == [TIME_COL, PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL]
            assert_equal(model.last_predicted_X_, test_df)
            assert_equal(model.cached_predictions_, predicted)
            log_capture.check()  # no log messages (not using cached predictions)

        y_true = test_df[VALUE_COL]
        y_pred = predicted[PREDICTED_COL]

        err = calc_pred_err(y_true, y_pred)
        enum = EvaluationMetricEnum.Correlation
        assert err[enum.get_metric_name()] > 0.50
        enum = EvaluationMetricEnum.MeanAbsoluteError
        assert err[enum.get_metric_name()] < 10.0
        enum = EvaluationMetricEnum.RootMeanSquaredError
        assert err[enum.get_metric_name()] < 10.0
        enum = EvaluationMetricEnum.MedianAbsoluteError
        assert err[enum.get_metric_name()] < 10.0

        # Uses cached predictions
        with LogCapture(LOGGER_NAME) as log_capture:
            assert_equal(model.predict(test_df), predicted)
            log_capture.check(
                (LOGGER_NAME, LoggingLevelEnum.DEBUG.name, "Returning cached predictions.")
            )

        # Predicts on a different dataset
        with LogCapture(LOGGER_NAME) as log_capture:
            predicted = model.predict(train_df)
            assert_equal(model.last_predicted_X_, train_df)
            assert_equal(model.cached_predictions_, predicted)
            log_capture.check()  # no log messages (not using cached predictions)

        # .fit() clears the cached result
        model.fit(train_df, time_col=TIME_COL, value_col=VALUE_COL)
        assert model.last_predicted_X_ is None
        assert model.cached_predictions_ is None


def test_predict_interaction(daily_data):
    """Tests interaction between predict date and parameter `d`.
    Arima can not predict below `d`."""
    model = AutoArimaEstimator(d=10)
    df = daily_data["df"]
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"]
    model.fit(train_df, time_col=TIME_COL, value_col=VALUE_COL)

    # predict start date < d
    # Predicted, lower and upper CI values of the first 4 (10-6) days should be NaN
    predicted = model.predict(df[6:])
    print(predicted.head(10))
    assert (predicted[[PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL]][0:4]).isnull().values.all()
    assert not (predicted[[PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL]][5:10]).isnull().values.any()

    # predict start date > d
    # Predicted, lower and upper CI values should not be NaN
    predicted = model.predict(df[12:])
    assert not (predicted[[PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL]][0:4]).isnull().values.any()

    # predict start date > train end date
    # Predicted, lower and upper CI values should not be NaN
    predicted = model.predict(test_df[5:])
    assert not (predicted[[PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL]][0:4]).isnull().values.any()


def test_forecast_via_arima_freq(params):
    frequencies = ["H", "D", "M"]
    for freq in frequencies:
        df = generate_df_for_tests(
            freq=freq,
            periods=50)
        train_df = df["train_df"]
        test_df = df["test_df"]

        # tests model fit and predict work without error
        model = AutoArimaEstimator(**params)
        try:
            model.fit(train_df, time_col=TIME_COL, value_col=VALUE_COL)
            pred = model.predict(test_df)
        except Exception:
            print(f"Failed for frequency {freq}")
            raise

        assert list(pred.columns) == [TIME_COL, PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL]
        assert pred[TIME_COL].equals(test_df[TIME_COL])
        model.summary()
