import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
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
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.sklearn.estimator.prophet_estimator import ProphetEstimator


try:
    import prophet  # noqa
except ModuleNotFoundError:
    pass


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
@pytest.fixture
def params():
    holidays = pd.DataFrame({
        "ds": pd.to_datetime(["2018-12-25", "2019-12-25", "2020-12-25"]),
        "holiday": ["christmas", "christmas", "christmas"],
        "lower_window": [-2, -2, -2],
        "upper_window": [2, 2, 2],
    })
    return {
        "growth": "linear",
        "changepoints": ["2018-01-02", "2018-01-04"],
        "n_changepoints": None,
        "changepoint_range": 0.7,
        "yearly_seasonality": False,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays": holidays,
        "seasonality_mode": "multiplicative",
        "seasonality_prior_scale": 5.0,
        "holidays_prior_scale": 5.0,
        "changepoint_prior_scale": 0.10,
        "mcmc_samples": 0,
        "uncertainty_samples": 1000
    }


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
@pytest.fixture
def params_multiple_regressors():
    add_regressor_dict = {
        "regressor1": {
            "prior_scale": 10,
            "standardize": True,
            "mode": 'additive'
        },
        "regressor2": {
            "prior_scale": 15,
            "standardize": False,
            "mode": 'additive'
        },
        "regressor3": {}
    }
    return {
        "add_regressor_dict": add_regressor_dict
    }


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
@pytest.fixture
def params_add_seasonality():
    add_seasonality_dict = {
        'monthly': {
            'period': 30.5,
            'fourier_order': 5
        },
        'yearly': {
            'period': 365.25,
            'fourier_order': 10,
            'prior_scale': 0.2,
            'mode': 'additive'
        }
    }
    return {
        "add_seasonality_dict": add_seasonality_dict
    }


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
@pytest.fixture
def params_reg(params_add_seasonality, params_multiple_regressors):
    holidays = pd.DataFrame({
        "ds": pd.to_datetime(["2018-12-25", "2019-12-25", "2020-12-25"]),
        "holiday": ["christmas", "christmas", "christmas"],
        "lower_window": [-2, -2, -2],
        "upper_window": [2, 2, 2],
    })
    add_regressor_dict = params_multiple_regressors["add_regressor_dict"]
    add_seasonality_dict = params_add_seasonality["add_seasonality_dict"]
    return {
        "growth": "linear",
        "changepoints": ["2018-01-02", "2018-01-04"],
        "n_changepoints": None,
        "changepoint_range": 0.7,
        "yearly_seasonality": False,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays": holidays,
        "add_regressor_dict": add_regressor_dict,
        "add_seasonality_dict": add_seasonality_dict,
        "seasonality_mode": "multiplicative",
        "seasonality_prior_scale": 5.0,
        "holidays_prior_scale": 5.0,
        "changepoint_prior_scale": 0.10,
        "mcmc_samples": 0,
        "uncertainty_samples": 1000
    }


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
@pytest.fixture
def daily_data():
    return generate_df_for_tests(
        freq="D",
        periods=500,
        conti_year_origin=2018)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
@pytest.fixture
def daily_data_reg():
    return generate_df_with_reg_for_tests(
        freq="D",
        periods=500)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
@pytest.fixture
def X():
    return pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=11, freq="D"),
        VALUE_COL: np.arange(1, 12)
    })


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
@pytest.fixture
def X_reg():
    return pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=20, freq="D"),
        VALUE_COL: np.arange(20),
        "regressor1": np.exp(np.arange(20)),
        "regressor2": np.random.normal(size=20),
        "regressor3": 0.001 * np.random.normal(size=20)
    })


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_setup(params, X):
    """Checks if parameters are passed to Prophet correctly"""
    coverage = 0.99
    model = ProphetEstimator(
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params)

    # set_params must be able to replicate the init
    model2 = ProphetEstimator()
    model2.set_params(**dict(
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params))
    assert model2.__dict__ == model.__dict__

    model.fit(X)
    direct_model = prophet.Prophet(**params)

    model_params = model.model.__dict__
    direct_model_params = direct_model.__dict__

    # only need to check these
    assert model_params["growth"] == direct_model_params["growth"]
    assert model_params["changepoints"].equals(direct_model_params["changepoints"])
    assert model_params["n_changepoints"] == direct_model_params["n_changepoints"]
    assert model_params["specified_changepoints"] == direct_model_params["specified_changepoints"]
    assert model_params["changepoint_range"] == direct_model_params["changepoint_range"]
    assert model_params["yearly_seasonality"] == direct_model_params["yearly_seasonality"]
    assert model_params["weekly_seasonality"] == direct_model_params["weekly_seasonality"]
    assert model_params["daily_seasonality"] == direct_model_params["daily_seasonality"]
    assert model_params["holidays"].equals(direct_model_params["holidays"])
    assert model_params["seasonality_mode"] == direct_model_params["seasonality_mode"]
    assert model_params["seasonality_prior_scale"] == direct_model_params["seasonality_prior_scale"]
    assert model_params["changepoint_prior_scale"] == direct_model_params["changepoint_prior_scale"]
    assert model_params["holidays_prior_scale"] == direct_model_params["holidays_prior_scale"]
    assert model_params["mcmc_samples"] == direct_model_params["mcmc_samples"]
    assert model_params["uncertainty_samples"] == direct_model_params["uncertainty_samples"]

    # interval width is set by coverage
    assert model_params["interval_width"] == coverage


# test regressor and custom seasonality hyper parameters being passed to ProphetEstimator vs Direct model
@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_reg_seas_setup(params_reg, X_reg):
    """Checks if parameters are passed to Prophet correctly"""
    coverage = 0.99
    model = ProphetEstimator(score_func=mean_squared_error, coverage=coverage, null_model_params=None, **params_reg)
    model.fit(X_reg)

    # remove custom seasonalities and regressors before passing params to prophet
    params_no_reg_no_custom_seas = {key: value for (key, value) in params_reg.items()
                                    if key not in ['add_regressor_dict', 'add_seasonality_dict']}

    direct_model = prophet.Prophet(**params_no_reg_no_custom_seas)

    # add regressors in direct (prophet) model in the usual way (add_regressor method)
    for reg_col, reg_params in params_reg["add_regressor_dict"].items():
        direct_model.add_regressor(name=reg_col, **reg_params)

    # add custom seasonality in direct (prophet) model in the usual way (add_seasonality method)
    for seasonality_type, seasonality_params in params_reg["add_seasonality_dict"].items():
        direct_model.add_seasonality(name=seasonality_type, **seasonality_params)

    # fit direct_model to add weekly seasonality to the seasonalities attribute (only those passed
    # by add_seasonality are set during init)
    df = X_reg
    df.rename(columns={TIME_COL: "ds", VALUE_COL: "y"}, inplace=True)
    # Prophet requires "ds" and "y" as time and value columns respectively.
    direct_model.fit(df)

    # direct model regressor and seasonality params
    direct_model_reg = direct_model.extra_regressors
    direct_model_seasonalities = direct_model.seasonalities

    # ProphetEstimator model regressor and seasonality params
    model_reg = model.model.extra_regressors
    model_seasonalities = model.model.seasonalities

    # confirm if all regressors and their params are being passed accurately, including default params
    for reg_col in params_reg["add_regressor_dict"].keys():
        assert model_reg[reg_col]['mode'] == direct_model_reg[reg_col]['mode']
        assert model_reg[reg_col]['standardize'] == direct_model_reg[reg_col]['standardize']
        assert model_reg[reg_col]['prior_scale'] == direct_model_reg[reg_col]['prior_scale']

    # make sure seasonalities are matching
    assert direct_model_seasonalities == model_seasonalities


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_null_model(X):
    """Checks null model"""
    model = ProphetEstimator(null_model_params={"strategy": "quantile",
                                                "constant": None,
                                                "quantile": 0.8})
    model.fit(X)
    y = np.repeat(2.0, X.shape[0])
    null_score = model.null_model.score(X, y=y)
    assert null_score == mean_squared_error(y, np.repeat(9.0, X.shape[0]))

    # tests if different score function gets propagated to null model
    model = ProphetEstimator(score_func=mean_absolute_error,
                             null_model_params={"strategy": "quantile",
                                                "constant": None,
                                                "quantile": 0.8})
    model.fit(X)
    y = np.repeat(2.0, X.shape[0])
    null_score = model.null_model.score(X, y=y)
    assert null_score == mean_absolute_error(y, np.repeat(9.0, X.shape[0]))


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_score_function_null(daily_data):
    """Checks score function accuracy with null model"""
    model = ProphetEstimator(null_model_params={"strategy": "mean"})
    train_df = daily_data["train_df"]

    value_col = "y"
    time_col = "ts"

    model.fit(train_df, time_col=time_col, value_col=value_col)
    score = model.score(daily_data["test_df"], daily_data["test_df"][value_col])
    assert score == pytest.approx(0.42, rel=1e-2)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_score_function(daily_data):
    """Checks score function accuracy without null model"""
    model = ProphetEstimator()
    train_df = daily_data["train_df"]

    value_col = "y"
    time_col = "ts"

    model.fit(train_df, time_col=time_col, value_col=value_col)
    score = model.score(daily_data["test_df"], daily_data["test_df"][value_col])
    assert score == pytest.approx(5.77, rel=1e-1)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_summary(daily_data):
    """Checks summary function output without error"""
    model = ProphetEstimator()
    train_df = daily_data["train_df"]

    value_col = "y"
    time_col = "ts"
    model.summary()

    model.fit(train_df, time_col=time_col, value_col=value_col)
    model.summary()


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_fit_predict(daily_data):
    """Tests fit and predict."""
    model = ProphetEstimator()
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
    assert err[enum.get_metric_name()] < 2.5
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 2.5
    enum = EvaluationMetricEnum.MedianAbsoluteError
    assert err[enum.get_metric_name()] < 2.5

    # Uses cached predictions
    with LogCapture(LOGGER_NAME) as log_capture:
        assert_equal(model.predict(test_df), predicted)
        log_capture.check(
            (LOGGER_NAME, "DEBUG", "Returning cached predictions.")
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


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_forecast_via_prophet_daily_reg(
        daily_data_reg,
        params_add_seasonality,
        params_multiple_regressors):
    """Tests fit and predict with regressors and custom seasonality."""
    model = ProphetEstimator(
        score_func=mean_squared_error,
        null_model_params=None,
        **params_multiple_regressors,
        **params_add_seasonality)
    train_df = daily_data_reg["train_df"]
    test_df = daily_data_reg["test_df"]

    model.fit(train_df, time_col=TIME_COL, value_col=VALUE_COL)
    pred = model.predict(test_df)
    assert list(pred.columns) == [TIME_COL, PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL]

    y_true = test_df[VALUE_COL]
    y_pred = pred[PREDICTED_COL]

    err = calc_pred_err(y_true, y_pred)
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.50
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 2.5
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.MedianAbsoluteError
    assert err[enum.get_metric_name()] < 3.0


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_forecast_via_prophet_freq():
    """Tests prophet model at different frequencies"""
    holidays = pd.DataFrame({
        "ds": pd.to_datetime(["2018-12-25", "2019-12-25", "2020-12-25"]),
        "holiday": "christmas",
        "lower_window": -2,
        "upper_window": 2,
    })
    params = dict(
        coverage=0.9,
        growth="linear",
        n_changepoints=2,
        changepoint_range=0.9,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=holidays,
        seasonality_mode="additive",
        seasonality_prior_scale=5.0,
        holidays_prior_scale=5.0,
        changepoint_prior_scale=0.10,
        mcmc_samples=0,
        uncertainty_samples=10
    )
    # A wide variety of frequencies listed here:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    frequencies = [
        "B", "W", "W-SAT", "W-TUE", "M", "SM",
        "MS", "SMS", "CBMS", "BM", "B", "Q",
        "QS", "BQS", "BQ-AUG", "Y", "YS",
        "AS-SEP", "H", "BH", "T", "S"]
    for freq in frequencies:
        df = generate_df_for_tests(
            freq=freq,
            periods=50)
        train_df = df["train_df"]
        test_df = df["test_df"]

        # tests model fit and predict work without error
        model = ProphetEstimator(**params)
        try:
            model.fit(train_df, time_col=TIME_COL, value_col=VALUE_COL)
            pred = model.predict(test_df)
        except Exception:
            print(f"Failed for frequency {freq}")
            raise

        assert list(pred.columns) == [TIME_COL, PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL]
        assert pred[TIME_COL].equals(test_df[TIME_COL])
        model.summary()


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_forecast_via_prophet_no_uncertainty(
        daily_data):
    """Tests fit and predict with no uncertainty interval."""
    model = ProphetEstimator(
        score_func=mean_squared_error,
        null_model_params=None,
        uncertainty_samples=0,
        mcmc_samples=0)
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"]

    assert model.uncertainty_samples == 0
    assert model.mcmc_samples == 0

    model.fit(train_df, time_col=TIME_COL, value_col=VALUE_COL)
    model.predict(test_df)
    # The following asserts are temporarily disabled,
    # since they work under prophet >= 0.6 only.
    # assert PREDICTED_LOWER_COL not in pred.columns
    # assert PREDICTED_UPPER_COL not in pred.columns


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_plot_components(daily_data, params):
    """Test plot_components"""
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"]

    model = ProphetEstimator()
    model.fit(train_df)
    assert model.forecast is None

    model.predict(test_df)
    forecast = model.forecast
    expected_forecast_cols = \
        {"ds", "yhat", "yhat_lower", "yhat_upper", "trend", "trend_lower",
         "trend_upper", "weekly", "weekly_lower", "weekly_upper"}
    assert expected_forecast_cols.issubset(list(forecast.columns))

    fig = model.plot_components(uncertainty=True, plot_cap=False)
    assert fig

    direct_model = model.model
    direct_fig = direct_model.plot_components(
        fcst=forecast,
        uncertainty=True,
        plot_cap=False)
    assert direct_fig

    # Tests plot_components warnings
    model = ProphetEstimator()
    with pytest.raises(NotFittedError, match="The fit method has not been run yet."):
        model.plot_components()

    with pytest.raises(RuntimeError, match="The predict method has not been run yet."):
        model.fit(train_df)
        model.plot_components()
