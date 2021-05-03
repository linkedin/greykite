import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

import greykite.common.constants as cst
from greykite.algo.forecast.silverkite.forecast_simple_silverkite import SimpleSilverkiteForecast
from greykite.common.features.timeseries_features import convert_date_to_continuous_time
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import daily_data_reg
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator


@pytest.fixture
def params():
    silverkite = SimpleSilverkiteForecast()
    daily_event_df_dict = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=["India"],
        holidays_to_model_separately=["Easter Sunday", "Republic Day"],
        start_year=2017,
        end_year=2025,
        pre_num=2,
        post_num=2)
    autoreg_dict = {
        "lag_dict": {"orders": [7]},
        "agg_lag_dict": {
            "orders_list": [[7, 7*2, 7*3]],
            "interval_list": [(7, 7*2)]},
        "series_na_fill_func": lambda s: s.bfill().ffill()}
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow"],
            "quantiles": [0.05, 0.95],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    return {
        "time_properties": None,
        "freq": None,
        "forecast_horizon": None,
        "origin_for_time_vars": convert_date_to_continuous_time(datetime.datetime(2018, 1, 3)),
        "train_test_thresh": None,
        "training_fraction": None,
        "fit_algorithm_dict": {
            "fit_algorithm": "sgd",
            "fit_algorithm_params": {"alpha": 0.1}
        },
        "holidays_to_model_separately": ["New Year's Day", "Christmas Day"],
        "holiday_lookup_countries": ["UnitedStates"],
        "holiday_pre_num_days": 2,
        "holiday_post_num_days": 2,
        "holiday_pre_post_num_dict": {
            "New Year's Day": (7, 3)
        },
        "daily_event_df_dict": daily_event_df_dict,
        "changepoints_dict": None,
        "yearly_seasonality": "auto",
        "quarterly_seasonality": False,
        "monthly_seasonality": False,
        "weekly_seasonality": 3,
        "daily_seasonality": False,
        "max_daily_seas_interaction_order": None,
        "max_weekly_seas_interaction_order": None,
        "autoreg_dict": autoreg_dict,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "uncertainty_dict": uncertainty_dict,
        "growth_term": "linear",
        "regressor_cols": None,
        "feature_sets_enabled": None,
        "extra_pred_cols": ["ct1", "regressor1", "regressor2"]
    }


@pytest.fixture
def params2():
    silverkite = SimpleSilverkiteForecast()
    daily_event_df_dict = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=["India"],
        holidays_to_model_separately=["Easter Sunday", "Republic Day"],
        start_year=2017,
        end_year=2025,
        pre_num=2,
        post_num=2)
    autoreg_dict = {
        "lag_dict": {"orders": [7]},
        "agg_lag_dict": {
            "orders_list": [[7, 7*2, 7*3]],
            "interval_list": [(7, 7*2)]},
        "series_na_fill_func": lambda s: s.bfill().ffill()}
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow"],
            "quantiles": [0.05, 0.95],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    return {
        "time_properties": None,
        "freq": None,
        "forecast_horizon": 5,
        "origin_for_time_vars": convert_date_to_continuous_time(datetime.datetime(2018, 1, 1)),
        "train_test_thresh": None,
        "training_fraction": None,
        "fit_algorithm_dict": {
            "fit_algorithm": "ridge",
            "fit_algorithm_params": None
        },
        "holidays_to_model_separately": ["New Year's Day", "Christmas Day"],
        "holiday_lookup_countries": ["UnitedStates"],
        "holiday_pre_num_days": 2,
        "holiday_post_num_days": 2,
        "holiday_pre_post_num_dict": {
            "New Year's Day": (7, 3)
        },
        "daily_event_df_dict": daily_event_df_dict,
        "changepoints_dict": None,
        "yearly_seasonality": "auto",
        "quarterly_seasonality": False,
        "monthly_seasonality": False,
        "weekly_seasonality": 3,
        "daily_seasonality": False,
        "max_daily_seas_interaction_order": None,
        "max_weekly_seas_interaction_order": None,
        "autoreg_dict": autoreg_dict,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "uncertainty_dict": uncertainty_dict,
        "growth_term": "linear",
        "regressor_cols": None,
        "feature_sets_enabled": None,
        "extra_pred_cols": ["ct1", "regressor1", "regressor2"],
        "regression_weight_col": "ct2",
        "simulation_based": True,
    }


@pytest.fixture
def daily_data():
    return generate_df_for_tests(
        freq="D",
        periods=1000,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)


@pytest.fixture
def daily_data_with_reg():
    return daily_data_reg()


@pytest.fixture
def X():
    periods = 11
    return pd.DataFrame({
        cst.TIME_COL: pd.date_range("2018-01-01", periods=periods, freq="D"),
        cst.VALUE_COL: np.arange(1, periods + 1)
    })


def test_setup(params):
    """Tests __init__ and attributes set during fit"""
    coverage = 0.90
    silverkite = SimpleSilverkiteForecast()
    model = SimpleSilverkiteEstimator(
        silverkite=silverkite,
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params)

    assert model.silverkite == silverkite
    assert model.score_func == mean_squared_error
    assert model.coverage == coverage
    assert model.null_model_params is None

    # set_params must be able to replicate the init
    model2 = SimpleSilverkiteEstimator()
    model2.set_params(**dict(
        silverkite=silverkite,
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params))
    assert model2.__dict__ == model.__dict__

    initialized_params = model.__dict__
    initialized_params_subset = {
        k: v for k, v in initialized_params.items()
        if k in params.keys()}
    assert_equal(initialized_params_subset, params)

    assert model.model_dict is None
    assert model.pred_cols is None
    assert model.feature_cols is None
    assert model.coef_ is None

    train_df = daily_data_reg().get("train_df").copy()
    model.fit(train_df)
    assert model.fit_algorithm_dict == {
        "fit_algorithm": "sgd",
        "fit_algorithm_params": {"alpha": 0.1}}
    assert model.model_dict is not None
    assert type(model.model_dict["ml_model"]) == SGDRegressor
    assert model.model_dict["ml_model"].alpha == (
        params["fit_algorithm_dict"]["fit_algorithm_params"]["alpha"])
    assert model.model_dict["training_evaluation"] is not None
    assert model.model_dict["test_evaluation"] is None
    assert model.pred_cols is not None
    assert model.feature_cols is not None
    assert_frame_equal(model.df, train_df)
    assert model.coef_ is not None


def test_setup2(params2):
    """Tests __init__ and attributes set during fit"""
    coverage = 0.90
    silverkite = SimpleSilverkiteForecast()
    model = SimpleSilverkiteEstimator(
        silverkite=silverkite,
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params2)

    assert model.silverkite == silverkite
    assert model.score_func == mean_squared_error
    assert model.coverage == coverage
    assert model.null_model_params is None

    # set_params must be able to replicate the init
    model2 = SimpleSilverkiteEstimator()
    model2.set_params(**dict(
        silverkite=silverkite,
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params2))
    assert model2.__dict__ == model.__dict__

    initialized_params = model.__dict__
    initialized_params_subset = {
        k: v for k, v in initialized_params.items()
        if k in params2.keys()}
    assert_equal(initialized_params_subset, params2)

    assert model.model_dict is None
    assert model.pred_cols is None
    assert model.feature_cols is None
    assert model.coef_ is None

    train_df = daily_data_reg().get("train_df").copy()
    model.fit(train_df)
    assert model.fit_algorithm_dict == {
        "fit_algorithm": "ridge",
        "fit_algorithm_params": None}
    assert model.model_dict is not None
    assert model.model_dict["training_evaluation"] is not None
    assert model.model_dict["test_evaluation"] is None
    assert model.pred_cols is not None
    assert model.feature_cols is not None
    assert_frame_equal(model.df, train_df)
    assert model.coef_ is not None


def test_score_function_null(daily_data):
    """Tests fit and its compatibility with predict/score.
    Checks score function accuracy with null model
    """
    model = SimpleSilverkiteEstimator(
        null_model_params={"strategy": "mean"},
        fit_algorithm_dict={
            "fit_algorithm": "linear"
        },
        holidays_to_model_separately=[],
        yearly_seasonality=5,
        quarterly_seasonality=False,
        monthly_seasonality=False,
        weekly_seasonality=3,
        feature_sets_enabled=False)
    train_df = daily_data["train_df"]

    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.fit_algorithm_dict == {
        "fit_algorithm": "linear",
        "fit_algorithm_params": None
    }
    score = model.score(
        daily_data["test_df"],
        daily_data["test_df"][cst.VALUE_COL])
    assert score == pytest.approx(0.90, rel=1e-2)


def test_score_function(daily_data_with_reg):
    """Tests fit and its compatibility with predict/score.
    Checks score function accuracy without null model
    """
    model = SimpleSilverkiteEstimator(
        extra_pred_cols=["ct1", "regressor1", "regressor2"],
        fit_algorithm_dict={
            "fit_algorithm": "linear"
        },
        holidays_to_model_separately=[],
        yearly_seasonality=5,
        quarterly_seasonality=False,
        monthly_seasonality=False,
        weekly_seasonality=3,
        feature_sets_enabled=False)
    train_df = daily_data_with_reg["train_df"]
    test_df = daily_data_with_reg["test_df"]

    model.fit(
        X=train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)

    score = model.score(test_df, test_df[cst.VALUE_COL])
    pred_df = model.predict(test_df)
    assert list(pred_df.columns) == [cst.TIME_COL, cst.PREDICTED_COL]
    assert score == pytest.approx(mean_squared_error(
        pred_df[cst.PREDICTED_COL],
        test_df[cst.VALUE_COL]))
    assert score == pytest.approx(4.39, rel=1e-2)


def test_uncertainty(daily_data):
    """Runs a basic model with uncertainty intervals
    and checks coverage"""
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 10,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    model = SimpleSilverkiteEstimator(
        uncertainty_dict=uncertainty_dict,
        fit_algorithm_dict={
            "fit_algorithm": "linear"
        },
        holidays_to_model_separately=[],
        yearly_seasonality=5,
        quarterly_seasonality=False,
        monthly_seasonality=False,
        weekly_seasonality=3,
        feature_sets_enabled=False
    )
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"]

    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    predictions = model.predict(test_df)
    expected_forecast_cols = \
        {"ts", "y", "y_quantile_summary", "err_std", "forecast_lower", "forecast_upper"}
    assert expected_forecast_cols.issubset(list(model.forecast.columns))

    actual = daily_data["test_df"][cst.VALUE_COL]
    forecast_lower = predictions[cst.PREDICTED_LOWER_COL]
    forecast_upper = predictions[cst.PREDICTED_UPPER_COL]
    calc_pred_coverage = 100 * (
            (actual <= forecast_upper)
            & (actual >= forecast_lower)
    ).mean()
    assert round(calc_pred_coverage) == 95, "forecast coverage is incorrect"


def test_summary(daily_data):
    """Checks summary function returns without error"""
    model = SimpleSilverkiteEstimator(
        fit_algorithm_dict={
            "fit_algorithm_params": {"cv": 3}
        }
    )
    train_df = daily_data["train_df"]
    model.summary()

    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.fit_algorithm_dict == {
        "fit_algorithm": "ridge",
        "fit_algorithm_params": {"cv": 3}
    }
    model.summary()


def test_plot_components():
    """Tests plot_components.
    Because component plots are implemented in `base_silverkite_estimator.py,` the bulk of
    the testing is done there. This file only tests inheritance and compatibility of the
    trained_model generated by this estimator's fit.
    """
    daily_data = generate_df_with_reg_for_tests(
        freq="D",
        periods=20,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)
    train_df = daily_data.get("train_df").copy()
    model = SimpleSilverkiteEstimator(
        fit_algorithm_dict={
            "fit_algorithm": "linear"
        },
        yearly_seasonality=True,
        quarterly_seasonality=False,
        monthly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(train_df)

    # Test plot_components
    with pytest.warns(Warning) as record:
        title = "Custom component plot"
        fig = model.plot_components(names=["trend", "YEARLY_SEASONALITY", "DUMMY"], title=title)
        expected_rows = 3
        assert len(fig.data) == expected_rows
        assert [fig.data[i].name for i in range(expected_rows)] == \
               [cst.VALUE_COL, "trend", "YEARLY_SEASONALITY"]

        assert fig.layout.xaxis.title["text"] == cst.TIME_COL
        assert fig.layout.xaxis2.title["text"] == cst.TIME_COL
        assert fig.layout.xaxis3.title["text"] == "Time of year"

        assert fig.layout.yaxis.title["text"] == cst.VALUE_COL
        assert fig.layout.yaxis2.title["text"] == "trend"
        assert fig.layout.yaxis3.title["text"] == "yearly"

        assert fig.layout.title["text"] == title
        assert f"The following components have not been specified in the model: " \
               f"{{'DUMMY'}}, plotting the rest." in record[0].message.args[0]

    # Test plot_trend
    title = "Custom trend plot"
    fig = model.plot_trend(title=title)
    expected_rows = 2
    assert len(fig.data) == expected_rows
    assert [fig.data[i].name for i in range(expected_rows)] == [cst.VALUE_COL, "trend"]

    assert fig.layout.xaxis.title["text"] == cst.TIME_COL
    assert fig.layout.xaxis2.title["text"] == cst.TIME_COL

    assert fig.layout.yaxis.title["text"] == cst.VALUE_COL
    assert fig.layout.yaxis2.title["text"] == "trend"

    assert fig.layout.title["text"] == title

    # Test plot_seasonalities
    with pytest.warns(Warning):
        # suppresses the warning on seasonalities removed
        title = "Custom seasonality plot"
        fig = model.plot_seasonalities(title=title)
        expected_rows = 3
        assert len(fig.data) == expected_rows
        assert [fig.data[i].name for i in range(expected_rows)] == \
               [cst.VALUE_COL, "WEEKLY_SEASONALITY", "YEARLY_SEASONALITY"]

        assert fig.layout.xaxis.title["text"] == cst.TIME_COL
        assert fig.layout.xaxis2.title["text"] == "Day of week"
        assert fig.layout.xaxis3.title["text"] == "Time of year"

        assert fig.layout.yaxis.title["text"] == cst.VALUE_COL
        assert fig.layout.yaxis2.title["text"] == "weekly"
        assert fig.layout.yaxis3.title["text"] == "yearly"

        assert fig.layout.title["text"] == title
