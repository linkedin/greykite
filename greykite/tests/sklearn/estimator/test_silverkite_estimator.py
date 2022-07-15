import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

import greykite.common.constants as cst
from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.common.features.timeseries_features import convert_date_to_continuous_time
from greykite.common.features.timeseries_impute import impute_with_lags
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import daily_data_reg
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.sklearn.estimator.silverkite_estimator import SilverkiteEstimator
from greykite.sklearn.estimator.testing_utils import params_components


@pytest.fixture
def params():
    autoreg_dict = {
        "lag_dict": {"orders": [7]},
        "agg_lag_dict": {
            "orders_list": [[7, 7*2, 7*3]],
            "interval_list": [(7, 7*2)]},
        "series_na_fill_func": lambda s: s.bfill().ffill()}
    lagged_regressor_dict = {
        "regressor1": {
            "lag_dict": {"orders": [1, 2, 3]},
            "agg_lag_dict": {
                "orders_list": [[7, 7 * 2, 7 * 3]],
                "interval_list": [(8, 7 * 2)]},
            "series_na_fill_func": lambda s: s.bfill().ffill()},
        "regressor2": "auto"}
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow"],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    return {
        "origin_for_time_vars": convert_date_to_continuous_time(datetime.datetime(2018, 1, 3)),
        "extra_pred_cols": ["ct1", "regressor1", "regressor2"],
        "train_test_thresh": None,
        "training_fraction": None,
        "fit_algorithm_dict": {
            "fit_algorithm": "sgd",
            "fit_algorithm_params": {"alpha": 0.1},
        },
        "daily_event_df_dict": None,
        "changepoints_dict": None,
        "changepoint_detector": None,
        "fs_components_df": pd.DataFrame({
            "name": ["tow"],
            "period": [7.0],
            "order": [3],
            "seas_names": [None]}),
        "autoreg_dict": autoreg_dict,
        "lagged_regressor_dict": lagged_regressor_dict,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "uncertainty_dict": uncertainty_dict,
        "normalize_method": "zero_to_one",
        "adjust_anomalous_dict": None,
        "impute_dict": {
            "func": impute_with_lags,
            "params": {"orders": [7]}},
        "regression_weight_col": None,
        "forecast_horizon": 12,
        "simulation_based": True,
    }


@pytest.fixture
def params2():
    autoreg_dict = {
        "lag_dict": {"orders": [7]},
        "agg_lag_dict": {
            "orders_list": [[7, 7*2, 7*3]],
            "interval_list": [(7, 7*2)]},
        "series_na_fill_func": lambda s: s.bfill().ffill()}
    lagged_regressor_dict = {
        "regressor1": {
            "lag_dict": {"orders": [1, 2, 3]},
            "agg_lag_dict": {
                "orders_list": [[7, 7 * 2, 7 * 3]],
                "interval_list": [(8, 7 * 2)]},
            "series_na_fill_func": lambda s: s.bfill().ffill()},
        "regressor2": "auto"}
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow"],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    return {
        "origin_for_time_vars": convert_date_to_continuous_time(datetime.datetime(2018, 1, 1)),
        "extra_pred_cols": ["ct1", "regressor1", "regressor2"],
        "train_test_thresh": None,
        "training_fraction": None,
        "fit_algorithm_dict": {
            "fit_algorithm": "ridge",
            "fit_algorithm_params": None,
        },
        "daily_event_df_dict": None,
        "changepoints_dict": None,
        "changepoint_detector": None,
        "fs_components_df": pd.DataFrame({
            "name": ["tow"],
            "period": [7.0],
            "order": [3],
            "seas_names": [None]}),
        "autoreg_dict": autoreg_dict,
        "lagged_regressor_dict": lagged_regressor_dict,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "uncertainty_dict": uncertainty_dict,
        "normalize_method": "zero_to_one",
        "adjust_anomalous_dict": None,
        "impute_dict": {
            "func": impute_with_lags,
            "params": {"orders": [7]}},
        "regression_weight_col": "ct2",
        "forecast_horizon": 5,
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
    coverage = 0.95
    silverkite = SilverkiteForecast()
    model = SilverkiteEstimator(
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
    model2 = SilverkiteEstimator()
    model2.set_params(**dict(
        silverkite=silverkite,
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params))
    assert model2.__dict__ == model.__dict__

    initalized_params = model.__dict__
    initalized_params_subset = {
        k: v for k, v in initalized_params.items()
        if k in params.keys()}
    assert_equal(initalized_params_subset, params)

    assert model.model_dict is None
    assert model.pred_cols is None
    assert model.feature_cols is None
    assert model.coef_ is None

    train_df = daily_data_reg().get("train_df").copy()
    model.fit(train_df)
    assert model.fit_algorithm_dict == {
        "fit_algorithm": "sgd",
        "fit_algorithm_params": {"alpha": 0.1},
    }
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
    coverage = 0.95
    silverkite = SilverkiteForecast()
    model = SilverkiteEstimator(
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
    model2 = SilverkiteEstimator()
    model2.set_params(**dict(
        silverkite=silverkite,
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        **params2))
    assert model2.__dict__ == model.__dict__

    initalized_params = model.__dict__
    initalized_params_subset = {
        k: v for k, v in initalized_params.items()
        if k in params2.keys()}
    assert_equal(initalized_params_subset, params2)

    assert model.model_dict is None
    assert model.pred_cols is None
    assert model.feature_cols is None
    assert model.coef_ is None

    train_df = daily_data_reg().get("train_df").copy()
    model.fit(train_df)
    assert model.model_dict is not None
    assert model.model_dict["training_evaluation"] is not None
    assert model.model_dict["test_evaluation"] is None
    assert model.pred_cols is not None
    assert model.feature_cols is not None
    assert_frame_equal(model.df, train_df)
    assert model.coef_ is not None


def test_validate_inputs():
    """Test validate_inputs"""

    with pytest.warns(None) as record:
        SilverkiteEstimator()
        assert len(record) == 0  # no warnings

    with pytest.raises(ValueError) as record:
        fs_components_df = pd.DataFrame({
            "name": ["tod", "tow"],
            "period": [24.0, 7.0]})
        SilverkiteEstimator(fs_components_df=fs_components_df)
        fs_cols_not_found = {"order", "seas_names"}
        assert (f"fs_components_df is missing the following columns: "
                f"{fs_cols_not_found}" in record[0].message.args[0])

    with pytest.raises(ValueError) as record:
        fs_components_df = pd.DataFrame({
            "name": ["tod", "tow", "tow"],
            "period": [24.0, 7.0, 10.0],
            "order": [12, 4, 3],
            "seas_names": ["daily", "weekly", "weekly"]})
        SilverkiteEstimator(fs_components_df=fs_components_df)
        assert ("Found multiple rows in fs_components_df with same `names` and `seas_names`. "
                "Make sure these are unique." in record[0].message.args[0])


def test_score_function_null(daily_data):
    """Tests fit and its compatibility with predict/score.
    Checks score function accuracy with null model
    """
    model = SilverkiteEstimator(
        null_model_params={"strategy": "mean"},
        fit_algorithm_dict={
            "fit_algorithm_params": {"fit_intercept": False}
        }
    )
    assert model.fit_algorithm_dict == {
        "fit_algorithm_params": {"fit_intercept": False}
    }
    train_df = daily_data["train_df"]

    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.fit_algorithm_dict == {
        "fit_algorithm": "linear",
        "fit_algorithm_params": {"fit_intercept": False}
    }
    score = model.score(
        daily_data["test_df"],
        daily_data["test_df"][cst.VALUE_COL])
    assert score == pytest.approx(0.90, rel=1e-2)


def test_score_function(daily_data_with_reg):
    """Tests fit and its compatibility with predict/score.
    Checks score function accuracy without null model
    """
    model = SilverkiteEstimator(
        extra_pred_cols=["ct1", "regressor1", "regressor2"],
        impute_dict={
            "func": impute_with_lags,
            "params": {"orders": [7]}}
    )
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
    assert score == pytest.approx(4.6, rel=1e-2)


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
    model = SilverkiteEstimator(uncertainty_dict=uncertainty_dict)
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"]

    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    predictions = model.predict(test_df)
    expected_forecast_cols = \
        {"ts", "y", cst.QUANTILE_SUMMARY_COL, "err_std", "forecast_lower", "forecast_upper"}
    assert expected_forecast_cols.issubset(list(model.forecast.columns))

    actual = daily_data["test_df"][cst.VALUE_COL]
    forecast_lower = predictions[cst.PREDICTED_LOWER_COL]
    forecast_upper = predictions[cst.PREDICTED_UPPER_COL]
    calc_pred_coverage = 100 * (
        (actual <= forecast_upper)
        & (actual >= forecast_lower)
        ).mean()
    assert round(calc_pred_coverage) == 97, "forecast coverage is incorrect"


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
    params_daily = params_components()
    fit_algorithm = params_daily.pop("fit_algorithm", "linear")
    fit_algorithm_params = params_daily.pop("fit_algorithm_params", None)
    params_daily["fit_algorithm_dict"] = {
        "fit_algorithm": fit_algorithm,
        "fit_algorithm_params": fit_algorithm_params,
    }
    # removing daily seasonality terms
    params_daily["fs_components_df"] = pd.DataFrame({
        "name": ["tow", "ct1"],
        "period": [7.0, 1.0],
        "order": [4, 5],
        "seas_names": ["weekly", "yearly"]})
    model = SilverkiteEstimator(**params_daily)
    with pytest.warns(Warning):
        # suppresses sklearn warning on `iid` parameter for ridge hyperparameter_grid search
        model.fit(train_df)

    # Test plot_components
    with pytest.warns(Warning) as record:
        title = "Custom component plot"
        fig = model.plot_components(names=["trend", "YEARLY_SEASONALITY", "DUMMY"], title=title)
        expected_rows = 3
        assert len(fig.data) == expected_rows + 1  # includes changepoints
        assert [fig.data[i].name for i in range(expected_rows)] == \
               [cst.VALUE_COL, "trend", "YEARLY_SEASONALITY"]

        assert fig.layout.xaxis.title["text"] == cst.TIME_COL
        assert fig.layout.xaxis2.title["text"] == cst.TIME_COL
        assert fig.layout.xaxis3.title["text"] == "Time of year"

        assert fig.layout.yaxis.title["text"] == cst.VALUE_COL
        assert fig.layout.yaxis2.title["text"] == "trend"
        assert fig.layout.yaxis3.title["text"] == "yearly"

        assert fig.layout.title["text"] == title
        assert fig.layout.title["x"] == 0.5
        assert f"The following components have not been specified in the model: " \
               f"{{'DUMMY'}}, plotting the rest." in record[0].message.args[0]

    # Test plot_trend
    title = "Custom trend plot"
    fig = model.plot_trend(title=title)
    expected_rows = 2
    assert len(fig.data) == expected_rows + 1  # includes changepoints
    assert [fig.data[i].name for i in range(expected_rows)] == [cst.VALUE_COL, "trend"]

    assert fig.layout.xaxis.title["text"] == cst.TIME_COL
    assert fig.layout.xaxis2.title["text"] == cst.TIME_COL

    assert fig.layout.yaxis.title["text"] == cst.VALUE_COL
    assert fig.layout.yaxis2.title["text"] == "trend"

    assert fig.layout.title["text"] == title
    assert fig.layout.title["x"] == 0.5

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
        assert fig.layout.title["x"] == 0.5


def test_autoreg(daily_data):
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

    model = SilverkiteEstimator(
        uncertainty_dict=uncertainty_dict,
        autoreg_dict="auto")
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"][:20]
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    trained_model = model.model_dict
    pred_cols = trained_model["pred_cols"]

    expected_autoreg_terms = {
        "y_lag30", "y_lag31", "y_lag32",
        "y_avglag_35_42_49", "y_avglag_30_to_36", "y_avglag_37_to_43"}
    assert expected_autoreg_terms.issubset(pred_cols)

    predictions = model.predict(test_df)
    expected_forecast_cols = {
        "ts", "y", cst.QUANTILE_SUMMARY_COL, "err_std", "forecast_lower",
        "forecast_upper"}

    assert expected_forecast_cols.issubset(list(model.forecast.columns))

    actual = test_df[cst.VALUE_COL]
    forecast_lower = predictions[cst.PREDICTED_LOWER_COL]
    forecast_upper = predictions[cst.PREDICTED_UPPER_COL]
    calc_pred_coverage = 100 * (
        (actual <= forecast_upper)
        & (actual >= forecast_lower)
        ).mean()
    assert round(calc_pred_coverage) >= 75, "forecast coverage is incorrect"

    # Simulation based, default forecast horizon
    model = SilverkiteEstimator(
        uncertainty_dict=uncertainty_dict,
        autoreg_dict="auto",
        simulation_based=True)
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"][:20]
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    trained_model = model.model_dict
    pred_cols = trained_model["pred_cols"]

    expected_autoreg_terms = {
         "y_lag1", "y_lag2", "y_lag3", "y_avglag_7_14_21", "y_avglag_1_to_7", "y_avglag_8_to_14"}
    assert expected_autoreg_terms.issubset(pred_cols)

    # Passes forecast horizon of 10
    model = SilverkiteEstimator(
        uncertainty_dict=uncertainty_dict,
        autoreg_dict="auto",
        forecast_horizon=10)
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"][:20]
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    trained_model = model.model_dict
    pred_cols = trained_model["pred_cols"]

    expected_autoreg_terms = {
        "y_lag10", "y_lag11", "y_lag12", "y_avglag_14_21_28", "y_avglag_10_to_16", "y_avglag_17_to_23"}
    assert expected_autoreg_terms.issubset(pred_cols)

    # Passes forecast horizon of 10, and simulation-based True
    model = SilverkiteEstimator(
        uncertainty_dict=uncertainty_dict,
        autoreg_dict="auto",
        forecast_horizon=10,
        simulation_based=True)
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"][:20]
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    trained_model = model.model_dict
    pred_cols = trained_model["pred_cols"]

    expected_autoreg_terms = {
        "y_lag1", "y_lag2", "y_lag3", "y_avglag_7_14_21", "y_avglag_1_to_7", "y_avglag_8_to_14"}
    assert expected_autoreg_terms.issubset(pred_cols)


def test_lagged_regressors(daily_data_with_reg, params):
    """Tests a basic model with lagged regressors"""
    train_df = daily_data_with_reg["train_df"]
    test_df = daily_data_with_reg["test_df"][:20]

    # default forecast horizon, no uncertainty
    model = SilverkiteEstimator(
        lagged_regressor_dict=params["lagged_regressor_dict"])
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    trained_model = model.model_dict
    assert trained_model["lagged_regressor_dict"] == params["lagged_regressor_dict"]
    pred_cols = trained_model["pred_cols"]
    expected_lagged_regression_terms = {
        "regressor1_lag1",
        "regressor1_lag2",
        "regressor1_lag3",
        "regressor1_avglag_7_14_21",
        "regressor1_avglag_8_to_14",
        "regressor2_lag35",
        "regressor2_avglag_35_42_49",
        "regressor2_avglag_30_to_36"
    }
    assert expected_lagged_regression_terms.issubset(pred_cols)

    model.predict(test_df)
    expected_forecast_cols = {"ts", "y"}
    assert expected_forecast_cols.issubset(list(model.forecast.columns))

    # Passes forecast horizon of 10, and uncertainty dict
    model = SilverkiteEstimator(
        uncertainty_dict=params["uncertainty_dict"],
        lagged_regressor_dict=params["lagged_regressor_dict"],
        forecast_horizon=10)
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    trained_model = model.model_dict
    pred_cols = trained_model["pred_cols"]
    expected_lagged_regression_terms = {
        "regressor1_lag1",
        "regressor1_lag2",
        "regressor1_lag3",
        "regressor1_avglag_7_14_21",
        "regressor1_avglag_8_to_14",
        "regressor2_lag35",
        "regressor2_avglag_35_42_49",
        "regressor2_avglag_30_to_36"
    }
    assert expected_lagged_regression_terms.issubset(pred_cols)

    model.predict(test_df)
    expected_forecast_cols = {"ts", "y", cst.QUANTILE_SUMMARY_COL, "err_std",
                              "forecast_lower", "forecast_upper"}
    assert expected_forecast_cols.issubset(list(model.forecast.columns))


def test_various_predictor_settings(daily_data_with_reg, params):
    """Tests a basic model with lagged regressors"""
    train_df = daily_data_with_reg["train_df"]
    test_df = daily_data_with_reg["test_df"][:20]

    # fit default model
    model = SilverkiteEstimator()
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    trained_model = model.model_dict
    assert "ct1" in trained_model["pred_cols"]

    # drop "ct1"
    model = SilverkiteEstimator(
        drop_pred_cols="ct1")
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    trained_model = model.model_dict
    assert "ct1" not in trained_model["pred_cols"]

    # fit a model with explicit predictors
    model = SilverkiteEstimator(
        explicit_pred_cols=["ct1", "ct2"],
        uncertainty_dict=params["uncertainty_dict"])
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert model.forecast is None

    trained_model = model.model_dict
    assert set(trained_model["pred_cols"]) == set(["ct1", "ct2"])

    model.predict(test_df)
    expected_forecast_cols = {
        "ts", "y", cst.QUANTILE_SUMMARY_COL, "err_std", "forecast_lower",
        "forecast_upper"}
    assert expected_forecast_cols.issubset(list(model.forecast.columns))


def test_validate_fs_components_df():
    """Tests validate_fs_components_df function"""
    model = SilverkiteEstimator()
    with pytest.warns(None) as record:
        fs_components_df = pd.DataFrame({
            "name": ["tod", "tow"],
            "period": [24.0, 7.0],
            "order": [12, 4],
            "seas_names": ["daily", "weekly"]})
        model.validate_fs_components_df(fs_components_df)
        assert len(record) == 0

    fs_cols_not_found = ["order", "seas_names"]
    with pytest.raises(ValueError) as record:
        fs_components_df = pd.DataFrame({
            "name": ["tod", "tow"],
            "period": [24.0, 7.0]})
        model.validate_fs_components_df(fs_components_df)
        assert (f"fs_components_df is missing the following columns: {fs_cols_not_found}"
                in record[0].message.args[0])

    with pytest.raises(ValueError, match="Found multiple rows in fs_components_df with the same "
                                         "`names` and `seas_names`. Make sure these are unique."):
        fs_components_df = pd.DataFrame({
            "name": ["tod", "tow", "tow"],
            "period": [24.0, 7.0, 10.0],
            "order": [12, 4, 3],
            "seas_names": ["daily", "weekly", "weekly"]})
        model.validate_fs_components_df(fs_components_df)


def test_past_df(daily_data):
    """Tests ``past_df`` is passed."""
    model = SilverkiteEstimator(past_df=daily_data["df"])
    assert model.past_df.equals(daily_data["df"])


def test_x_mat_in_predict(daily_data):
    """Tests to check if prediction phase design matrix is returned."""
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 10,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    model = SilverkiteEstimator(
        uncertainty_dict=uncertainty_dict,
        fs_components_df=None,
        extra_pred_cols=["ct1", "C(dow == 1)"],
        autoreg_dict=None)

    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"][:20]
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)

    pred_df = model.predict(test_df)
    cols = ["ts", cst.QUANTILE_SUMMARY_COL, "err_std", "forecast_lower", "forecast_upper"]
    assert_equal(model.forecast[cols], pred_df[cols])
    assert (model.forecast["y"].values == pred_df["forecast"].values).all()

    forecast_x_mat = model.forecast_x_mat
    assert list(forecast_x_mat.columns) == [
        "Intercept", "C(dow == 1)[T.True]", "ct1"]
    assert len(forecast_x_mat) == len(pred_df)

    assert len(model.forecast_x_mat) == 20

    # Predicts with a smaller length
    pred_df = model.predict(test_df[:5])
    assert len(model.forecast_x_mat) == 5
    assert len(pred_df) == 5
