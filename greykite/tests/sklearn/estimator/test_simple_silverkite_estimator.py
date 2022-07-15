import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

import greykite.common.constants as cst
from greykite.algo.forecast.silverkite.forecast_simple_silverkite import SimpleSilverkiteForecast
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.data_loader import DataLoader
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
        "lagged_regressor_dict": lagged_regressor_dict,
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
        "lagged_regressor_dict": lagged_regressor_dict,
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
        {"ts", "y", cst.QUANTILE_SUMMARY_COL, "err_std", "forecast_lower", "forecast_upper"}
    assert expected_forecast_cols.issubset(list(model.forecast.columns))

    actual = daily_data["test_df"][cst.VALUE_COL]
    forecast_lower = predictions[cst.PREDICTED_LOWER_COL]
    forecast_upper = predictions[cst.PREDICTED_UPPER_COL]
    calc_pred_coverage = 100 * (
            (actual <= forecast_upper)
            & (actual >= forecast_lower)
    ).mean()
    assert round(calc_pred_coverage) == 95, "forecast coverage is incorrect"


def test_normalize_method(daily_data):
    """Runs a basic model with normalize method"""
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
        normalize_method="statistical",
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
        {"ts", "y", cst.QUANTILE_SUMMARY_COL, "err_std", "forecast_lower", "forecast_upper"}
    assert expected_forecast_cols.issubset(list(model.forecast.columns))

    actual = daily_data["test_df"][cst.VALUE_COL]
    forecast_lower = predictions[cst.PREDICTED_LOWER_COL]
    forecast_upper = predictions[cst.PREDICTED_UPPER_COL]
    calc_pred_coverage = 100 * (
            (actual <= forecast_upper)
            & (actual >= forecast_lower)
    ).mean()
    assert round(calc_pred_coverage) == 95, "forecast coverage is incorrect"
    assert model.model_dict["normalize_method"] == "statistical"


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
        assert fig.layout.title["x"] == 0.5
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


def test_past_df(daily_data):
    """Tests ``past_df`` is passed."""
    model = SimpleSilverkiteEstimator(past_df=daily_data["df"])
    assert model.past_df.equals(daily_data["df"])


def test_x_mat_in_predict(daily_data):
    """Tests to check if prediction phase design matrix is returned."""
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"][:20]

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
        fit_algorithm_dict={
            "fit_algorithm": "linear"
        },
        yearly_seasonality=False,
        quarterly_seasonality=False,
        monthly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        holiday_lookup_countries=None,
        extra_pred_cols=["C(dow == 1)", "ct1"],
        uncertainty_dict=uncertainty_dict
    )

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


def test_uncertainty_with_nonstandard_cols(daily_data):
    model = SimpleSilverkiteEstimator(coverage=0.95)
    df = daily_data["df"].rename(columns={
        cst.TIME_COL: "t",
        cst.VALUE_COL: "z"
    })
    model.fit(
        df,
        time_col="t",
        value_col="z"
    )
    pred_df = model.predict(df)
    assert cst.PREDICTED_LOWER_COL in pred_df
    assert cst.PREDICTED_UPPER_COL in pred_df


def test_auto_config():
    df = DataLoader().load_peyton_manning()
    df[cst.TIME_COL] = pd.to_datetime(df[cst.TIME_COL])
    model = SimpleSilverkiteEstimator(
        forecast_horizon=7,
        auto_holiday=True,
        holidays_to_model_separately="auto",
        holiday_lookup_countries="auto",
        holiday_pre_num_days=2,
        holiday_post_num_days=2,
        daily_event_df_dict=dict(
            custom_event=pd.DataFrame({
                EVENT_DF_DATE_COL: pd.to_datetime(["2010-03-03", "2011-03-03", "2012-03-03"]),
                EVENT_DF_LABEL_COL: "threethree"
            })
        ),
        auto_growth=True,
        growth_term="quadratic",
        changepoints_dict=dict(
            method="uniform",
            n_changepoints=2
        ),
        auto_seasonality=True,
        yearly_seasonality=0,
        quarterly_seasonality="auto",
        monthly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=5
    )
    model.fit(
        X=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL
    )
    # Seasonality is overridden by auto seasonality.
    assert model.model_dict["fs_components_df"][["name", "period", "order", "seas_names"]].equals(pd.DataFrame({
        "name": ["tow", "toq", "ct1"],
        "period": [7.0, 1.0, 1.0],
        "order": [3, 1, 6],
        "seas_names": ["weekly", "quarterly", "yearly"]
    }))
    # Growth is overridden by auto growth.
    assert "ct1" in model.model_dict["x_mat"].columns
    assert model.model_dict["changepoints_dict"]["method"] == "custom"
    # Holidays is overridden by auto seasonality.
    assert len(model.model_dict["daily_event_df_dict"]) == 198
    assert "custom_event" in model.model_dict["daily_event_df_dict"]
    assert "China_Chinese New Year" in model.model_dict["daily_event_df_dict"]


def test_quantile_regression_uncertainty_model():
    """Tests the quantile regression uncertainty model."""
    df = DataLoader().load_peyton_manning().iloc[-365:].reset_index(drop=True)
    df[cst.TIME_COL] = pd.to_datetime(df[cst.TIME_COL])

    # Residual based
    model = SimpleSilverkiteEstimator(
        coverage=0.99,
        uncertainty_dict=dict(
            uncertainty_method="quantile_regression",
            params=dict(
                is_residual_based=True
            )
        )
    )
    model.fit(
        X=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL
    )
    assert model.uncertainty_model is not None
    assert len(model.uncertainty_model.models) == 2
    df_fut = pd.DataFrame({
        "ts": pd.date_range(df["ts"].max(), freq="D", periods=15),
        "y": np.nan
    }).iloc[1:].reset_index(drop=True)
    pred = model.predict(df_fut)
    assert cst.PREDICTED_LOWER_COL in pred.columns
    assert cst.PREDICTED_UPPER_COL in pred.columns
    assert all(pred[cst.PREDICTED_LOWER_COL] <= pred[cst.PREDICTED_UPPER_COL])

    # Not residual based
    model = SimpleSilverkiteEstimator(
        coverage=0.95,
        uncertainty_dict=dict(
            uncertainty_method="quantile_regression",
            params=dict(
                is_residual_based=False
            )
        )
    )
    model.fit(
        X=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL
    )
    assert model.uncertainty_model is not None
    assert len(model.uncertainty_model.models) == 2
    pred = model.predict(df)
    assert cst.PREDICTED_LOWER_COL in pred.columns
    assert cst.PREDICTED_UPPER_COL in pred.columns
    assert all(pred[cst.PREDICTED_LOWER_COL].round(5) <= pred[cst.PREDICTED_UPPER_COL].round(5))
