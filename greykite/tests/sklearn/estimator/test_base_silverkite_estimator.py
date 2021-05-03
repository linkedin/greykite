import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from testfixtures import LogCapture

import greykite.common.constants as cst
from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.common.data_loader import DataLoader
from greykite.common.features.timeseries_features import convert_date_to_continuous_time
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import daily_data_reg
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.sklearn.estimator.base_silverkite_estimator import BaseSilverkiteEstimator
from greykite.sklearn.estimator.testing_utils import params_components


@pytest.fixture
def params():
    autoreg_dict = {
        "lag_dict": {"orders": [7]},
        "agg_lag_dict": {
            "orders_list": [[7, 7 * 2, 7 * 3]],
            "interval_list": [(7, 7 * 2)]},
        "series_na_fill_func": lambda s: s.bfill().ffill()}
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
        "fit_algorithm": "sgd",
        "fit_algorithm_params": {"alpha": 0.1},
        "daily_event_df_dict": None,
        "changepoints_dict": None,
        "fs_components_df": pd.DataFrame({
            "name": ["tow"],
            "period": [7.0],
            "order": [3],
            "seas_names": [None]}),
        "autoreg_dict": autoreg_dict,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "uncertainty_dict": uncertainty_dict
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


@pytest.fixture
def df_pt():
    """fetches the Peyton Manning pageview data"""
    dl = DataLoader()
    return dl.load_peyton_manning()


def test_init(params):
    """Checks if parameters are passed to BaseSilverkiteEstimator correctly"""
    coverage = 0.95
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow"],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    model = BaseSilverkiteEstimator(
        score_func=mean_squared_error,
        coverage=coverage,
        null_model_params=None,
        uncertainty_dict=uncertainty_dict)

    assert model.score_func == mean_squared_error
    assert model.coverage == coverage
    assert model.null_model_params is None
    assert model.uncertainty_dict == uncertainty_dict

    assert model.model_dict is None
    assert model.pred_cols is None
    assert model.feature_cols is None
    assert model.df is None
    assert model.coef_ is None


def test_null_model(X):
    """Checks null model"""
    model = BaseSilverkiteEstimator(null_model_params={
        "strategy": "quantile",
        "constant": None,
        "quantile": 0.8})

    model.fit(X)
    y = np.repeat(2.0, X.shape[0])
    null_score = model.null_model.score(X, y=y)
    assert null_score == mean_squared_error(y, np.repeat(9.0, X.shape[0]))

    # tests if different score function gets propagated to null model
    model = BaseSilverkiteEstimator(
        score_func=mean_absolute_error,
        null_model_params={"strategy": "quantile",
                           "constant": None,
                           "quantile": 0.8})
    model.fit(X)
    y = np.repeat(2.0, X.shape[0])
    null_score = model.null_model.score(X, y=y)
    assert null_score == mean_absolute_error(y, np.repeat(9.0, X.shape[0]))
    # checks that `df` is set
    assert_equal(X, model.df)


def test_fit_predict(daily_data):
    """Checks fit and predict function with null model"""
    model = BaseSilverkiteEstimator(null_model_params={"strategy": "mean"})
    train_df = daily_data["train_df"]
    test_df = daily_data["test_df"]
    assert model.last_predicted_X_ is None
    assert model.cached_predictions_ is None

    with pytest.raises(
            NotFittedError,
            match="Call `fit` before calling `predict`."):
        model.predict(train_df)

    # Every subclass `fit` follows these steps
    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    # Checks that `df` is set, but other variables aren't
    assert_equal(model.df, train_df)
    assert model.pred_cols is None
    assert model.feature_cols is None
    assert model.coef_ is None

    with pytest.raises(ValueError, match="Must set `self.model_dict` before calling this function."):
        model.finish_fit()

    silverkite = SilverkiteForecast()
    model.model_dict = silverkite.forecast(
        df=train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        origin_for_time_vars=None,
        extra_pred_cols=None,
        train_test_thresh=None,
        training_fraction=None,
        fit_algorithm="linear",
        fit_algorithm_params=None,
        daily_event_df_dict=None,
        changepoints_dict=None,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 3, 5],
            "seas_names": ["daily", "weekly", "yearly"]}),
        autoreg_dict=None,
        min_admissible_value=None,
        max_admissible_value=None,
        uncertainty_dict=None
    )

    with pytest.raises(
            NotFittedError,
            match="Subclass must call `finish_fit` inside the `fit` method."):
        model.predict(train_df)
    assert model.last_predicted_X_ is not None  # attempted prediction
    assert model.cached_predictions_ is None

    model.finish_fit()
    # Checks that other variables are set
    assert_equal(model.pred_cols, model.model_dict["pred_cols"])
    assert_equal(model.feature_cols, model.model_dict["x_mat"].columns)
    assert_equal(model.coef_, pd.DataFrame(
        model.model_dict["ml_model"].coef_,
        index=model.feature_cols))

    # Predicts on a new dataset
    with LogCapture(cst.LOGGER_NAME) as log_capture:
        predicted = model.predict(test_df)
        assert_equal(model.last_predicted_X_, test_df)
        assert_equal(model.cached_predictions_, predicted)
        log_capture.check()  # no log messages (not using cached predictions)

    # Uses cached predictions
    with LogCapture(cst.LOGGER_NAME) as log_capture:
        assert_equal(model.predict(test_df), predicted)
        log_capture.check(
            (cst.LOGGER_NAME, "DEBUG", "Returning cached predictions.")
        )

    # Predicts on a different dataset
    with LogCapture(cst.LOGGER_NAME) as log_capture:
        predicted = model.predict(train_df)
        assert_equal(model.last_predicted_X_, train_df)
        assert_equal(model.cached_predictions_, predicted)
        log_capture.check()  # no log messages (not using cached predictions)

    # .fit() clears the cached result
    model.fit(train_df, time_col=cst.TIME_COL, value_col=cst.VALUE_COL)
    assert model.last_predicted_X_ is None
    assert model.cached_predictions_ is None


def test_score_function(daily_data_with_reg):
    """Checks score function without null model, with regressors"""
    model = BaseSilverkiteEstimator()
    train_df = daily_data_with_reg["train_df"]
    test_df = daily_data_with_reg["test_df"]

    # every subclass `fit` follows these steps
    model.fit(
        X=train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    silverkite = SilverkiteForecast()
    model.model_dict = silverkite.forecast(
        df=train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        origin_for_time_vars=None,
        extra_pred_cols=["ct1", "regressor1", "regressor2"],
        train_test_thresh=None,
        training_fraction=None,
        fit_algorithm="linear",
        fit_algorithm_params=None,
        daily_event_df_dict=None,
        changepoints_dict=None,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 3, 5],
            "seas_names": ["daily", "weekly", "yearly"]}),
        autoreg_dict=None,
        min_admissible_value=None,
        max_admissible_value=None,
        uncertainty_dict=None
    )
    model.finish_fit()

    score = model.score(test_df, test_df[cst.VALUE_COL])
    pred_df = model.predict(test_df)
    assert list(pred_df.columns) == [cst.TIME_COL, cst.PREDICTED_COL]
    assert score == pytest.approx(mean_squared_error(
        pred_df[cst.PREDICTED_COL],
        test_df[cst.VALUE_COL]))
    assert score == pytest.approx(4.6, rel=1e-1)


def test_set_uncertainty_dict(daily_data):
    """Tests __set_uncertainty_dict"""
    train_df = daily_data["train_df"]

    # both provided
    coverage = 0.95
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 20,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    model = BaseSilverkiteEstimator(
        coverage=coverage,
        uncertainty_dict=uncertainty_dict)
    model.fit(train_df)
    expected_dict = uncertainty_dict
    assert_equal(model.uncertainty_dict, expected_dict)
    assert_equal(model.coverage, coverage)

    # only coverage provided
    coverage = 0.90
    uncertainty_dict = None
    model = BaseSilverkiteEstimator(
        coverage=coverage,
        uncertainty_dict=uncertainty_dict)
    model.fit(train_df)
    expected_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.05, 0.95],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    assert_equal(model.uncertainty_dict, expected_dict)
    assert_equal(model.coverage, coverage)

    # both missing
    coverage = None
    uncertainty_dict = None
    model = BaseSilverkiteEstimator(
        coverage=coverage,
        uncertainty_dict=uncertainty_dict)
    model.fit(train_df)
    expected_dict = None
    assert_equal(model.uncertainty_dict, expected_dict)
    assert_equal(model.coverage, None)

    # only uncertainty provided
    coverage = None
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.05, 0.95],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    model = BaseSilverkiteEstimator(
        coverage=coverage,
        uncertainty_dict=uncertainty_dict)
    model.fit(train_df)
    expected_dict = uncertainty_dict
    assert_equal(model.uncertainty_dict, expected_dict)
    assert_equal(model.coverage, 0.90)


def test_summary(daily_data):
    """Checks summary function returns without error"""
    model = BaseSilverkiteEstimator()
    train_df = daily_data["train_df"]
    model.summary()

    model.fit(
        train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    model.summary()


def test_silverkite_with_components_daily_data():
    """Tests get_components, plot_components, plot_trend,
    plot_seasonalities with daily data and missing input values.
    """
    daily_data = generate_df_with_reg_for_tests(
        freq="D",
        periods=20,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)
    train_df = daily_data["train_df"].copy()
    train_df.loc[[2, 4, 7], cst.VALUE_COL] = np.nan  # creates missing values

    params_daily = params_components()  # SilverkiteEstimator parameters
    # converts into parameters for `forecast_silverkite`
    coverage = params_daily.pop("coverage")
    # removes daily seasonality terms
    params_daily["fs_components_df"] = pd.DataFrame({
        "name": ["tow", "ct1"],
        "period": [7.0, 1.0],
        "order": [4, 5],
        "seas_names": ["weekly", "yearly"]})

    model = BaseSilverkiteEstimator(
        coverage=coverage,
        uncertainty_dict=params_daily["uncertainty_dict"])

    with pytest.raises(
            NotFittedError,
            match="Call `fit` before calling `plot_components`."):
        model.plot_components()

    with pytest.warns(Warning):
        # suppress warnings from conf_interval.py and sklearn
        # a subclass's fit() method will have these steps
        model.fit(
            X=train_df,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL)
        silverkite = SilverkiteForecast()
        model.model_dict = silverkite.forecast(
            df=train_df,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            **params_daily)
        model.finish_fit()

    # Tests plot_components
    with pytest.warns(Warning) as record:
        title = "Custom component plot"
        model._set_silverkite_diagnostics_params()
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
        assert f"The following components have not been specified in the model: " \
               f"{{'DUMMY'}}, plotting the rest." in record[0].message.args[0]

    # Missing component error
    with pytest.raises(
            ValueError,
            match="None of the provided components have been specified in the model."):
        model.plot_components(names=["DUMMY"])

    # Tests plot_trend
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

    # Tests plot_seasonalities
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

    # Component plot error if `fit_algorithm` is "rf" or "gradient_boosting"
    params_daily["fit_algorithm"] = "rf"
    model = BaseSilverkiteEstimator(
        coverage=coverage,
        uncertainty_dict=params_daily["uncertainty_dict"])
    with pytest.warns(Warning):
        # suppress warnings from conf_interval.py and sklearn
        # a subclass's fit() method will have these steps
        model.fit(
            X=train_df,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL)
        model.model_dict = silverkite.forecast(
            df=train_df,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            **params_daily)
        model.finish_fit()
    assert model.coef_ is None
    with pytest.raises(
            NotImplementedError,
            match="Component plot has only been implemented for additive linear models."):
        model.plot_components()

    with pytest.raises(
            NotImplementedError,
            match="Component plot has only been implemented for additive linear models."):
        model.plot_trend()

    with pytest.raises(
            NotImplementedError,
            match="Component plot has only been implemented for additive linear models."):
        model.plot_seasonalities()


def test_silverkite_with_components_hourly_data():
    """Tests get_components, plot_components, plot_trend,
    plot_seasonalities with hourly data
    """
    hourly_data = generate_df_with_reg_for_tests(
        freq="H",
        periods=24 * 4,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)
    train_df = hourly_data.get("train_df").copy()
    params_hourly = params_components()

    # converts into parameters for `forecast_silverkite`
    coverage = params_hourly.pop("coverage")
    model = BaseSilverkiteEstimator(
        coverage=coverage,
        uncertainty_dict=params_hourly["uncertainty_dict"])
    model.fit(
        X=train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    silverkite = SilverkiteForecast()
    model.model_dict = silverkite.forecast(
        df=train_df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        **params_hourly)
    model.finish_fit()

    # Test plot_components
    with pytest.warns(Warning) as record:
        title = "Custom component plot"
        fig = model.plot_components(names=["trend", "DAILY_SEASONALITY", "DUMMY"], title=title)
        expected_rows = 3 + 1  # includes changepoints
        assert len(fig.data) == expected_rows
        assert [fig.data[i].name for i in range(expected_rows)] == \
               [cst.VALUE_COL, "trend", "DAILY_SEASONALITY", "trend change point"]

        assert fig.layout.xaxis.title["text"] == cst.TIME_COL
        assert fig.layout.xaxis2.title["text"] == cst.TIME_COL
        assert fig.layout.xaxis3.title["text"] == "Hour of day"

        assert fig.layout.yaxis.title["text"] == cst.VALUE_COL
        assert fig.layout.yaxis2.title["text"] == "trend"
        assert fig.layout.yaxis3.title["text"] == "daily"

        assert fig.layout.title["text"] == title
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

    # Test plot_seasonalities
    with pytest.warns(Warning):
        # suppresses the warning on seasonalities removed
        title = "Custom seasonality plot"
        fig = model.plot_seasonalities(title=title)
        expected_rows = 4
        assert len(fig.data) == expected_rows
        assert [fig.data[i].name for i in range(expected_rows)] == \
               [cst.VALUE_COL, "DAILY_SEASONALITY", "WEEKLY_SEASONALITY", "YEARLY_SEASONALITY"]

        assert fig.layout.xaxis.title["text"] == cst.TIME_COL
        assert fig.layout.xaxis2.title["text"] == "Hour of day"
        assert fig.layout.xaxis3.title["text"] == "Day of week"
        assert fig.layout.xaxis4.title["text"] == "Time of year"

        assert fig.layout.yaxis.title["text"] == cst.VALUE_COL
        assert fig.layout.yaxis2.title["text"] == "daily"
        assert fig.layout.yaxis3.title["text"] == "weekly"
        assert fig.layout.yaxis4.title["text"] == "yearly"

        assert fig.layout.title["text"] == title


def test_plot_trend_changepoint_detection(df_pt):
    model = BaseSilverkiteEstimator()
    model.fit(
        X=df_pt,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    params = {
        "changepoints_dict": {"method": "auto"}}
    silverkite = SilverkiteForecast()
    model.model_dict = silverkite.forecast(
        df=df_pt,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        **params)
    model.finish_fit()
    fig = model.plot_trend_changepoint_detection()
    assert fig is not None
    assert fig.layout.title["text"] == "Timeseries Plot with detected trend change points"
    assert fig.layout.yaxis.title["text"] == cst.VALUE_COL
    assert fig.layout.xaxis.title["text"] == "Dates"
    # tests given parameters
    fig = model.plot_trend_changepoint_detection(
        dict(trend_change=False))
    assert fig is not None
    assert fig.layout.title["text"] == "Timeseries Plot"
    assert fig.layout.yaxis.title["text"] == cst.VALUE_COL
    assert fig.layout.xaxis.title["text"] == "Dates"


def test_model_summary(df_pt):
    model = BaseSilverkiteEstimator()
    model.fit(
        X=df_pt.iloc[:100],  # speeds up
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    params = {
        "fit_algorithm": "linear",
        "training_fraction": 0.8}
    silverkite = SilverkiteForecast()
    model.model_dict = silverkite.forecast(
        df=df_pt.iloc[:100],
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        **params)
    model.finish_fit()
    summary = model.summary()
    summary.__str__()
    summary.__repr__()
    assert summary is not None


def test_pred_category(df_pt):
    model = BaseSilverkiteEstimator()
    # property is not available without fitting.
    with pytest.raises(
            NotFittedError,
            match="Must fit before getting predictor category."):
        print(model.pred_category)
    model.fit(
        X=df_pt.iloc[:100],  # speeds up
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    params = {
        "fit_algorithm": "linear",
        "training_fraction": 0.8,
        "extra_pred_cols": ["ct1", "x", "x:ct1"]}
    df_pt["x"] = np.random.randn(df_pt.shape[0])
    silverkite = SilverkiteForecast()
    model.model_dict = silverkite.forecast(
        df=df_pt.iloc[:100],
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        **params)
    model.extra_pred_cols = ["ct1", "x", "x:ct1"]  # set in subclass initialization
    # _pred_category is None before trying to access pred_category
    assert model._pred_category is None
    model.finish_fit()
    pred_category = model.pred_category
    # _pred_category is updated after trying to access pred_category
    assert model._pred_category is not None
    assert pred_category["intercept"] == ["Intercept"]
    assert pred_category["time_features"] == ["ct1", "x:ct1"]
    assert pred_category["event_features"] == []
    assert pred_category["trend_features"] == ["ct1", "x:ct1"]
    assert pred_category["seasonality_features"] == ["sin1_tod_daily",
                                                     "cos1_tod_daily",
                                                     "sin2_tod_daily",
                                                     "cos2_tod_daily",
                                                     "sin3_tod_daily",
                                                     "cos3_tod_daily",
                                                     "sin1_tow_weekly",
                                                     "cos1_tow_weekly",
                                                     "sin2_tow_weekly",
                                                     "cos2_tow_weekly",
                                                     "sin3_tow_weekly",
                                                     "cos3_tow_weekly",
                                                     "sin1_toy_yearly",
                                                     "cos1_toy_yearly",
                                                     "sin2_toy_yearly",
                                                     "cos2_toy_yearly",
                                                     "sin3_toy_yearly",
                                                     "cos3_toy_yearly",
                                                     "sin4_toy_yearly",
                                                     "cos4_toy_yearly",
                                                     "sin5_toy_yearly",
                                                     "cos5_toy_yearly"]
    assert pred_category["lag_features"] == []
    assert pred_category["regressor_features"] == ["x", "x:ct1"]
    assert pred_category["interaction_features"] == ["x:ct1"]
