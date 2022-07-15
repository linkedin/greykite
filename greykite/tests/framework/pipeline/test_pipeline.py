import datetime
import math
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import randint as sp_randint
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import ADJUSTMENT_DELTA_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import FRACTION_OUTSIDE_TOLERANCE
from greykite.common.constants import METRIC_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import QUANTILE_SUMMARY_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.constants import R2_null_model_score
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import add_finite_filter_to_scorer
from greykite.common.evaluation import add_preaggregation_to_scorer
from greykite.common.python_utils import assert_equal
from greykite.common.python_utils import unique_elements_in_list
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.constants import FRACTION_OUTSIDE_TOLERANCE_NAME
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries
from greykite.framework.pipeline.pipeline import forecast_pipeline
from greykite.framework.templates.prophet_template import ProphetTemplate
from greykite.framework.utils.framework_testing_utils import check_forecast_pipeline_result
from greykite.framework.utils.framework_testing_utils import mock_pipeline
from greykite.sklearn.estimator.null_model import DummyEstimator
from greykite.sklearn.estimator.prophet_estimator import ProphetEstimator
from greykite.sklearn.estimator.silverkite_estimator import SilverkiteEstimator
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator
from greykite.sklearn.transform.column_selector import ColumnSelector
from greykite.sklearn.transform.drop_degenerate_transformer import DropDegenerateTransformer
from greykite.sklearn.transform.dtype_column_selector import DtypeColumnSelector
from greykite.sklearn.transform.normalize_transformer import NormalizeTransformer
from greykite.sklearn.transform.null_transformer import NullTransformer
from greykite.sklearn.transform.pandas_feature_union import PandasFeatureUnion
from greykite.sklearn.transform.zscore_outlier_transformer import ZscoreOutlierTransformer


try:
    import prophet  # noqa
except ModuleNotFoundError:
    pass


@pytest.fixture
def df():
    """8 months of daily data"""
    data = generate_df_for_tests(freq="D", periods=30*8)
    df = data["df"][[TIME_COL, VALUE_COL]]
    return df


@pytest.fixture
def df_reg():
    """100 days of hourly data with regressors"""
    data = generate_df_with_reg_for_tests(
        freq="H",
        periods=24*100,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols = ["regressor1", "regressor2", "regressor_bool", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols
    df = data["df"][keep_cols]
    return df


def get_dummy_pipeline(
        include_preprocessing=False,
        regressor_cols=None,
        lagged_regressor_cols=None):
    """Returns a ``pipeline`` argument to ``forecast_pipeline``
    that uses ``DummyEstimator`` to make it easy to unit test
    ``forecast_pipeline``.

    Parameters
    ----------
    include_preprocessing : `bool`, default False
        If True, includes preprocessing steps.
    regressor_cols : `list` [`str`] or None, default None
        Names of regressors in ``df`` passed to ``forecast_pipeline``.
        Only used if ``include_preprocessing=True``.
    lagged_regressor_cols : `list` [`str`] or None, default None
        Names of lagged regressor columns in ``df`` passed to ``forecast_pipeline``.
        Only used if ``include_preprocessing=True``.

    Returns
    -------
    pipeline : `sklearn.pipeline.Pipeline`
        sklearn Pipeline for univariate forecasting.
    """
    if regressor_cols is None:
        regressor_cols = []
    if lagged_regressor_cols is None:
        lagged_regressor_cols = []
    all_reg_cols = unique_elements_in_list(regressor_cols + lagged_regressor_cols)
    steps = []
    if include_preprocessing:
        steps += [
            ("input", PandasFeatureUnion([
                ("date", Pipeline([
                    ("select_date", ColumnSelector([TIME_COL]))  # leaves time column unmodified
                ])),
                ("response", Pipeline([  # applies outlier and null transformation to value column
                    ("select_val", ColumnSelector([VALUE_COL])),
                    ("outlier", ZscoreOutlierTransformer()),
                    ("null", NullTransformer())
                ])),
                ("regressors_numeric", Pipeline([
                    ("select_reg", ColumnSelector(all_reg_cols)),
                    ("select_reg_numeric", DtypeColumnSelector(include="number")),
                    ("outlier", ZscoreOutlierTransformer()),
                    ("normalize", NormalizeTransformer()),  # no normalization by default
                    ("null", NullTransformer())
                ])),
                ("regressors_other", Pipeline([
                    ("select_reg", ColumnSelector(all_reg_cols)),
                    ("select_reg_non_numeric", DtypeColumnSelector(exclude="number"))
                ]))
            ])),
            ("degenerate", DropDegenerateTransformer()),  # default `drop_degenerate=False`
        ]
    steps += [
        ("estimator", DummyEstimator())  # predicts a constant
    ]
    return Pipeline(steps)


def test_validate_pipeline_input():
    """Tests for validate_pipeline_input function"""
    df = pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=1000, freq="D"),
        VALUE_COL: np.arange(1000),
        "regressor1": np.random.normal(size=1000),
        "regressor2": np.random.normal(size=1000),
        "regressor3": np.random.normal(size=1000),
    })
    hyperparameter_grid = {
        "estimator__weekly_seasonality": [True],
        "estimator__daily_seasonality": [True, False],
        "estimator__n_changepoints": [0],  # to speed up test case, remove for better fit
        "estimator__uncertainty_samples": [10],  # to speed up test case
        "estimator__add_regressor_dict": [{
            "regressor1": {
                "prior_scale": 10,
                "standardize": True,
                "mode": "additive"
            },
            "regressor2": {
                "prior_scale": 15,
                "standardize": False,
                "mode": "additive"
            },
            "regressor3": {}
        }]
    }

    # some parameters can be None
    result = mock_pipeline(
        df,
        hyperparameter_grid=None,
        coverage=None,
        relative_error_tolerance=None)
    assert result["hyperparameter_grid"] is None
    assert result["coverage"] is None
    assert result["relative_error_tolerance"] is None

    result = mock_pipeline(df, hyperparameter_grid=hyperparameter_grid)
    assert result["hyperparameter_budget"] is None
    assert result["freq"] is None
    assert result["forecast_horizon"] == 30
    assert result["test_horizon"] == 30
    assert result["periods_between_train_test"] == 0
    assert result["cv_horizon"] == 30
    assert result["cv_use_most_recent_splits"] is False
    assert result["train_end_date"] is None
    assert result["relative_error_tolerance"] is None
    assert hyperparameter_grid == result["hyperparameter_grid"]

    with pytest.raises(ValueError, match="coverage must be between 0 and 1"):
        mock_pipeline(df, coverage=-1)

    with pytest.raises(ValueError, match="relative_error_tolerance must non-negative"):
        mock_pipeline(df, relative_error_tolerance=-1)

    with pytest.raises(ValueError, match="forecast_horizon must be >= 1"):
        mock_pipeline(df, forecast_horizon=0)

    with pytest.raises(ValueError, match="test_horizon must be >= 0"):
        mock_pipeline(df, test_horizon=-1)

    with pytest.raises(ValueError, match="cv_horizon must be >= 0"):
        mock_pipeline(df, cv_horizon=-1)

    with pytest.warns(Warning) as record:
        mock_pipeline(df, forecast_horizon=501)
        assert "Not enough training data to forecast the full forecast_horizon" in record[0].message.args[0]

    with pytest.warns(Warning) as record:
        result = mock_pipeline(df, test_horizon=1001)
        assert result["test_horizon"] == math.floor(df.shape[0] * 0.2)
        assert "test_horizon should never be larger than forecast_horizon" in record[0].message.args[0]

    with pytest.warns(Warning) as record:
        mock_pipeline(df, forecast_horizon=100, test_horizon=101)
        assert "test_horizon should never be larger than forecast_horizon" in record[0].message.args[0]

    with pytest.warns(Warning) as record:
        mock_pipeline(df, forecast_horizon=340, test_horizon=340)
        assert "test_horizon should be <= than 1/3 of the data set size" in record[0].message.args[0]

    with pytest.warns(Warning) as record:
        mock_pipeline(df, forecast_horizon=340, test_horizon=0)
        assert "No data selected for test" in record[0].message.args[0]

    with pytest.warns(Warning) as record:
        mock_pipeline(df, forecast_horizon=340, test_horizon=0, cv_horizon=0)
        assert "Both CV and backtest are skipped" in record[0].message.args[0]

    with pytest.raises(ValueError, match="periods_between_train_test must be >= 0"):
        mock_pipeline(df, periods_between_train_test=-1)


def test_input(df, df_reg):
    """Tests whether input parameters are properly set in pipeline.

    Parameters tested:

        - df
        - time_col
        - value_col
        - date_format
        - tz
        - freq
        - anomaly_info
        - regressor_cols
        - lagged_regressor_cols
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dummy_pipeline = get_dummy_pipeline()
        # anomaly adjustment adds 10.0 to every record
        adjustment_size = 10.0
        anomaly_df = pd.DataFrame({
            START_TIME_COL: [df[TIME_COL].min()],
            END_TIME_COL: [df[TIME_COL].max()],
            ADJUSTMENT_DELTA_COL: [adjustment_size],
            METRIC_COL: [VALUE_COL]})
        anomaly_info = {
            "value_col": VALUE_COL,
            "anomaly_df": anomaly_df,
            "start_time_col": START_TIME_COL,
            "end_time_col": END_TIME_COL,
            "adjustment_delta_col": ADJUSTMENT_DELTA_COL,
            "filter_by_dict": {METRIC_COL: VALUE_COL},
            "adjustment_method": "add"}
        result = forecast_pipeline(
            df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            date_format=None,
            tz=None,
            freq=None,
            anomaly_info=anomaly_info,
            pipeline=dummy_pipeline,
            hyperparameter_grid=None,
            forecast_horizon=10,
            test_horizon=10,
            cv_horizon=0)
        ts: UnivariateTimeSeries = result.timeseries
        assert_equal(ts.anomaly_info, anomaly_info)
        assert_equal(
            ts.df[VALUE_COL].values,
            (df[VALUE_COL] + adjustment_size).values,
            check_names=False)
        assert ts.df[TIME_COL][0] == df[TIME_COL].min()
        assert ts.original_time_col == TIME_COL
        assert ts.original_value_col == VALUE_COL
        assert ts.freq == "D"
        assert ts.time_stats["data_points"] == df.shape[0]
        assert not hasattr(ts.df.index.dtype, "tz")

    # with gaps, regressors, custom column names, date format, tz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg_cols = ["regressor1", "regressor2", "regressor_bool", "regressor_categ"]
        lag_reg_cols = ["regressor1", "regressor_bool"]
        df_reg = df_reg.rename({
            TIME_COL: "custom_time_col",
            VALUE_COL: "custom_value_col"
        }, axis=1)
        # changes date to string representation
        df_reg["custom_time_col"] = df_reg["custom_time_col"].dt.strftime("%Y-%m-%d-%H")
        # drops some rows to create gaps in input data
        drop_indices = [10, 20, 40, 41, 42, 43, 48, 49, 50, 55]
        df_reg.drop(drop_indices, axis=0, inplace=True)
        dummy_pipeline = get_dummy_pipeline(
            include_preprocessing=True,  # fills in gaps
            regressor_cols=reg_cols,
            lagged_regressor_cols=lag_reg_cols)
        result = forecast_pipeline(
            df_reg,
            time_col="custom_time_col",
            value_col="custom_value_col",
            date_format="%Y-%m-%d-%H",
            tz="Europe/Berlin",
            freq="H",
            anomaly_info=None,
            pipeline=dummy_pipeline,
            regressor_cols=reg_cols,
            lagged_regressor_cols=lag_reg_cols,
            hyperparameter_grid=None,
            forecast_horizon=10,
            test_horizon=10,
            cv_horizon=0)
        ts: UnivariateTimeSeries = result.timeseries
        assert ts.original_time_col == "custom_time_col"
        assert ts.original_value_col == "custom_value_col"
        assert ts.regressor_cols == reg_cols
        assert ts.lagged_regressor_cols == lag_reg_cols
        assert ts.freq == "H"
        assert ts.time_stats["min_timestamp"] == pd.to_datetime(df_reg["custom_time_col"]).min()
        assert ts.time_stats["max_timestamp"] == pd.to_datetime(df_reg["custom_time_col"]).max()
        assert ts.time_stats["added_timepoints"] == 10
        assert ts.time_stats["data_points"] == 2400
        assert ts.df.index.dtype.tz is not None


def test_train_end_date_gap():
    """Tests the parameters `train_end_date` and `periods_between_train_test`
    on a dataset without regressors.
    """
    data = generate_df_for_tests(
        freq="D",
        periods=30,
        train_start_date=datetime.datetime(2018, 1, 1))
    df = data["df"][[TIME_COL, VALUE_COL]].copy()
    df.loc[df.tail(5).index, VALUE_COL] = np.nan
    pipeline = get_dummy_pipeline(include_preprocessing=True)

    # No train_end_date
    with pytest.warns(UserWarning) as record:
        result = forecast_pipeline(
            df,
            train_end_date=None,
            pipeline=pipeline,
            forecast_horizon=10)
        ts = result.timeseries
        assert f"{ts.original_value_col} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]
        assert ts.train_end_date == datetime.datetime(2018, 1, 25)
        assert result.forecast.test_evaluation is None

    # train_end_date later than last date in df
    with pytest.warns(UserWarning) as record:
        train_end_date = datetime.datetime(2018, 2, 10)
        result = forecast_pipeline(
            df,
            train_end_date=train_end_date,
            pipeline=pipeline,
            forecast_horizon=5)
        ts = result.timeseries
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]
        assert ts.train_end_date == datetime.datetime(2018, 1, 25)
        assert result.forecast.test_evaluation is None

    # `train_end_date` before the last date in df
    train_end_date = datetime.datetime(2018, 1, 20)
    result = forecast_pipeline(
        df,
        train_end_date=train_end_date,
        pipeline=pipeline)
    ts = result.timeseries
    assert ts.train_end_date == datetime.datetime(2018, 1, 20)
    assert result.forecast.test_evaluation is not None

    # Tests `periods_between_train_test`
    forecast_horizon = 2
    test_horizon = 3
    periods_between_train_test = 4
    result = forecast_pipeline(
        df,
        forecast_horizon=forecast_horizon,
        test_horizon=test_horizon,
        train_end_date=train_end_date,
        periods_between_train_test=periods_between_train_test,
        pipeline=pipeline)
    df_train = df[df[TIME_COL] <= train_end_date]
    n_cols = len(["ts", "actual", "forecast"])
    train_size = df_train.shape[0]
    assert result.backtest.df_train.shape == (train_size - test_horizon - periods_between_train_test, n_cols)
    assert result.backtest.df_test.shape == (test_horizon, n_cols)
    assert result.backtest.df.shape == (train_size, n_cols)
    assert result.backtest.train_end_date == df_train.iloc[-(test_horizon + periods_between_train_test + 1)][TIME_COL]
    assert result.backtest.test_start_date == df_train.iloc[-test_horizon][TIME_COL]
    assert result.forecast.df_train.shape == (train_size, n_cols)
    assert result.forecast.df_test.shape == (forecast_horizon, n_cols)
    assert result.forecast.df.shape == (train_size + periods_between_train_test + forecast_horizon, n_cols)
    assert result.forecast.train_end_date == df_train.iloc[-1][TIME_COL]
    expected_forecast_test_start_date = pd.date_range(
        start=result.forecast.train_end_date,
        periods=periods_between_train_test + 2,
        freq=result.timeseries.freq)[-1]
    assert result.forecast.test_start_date == expected_forecast_test_start_date


def test_train_end_date_gap_regressors():
    """Tests the parameters `train_end_date` and `periods_between_train_test`
    on a dataset with regressors.
    """
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=60,
        train_start_date=datetime.datetime(2018, 1, 1),
        remove_extra_cols=True)
    regressor_cols = ["regressor1", "regressor2", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + regressor_cols
    df = data["df"][keep_cols].copy()
    # Setting NaN values at the end, omitting `regressor_categ` as
    # we do not have a null transformer for categorical variables yet
    df.loc[df.tail(2).index, "regressor1"] = np.nan
    df.loc[df.tail(5).index, "regressor2"] = np.nan
    available_forecast_horizon = 8
    df.loc[df.tail(available_forecast_horizon).index, VALUE_COL] = np.nan

    # Default `train_end_date`, default `regressor_cols`
    with pytest.warns(UserWarning) as record:
        result = forecast_pipeline(
            df=df,
            train_end_date=None,
            regressor_cols=None,
            pipeline=get_dummy_pipeline(include_preprocessing=True),
            forecast_horizon=10)
        ts = result.timeseries
        assert f"{ts.original_value_col} column of the provided TimeSeries contains " \
               f"null values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]
        assert ts.train_end_date == datetime.datetime(2018, 2, 21)
        assert ts.last_date_for_reg is None
        assert result.forecast.test_evaluation is None

    # `train_end_date` later than last date in df, all available `regressor_cols`
    with pytest.warns(UserWarning) as record:
        train_end_date = datetime.datetime(2018, 3, 10)
        result = forecast_pipeline(
            df,
            train_end_date=train_end_date,
            regressor_cols=regressor_cols,
            pipeline=get_dummy_pipeline(
                include_preprocessing=True,
                regressor_cols=regressor_cols),
            forecast_horizon=5)
        ts = result.timeseries
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]
        assert ts.train_end_date == datetime.datetime(2018, 2, 21)
        assert ts.last_date_for_reg == datetime.datetime(2018, 3, 1)
        forecast = result.forecast
        assert forecast.df[TIME_COL].max() == datetime.datetime(2018, 2, 26)
        assert forecast.test_evaluation is None

    # `train_end_date` in between last date in df and last date before null;
    # user passes no `regressor_cols`
    with pytest.warns(UserWarning) as record:
        train_end_date = datetime.datetime(2018, 2, 26)
        result = forecast_pipeline(
            df=df,
            train_end_date=train_end_date,
            regressor_cols=[],
            pipeline=get_dummy_pipeline(
                include_preprocessing=True,
                regressor_cols=[]),
            forecast_horizon=5)
        ts = result.timeseries
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]
        assert ts.train_end_date == datetime.datetime(2018, 2, 21)
        assert ts.last_date_for_reg is None
        forecast = result.forecast
        assert forecast.df[TIME_COL].max() == datetime.datetime(2018, 2, 26)
        assert forecast.test_evaluation is None

    # `train_end_date` smaller than last date before null,
    # using a subset of the `regressor_cols`.
    train_end_date = datetime.datetime(2018, 2, 17)
    regressor_cols = ["regressor2"]
    result = forecast_pipeline(
        df=df,
        train_end_date=train_end_date,
        regressor_cols=regressor_cols,
        pipeline=get_dummy_pipeline(
            include_preprocessing=True,
            regressor_cols=regressor_cols),
        forecast_horizon=5)
    ts = result.timeseries
    assert ts.train_end_date == datetime.datetime(2018, 2, 17)
    assert ts.last_date_for_reg == datetime.datetime(2018, 2, 24)
    forecast = result.forecast
    assert forecast.df[TIME_COL].max() == datetime.datetime(2018, 2, 22)
    assert forecast.forecast_horizon == 5
    assert forecast.test_evaluation is not None

    # `periods_between_train_test` is provided, attempts to predict beyond
    # the last known regressor value
    regressor_cols = ["regressor1", "regressor2"]
    forecast_horizon = 5
    periods_between_train_test = 2
    test_horizon = 3
    result = forecast_pipeline(
        df,
        forecast_horizon=forecast_horizon,
        test_horizon=test_horizon,
        regressor_cols=regressor_cols,
        periods_between_train_test=periods_between_train_test,
        pipeline=get_dummy_pipeline(
            include_preprocessing=True,
            regressor_cols=regressor_cols))
    n_cols = len(["ts", "actual", "forecast"])
    # Number of rows up to last non NaN value column entry
    train_size = df.shape[0] - available_forecast_horizon
    df_train = df.iloc[:train_size]
    assert result.backtest.df_train.shape == (train_size - test_horizon - periods_between_train_test, n_cols)
    # Backtest data corresponds to `df_test`.
    assert result.backtest.df_test.shape == (test_horizon, n_cols)
    assert result.backtest.df.shape == (train_size, n_cols)
    assert result.backtest.train_end_date == df_train.iloc[-(test_horizon + periods_between_train_test + 1)][TIME_COL]
    assert result.backtest.test_start_date == df_train.iloc[-test_horizon][TIME_COL]
    # Forecast horizon (5) + periods_between_train_test (2) extends beyond the
    # length of available regressors (6). Skips (2), then predicts the next
    # 4 before running out available regressors.
    # Future prediction can not exceed length of available regressors.
    assert result.forecast.df_train.shape == (train_size, n_cols)
    assert result.forecast.df_test.shape == (4, n_cols)
    assert result.forecast.df.iloc[-1][TIME_COL] == result.timeseries.last_date_for_reg
    assert result.forecast.train_end_date == df_train.iloc[-1][TIME_COL]
    expected_forecast_test_start_date = pd.date_range(
        start=result.forecast.train_end_date,
        periods=periods_between_train_test + 2,
        freq=result.timeseries.freq)[-1]
    assert result.forecast.test_start_date == expected_forecast_test_start_date


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_exceptions(df):
    """Tests error messages when CV is skipped and there are
    multiple hyperparameter options.

    Parameters tested:

        - hyperparameter_grid = dict, list of dict with distribution
        - cv_horizon=0
    """
    with pytest.raises(
            ValueError,
            match="CV is required to identify the best model because there are multiple options"):
        hyperparameter_grid = {
            "estimator__n_changepoints": [0],
            "estimator__uncertainty_samples": [10, 20]}
        forecast_pipeline(
            df,
            estimator=ProphetEstimator(),
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=None,
            forecast_horizon=24,
            test_horizon=10,
            cv_horizon=0)

    with pytest.raises(
            ValueError,
            match="CV is required to identify the best model because `hyperparameter_grid` contains"):
        hyperparameter_grid = [{
            "estimator__strategy": ["constant"],
            "estimator__constant": sp_randint(1, 3, 4)}]
        forecast_pipeline(
            df,
            pipeline=get_dummy_pipeline(),
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=None,
            forecast_horizon=24,
            test_horizon=10,
            cv_horizon=0)


def test_hyperparameter_grid(df):
    """Tests `forecast_pipeline` with various hyperparmeter_grid options.

    Parameters tested:

        - hyperparameter_grid=None, dict, list of dict
    """
    # hyperparameter_grid is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = forecast_pipeline(
            df,
            pipeline=get_dummy_pipeline(),
            hyperparameter_grid=None,
            hyperparameter_budget=1,
            forecast_horizon=10,
            test_horizon=10,
            cv_horizon=10)
        # default values are used
        backtest_params = result.grid_search.best_estimator_.get_params()
        forecast_params = result.model.get_params()
        for params in [backtest_params, forecast_params]:
            assert params["estimator__constant"] is None
            assert params["estimator__quantile"] is None
            assert params["estimator__strategy"] == "mean"

    # hyperparameter_grid is a dict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hyperparameter_grid = {
            "input__response__outlier__z_cutoff": [4.0],  # pre-processing param
            "estimator__constant": [None],
            "estimator__quantile": [0.9],
            "estimator__strategy": ["quantile"]}
        result = forecast_pipeline(
            df,
            pipeline=get_dummy_pipeline(include_preprocessing=True),
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=1,
            forecast_horizon=10,
            test_horizon=10,
            cv_horizon=10)

        # tests if parameters are set from hyperparameter_grid
        backtest_params = result.grid_search.best_estimator_.get_params()
        forecast_params = result.model.get_params()
        for params in [backtest_params, forecast_params]:
            assert params["input__response__outlier__z_cutoff"] == 4.0
            assert params["estimator__constant"] is None
            assert params["estimator__quantile"] == 0.9
            assert params["estimator__strategy"] == "quantile"

    # hyperparameter_grid is a list of dict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hyperparameter_grid = [
            {"input__response__outlier__z_cutoff": [4.0]},
            {
                "input__response__outlier__z_cutoff": [4.0],
                "estimator__constant": [None],
                "estimator__quantile": [0.9],
                "estimator__strategy": ["quantile"]
            }
        ]
        result = forecast_pipeline(
            df,
            pipeline=get_dummy_pipeline(include_preprocessing=True),
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=2,
            forecast_horizon=10,
            test_horizon=10,
            cv_horizon=10)

        # tests if parameters are set from hyperparameter_grid
        backtest_params = result.grid_search.best_estimator_.get_params()
        forecast_params = result.model.get_params()
        for params in [backtest_params, forecast_params]:
            assert params["input__response__outlier__z_cutoff"] == 4.0

        assert result.grid_search.n_iter == 2


def test_cv_backtest(df):
    """Tests `forecast_pipeline` with various combinations of CV and backtest.

    Parameters tested:

        - test_horizon
        - cv_horizon
        - cv_min_train_periods
        - cv_expanding_window
        - cv_use_most_recent_splits
        - cv_periods_between_splits
        - cv_periods_between_train_test
        - cv_max_splits
    """
    # Both CV and backtest
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_horizon = 11
        cv_horizon = 10
        result = forecast_pipeline(
            df,
            pipeline=get_dummy_pipeline(),
            forecast_horizon=24,
            test_horizon=test_horizon,
            cv_horizon=cv_horizon,
            cv_expanding_window=True,  # custom values
            cv_use_most_recent_splits=True,
            cv_min_train_periods=30,
            cv_periods_between_splits=180,
            cv_periods_between_train_test=1,
            cv_max_splits=3)
        backtest_params = result.grid_search.best_estimator_.get_params()
        forecast_params = result.model.get_params()
        for params in [backtest_params, forecast_params]:
            assert params["estimator__constant"] is None
            assert params["estimator__quantile"] is None
            assert params["estimator__strategy"] == "mean"
        assert result.backtest.forecast_horizon == test_horizon
        assert result.grid_search.cv_results_ is not None
        expected_cv_params = {
            "forecast_horizon": cv_horizon,
            "expanding_window": True,
            "use_most_recent_splits": True,
            "min_train_periods": 30,
            "periods_between_splits": 180,
            "periods_between_train_test": 1,
            "max_splits": 3,
            "min_splits": 1}
        for param, value in expected_cv_params.items():
            assert getattr(result.grid_search.cv, param) == value

    # Only CV (skip backtest)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_horizon = 0  # skip backtest
        cv_horizon = 10
        result = forecast_pipeline(
            df,
            pipeline=get_dummy_pipeline(),
            forecast_horizon=24,
            test_horizon=test_horizon,
            cv_horizon=cv_horizon,
            cv_expanding_window=False,
            cv_use_most_recent_splits=False,
            cv_min_train_periods=None,
            cv_periods_between_splits=None,
            cv_periods_between_train_test=100,
            cv_max_splits=10)
        backtest_params = result.grid_search.best_estimator_.get_params()
        forecast_params = result.model.get_params()
        for params in [backtest_params, forecast_params]:
            assert params["estimator__constant"] is None
            assert params["estimator__quantile"] is None
            assert params["estimator__strategy"] == "mean"
        assert result.backtest is None
        assert result.grid_search.cv_results_ is not None
        expected_cv_params = {
            "forecast_horizon": cv_horizon,
            "expanding_window": False,
            "use_most_recent_splits": False,
            "min_train_periods": 20,       # auto-populated by RollingTimeSeriesSplit
            "periods_between_splits": 10,  # auto-populated by RollingTimeSeriesSplit
            "periods_between_train_test": 100,
            "max_splits": 10,
            "min_splits": 1}
        for param, value in expected_cv_params.items():
            assert getattr(result.grid_search.cv, param) == value

    # Only backtest (skip CV)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_horizon = 11
        cv_horizon = 0  # skip CV
        result = forecast_pipeline(
            df,
            pipeline=get_dummy_pipeline(),
            forecast_horizon=24,
            test_horizon=test_horizon,
            cv_horizon=cv_horizon,
            cv_expanding_window=True,
            cv_use_most_recent_splits=True,
            cv_min_train_periods=30,
            cv_periods_between_splits=180,
            cv_periods_between_train_test=1,
            cv_max_splits=3)
        backtest_params = result.grid_search.best_estimator_.get_params()
        forecast_params = result.model.get_params()
        for params in [backtest_params, forecast_params]:
            assert params["estimator__constant"] is None
            assert params["estimator__quantile"] is None
            assert params["estimator__strategy"] == "mean"
        assert result.backtest.forecast_horizon == test_horizon
        # dummy RandomizedSearchCV object has only the applicable attributes
        assert result.grid_search.cv is None
        assert not hasattr(result.grid_search, "cv_results_")
        assert not hasattr(result.grid_search, "best_index_")
        assert not hasattr(result.grid_search, "scorer_")
        assert not hasattr(result.grid_search, "refit_time_")
        assert result.grid_search.best_params_ == {}
        assert result.grid_search.n_splits_ == 0

        cv_max_splits = 0  # skips CV using cv_max_splits
        result = forecast_pipeline(
            df,
            pipeline=get_dummy_pipeline(),
            forecast_horizon=24,
            test_horizon=test_horizon,
            cv_horizon=3,
            cv_expanding_window=True,
            cv_use_most_recent_splits=True,
            cv_min_train_periods=30,
            cv_periods_between_splits=180,
            cv_periods_between_train_test=1,
            cv_max_splits=cv_max_splits)
        assert result.grid_search.cv is None
        assert result.grid_search.n_splits_ == 0

    # No CV or backtest (not allowed)
    # Already checked by test case in `test_validate_pipeline_input`.


def test_model_forecast_evaluation(df):
    """Tests whether model, forecast, evaluation parameters are properly set in pipeline:

    Parameters tested:

        # model
        - pipeline
        - estimator_name
        - hyperparameter_grid
        - hyperparameter_budget
        - n_jobs
        - verbose
        # forecast
        - forecast_horizon
        - coverage
        - agg_periods
        - agg_func
        # evaluation
        - score_func
        - score_func_greater_is_better
        - cv_report_metrics
        - null_model_params
        - relative_error_tolerance
    """
    # Checks that grid search scores are as expected.
    # 4 options in the grid, score_func_greater_is_better=True
    hyperparameter_grid = [
        {
            "estimator__strategy": ["quantile"],
            "estimator__quantile": [0.9]
        },
        {
            "estimator__strategy": ["mean"]
        },
        {
            "estimator__strategy": ["constant"],
            "estimator__constant": [1.0, 2.0]
        }
    ]
    pipeline = get_dummy_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metric = EvaluationMetricEnum.MeanAbsolutePercentError
        result = forecast_pipeline(
            df,
            pipeline=pipeline,
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=None,  # full grid search
            n_jobs=-1,
            verbose=2,
            forecast_horizon=20,
            coverage=None,
            agg_periods=7,
            agg_func=np.sum,
            score_func=metric.name,
            score_func_greater_is_better=metric.get_metric_greater_is_better(),
            cv_report_metrics=CV_REPORT_METRICS_ALL,
            null_model_params=None,
            relative_error_tolerance=0.02)
        expected_report_metrics = [enum.get_metric_name() for enum in EvaluationMetricEnum]
        expected_report_metrics += [FRACTION_OUTSIDE_TOLERANCE_NAME]
        check_forecast_pipeline_result(
            result,
            coverage=None,
            expected_grid_size=4,
            lower_bound_cv=0.0,  # MAPE is non-negative
            score_func=metric.name,
            greater_is_better=metric.get_metric_greater_is_better(),
            cv_report_metrics_names=expected_report_metrics,
            relative_error_tolerance=0.02)
        df_train = result.backtest.df_train
        df_test = result.backtest.df_test
        for enum in EvaluationMetricEnum:
            scorer = enum.get_metric_func()
            # Note: `agg_func` is used in CV evaluation but currently not used
            #   in reporting backtest/forecast metrics. Check when this is enabled.
            # scorer = add_preaggregation_to_scorer(scorer, agg_periods=7, agg_func=np.sum)
            expected_score = scorer(df_train[ACTUAL_COL], df_train[PREDICTED_COL])  # train score
            assert result.backtest.train_evaluation[enum.get_metric_name()] == expected_score
            expected_score = scorer(df_test[ACTUAL_COL], df_test[PREDICTED_COL])  # test score
            assert result.backtest.test_evaluation[enum.get_metric_name()] == expected_score
        assert (result.backtest.test_evaluation[FRACTION_OUTSIDE_TOLERANCE]
                == result.backtest.test_evaluation[EvaluationMetricEnum.FractionOutsideTolerance2.get_metric_name()])
        assert (result.forecast.train_evaluation[FRACTION_OUTSIDE_TOLERANCE]
                == result.forecast.train_evaluation[EvaluationMetricEnum.FractionOutsideTolerance2.get_metric_name()])
    assert len(result.grid_search.cv_results_.keys()) == 206

    # Tests estimator_name, coverage, null_model_params,
    # score_func callable, score_func_greater_is_better=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hyperparameter_grid = {
            "estimator__fit_algorithm_dict": [
                {
                    "fit_algorithm": "linear",
                    "fit_algorithm_params": {"fit_intercept": False}
                },
                {
                    "fit_algorithm": "ridge",
                },
            ],
            "estimator__extra_pred_cols": [
                ["ct1", "is_weekend"],
                ["ct1"],
                ["ct2"],
            ]
        }
        null_model_params = {"strategy": "mean"}
        coverage = 0.80
        hyperparameter_budget = 2  # limited grid search
        result = forecast_pipeline(
            df,
            estimator=SilverkiteEstimator(),
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=hyperparameter_budget,
            n_jobs=1,
            verbose=1,
            forecast_horizon=10,
            coverage=coverage,
            agg_periods=2,
            agg_func=np.max,
            score_func=explained_variance_score,
            score_func_greater_is_better=True,
            cv_report_metrics=None,
            null_model_params=null_model_params,
            relative_error_tolerance=None,
            cv_max_splits=1)
        check_forecast_pipeline_result(
            result,
            coverage=coverage,
            strategy=null_model_params["strategy"],
            expected_grid_size=hyperparameter_budget,
            upper_bound_cv=1.0,  # highest possible explained variance
            score_func=explained_variance_score,
            greater_is_better=True,
            cv_report_metrics_names=None,
            relative_error_tolerance=None)
        # Fewer metrics than before, because cv_report_metrics=None
        assert len(result.grid_search.cv_results_.keys()) == 14
        assert "mean_test_CORR" not in result.grid_search.cv_results_.keys()
        assert R2_null_model_score in result.backtest.test_evaluation

    # Tests estimator_name, coverage, score_func=FRACTION_OUTSIDE_TOLERANCE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = forecast_pipeline(
            df,
            estimator=SilverkiteEstimator(),
            forecast_horizon=10,
            coverage=coverage,
            agg_periods=2,
            agg_func=np.max,
            score_func=FRACTION_OUTSIDE_TOLERANCE,
            score_func_greater_is_better=False,
            relative_error_tolerance=0.02,
            cv_max_splits=1)
        assert f"mean_test_{FRACTION_OUTSIDE_TOLERANCE_NAME}" in result.grid_search.cv_results_.keys()


def test_cv_error_calculation(df):
    """Tests whether error metrics are properly calculated in CV
    This is done by using backtest to get predicted values, and
    evaluating the error metrics directly.
    """
    pipeline = get_dummy_pipeline()
    # test with and without aggregation
    test_cases = [
        # (agg_periods, agg_func)
        (None, None),
        (7, np.sum)
    ]
    for agg_periods, agg_func in test_cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metric = EvaluationMetricEnum.MeanAbsolutePercentError
            params = dict(
                pipeline=pipeline,
                forecast_horizon=20,
                coverage=None,
                agg_periods=agg_periods,
                agg_func=agg_func,
                score_func=metric.name,
                score_func_greater_is_better=metric.get_metric_greater_is_better(),
                cv_report_metrics=CV_REPORT_METRICS_ALL,
                null_model_params=None,
                relative_error_tolerance=0.02)
            backtest = forecast_pipeline(
                df,
                test_horizon=20,
                cv_horizon=0,
                **params).backtest
            # Single CV split. CV test = backtest test
            grid_search = forecast_pipeline(
                df,
                test_horizon=0,
                cv_horizon=20,
                cv_min_train_periods=df.shape[0]-20,
                **params).grid_search
            df_train = backtest.df_train
            df_test = backtest.df_test
            scorer = metric.get_metric_func()
            if agg_periods is not None:
                scorer = add_preaggregation_to_scorer(scorer, agg_periods=agg_periods, agg_func=agg_func)
            expected_score = scorer(df_train[ACTUAL_COL], df_train[PREDICTED_COL])
            assert grid_search.cv_results_[f"mean_train_{metric.get_metric_name()}"][0] == expected_score
            expected_score = scorer(df_test[ACTUAL_COL], df_test[PREDICTED_COL])
            assert grid_search.cv_results_[f"mean_test_{metric.get_metric_name()}"][0] == expected_score


# Integration tests below
def test_default():
    """Tests forecast_pipeline with short forecast horizon"""
    data = generate_df_for_tests(freq="D", periods=20)
    df = data["df"][[TIME_COL, VALUE_COL]]
    result = forecast_pipeline(
        df,
        forecast_horizon=2,
        n_jobs=1,
        # "ridge" fit_algorithm needs enough data to run internal CV
        cv_min_train_periods=16)
    check_forecast_pipeline_result(
        result,
        greater_is_better=False,
        expected_grid_size=1)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_simple():
    """Tests forecast_pipeline function with Prophet and default parameters"""
    data = generate_df_for_tests(freq="H", periods=24*10)
    df = data["df"][[TIME_COL, VALUE_COL]]
    hyperparameter_grid = {
        "estimator__weekly_seasonality": [True],
        "estimator__daily_seasonality": [True, False],
        "estimator__n_changepoints": [0],  # to speed up test case, remove for better fit
        "estimator__uncertainty_samples": [10]  # to speed up test case
    }

    # run pipeline with 1 CV fold, 2 parameter sets. Using default horizons
    with pytest.warns(Warning) as record:
        result = forecast_pipeline(
            df,
            score_func=EvaluationMetricEnum.MeanSquaredError.name,
            estimator=ProphetEstimator(),
            hyperparameter_grid=hyperparameter_grid,
            cv_min_train_periods=24*8,  # to speed up test case
            cv_report_metrics=[
                EvaluationMetricEnum.MeanAbsoluteError.name],
            cv_max_splits=1)
        assert "There is only one CV split" in record[0].message.args[0]
        backtest_prophet_model = result.grid_search.best_estimator_.steps[-1][-1].model
        assert backtest_prophet_model.daily_seasonality
        assert backtest_prophet_model.weekly_seasonality
        assert backtest_prophet_model.n_changepoints == 0
        assert backtest_prophet_model.uncertainty_samples == 10
        check_forecast_pipeline_result(
            result,
            expected_grid_size=2,
            lower_bound_cv=0.0,  # default MAPE is non-negative
            score_func=EvaluationMetricEnum.MeanSquaredError.name,
            greater_is_better=False,
            cv_report_metrics_names=[
                EvaluationMetricEnum.MeanAbsoluteError.get_metric_name(),
            ]
        )
        # Total data is 240 (24*10) rows. All of it is used for training;
        # 24 new rows created for testing based on default horizon setting. Test horizon & CV horizon is 24
        # Similary, Backtest training has 240-24 = 216 rows.
        assert result.forecast.df_train.shape == (240, 5)
        assert result.forecast.df_test.shape == (24, 5)
        assert result.backtest.df_train.shape == (216, 5)
        assert result.backtest.df_test.shape == (24, 5)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_with_regressor():
    """Tests forecast_pipeline function with Prophet,
    input regressors, and default parameters
    """
    data = generate_df_with_reg_for_tests(
        freq="H",
        periods=24*16,
        train_frac=0.8,
        conti_year_origin=2018,
        remove_extra_cols=True,
        mask_test_actuals=True)
    # select relevant columns for testing
    regressor_cols = ["regressor1", "regressor2", "regressor3"]
    relevant_cols = [TIME_COL, VALUE_COL] + regressor_cols
    df = data["df"][relevant_cols]

    hyperparameter_grid = {
        "estimator__weekly_seasonality": [True],
        "estimator__daily_seasonality": [True, False],
        "estimator__n_changepoints": [0],  # to speed up test case, remove for better fit
        "estimator__uncertainty_samples": [10],  # to speed up test case
        "estimator__add_regressor_dict": [{
            "regressor1": {
                "prior_scale": 10,
                "standardize": True,
                "mode": "additive"
            },
            "regressor2": {
              "prior_scale": 15,
              "standardize": False,
              "mode": "additive"
            },
            "regressor3": {}
        }]
    }
    # run pipeline with 1 CV fold, 2 parameter sets. Using default horizons
    result = forecast_pipeline(
        df,
        estimator=ProphetEstimator(),
        regressor_cols=regressor_cols,
        hyperparameter_grid=hyperparameter_grid,
        cv_min_train_periods=24*8)

    backtest_prophet_estimator = result.grid_search.best_estimator_.steps[-1][-1]
    assert [backtest_prophet_estimator.add_regressor_dict] == hyperparameter_grid["estimator__add_regressor_dict"]
    check_forecast_pipeline_result(
        result,
        coverage=0.95,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        greater_is_better=False)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_complex():
    """Tests forecast_pipeline function with Prophet,
    custom parameters, missing data, and holidays
    """
    num_periods = 17*7 - 2  # not a whole number of weeks
    data = generate_df_for_tests(freq="D", periods=num_periods)
    df = data["df"][[TIME_COL, VALUE_COL]].rename({
        TIME_COL: "custom_time_col",  # non-standard column names
        VALUE_COL: "custom_value_col"
    }, axis=1)
    df["custom_time_col"] = [x._date_repr for x in df["custom_time_col"]]  # change date to string representation

    # drops some rows to create gaps in input data
    drop_indices = [10, 20, 40, 41, 42, 43, 48, 49, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 101, 110, 111, 112]
    df.drop(drop_indices, axis=0, inplace=True)
    assert df.shape == (num_periods - len(drop_indices), 2)

    # Without catching the following code produces about 30 warnings.
    # For brevity, check a few warnings
    with pytest.warns(Warning) as record:
        # creates holiday features
        holidays = ProphetTemplate().get_prophet_holidays(
            year_list=[2018, 2019],
            countries=["UnitedStates", "India"],
            lower_window=0,
            upper_window=1)
        coverage = 0.1  # low value to make sure defaults are overridden
        hyperparameter_grid = {
            "estimator__seasonality_mode": ["multiplicative"],
            "estimator__holidays": [holidays],
            "estimator__n_changepoints": [0],  # to speed up test case, remove for better fit
            "estimator__uncertainty_samples": [10]  # to speed up test case
        }

        # runs pipeline with 1 CV fold, 1 parameter set
        result = forecast_pipeline(
            df,
            time_col="custom_time_col",
            value_col="custom_value_col",
            date_format="%Y-%m-%d",  # possible to specify, not recommended
            tz=None,
            freq="D",  # recommended to specify when there are missing data
            pipeline=None,
            estimator=ProphetEstimator(),
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=1,
            n_jobs=-1,
            forecast_horizon=3 * 7,
            coverage=coverage,
            test_horizon=2 * 7,
            agg_periods=7,
            agg_func=np.max,
            score_func=EvaluationMetricEnum.MeanSquaredError.name,
            null_model_params={"strategy": "quantile", "quantile": 0.5},  # uses null model
            cv_horizon=3 * 7,
            cv_expanding_window=True,
            cv_min_train_periods=8 * 7,
            cv_periods_between_splits=7,
            cv_periods_between_train_test=3 * 7)
        check_forecast_pipeline_result(
            result,
            coverage=coverage,
            strategy="quantile",
            expected_grid_size=1,
            lower_bound_cv=0.0,
            score_func=EvaluationMetricEnum.MeanSquaredError.name,
            greater_is_better=False)
        assert "There is only one CV split" in record[0].message.args[0]


def test_silverkite_longterm():
    # testing long term forecast
    # generate 3 years of data
    data = generate_df_for_tests(freq="1D", periods=3 * 52 * 7)
    df = data["df"]
    coverage = 0.1  # low value to make sure defaults are overridden
    hyperparameter_grid = {
        "estimator__origin_for_time_vars": [2018],
        "estimator__fs_components_df": [
            pd.DataFrame({
                "name": ["tow", "conti_year"],
                "period": [7.0, 1.0],
                "order": [3, 5],
                "seas_names": ["weekly", "yearly"]})],
        "estimator__extra_pred_cols": [["ct_sqrt"]],
        "estimator__fit_algorithm_dict": [{"fit_algorithm": "linear"}]
    }

    # run pipeline with 42 CV folds, 1 parameter set
    forecast_horizon = 52 * 7
    cv_horizon = forecast_horizon
    test_horizon = forecast_horizon
    periods_between_train_test = 2
    with pytest.warns(UserWarning) as record:
        result = forecast_pipeline(
            df,
            time_col="ts",
            value_col="y",
            date_format=None,  # not recommended, but possible to specify
            freq=None,
            estimator=SilverkiteEstimator(),
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=1,
            n_jobs=-1,
            forecast_horizon=forecast_horizon,
            coverage=coverage,
            test_horizon=test_horizon,
            periods_between_train_test=periods_between_train_test,
            agg_periods=7,
            agg_func=np.mean,
            score_func=mean_absolute_error,  # callable score_func
            score_func_greater_is_better=False,
            null_model_params=None,
            cv_horizon=cv_horizon,
            cv_expanding_window=True,
            cv_min_train_periods=8 * 7,
            cv_periods_between_splits=7,
            cv_periods_between_train_test=3 * 7,
            cv_max_splits=None)

        # gathers all warning messages
        all_warnings = ""
        for i in range(len(record)):
            all_warnings += record[i].message.args[0]
        assert "`min_train_periods` is too small for your `forecast_horizon`. Should be at " \
               "least 728=2*`forecast_horizon`." in all_warnings
        assert "There is a high number of CV splits (41). If training is slow, increase " \
               "`periods_between_splits` or `min_train_periods`, or decrease `max_splits`" in all_warnings
        check_forecast_pipeline_result(
            result,
            coverage=coverage,
            expected_grid_size=1,
            lower_bound_cv=0.0,
            score_func=mean_absolute_error,
            greater_is_better=False)

    expected_backtest_train_size = result.timeseries.fit_df.shape[0] - test_horizon - periods_between_train_test
    assert result.backtest.estimator.model_dict["x_mat"].shape[0] == expected_backtest_train_size
    expected_forecast_train_size = result.timeseries.fit_df.shape[0]
    assert result.forecast.estimator.model_dict["x_mat"].shape[0] == expected_forecast_train_size


def test_silverkite_regressor():
    """Tests forecast_pipeline with silverkite and input regressors,
    autoregression and lagged regressors"""
    data = generate_df_with_reg_for_tests(
        freq="1D",
        periods=20 * 7,  # short-term: 20 weeks of data
        remove_extra_cols=True,
        mask_test_actuals=True)
    regressor_cols = ["regressor1", "regressor2", "regressor_categ"]
    lagged_regressor_cols = ["regressor1", "regressor2"]
    keep_cols = [TIME_COL, VALUE_COL] + regressor_cols
    df = data["df"][keep_cols]
    coverage = 0.1
    hyperparameter_grid = {
        "estimator__origin_for_time_vars": [None],  # inferred from training data
        "estimator__fs_components_df": [
            pd.DataFrame({
                "name": ["tow", "conti_year"],
                "period": [7.0, 1.0],
                "order": [3, 0],
                "seas_names": ["weekly", None]})],
        "estimator__extra_pred_cols": [
            regressor_cols,
            regressor_cols + ["ct_sqrt"]
        ],  # two cases: no growth term and single growth term
        "estimator__fit_algorithm_dict": [{"fit_algorithm": "linear"}],
        "estimator__autoreg_dict": [{
            "lag_dict": {"orders": [7]},
            "agg_lag_dict": {
                "orders_list": [[7, 7*2, 7*3]],
                "interval_list": [(7, 7*2)]},
            "series_na_fill_func": lambda s: s.bfill().ffill()}],
        "estimator__lagged_regressor_dict": [{
            "regressor1": {
                "lag_dict": {"orders": [1, 2, 3]},
                "agg_lag_dict": {
                    "orders_list": [[7, 7 * 2, 7 * 3]],
                    "interval_list": [(8, 7 * 2)]},
                "series_na_fill_func": lambda s: s.bfill().ffill()},
            "regressor2": "auto"
        }]
    }
    test_horizon = 2 * 7
    periods_between_train_test = 2
    # Runs pipeline with 2 (of 3) CV folds, 1 parameter set
    result = forecast_pipeline(
        df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        date_format=None,  # not recommended, but possible to specify
        freq=None,
        regressor_cols=regressor_cols,
        lagged_regressor_cols=lagged_regressor_cols,
        estimator=SilverkiteEstimator(),
        hyperparameter_grid=hyperparameter_grid,
        hyperparameter_budget=1,
        n_jobs=1,
        forecast_horizon=3 * 7,
        coverage=coverage,
        test_horizon=test_horizon,
        periods_between_train_test=periods_between_train_test,
        agg_periods=7,
        agg_func=np.mean,
        score_func=mean_absolute_error,  # callable score_func
        score_func_greater_is_better=False,
        null_model_params=None,
        cv_horizon=1 * 7,
        cv_expanding_window=True,
        cv_min_train_periods=8 * 7,
        cv_periods_between_splits=7,
        cv_periods_between_train_test=3 * 7,
        cv_max_splits=2)
    check_forecast_pipeline_result(
        result,
        coverage=coverage,
        expected_grid_size=1,
        lower_bound_cv=0.0,
        score_func=mean_absolute_error,
        greater_is_better=False)

    expected_backtest_train_size = result.timeseries.fit_df.shape[0] - test_horizon - periods_between_train_test
    assert result.backtest.estimator.model_dict["x_mat"].shape[0] == expected_backtest_train_size
    expected_forecast_train_size = result.timeseries.fit_df.shape[0]
    assert result.forecast.estimator.model_dict["x_mat"].shape[0] == expected_forecast_train_size

    model = result.model.steps[-1][-1]
    trained_model = model.model_dict
    pred_cols = trained_model["pred_cols"]
    expected_feature_cols = {
        # regressor columns
        "regressor1",
        "regressor2",
        "regressor_categ",
        # lagged regressor columns
        "regressor1_lag1",
        "regressor1_lag2",
        "regressor1_lag3",
        "regressor1_avglag_7_14_21",
        "regressor1_avglag_8_to_14",
        "regressor2_lag35",
        "regressor2_avglag_35_42_49",
        "regressor2_avglag_30_to_36"
    }
    assert expected_feature_cols.issubset(pred_cols)


def test_silverkite_regressor_with_missing_values():
    """Tests forecast_pipeline with silverkite and input regressors and lagged regressors.
    In particular, lagged regressor columns are not a subset of regressor columns.
    Multiple lag order + forecast horizon combinations are tested for warnings."""
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=20 * 7,
        train_start_date=datetime.datetime(2018, 1, 1),
        remove_extra_cols=True,
        mask_test_actuals=False)
    regressor_cols_all = ["regressor1", "regressor2"]
    regressor_cols = ["regressor1"]
    lagged_regressor_cols = ["regressor2"]
    keep_cols = [TIME_COL, VALUE_COL] + regressor_cols_all
    df = data["df"][keep_cols].copy()
    # Setting NaN values at the end
    # VALUE_COL and regressor2 have the same length
    # regressor1 has 5 more future values than VALUE_COL and regressor2
    # Therefore, the max forecast horizon is 5
    df.loc[df.tail(8).index, VALUE_COL] = np.nan
    df.loc[df.tail(3).index, "regressor1"] = np.nan
    df.loc[df.tail(8).index, "regressor2"] = np.nan

    hyperparameter_grid = {
        "estimator__fs_components_df": [None],
        "estimator__extra_pred_cols": [
            regressor_cols
        ],  # two cases: no growth term and single growth term
        "estimator__fit_algorithm_dict": [{"fit_algorithm": "linear"}],
        "estimator__lagged_regressor_dict": [{
            "regressor2": {
                "lag_dict": {"orders": [3]},
                "agg_lag_dict": {
                    "orders_list": [[7, 7 * 2, 7 * 3]],
                    "interval_list": [(8, 7 * 2)]},
                "series_na_fill_func": lambda s: s.bfill().ffill()}

        }, {
            "regressor2": {
                "lag_dict": {"orders": [5]},
                "agg_lag_dict": {
                    "orders_list": [[7, 7 * 2, 7 * 3]],
                    "interval_list": [(8, 7 * 2)]},
                "series_na_fill_func": lambda s: s.bfill().ffill()}

        }]
    }

    # When any minimal lagged regressor order in the grids is less than forecast_horizon,
    # there should a warning of lagged regressor columns being imputed
    forecast_horizon = 5
    test_horizon = 7
    periods_between_train_test = 0
    with pytest.warns(Warning) as record:
        result = forecast_pipeline(
            df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            regressor_cols=regressor_cols,
            lagged_regressor_cols=lagged_regressor_cols,
            estimator=SilverkiteEstimator(),
            hyperparameter_grid=hyperparameter_grid,
            forecast_horizon=forecast_horizon,
            test_horizon=test_horizon,
            periods_between_train_test=periods_between_train_test,
            cv_max_splits=2
        )
        all_warnings = "".join([warn.message.args[0] for warn in record])
        assert "test_horizon should never be larger than forecast_horizon" in all_warnings
        assert "Trained model's `min_lagged_regressor_order` (3) is less than the size of `fut_df` (5)" in all_warnings
    assert result.model[-1].model_dict["min_lagged_regressor_order"] == 5

    # Checks model
    expected_pred_cols = [
        "regressor1",
        "regressor2_lag5",
        "regressor2_avglag_7_14_21",
        "regressor2_avglag_8_to_14"
    ]
    assert result.model[-1].model_dict["pred_cols"] == expected_pred_cols

    expected_train_size = result.timeseries.fit_df.shape[0]  # 132
    assert result.timeseries.fit_df.shape == (expected_train_size, 4)  # 132
    assert result.model[-1].model_dict["x_mat"].shape == (expected_train_size, 5)  # 132

    # Checks backtest
    expected_backtest_train_size = expected_train_size - test_horizon - periods_between_train_test  # 132 - 5 - 0
    assert result.backtest.estimator.model_dict["x_mat"].shape == (expected_backtest_train_size, 5)  # 127
    assert result.backtest.df_test.shape == (test_horizon, 5)  # 5

    # Checks forecast
    assert result.forecast.df_test.shape == (forecast_horizon, 5)  # 5

    # Checks key dates
    assert result.timeseries.train_end_date == datetime.datetime(2018, 5, 12)
    assert result.backtest.df_test[TIME_COL].iloc[0] == datetime.datetime(2018, 5, 6)
    assert result.forecast.df_test[TIME_COL].iloc[0] == datetime.datetime(2018, 5, 13)

    # More edge cases
    # When overall minimal lagged regressor order for all grids is at least forecast_horizon,
    # there is no warning
    forecast_horizon = 3
    test_horizon = 3
    periods_between_train_test = 0
    with pytest.warns(Warning) as record:
        result = forecast_pipeline(
            df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            regressor_cols=regressor_cols,
            lagged_regressor_cols=lagged_regressor_cols,
            estimator=SilverkiteEstimator(),
            hyperparameter_grid=hyperparameter_grid,
            forecast_horizon=forecast_horizon,
            test_horizon=test_horizon,
            periods_between_train_test=periods_between_train_test,
            cv_max_splits=2
        )
        all_warnings = "".join([warn.message.args[0] for warn in record])
        assert "Trained model's `min_lagged_regressor_order` (3) is less than the size of `fut_df` (3)" not in all_warnings
    assert result.model[-1].model_dict["min_lagged_regressor_order"] == 5


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_custom_pipeline():
    """Tests forecast_pipeline function with custom pipeline"""
    data = generate_df_for_tests(freq="D", periods=30*8)  # 8 months
    df = data["df"][[TIME_COL, VALUE_COL]]
    score_func = add_finite_filter_to_scorer(mean_squared_error)
    coverage = 0.1  # low value to make sure defaults are overridden
    hyperparameter_grid = {
        "estimator__seasonality_mode": ["additive"],
        "estimator__n_changepoints": [0],  # to speed up test case, remove for better fit
        "estimator__uncertainty_samples": [10]  # to speed up test case
    }

    # it's possible, but not recommended, to write your own pipeline for whatever reason
    pipeline = Pipeline([
        # the final step in the pipeline must be called "estimator"
        ("estimator", ProphetEstimator(
            score_func=score_func,
            coverage=coverage,
            null_model_params=None))
    ])

    with pytest.warns(UserWarning) as record:
        result = forecast_pipeline(
            df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            date_format=None,
            tz=None,
            freq="D",
            pipeline=pipeline,
            estimator=ProphetEstimator(),  # IGNORED, uses pipeline's estimator instead
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=1,
            n_jobs=-1,
            forecast_horizon=None,
            coverage=0.99,  # IGNORED, uses pipeline's coverage instead
            test_horizon=None,
            agg_periods=None,
            agg_func=None,
            score_func=EvaluationMetricEnum.MeanAbsoluteError.name,
            score_func_greater_is_better=False,
            null_model_params={"strategy": "quantile", "quantile": 0.5},  # IGNORED, uses pipeline's null model instead
            cv_horizon=None,
            cv_expanding_window=False,
            cv_use_most_recent_splits=True,
            cv_min_train_periods=None,
            cv_periods_between_splits=200,
            cv_periods_between_train_test=0,
            cv_max_splits=3)
        check_forecast_pipeline_result(
            result,
            coverage=coverage,
            expected_grid_size=1,
            lower_bound_cv=0.0,
            score_func=EvaluationMetricEnum.MeanAbsoluteError.name,
            greater_is_better=False)
        assert "There is only one CV split" in record[0].message.args[0]

        # tests whether the pipeline properties were used
        estimator = result.model.steps[-1][-1]
        assert estimator.null_model_params is None
        assert estimator.coverage is coverage
        assert estimator.score_func([1.0], [3.0]) == 4.0  # confirms MSE is used, not MAE


def test_forecast_pipeline_coverage():
    """Forecast result with the following settings:

        * Hourly forecast with CV and backtest
        * 1 hyperparameter sets, 1 CV splits
        * All relevant `cv_report_metrics`.
        * Uses `SimpleSilverkiteEstimator`
        * Has prediction intervals
    """
    data = generate_df_for_tests(freq="H", periods=24*50)
    df = data["df"][[TIME_COL, VALUE_COL]]
    hyperparameter_grid = {
        "estimator__weekly_seasonality": [False],
        "estimator__daily_seasonality": [False],
    }

    def forecast_result(coverage):
        result = forecast_pipeline(
            df,
            score_func=EvaluationMetricEnum.MeanSquaredError.name,
            estimator=SimpleSilverkiteEstimator(),
            forecast_horizon=24*2,
            coverage=coverage,
            test_horizon=24,
            hyperparameter_grid=hyperparameter_grid,
            cv_horizon=24,
            cv_min_train_periods=24*30,
            cv_report_metrics=None,
            cv_max_splits=1,
            hyperparameter_budget=1,
            relative_error_tolerance=0.02,
            n_jobs=1,
        )
        return result
    result = forecast_result(coverage=0.9)
    check_forecast_pipeline_result(
            result,
            coverage=0.9,
            expected_grid_size=1,
            lower_bound_cv=0.0,
            score_func=EvaluationMetricEnum.MeanSquaredError.name,
            greater_is_better=False)

    result = forecast_result(coverage=0.8)
    check_forecast_pipeline_result(
            result,
            coverage=0.8,
            expected_grid_size=1,
            lower_bound_cv=0.0,
            score_func=EvaluationMetricEnum.MeanSquaredError.name,
            greater_is_better=False)


def test_pipeline_end2end_predict():
    """Tests for `forecast_pipeline` predictions and design matrix validity.
    """
    warnings.simplefilter("ignore")

    # Generates data
    forecast_horizon = 20
    data = generate_df_for_tests(freq="H", periods=24*10)
    train_df = data["train_df"][[TIME_COL, VALUE_COL]]
    test_df = data["test_df"][[TIME_COL, VALUE_COL]][:forecast_horizon]

    # Sets hyperparameter grid
    hyperparameter_grid = {
        "estimator__weekly_seasonality": [False],
        "estimator__daily_seasonality": [False],
        "estimator__holiday_lookup_countries": [None],
        "estimator__extra_pred_cols": [["ct1", "ct2"]]
    }

    # Fits and predicts using pipeline
    result = forecast_pipeline(
        df=train_df,
        score_func=EvaluationMetricEnum.MeanSquaredError.name,
        estimator=SimpleSilverkiteEstimator(),
        forecast_horizon=forecast_horizon,
        coverage=0.95,
        test_horizon=None,
        hyperparameter_grid=hyperparameter_grid,
        cv_horizon=24,
        cv_min_train_periods=24*30,
        cv_report_metrics=None,
        cv_max_splits=1,
        hyperparameter_budget=1,
        relative_error_tolerance=0.02,
        n_jobs=1,
    )

    # Tests to see if parameters are set from hyperparameter_grid
    backtest_params = result.grid_search.best_estimator_.get_params()
    forecast_params = result.model.get_params()
    for params in [backtest_params, forecast_params]:
        assert params["estimator__weekly_seasonality"] is False
        assert params["estimator__daily_seasonality"] is False
        assert params["estimator__holiday_lookup_countries"] is None
        assert params["estimator__extra_pred_cols"] == ["ct1", "ct2"]

    # Gets the final estimator
    trained_estimator = result.model[-1]

    forecast_x_mat = trained_estimator.forecast_x_mat
    # Regression coefficients
    model_coef = trained_estimator.coef_
    # Regression coefficients directly from ML models
    ml_model_coef = trained_estimator.model_dict["ml_model"].coef_
    # Extracts original forecasts from the pipeline
    pipeline_original_forecast = trained_estimator.forecast[-forecast_horizon:].reset_index(drop=True)

    # The returned coefficients from ML model must be the same as final coefficients
    assert (ml_model_coef == model_coef[0].values).all()

    # Checks to see if multiplying the variables with regression coefficients is possible
    # and works as expected
    # Multiplies each row of the design matrix by regression coefficients (element-wise)
    # and returns a dataframe of same shape as ``forecast_x_mat``
    forecast_x_mat_weighted = forecast_x_mat * ml_model_coef

    assert list(forecast_x_mat) == ["Intercept", "ct1", "ct2"]
    assert list(model_coef.index) == ["Intercept", "ct1", "ct2"]
    # We expect the forecasted data and design matrix having a size equal to
    # "trained data size" + "forecast horizon"
    assert len(forecast_x_mat) == len(train_df) + forecast_horizon
    assert forecast_x_mat_weighted.shape == forecast_x_mat.shape
    assert len(trained_estimator.forecast) == len(train_df) + forecast_horizon

    # Checks to see if the manually calculated forecast is consistent
    # with returned forecast in terms of correlation
    # Note that they will not be the same scale due to (affine) transformations
    calculated_forecast = forecast_x_mat_weighted.sum(axis=1)
    forecast_corr = calculated_forecast.corr(trained_estimator.forecast["y"])
    assert round(forecast_corr, 3) == 1.0

    pred_df = trained_estimator.predict(test_df[:forecast_horizon])
    pred_df.rename(columns={"forecast": "y"}, inplace=True)
    cols = ["ts", "y", "forecast_lower", "forecast_upper", QUANTILE_SUMMARY_COL, "err_std"]

    # We expect the predictions to be the same as original predictions
    # since we are passing the same test data
    assert_equal(pred_df[cols], pipeline_original_forecast[cols])

    # Tries a new forecast horizon (10)
    new_pred_df = trained_estimator.predict(test_df[:10])
    new_pred_df.rename(columns={"forecast": "y"}, inplace=True)
    cols = ["ts", "y", "forecast_lower", "forecast_upper", QUANTILE_SUMMARY_COL, "err_std"]

    # Checks to see if forecat and design matrix are updated in the estimator
    assert len(new_pred_df) == 10
    assert len(trained_estimator.forecast) == 10
    assert len(trained_estimator.forecast_x_mat) == 10
    # We expect the predictions to be the same as the first 10 rows of original predictions,
    # since we are passing the first 10 rows of the same test data
    assert_equal(new_pred_df[cols], pipeline_original_forecast[:10][cols])
