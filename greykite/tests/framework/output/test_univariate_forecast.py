import datetime
import math
import sys
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from pandas.util.testing import assert_series_equal
from sklearn.pipeline import Pipeline

from greykite.common import constants as cst
from greykite.common.evaluation import ElementwiseEvaluationMetricEnum
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import gen_sliced_df
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries
from greykite.framework.output.univariate_forecast import UnivariateForecast
from greykite.framework.pipeline.utils import get_forecast
from greykite.sklearn.estimator.prophet_estimator import ProphetEstimator
from greykite.sklearn.estimator.silverkite_estimator import SilverkiteEstimator


try:
    import prophet  # noqa
except ModuleNotFoundError:
    pass


@pytest.fixture
def df():
    return pd.DataFrame({
        cst.TIME_COL: [
            datetime.datetime(2018, 1, 1),
            datetime.datetime(2018, 1, 2),
            datetime.datetime(2018, 1, 3),
            datetime.datetime(2018, 1, 4)],
        cst.ACTUAL_COL: [1, 2, 3, 4],
        cst.PREDICTED_COL: [1, 4, 1, 2],
        cst.PREDICTED_LOWER_COL: [1, 1, 1, 1],
        cst.PREDICTED_UPPER_COL: [4, 5, 4, 4],
        cst.NULL_PREDICTED_COL: [1.5, 1.5, 1.5, 1.5]
    })


@pytest.fixture
def df2():
    return pd.DataFrame({
        cst.TIME_COL: pd.date_range(start="2018-01-01", periods=7),
        cst.ACTUAL_COL:
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        cst.PREDICTED_COL:
            [1.0, 4.0, 3.0, 2.0, 3.0, 4.0, 8.0],
        cst.PREDICTED_LOWER_COL:
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        cst.PREDICTED_UPPER_COL:
            [4.0, 5.0, 4.0, 4.0, 5.0, 6.0, 9.0],
        cst.NULL_PREDICTED_COL:
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    })


def test_univariate_forecast(df):
    """Checks univariate forecast class"""
    # Without test_start_date
    forecast = UnivariateForecast(
        df,
        train_end_date=datetime.datetime(2018, 1, 2),
        test_start_date=None,
        forecast_horizon=2)

    assert forecast.forecast_horizon == 2
    assert forecast.df_train.shape == (2, 6)
    assert forecast.df_test.shape == (2, 6)
    assert forecast.relative_error_tolerance is None

    # evaluation metrics
    enum = EvaluationMetricEnum.Correlation
    assert forecast.train_evaluation[enum.get_metric_name()] == 1.0
    assert forecast.test_evaluation[enum.get_metric_name()] == 1.0
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert forecast.train_evaluation[enum.get_metric_name()] == 1.0
    assert forecast.test_evaluation[enum.get_metric_name()] == 2.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert forecast.train_evaluation[enum.get_metric_name()] == math.sqrt(2)
    assert forecast.test_evaluation[enum.get_metric_name()] == 2.0
    enum = EvaluationMetricEnum.MedianAbsoluteError
    assert forecast.train_evaluation[enum.get_metric_name()] == 1.0
    assert forecast.test_evaluation[enum.get_metric_name()] == 2.0
    enum = EvaluationMetricEnum.MeanAbsolutePercentError
    assert forecast.train_evaluation[enum.get_metric_name()] == 50.0
    assert forecast.test_evaluation[enum.get_metric_name()] == pytest.approx(58.33333, 1e-4)
    assert forecast.train_evaluation[cst.R2_null_model_score] == -7.0
    assert forecast.test_evaluation[cst.R2_null_model_score] == pytest.approx(0.058824, 1e-4)
    assert forecast.train_evaluation[cst.FRACTION_OUTSIDE_TOLERANCE] is None
    assert forecast.test_evaluation[cst.FRACTION_OUTSIDE_TOLERANCE] is None
    # validation metrics
    assert forecast.train_evaluation[cst.PREDICTION_BAND_WIDTH] == 250.0
    assert forecast.test_evaluation[cst.PREDICTION_BAND_WIDTH] == 87.5
    assert forecast.train_evaluation[cst.PREDICTION_BAND_COVERAGE] == 0.5
    assert forecast.test_evaluation[cst.PREDICTION_BAND_COVERAGE] == 0.5
    assert forecast.train_evaluation[cst.LOWER_BAND_COVERAGE] == 0.5
    assert forecast.test_evaluation[cst.LOWER_BAND_COVERAGE] == 0.0
    assert forecast.train_evaluation[cst.UPPER_BAND_COVERAGE] == 0.0
    assert forecast.test_evaluation[cst.UPPER_BAND_COVERAGE] == 0.5
    assert forecast.train_evaluation[cst.COVERAGE_VS_INTENDED_DIFF] == pytest.approx(-0.45)
    assert forecast.test_evaluation[cst.COVERAGE_VS_INTENDED_DIFF] == pytest.approx(-0.45)

    # With test_start_date, relative_error_tolerance
    with pytest.warns(UserWarning):
        forecast = UnivariateForecast(
            df,
            train_end_date=datetime.datetime(2018, 1, 2),
            test_start_date=datetime.datetime(2018, 1, 4),
            relative_error_tolerance=0.05)

        assert forecast.forecast_horizon is None
        assert forecast.df_train.shape == (2, 6)
        assert forecast.df_test.shape == (1, 6)
        assert forecast.relative_error_tolerance == 0.05

        # evaluation metrics (train_metrics remain the same, test_metrics change)
        enum = EvaluationMetricEnum.Correlation
        assert forecast.train_evaluation[enum.get_metric_name()] == 1.0
        assert forecast.test_evaluation[enum.get_metric_name()] is None
        enum = EvaluationMetricEnum.MeanAbsoluteError
        assert forecast.train_evaluation[enum.get_metric_name()] == 1.0
        assert forecast.test_evaluation[enum.get_metric_name()] == 2.0
        enum = EvaluationMetricEnum.RootMeanSquaredError
        assert forecast.train_evaluation[enum.get_metric_name()] == math.sqrt(2)
        assert forecast.test_evaluation[enum.get_metric_name()] == 2.0
        enum = EvaluationMetricEnum.MedianAbsoluteError
        assert forecast.train_evaluation[enum.get_metric_name()] == 1.0
        assert forecast.test_evaluation[enum.get_metric_name()] == 2.0
        enum = EvaluationMetricEnum.MeanAbsolutePercentError
        assert forecast.train_evaluation[enum.get_metric_name()] == 50.0
        assert forecast.test_evaluation[enum.get_metric_name()] == 50.0
        assert forecast.train_evaluation[cst.R2_null_model_score] == -7.0
        assert forecast.test_evaluation[cst.R2_null_model_score] == 0.36
        assert forecast.train_evaluation[cst.FRACTION_OUTSIDE_TOLERANCE] == 0.5
        assert forecast.test_evaluation[cst.FRACTION_OUTSIDE_TOLERANCE] == 1.0
        # validation metrics
        assert forecast.train_evaluation[cst.PREDICTION_BAND_WIDTH] == 250.0
        assert forecast.test_evaluation[cst.PREDICTION_BAND_WIDTH] == 75.0
        assert forecast.train_evaluation[cst.PREDICTION_BAND_COVERAGE] == 0.5
        assert forecast.test_evaluation[cst.PREDICTION_BAND_COVERAGE] == 0.0
        assert forecast.train_evaluation[cst.LOWER_BAND_COVERAGE] == 0.5
        assert forecast.test_evaluation[cst.LOWER_BAND_COVERAGE] == 0.0
        assert forecast.train_evaluation[cst.UPPER_BAND_COVERAGE] == 0.0
        assert forecast.test_evaluation[cst.UPPER_BAND_COVERAGE] == 0.0
        assert forecast.train_evaluation[cst.COVERAGE_VS_INTENDED_DIFF] == pytest.approx(-0.45)
        assert forecast.test_evaluation[cst.COVERAGE_VS_INTENDED_DIFF] == pytest.approx(-0.95)


def test_subset_columns(df):
    """Tests if intervals and null prediction are truly optional,
    and relative_error_tolerance parameter"""
    forecast = UnivariateForecast(df[[cst.TIME_COL, cst.ACTUAL_COL, cst.PREDICTED_COL]],
                                  predicted_lower_col=None,
                                  predicted_upper_col=None,
                                  null_model_predicted_col=None,
                                  train_end_date=datetime.datetime(2018, 1, 2),
                                  relative_error_tolerance=0.7)

    forecast_full = UnivariateForecast(df, train_end_date=datetime.datetime(2018, 1, 2))

    for enum in EvaluationMetricEnum:
        assert forecast.train_evaluation[enum.get_metric_name()] == forecast_full.train_evaluation[enum.get_metric_name()]
        assert forecast.test_evaluation[enum.get_metric_name()] == forecast_full.test_evaluation[enum.get_metric_name()]
    for metric in [cst.R2_null_model_score, cst.PREDICTION_BAND_WIDTH, cst.PREDICTION_BAND_COVERAGE, cst.LOWER_BAND_COVERAGE,
                   cst.UPPER_BAND_COVERAGE, cst.COVERAGE_VS_INTENDED_DIFF]:
        assert forecast.train_evaluation[metric] is None
        assert forecast.test_evaluation[metric] is None

    assert forecast.relative_error_tolerance == 0.7
    assert forecast.train_evaluation[cst.FRACTION_OUTSIDE_TOLERANCE] == 0.5
    assert forecast.test_evaluation[cst.FRACTION_OUTSIDE_TOLERANCE] == 0.0


def test_input_validation(df):
    """Tests input validation"""
    with pytest.raises(ValueError, match="`coverage` must be provided"):
        UnivariateForecast(df, train_end_date=datetime.datetime(2018, 1, 2), coverage=None)

    with pytest.raises(ValueError, match="`coverage` must be between 0.0 and 1.0"):
        UnivariateForecast(df, train_end_date=datetime.datetime(2018, 1, 2), coverage=80.0)

    with pytest.raises(ValueError, match="2018-01-05 is not found in time column"):
        UnivariateForecast(df, train_end_date="2018-01-05")

    with pytest.raises(ValueError, match="Column not found in data frame"):
        UnivariateForecast(df, actual_col="not_a_column")


def test_no_train_end_date(df):
    """Tests if train end date can be None"""
    forecast = UnivariateForecast(
        df,
        train_end_date=None)
    forecast2 = UnivariateForecast(
        df,
        train_end_date=datetime.datetime(2018, 1, 4))
    assert_equal(forecast.train_evaluation, forecast2.train_evaluation)
    assert forecast.test_evaluation is None


def test_partial_test_data():
    """Tests if forecast evaluation can handle partially missing data"""
    df = pd.DataFrame({
        cst.TIME_COL: ["2018-01-01", datetime.datetime(2018, 1, 2), "2018-01-03", "2018-01-04", "2018-01-05"],
        cst.ACTUAL_COL: [1, 2, 3, 2, np.nan],
        cst.PREDICTED_COL: [1, 4, 1, 2, 4],
        cst.PREDICTED_LOWER_COL: [1, 1, 1, 1, 2],
        cst.PREDICTED_UPPER_COL: [4, 5, 4, 4, 6],
        cst.NULL_PREDICTED_COL: [1.5, 1.5, 1.5, 1.5, 1.5]
    })

    with pytest.warns(UserWarning) as record:
        forecast = UnivariateForecast(df, train_end_date=datetime.datetime(2018, 1, 2))
        forecast2 = UnivariateForecast(df.iloc[:4, ], train_end_date=datetime.datetime(2018, 1, 2))
        assert forecast.test_na_count == 1
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0:2]
        assert_equal(forecast.train_evaluation, forecast2.train_evaluation)
        assert_equal(forecast.test_evaluation, forecast2.test_evaluation)


def test_no_test_data():
    """Tests if test evaluation is skipped when there are no test data"""
    df = pd.DataFrame({
        cst.TIME_COL: ["2018-01-01", datetime.datetime(2018, 1, 2), "2018-01-03", "2018-01-04"],
        cst.ACTUAL_COL: [1, 2, np.nan, np.nan],
        cst.PREDICTED_COL: [1, 4, 1, 2],
        cst.PREDICTED_LOWER_COL: [1, 1, 1, 1],
        cst.PREDICTED_UPPER_COL: [4, 5, 4, 4],
        cst.NULL_PREDICTED_COL: [1.5, 1.5, 1.5, 1.5]
    })
    forecast = UnivariateForecast(df, train_end_date=datetime.datetime(2018, 1, 2))
    assert forecast.test_na_count == 2
    assert forecast.train_evaluation is not None
    assert forecast.test_evaluation is None


def test_custom_loss_function(df):
    """Tests the custom loss function argument"""
    def custom_loss(y_pred, y_true):
        """Root mean absolute error"""
        return np.sqrt(np.sum(np.abs(np.array(y_pred) - np.array(y_true))))
    forecast = UnivariateForecast(df, train_end_date=datetime.datetime(2018, 1, 2), r2_loss_function=custom_loss)
    assert forecast.train_evaluation[cst.R2_null_model_score] == 1 - math.sqrt(2)
    assert forecast.test_evaluation[cst.R2_null_model_score] == 0


def test_plot(df):
    """Tests plot function"""
    forecast = UnivariateForecast(df, train_end_date=datetime.datetime(2018, 1, 2))
    fig = forecast.plot()
    assert fig is not None

    forecast = UnivariateForecast(df, train_end_date=datetime.datetime(2018, 1, 4))
    fig = forecast.plot(vertical_line_color="green")
    assert fig is not None


def test_get_grouping_evaluation(df2):
    """Tests get_grouping_evaluation function"""
    forecast = UnivariateForecast(df2, train_end_date=datetime.datetime(2018, 1, 5))

    # MAPE, groupby_time_feature, train set
    metric = EvaluationMetricEnum.MeanAbsolutePercentError
    metric_name = metric.get_metric_name()
    grouped_df = forecast.get_grouping_evaluation(
        score_func=metric.get_metric_func(),
        score_func_name=metric_name,
        which="train",
        groupby_time_feature="dow")
    expected = pd.DataFrame({
        "dow": [1, 2, 3, 4, 5],  # Monday, Tuesday, etc. Time feature is used as column name
        f"train {metric_name}": [0.0, 100.0, 0.0, 50.0, 40.0]
    })
    assert_equal(grouped_df, expected)

    # MSE, groupby_sliding_window_size
    metric = EvaluationMetricEnum.MeanSquaredError
    metric_name = metric.get_metric_name()
    grouped_df = forecast.get_grouping_evaluation(
        score_func=metric.get_metric_func(),
        score_func_name=metric_name,
        which="train",
        groupby_sliding_window_size=2)
    expected = pd.DataFrame({
        f"{cst.TIME_COL}_downsample": [
            datetime.datetime(2018, 1, 1),
            datetime.datetime(2018, 1, 3),
            datetime.datetime(2018, 1, 5)],
        f"train {metric_name}": [0.0, 2.0, 4.0]
    })
    assert_equal(grouped_df, expected)

    # MAE, groupby_custom_column, test set
    forecast = UnivariateForecast(df2, train_end_date=datetime.datetime(2018, 1, 2))
    metric = EvaluationMetricEnum.MeanAbsoluteError
    custom_groups = pd.Series(["g1", "g2", "g1", "g3", "g2"], name="custom_groups")
    grouped_df = forecast.get_grouping_evaluation(
        score_func=metric.get_metric_func(),
        score_func_name=None,
        which="test",
        groupby_custom_column=custom_groups)
    expected = pd.DataFrame({
        "custom_groups": ["g1", "g2", "g3"],
        "test metric": [1.0, 1.5, 2.0]
    })
    assert_equal(grouped_df, expected)


def test_plot_grouping_evaluation(df2):
    """Tests plot_grouping_evaluation function"""
    forecast = UnivariateForecast(df2, train_end_date=datetime.datetime(2018, 1, 5))
    # MAPE, groupby_time_feature, train set
    metric = EvaluationMetricEnum.MeanAbsolutePercentError
    metric_name = metric.get_metric_name()
    fig = forecast.plot_grouping_evaluation(
        score_func=metric.get_metric_func(),
        score_func_name=metric_name,
        which="train",
        groupby_time_feature="dow")
    assert fig.data[0].name == f"train {metric_name}"
    assert fig.layout.xaxis.title.text == "dow"
    assert fig.layout.yaxis.title.text == f"train {metric_name}"
    assert fig.layout.title.text == f"train {metric_name} vs dow"
    assert fig.layout.title.x == 0.5
    assert fig.data[0].x.shape[0] == 5

    # MSE, groupby_sliding_window_size, train set
    metric = EvaluationMetricEnum.MeanSquaredError
    metric_name = metric.get_metric_name()
    fig = forecast.plot_grouping_evaluation(
        score_func=metric.get_metric_func(),
        score_func_name=metric_name,
        which="train",
        groupby_sliding_window_size=2)  # there are 5 training points, so this creates groups of size (1, 2, 2)
    assert fig.data[0].name == f"train {metric_name}"
    assert fig.layout.xaxis.title.text == f"{cst.TIME_COL}_downsample"
    assert fig.layout.yaxis.title.text == f"train {metric_name}"
    assert fig.layout.title.text == f"train {metric_name} vs {cst.TIME_COL}_downsample"
    assert fig.layout.title.x == 0.5
    assert fig.data[0].x.shape[0] == 3

    # MAE, groupby_custom_column, test set
    forecast = UnivariateForecast(df2, train_end_date=datetime.datetime(2018, 1, 2))
    metric = EvaluationMetricEnum.MeanAbsoluteError
    metric_name = metric.get_metric_name()
    custom_groups = pd.Series(["g1", "g2", "g1", "g3", "g2"], name="custom_groups")
    fig = forecast.plot_grouping_evaluation(
        groupby_custom_column=custom_groups,
        score_func=metric.get_metric_func(),
        score_func_name=metric_name,
        which="test",
        title=None)
    assert fig.data[0].name == f"test {metric_name}"
    assert fig.layout.xaxis.title.text == "custom_groups"
    assert fig.layout.yaxis.title.text == f"test {metric_name}"
    assert fig.layout.title.text == f"test {metric_name} vs custom_groups"
    assert fig.layout.title.x == 0.5
    assert fig.data[0].x.shape[0] == 3

    # custom xlabel, ylabel, title
    fig = forecast.plot_grouping_evaluation(
        groupby_custom_column=custom_groups,
        score_func=metric.get_metric_func(),
        score_func_name=metric_name,
        which="test",
        xlabel="Custom labels",
        ylabel="Mean Absolute Error of y",
        title="Mean Absolute Error of y by Custom labels")
    assert fig.layout.xaxis.title.text == "Custom labels"
    assert fig.layout.yaxis.title.text == "Mean Absolute Error of y"
    assert fig.layout.title.text == "Mean Absolute Error of y by Custom labels"
    assert fig.layout.title.x == 0.5


def test_autocomplete_map_func_dict(df2):
    """Tests autocomplete_map_func_dict function"""
    map_func_dict = {
        "residual": ElementwiseEvaluationMetricEnum.Residual.name,
        "squared_error": ElementwiseEvaluationMetricEnum.SquaredError.name,
        "coverage": ElementwiseEvaluationMetricEnum.Coverage.name,
        "custom_metric": lambda row: (row[cst.ACTUAL_COL] - row[cst.PREDICTED_COL])**4
    }

    df_renamed = df2.rename({
        cst.TIME_COL: "custom_time_col",
        cst.ACTUAL_COL: "custom_actual_col",
        cst.PREDICTED_COL: "custom_predicted_col",
        cst.PREDICTED_LOWER_COL: "custom_predicted_lower_col",
        cst.PREDICTED_UPPER_COL: "custom_predicted_upper_col",
        cst.NULL_PREDICTED_COL: "custom_null_predicted_col",
    })

    forecast = UnivariateForecast(df_renamed, train_end_date=datetime.datetime(2018, 1, 5))
    map_func_dict = forecast.autocomplete_map_func_dict(map_func_dict)

    actual = df2.apply(map_func_dict["residual"], axis=1)
    expected = (df2[cst.ACTUAL_COL] - df2[cst.PREDICTED_COL])
    assert_series_equal(actual, expected)

    actual = df2.apply(map_func_dict["squared_error"], axis=1)
    expected = (df2[cst.ACTUAL_COL] - df2[cst.PREDICTED_COL]).pow(2)
    assert_series_equal(actual, expected)

    actual = df2.apply(map_func_dict["coverage"], axis=1)
    expected = ((df2[cst.ACTUAL_COL] > df2[cst.PREDICTED_LOWER_COL]) & (df2[cst.ACTUAL_COL] < df2[cst.PREDICTED_UPPER_COL])).astype('float')
    assert_series_equal(actual, expected)

    actual = df2.apply(map_func_dict["custom_metric"], axis=1)
    expected = (df2[cst.ACTUAL_COL] - df2[cst.PREDICTED_COL]).pow(4)
    assert_series_equal(actual, expected)

    assert forecast.autocomplete_map_func_dict(None) is None

    valid_names = ", ".join(ElementwiseEvaluationMetricEnum.__dict__["_member_names_"])
    with pytest.raises(ValueError, match=f"unknown_func is not a recognized elementwise "
                                         f"evaluation metric. Must be one of: {valid_names}"):
        map_func_dict = {"unknown_func": "unknown_func"}
        forecast.autocomplete_map_func_dict(map_func_dict)


def test_get_flexible_grouping_evaluation(df2):
    """Tests get_flexible_grouping_evaluation function"""
    forecast = UnivariateForecast(df2, train_end_date=datetime.datetime(2018, 1, 5))
    # Checks residual quantiles, MSE/median squared error, and coverage
    map_func_dict = {
        "residual": ElementwiseEvaluationMetricEnum.Residual.name,
        "squared_error": ElementwiseEvaluationMetricEnum.SquaredError.name,
        "coverage": ElementwiseEvaluationMetricEnum.Coverage.name
    }
    agg_kwargs = {
        "residual_mean": pd.NamedAgg(column="residual", aggfunc=np.nanmean),
        "residual_q05": pd.NamedAgg(column="residual", aggfunc=partial(np.nanquantile, q=0.05)),
        "residual_q95": pd.NamedAgg(column="residual", aggfunc=partial(np.nanquantile, q=0.95)),
        "MSE": pd.NamedAgg(column="squared_error", aggfunc=np.nanmean),
        "median_squared_error": pd.NamedAgg(column="squared_error", aggfunc=np.nanmedian),
        "coverage": pd.NamedAgg(column="coverage", aggfunc=np.nanmean),
    }

    result = forecast.get_flexible_grouping_evaluation(
        which="train",
        groupby_time_feature="dow",
        groupby_sliding_window_size=None,
        groupby_custom_column=None,
        map_func_dict=map_func_dict,
        agg_kwargs=agg_kwargs,
        extend_col_names=False)
    expected = pd.DataFrame({
        # Only one value per group, so the mean/median/quantiles are the same
        "residual_mean": [0.0, -2.0, 0.0, 2.0, 2.0],
        "residual_q05": [0.0, -2.0, 0.0, 2.0, 2.0],
        "residual_q95": [0.0, -2.0, 0.0, 2.0, 2.0],
        "MSE": [0.0, 4.0, 0.0, 4.0, 4.0],
        "median_squared_error": [0.0, 4.0, 0.0, 4.0, 4.0],
        "coverage": [0.0, 1.0, 1.0, 0.0, 0.0],
    }, index=pd.Series([1, 2, 3, 4, 5], name="dow"))
    assert_frame_equal(result, expected)

    # Equivalent way to specify `map_func_dict` (without autocomplete)
    map_func_dict = {
        "residual": lambda row: ElementwiseEvaluationMetricEnum.Residual.get_metric_func()(
            row[forecast.actual_col],
            row[forecast.predicted_col]),
        "squared_error": lambda row: ElementwiseEvaluationMetricEnum.SquaredError.get_metric_func()(
            row[forecast.actual_col],
            row[forecast.predicted_col]),
        "coverage": lambda row: ElementwiseEvaluationMetricEnum.Coverage.get_metric_func()(
            row[forecast.actual_col],
            row[forecast.predicted_lower_col],
            row[forecast.predicted_upper_col]),
    }
    result = forecast.get_flexible_grouping_evaluation(
        which="train",
        groupby_time_feature="dow",
        groupby_sliding_window_size=None,
        groupby_custom_column=None,
        map_func_dict=map_func_dict,
        agg_kwargs=agg_kwargs,
        extend_col_names=False)
    assert_frame_equal(result, expected)

    # Equivalent way to specify `map_func_dict` (without autocomplete)
    map_func_dict = {
        "residual": lambda row: row[cst.ACTUAL_COL] - row[cst.PREDICTED_COL],
        "squared_error": lambda row: (row[cst.ACTUAL_COL] - row[cst.PREDICTED_COL])**2,
        "coverage": lambda row: 1.0 if row[cst.PREDICTED_LOWER_COL] < row[cst.ACTUAL_COL] < row[cst.PREDICTED_UPPER_COL] else 0.0
    }
    result = forecast.get_flexible_grouping_evaluation(
        which="train",
        groupby_time_feature="dow",
        groupby_sliding_window_size=None,
        groupby_custom_column=None,
        map_func_dict=map_func_dict,
        agg_kwargs=agg_kwargs,
        extend_col_names=False)
    assert_frame_equal(result, expected)

    # Groupby sliding window
    result = forecast.get_flexible_grouping_evaluation(
        which="train",
        groupby_time_feature=None,
        groupby_sliding_window_size=3,
        groupby_custom_column=None,
        map_func_dict=map_func_dict,
        agg_kwargs=agg_kwargs,
        extend_col_names=False)
    expected = pd.DataFrame({
        "residual_mean": [-1.0, 4/3],
        "residual_q05": [-1.9, 0.2],
        "residual_q95": [-0.1, 2.0],
        "MSE": [2.0, 2.0 + 2/3],
        "median_squared_error": [2.0, 4.0],
        "coverage": [0.5, 1/3],
    }, index=pd.DatetimeIndex(["2018-01-01", "2018-01-04"], name="ts_downsample"))
    assert_frame_equal(result, expected)

    # On test set with custom groupby column
    custom_groups = pd.Series(["val1"], name="value_group").repeat(forecast.df_test.shape[0])
    result = forecast.get_flexible_grouping_evaluation(
        which="test",
        groupby_time_feature=None,
        groupby_sliding_window_size=None,
        groupby_custom_column=custom_groups,
        map_func_dict=map_func_dict,
        agg_kwargs=agg_kwargs)

    colindex = pd.Index(
        ["residual_mean", "residual_q05", "residual_q95",
         "MSE", "median_squared_error", "coverage"])
    expected = pd.DataFrame(
        [[0.5, -0.85, 1.85, 2.5, 2.5, 0.5]],
        columns=colindex,
        index=pd.Series(["val1"], name=custom_groups.name))
    assert_frame_equal(result, expected)


def test_plot_flexible_grouping_evaluation():
    """Tests plot_flexible_grouping_evaluation function"""
    df = gen_sliced_df(sample_size_dict={"a": 300, "b": 200, "c": 300, "d": 80, "e": 300})
    actual_col = "y"
    predicted_col = "y_hat"
    groupby_col = "x"
    groupby_col2 = "z"
    df = df[[actual_col, predicted_col, groupby_col, groupby_col2]]
    df[cst.TIME_COL] = pd.date_range(start="2020-01-01", periods=df.shape[0], freq="D")
    end_index = math.floor(df.shape[0] * 0.8)
    forecast = UnivariateForecast(
        df,
        train_end_date=df[cst.TIME_COL][end_index],
        time_col=cst.TIME_COL,
        actual_col=actual_col,
        predicted_col=predicted_col,
        predicted_lower_col=None,
        predicted_upper_col=None,
        null_model_predicted_col=None)

    # MSE and quantiles of squared error
    metric_col = "squared_err"
    map_func_dict = {metric_col: ElementwiseEvaluationMetricEnum.SquaredError.name}
    agg_kwargs = {f"Q{quantile}": pd.NamedAgg(column=metric_col, aggfunc=partial(np.nanquantile, q=quantile)) for quantile in [0.1, 0.9]}
    agg_kwargs.update({"mean": pd.NamedAgg(column=metric_col, aggfunc=np.nanmean)})

    # group by "dom", "auto-fill" styling
    fig = forecast.plot_flexible_grouping_evaluation(
        which="train",
        groupby_time_feature="dom",
        groupby_sliding_window_size=None,
        groupby_custom_column=None,
        map_func_dict=map_func_dict,
        agg_kwargs=agg_kwargs,
        extend_col_names=False,
        y_col_style_dict="auto-fill",
        default_color="rgba(0, 145, 202, 1.0)",
        xlabel=None,
        ylabel=metric_col,
        title=None,
        showlegend=True)

    assert [fig.data[i].name for i in range(len(fig.data))] == ["Q0.1", "mean", "Q0.9"]
    assert fig.layout.xaxis.title.text == "dom"
    assert fig.layout.yaxis.title.text == metric_col
    assert fig.layout.title.text == f"{metric_col} vs dom"
    assert fig.layout.title.x == 0.5
    assert fig.data[0].x.shape[0] == 31  # 31 unique days in month
    assert fig.data[1].line["color"] == "rgba(0, 145, 202, 1.0)"
    assert fig.data[1].fill == "tonexty"  # from auto-fill
    assert fig.layout.showlegend

    # group by sliding window, "auto" styling
    # provide default color, xlabel, hide legend
    fig = forecast.plot_flexible_grouping_evaluation(
        which="train",
        groupby_time_feature=None,
        groupby_sliding_window_size=7,
        groupby_custom_column=None,
        map_func_dict=map_func_dict,
        agg_kwargs=agg_kwargs,
        extend_col_names=False,
        y_col_style_dict="auto",
        default_color="rgba(145, 0, 202, 1.0)",
        xlabel="ts",
        ylabel=None,
        title=None,
        showlegend=False)

    assert [fig.data[i].name for i in range(len(fig.data))] == ["Q0.1", "mean", "Q0.9"]
    assert fig.layout.xaxis.title.text == "ts"
    assert fig.layout.yaxis.title.text is None
    assert fig.layout.title.text is None
    assert fig.layout.title.x == 0.5
    assert fig.data[0].x[0] == datetime.datetime(2020, 1, 1, 0, 0)
    assert fig.data[1].line["color"] == "rgba(145, 0, 202, 1.0)"
    assert fig.data[1].fill is None
    assert not fig.layout.showlegend

    # custom groups, "plotly" styling, provide ylabel, title
    fig = forecast.plot_flexible_grouping_evaluation(
        which="train",
        groupby_time_feature=None,
        groupby_sliding_window_size=None,
        groupby_custom_column=forecast.df_train["x"],
        map_func_dict=map_func_dict,
        agg_kwargs=agg_kwargs,
        extend_col_names=False,
        y_col_style_dict="plotly",
        default_color=None,
        xlabel=None,
        ylabel=metric_col,
        title="custom title",
        showlegend=True)

    assert [fig.data[i].name for i in range(len(fig.data))] == ["Q0.1", "Q0.9", "mean"]  # not sorted
    assert fig.layout.xaxis.title.text == "x"
    assert fig.layout.yaxis.title.text == metric_col
    assert fig.layout.title.text == "custom title"
    assert fig.layout.title.x == 0.5
    assert list(fig.data[0].x) == list("abcde")
    assert fig.data[0].line["color"] is None  # color is up to plotly
    assert fig.data[1].fill is None
    assert fig.layout.showlegend

    # test set, absolute percent error, custom `y_col_style_dict` styling
    metric_col = "squared_error"
    map_func_dict = {
        metric_col: ElementwiseEvaluationMetricEnum.AbsolutePercentError.name
    }
    agg_kwargs = {
        "median": pd.NamedAgg(column=metric_col, aggfunc=np.nanmedian),
        "mean": pd.NamedAgg(column=metric_col, aggfunc=np.nanmean),
    }
    y_col_style_dict = {
        "median": {
            "mode": "lines+markers",
            "line": {
                "color": "rgba(202, 145, 0, 0.5)"
            }
        },
        "mean": {
            "mode": "lines+markers",
            "line": {
                "color": "rgba(0, 145, 202, 1.0)"
            }
        },
    }
    with pytest.warns(UserWarning, match="true_val is less than 1e-8"):
        fig = forecast.plot_flexible_grouping_evaluation(
            which="test",
            groupby_time_feature="dow",
            groupby_sliding_window_size=None,
            groupby_custom_column=None,
            map_func_dict=map_func_dict,
            agg_kwargs=agg_kwargs,
            extend_col_names=False,
            y_col_style_dict=y_col_style_dict,
            xlabel="x value",
            ylabel="y value",
            title="error plot",
            showlegend=True)
        assert [fig.data[i].name for i in range(len(fig.data))] == ["median", "mean"]  # not sorted
        assert fig.layout.xaxis.title.text == "x value"
        assert fig.layout.yaxis.title.text == "y value"
        assert fig.layout.title.text == "error plot"
        assert fig.layout.title.x == 0.5
        assert len(fig.data[0].x) == 7
        assert fig.data[0].mode == "lines+markers"
        assert fig.data[1].mode == "lines+markers"
        assert fig.data[0].line["color"] == y_col_style_dict["median"]["line"]["color"]
        assert fig.data[1].line["color"] == y_col_style_dict["mean"]["line"]["color"]
        assert fig.data[1].fill is None
        assert fig.layout.showlegend

    # median actual vs forecast value by group
    agg_kwargs = {
        "y_median": pd.NamedAgg(column="y", aggfunc=np.nanmedian),
        "y_hat_median": pd.NamedAgg(column="y_hat", aggfunc=np.nanmedian),
    }
    fig = forecast.plot_flexible_grouping_evaluation(
        which="train",
        groupby_time_feature="dow",
        groupby_sliding_window_size=None,
        groupby_custom_column=None,
        map_func_dict=None,
        agg_kwargs=agg_kwargs,
        extend_col_names=True,
        y_col_style_dict="plotly",
        xlabel=None,
        ylabel=forecast.ylabel,
        title="true vs actual by dow",
        showlegend=True)
    assert [fig.data[i].name for i in range(len(fig.data))] == ["y_median", "y_hat_median"]
    assert fig.layout.xaxis.title.text == "dow"
    assert fig.layout.yaxis.title.text == "y"
    assert fig.layout.title.text == "true vs actual by dow"
    assert fig.layout.title.x == 0.5
    assert len(fig.data[0].x) == 7
    assert fig.layout.showlegend


def test_make_univariate_time_series(df):
    """Tests make_univariate_time_series function"""
    forecast = UnivariateForecast(df, train_end_date=datetime.datetime(2018, 1, 2))
    ts = UnivariateTimeSeries()
    ts.load_data(pd.DataFrame({
        cst.TIME_COL: df[cst.TIME_COL],
        cst.VALUE_COL: df[cst.PREDICTED_COL]
    }), cst.TIME_COL, cst.VALUE_COL)
    assert forecast.make_univariate_time_series().df.equals(ts.df)


def test_plot_components():
    """Test plot_components of UnivariateForecast class"""
    X = pd.DataFrame({
        cst.TIME_COL: pd.date_range("2018-01-01", periods=10, freq="D"),
        cst.VALUE_COL: np.arange(1, 11)
    })
    coverage = 0.95

    # Test Silverkite
    trained_model = Pipeline([("estimator", SilverkiteEstimator(coverage=coverage))])
    with pytest.warns(Warning) as record:
        trained_model.fit(X, X[cst.VALUE_COL])
        assert "No slice had sufficient sample size" in record[0].message.args[0]
    forecast = get_forecast(X, trained_model)

    with pytest.warns(Warning) as record:
        title = "Custom component plot"
        fig = forecast.plot_components(names=["trend", "YEARLY_SEASONALITY", "DUMMY"], title=title)

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


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_plot_components_prophet():
    X = pd.DataFrame({
        cst.TIME_COL: pd.date_range("2018-01-01", periods=10, freq="D"),
        cst.VALUE_COL: np.arange(1, 11)
    })
    coverage = 0.95

    # Test Prophet
    trained_model = Pipeline([("estimator", ProphetEstimator(coverage=coverage))])
    trained_model.fit(X, X[cst.VALUE_COL])
    forecast = get_forecast(X, trained_model)
    fig = forecast.plot_components()
    assert fig is not None
