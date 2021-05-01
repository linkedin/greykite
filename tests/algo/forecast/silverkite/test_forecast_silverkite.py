import datetime
from datetime import timedelta

import matplotlib
import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal

from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector
from greykite.algo.changepoint.adalasso.changepoints_utils import get_changepoint_dates_from_changepoints_dict
from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import cols_interact
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import generate_holiday_events
from greykite.common.constants import ADJUSTMENT_DELTA_COL
from greykite.common.constants import END_DATE_COL
from greykite.common.constants import ERR_STD_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import START_DATE_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import calc_pred_err
from greykite.common.features.adjust_anomalous_data import adjust_anomalous_data
from greykite.common.features.normalize import normalize_df
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.features.timeseries_features import fourier_series_multi_fcn
from greykite.common.features.timeseries_features import get_changepoint_string
from greykite.common.features.timeseries_features import get_evenly_spaced_changepoints_values
from greykite.common.features.timeseries_features import get_fourier_col_name
from greykite.common.features.timeseries_features import get_holidays
from greykite.common.features.timeseries_impute import impute_with_lags
from greykite.common.features.timeseries_lags import build_autoreg_df
from greykite.common.python_utils import assert_equal
from greykite.common.python_utils import get_pattern_cols
from greykite.common.testing_utils import generate_anomalous_data
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_holidays
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.common.testing_utils import generic_test_adjust_anomalous_data
from greykite.common.viz.timeseries_plotting_mpl import plt_compare_timeseries


matplotlib.use("agg")  # noqa: E402
import matplotlib.pyplot as plt  # isort:skip


@pytest.fixture
def hourly_data():
    """Generate 500 days of hourly data for tests"""
    return generate_df_for_tests(
        freq="H",
        periods=24 * 500,
        train_start_date=datetime.datetime(2018, 7, 1),
        conti_year_origin=2018)


def plt_comparison_forecast_vs_observed(
        fut_df,
        test_df,
        file_name=None,
        plt_title=""):
    """A local function for comparing forecasts with observed on test set.
       This function is only for tests here.
       Imports are delibrately put inside the function as the function
       will be run only selectively if user descides to make plots
    :param fut_df: pd.DataFrame
        dataframe with predictions
        expected to have a VALUE_COL at least
    :param test_df: pd.DataFrame
        dataframe which includes the observed values
        expected to have at least two columns: TIME_COL, VALUE_COL
    :param file_name: Optional[str]
        File name for the plot to be saved
    :param plt_title: str
        title of the plot, default: ""
    """
    fut_df[TIME_COL] = test_df[TIME_COL]
    plt_compare_timeseries(
        df_dict={"obs": test_df, "forecast": fut_df},
        time_col=TIME_COL,
        value_col=VALUE_COL,
        colors_dict={"obs": "red", "forecast": "green"},
        plt_title=plt_title)
    if file_name is not None:
        plt.savefig(file_name)
        plt.close()


def plt_check_ci(fut_df, test_df):
    """A local function for creating conf. interval plots within this test file.
    :param fut_df: pd.DataFrame
        the dataframe which includes future predictions in its VALUE_COL column
    :param test_df: pd.DataFrame
        the dataframe which includes true values in its VALUE_COL column
    """
    # imports are done within the function as the function is not
    # automatically run when tests run
    import plotly

    from greykite.common.constants import ACTUAL_COL
    from greykite.common.constants import PREDICTED_COL
    from greykite.common.constants import PREDICTED_LOWER_COL
    from greykite.common.constants import PREDICTED_UPPER_COL
    from greykite.common.viz.timeseries_plotting import plot_forecast_vs_actual

    # splitting the ci column to create lower and upper columns
    ci_df = pd.DataFrame(fut_df["y_quantile_summary"].tolist())
    assert ci_df.shape[1] == 2, "ci_df must have exactly two columns"
    ci_df.columns = [PREDICTED_LOWER_COL, PREDICTED_UPPER_COL]
    # adding necessary columns
    ci_df[ACTUAL_COL] = test_df["y"]
    ci_df[PREDICTED_COL] = fut_df["y"]
    ci_df[TIME_COL] = test_df[TIME_COL]

    fig = plot_forecast_vs_actual(
        df=ci_df,
        time_col=TIME_COL,
        actual_col=ACTUAL_COL,
        predicted_col=PREDICTED_COL,
        predicted_lower_col=PREDICTED_LOWER_COL,
        predicted_upper_col=PREDICTED_UPPER_COL,
        ylabel=VALUE_COL,
        train_end_date=None,
        title=None,
        actual_points_color="red",
        actual_points_size=2.0,
        forecast_curve_color="blue",
        actual_color_opacity=0.6,
        ci_band_color="rgba(0, 0, 200, 0.4)",
        ci_boundary_curve_color="rgb(56, 119, 166, 0.95)",  # blue navy color with opacity of 0.95
        ci_boundary_curve_width=0.5)

    plotly.offline.plot(fig)


def test_forecast_silverkite_hourly(hourly_data):
    """Tests silverkite on hourly data with linear model fit"""
    train_df = hourly_data["train_df"]
    test_df = hourly_data["test_df"]
    fut_time_num = hourly_data["fut_time_num"]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=datetime.datetime(2019, 6, 1),
        origin_for_time_vars=None,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5]}),
        extra_pred_cols=["ct_sqrt", "dow_hr", "ct1"])

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="H",
        new_external_regressor_df=None)
    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.3
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 6.0
    assert trained_model["x_mat"]["ct1"][0] == 0  # this should be true when origin_for_time_vars=None
    """
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df,
        test_df=test_df,
        file_name=None)
    """

    # with normalization
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=datetime.datetime(2019, 6, 1),
        origin_for_time_vars=None,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5]}),
        extra_pred_cols=["ct_sqrt", "dow_hr", "ct1"],
        normalize_method="min_max")

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="H",
        new_external_regressor_df=None)
    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.3
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 6.0
    assert trained_model["x_mat"]["ct1"][0] == 0  # this should be true when origin_for_time_vars=None
    """
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df,
        test_df=test_df,
        file_name=None)
    """


def test_forecast_silverkite_hourly_regressor():
    """Tests silverkite with regressors and random forest fit"""
    hourly_data = generate_df_with_reg_for_tests(
        freq="H",
        periods=24 * 500,
        train_start_date=datetime.datetime(2018, 7, 1),
        conti_year_origin=2018)
    regressor_cols = ["regressor1", "regressor_bool", "regressor_categ"]
    train_df = hourly_data["train_df"].reset_index(drop=True)
    test_df = hourly_data["test_df"].reset_index(drop=True)
    fut_time_num = hourly_data["fut_time_num"]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        training_fraction=None,
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5],
            "seas_names": None}),
        extra_pred_cols=["ct_sqrt", "dow_hr", "ct1"] + regressor_cols,
        fit_algorithm="rf",
        fit_algorithm_params={"min_samples_split": 3})

    assert trained_model["ml_model"].min_samples_split == 3

    # three equivalent ways of generating predictions
    result1 = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="H",
        new_external_regressor_df=test_df[regressor_cols])
    result2 = silverkite.predict_no_sim(
        fut_df=test_df[[TIME_COL, VALUE_COL]],
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=test_df[regressor_cols])
    result3 = silverkite.predict_no_sim(
        fut_df=test_df[[TIME_COL, VALUE_COL] + regressor_cols],
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=None)
    # checks for equality of contents, ignoring row/column order
    # `df` may contain extra columns not required by `silverkite.predict_n(_no_sim`
    # and VALUE_COL is the last column in `silverkite.predict_n(_no_sim` but the
    # original order in `df` is preserved
    assert_frame_equal(result1, result2, check_like=True)
    assert_frame_equal(result1, result3, check_like=True)

    err = calc_pred_err(test_df[VALUE_COL], result1[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.3
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 6.0
    # checks to make sure the frequency is set properly
    assert np.array_equal(result1[TIME_COL].values, test_df[TIME_COL].values)
    """
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df,
        test_df=test_df,
        file_name=None)
    """


def test_forecast_silverkite_freq():
    """Tests forecast_silverkite at different frequencies"""
    # A wide variety of frequencies listed here:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    frequencies = [
        "B", "W", "W-SAT", "W-TUE", "M", "SM",
        "MS", "SMS", "CBMS", "BM", "B", "Q",
        "QS", "BQS", "BQ-AUG", "Y", "YS",
        "AS-SEP", "BH", "T", "S"]
    periods = 50
    train_frac = 0.8
    train_test_thresh_index = int(periods * train_frac * 0.8)
    for freq in frequencies:
        df = generate_df_for_tests(
            freq=freq,
            periods=50,
            train_frac=0.8,
            train_start_date=datetime.datetime(2018, 5, 1))
        train_df = df["train_df"]
        test_df = df["test_df"]
        fut_time_num = df["fut_time_num"]

        silverkite = SilverkiteForecast()
        trained_model = silverkite.forecast(
            df=train_df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            train_test_thresh=train_df[TIME_COL][train_test_thresh_index],
            origin_for_time_vars=2018,
            fs_components_df=pd.DataFrame({
                "name": ["tod", "tow", "conti_year"],
                "period": [24.0, 7.0, 1.0],
                "order": [3, 0, 5]}),
            extra_pred_cols=["ct_sqrt"],
            changepoints_dict={
                "method": "uniform",
                "n_changepoints": 2,
                "continuous_time_col": "ct1",
                "growth_func": lambda x: x})

        changepoint_dates = get_changepoint_dates_from_changepoints_dict(
            changepoints_dict={
                "method": "uniform",
                "n_changepoints": 2,
                "continuous_time_col": "ct1",
                "growth_func": lambda x: x},
            df=train_df,
            time_col=TIME_COL
        )

        changepoint_cols = get_pattern_cols(trained_model["pred_cols"], "^changepoint")

        assert len(changepoint_cols) == 2
        string_format = get_changepoint_string(changepoint_dates)
        assert "changepoint0" + string_format[0] in trained_model["pred_cols"]
        assert "changepoint1" + string_format[1] in trained_model["pred_cols"]

        fut_df = silverkite.predict_n_no_sim(
            fut_time_num=fut_time_num,
            trained_model=trained_model,
            freq=freq,
            new_external_regressor_df=None)
        # checks silverkite.predict_n(_no_sim
        fut_df_via_predict = silverkite.predict_no_sim(
            fut_df=test_df,
            trained_model=trained_model)
        assert_frame_equal(
            fut_df[[TIME_COL]],
            fut_df_via_predict[[TIME_COL]],
            check_like=True)
        assert_frame_equal(
            fut_df,
            fut_df_via_predict,
            check_like=True)


def test_forecast_silverkite_changepoints():
    """Tests forecast_silverkite on peyton manning data
    (with changepoints and missing values)
    """
    dl = DataLoader()
    df_pt = dl.load_peyton_manning()

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=df_pt,
        time_col="ts",
        value_col="y",
        changepoints_dict={
            "method": "auto",
            "yearly_seasonality_order": 6,
            "resample_freq": "2D",
            "actual_changepoint_min_distance": "100D",
            "potential_changepoint_distance": "50D",
            "no_changepoint_proportion_from_end": 0.3
        }
    )
    # "df" preserves the original indices
    assert_equal(trained_model["df"].index, df_pt.index)
    # "df_dropna" drops the correct indices
    assert_equal(trained_model["df_dropna"].index, df_pt.dropna().index)

    changepoint_values = trained_model["normalized_changepoint_values"]
    df_length = trained_model["x_mat"]["ct1"].iloc[-1]
    cp_distance = timedelta(days=100) / (pd.to_datetime(df_pt["ts"].iloc[-1]) - pd.to_datetime(df_pt["ts"].iloc[0]))
    # has change points
    assert len(changepoint_values) >= 0
    # checks no change points at the end
    assert changepoint_values[-1] <= df_length * 0.7
    # checks change point distance is at least "100D"
    min_cp_dist = min([changepoint_values[i] - changepoint_values[i - 1] for i in range(1, len(changepoint_values))])
    assert min_cp_dist >= df_length * cp_distance
    # checks the number of change points is consistent with the change points detected by ChangepointDetector
    cd = ChangepointDetector()
    res = cd.find_trend_changepoints(
        df=df_pt,
        time_col="ts",
        value_col="y",
        yearly_seasonality_order=6,
        resample_freq="2D",
        actual_changepoint_min_distance="100D",
        potential_changepoint_distance="50D",
        no_changepoint_proportion_from_end=0.3
    )
    changepoint_dates = res["trend_changepoints"]
    assert len(changepoint_values) == len(changepoint_dates)


def test_forecast_silverkite_seasonality_changepoints():
    # test forecast_silverkite on peyton manning data
    dl = DataLoader()
    df_pt = dl.load_peyton_manning()
    silverkite = SilverkiteForecast()
    # seasonality changepoints is None if dictionary is not provided
    trained_model = silverkite.forecast(
        df=df_pt,
        time_col="ts",
        value_col="y",
        changepoints_dict={
            "method": "auto"
        },
        seasonality_changepoints_dict=None
    )
    assert trained_model["seasonality_changepoint_dates"] is None
    assert trained_model["seasonality_changepoint_result"] is None
    # all test cases below include seasonality changepoint detection.
    # without trend change points
    trained_model = silverkite.forecast(
        df=df_pt,
        time_col="ts",
        value_col="y",
        changepoints_dict=None,
        seasonality_changepoints_dict={}
    )
    assert trained_model["seasonality_changepoint_dates"] is not None
    assert trained_model["seasonality_changepoint_result"] is not None
    assert "weekly" in trained_model["seasonality_changepoint_dates"].keys()
    assert "yearly" in trained_model["seasonality_changepoint_dates"].keys()
    # with different seasonality change point parameters
    trained_model = silverkite.forecast(
        df=df_pt,
        time_col="ts",
        value_col="y",
        changepoints_dict={
            "method": "auto"
        },
        seasonality_changepoints_dict={
            "no_changepoint_distance_from_end": "730D"
        }
    )
    assert trained_model["seasonality_changepoint_dates"] is not None
    assert trained_model["seasonality_changepoint_result"] is not None
    assert "weekly" in trained_model["seasonality_changepoint_dates"].keys()
    assert "yearly" in trained_model["seasonality_changepoint_dates"].keys()
    no_changepoint_proportion_from_end = timedelta(days=730) / (
            pd.to_datetime(df_pt["ts"].iloc[-1]) - pd.to_datetime(df_pt["ts"].iloc[0]))
    last_date_to_have_changepoint = pd.to_datetime(df_pt["ts"].iloc[int(
        df_pt.shape[0] * (1 - no_changepoint_proportion_from_end))])
    for component in trained_model["seasonality_changepoint_dates"].keys():
        if len(trained_model["seasonality_changepoint_dates"][component]) > 0:
            assert trained_model["seasonality_changepoint_dates"][component][-1] <= last_date_to_have_changepoint
    # tests forecasting the future
    pred = silverkite.predict_no_sim(
        fut_df=pd.DataFrame({
            "ts": pd.date_range(start=df_pt["ts"].iloc[-1], periods=10, freq="D")
        }),
        trained_model=trained_model
    )
    assert pred.shape[0] == 10
    assert "y" in pred.columns


def test_forecast_silverkite_hourly_changepoint_uniform(hourly_data):
    """Tests forecast_silverkite with uniform changepoints"""
    train_df = hourly_data["train_df"]
    test_df = hourly_data["test_df"]
    fut_time_num = hourly_data["fut_time_num"]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=datetime.datetime(2019, 6, 1),
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5]}),
        extra_pred_cols=["ct_sqrt", "dow_hr"],
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 2,
            "continuous_time_col": "ct1",
            "growth_func": lambda x: x})

    changepoint_dates = get_changepoint_dates_from_changepoints_dict(
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 2,
            "continuous_time_col": "ct1",
            "growth_func": lambda x: x},
        df=train_df,
        time_col=TIME_COL
    )

    changepoint_cols = get_pattern_cols(trained_model["pred_cols"], "^changepoint")

    # checks that there are two changepoints
    assert len(changepoint_cols) == 2
    assert "changepoint0" + pd.to_datetime(changepoint_dates[0]).strftime('_%Y_%m_%d_%H') \
           in trained_model["pred_cols"]
    assert "changepoint1" + pd.to_datetime(changepoint_dates[1]).strftime('_%Y_%m_%d_%H') \
           in trained_model["pred_cols"]

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="H",
        new_external_regressor_df=None)
    # checks predict_n
    fut_df_via_predict = silverkite.predict_no_sim(
        fut_df=test_df,
        trained_model=trained_model)
    assert_frame_equal(
        fut_df[[TIME_COL]],
        fut_df_via_predict[[TIME_COL]],
        check_like=True)
    assert_frame_equal(
        fut_df,
        fut_df_via_predict,
        check_like=True)

    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.3
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 6.0
    """
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df,
        test_df=test_df,
        file_name=None)
    """


def test_forecast_silverkite_hourly_changepoint_custom(hourly_data):
    """Tests forecast_silverkite with custom changepoints"""
    train_df = hourly_data["train_df"]
    test_df = hourly_data["test_df"]
    fut_time_num = hourly_data["fut_time_num"]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=datetime.datetime(2019, 6, 1),
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5]}),
        extra_pred_cols=["ct_sqrt", "dow_hr"],
        changepoints_dict={
            "method": "custom",
            "dates": [train_df[TIME_COL][100],
                      train_df[TIME_COL][500],
                      train_df[TIME_COL][1000]],
            "continuous_time_col": "ct1",
            "growth_func": lambda x: x})

    changepoint_dates = get_changepoint_dates_from_changepoints_dict(
        changepoints_dict={
            "method": "custom",
            "dates": [train_df[TIME_COL][100],
                      train_df[TIME_COL][500],
                      train_df[TIME_COL][1000]],
            "continuous_time_col": "ct1",
            "growth_func": lambda x: x},
        df=train_df,
        time_col=TIME_COL
    )

    changepoint_cols = get_pattern_cols(trained_model["pred_cols"], "^changepoint")

    # checks that there are three changepoints
    assert len(changepoint_cols) == 3
    assert "changepoint0" + pd.to_datetime(changepoint_dates[0]).strftime('_%Y_%m_%d_%H') \
           in trained_model["pred_cols"]
    assert "changepoint1" + pd.to_datetime(changepoint_dates[1]).strftime('_%Y_%m_%d_%H') \
           in trained_model["pred_cols"]
    assert "changepoint2" + pd.to_datetime(changepoint_dates[2]).strftime('_%Y_%m_%d_%H') \
           in trained_model["pred_cols"]

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="H",
        new_external_regressor_df=None)
    # checks `silverkite.predict_n(_no_sim`
    fut_df_via_predict = silverkite.predict_no_sim(
        fut_df=test_df,
        trained_model=trained_model)
    assert_frame_equal(
        fut_df[[TIME_COL]],
        fut_df_via_predict[[TIME_COL]],
        check_like=True)
    assert_frame_equal(
        fut_df,
        fut_df_via_predict,
        check_like=True)

    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.3
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 6.0


def test_forecast_silverkite_hourly_changepoint_err(hourly_data):
    """Tests forecast_silverkite changepoint warnings and exceptions"""
    train_df = hourly_data["train_df"]
    silverkite = SilverkiteForecast()
    with pytest.raises(
            Exception,
            match="changepoint method must be specified"):
        silverkite.forecast(
            df=train_df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            train_test_thresh=datetime.datetime(2019, 6, 1),
            origin_for_time_vars=2018,
            fs_components_df=pd.DataFrame({
                "name": ["tod", "tow", "conti_year"],
                "period": [24.0, 7.0, 1.0],
                "order": [3, 0, 5]}),
            extra_pred_cols=["ct_sqrt", "dow_hr"],
            changepoints_dict={"n_changepoints": 2})

    with pytest.raises(
            NotImplementedError,
            match="changepoint method.*not recognized"):
        silverkite.forecast(
            df=train_df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            train_test_thresh=datetime.datetime(2019, 6, 1),
            origin_for_time_vars=2018,
            fs_components_df=pd.DataFrame({
                "name": ["tod", "tow", "conti_year"],
                "period": [24.0, 7.0, 1.0],
                "order": [3, 0, 5]}),
            extra_pred_cols=["ct_sqrt", "dow_hr"],
            changepoints_dict={"method": "not implemented"})

    with pytest.warns(Warning) as record:
        silverkite = SilverkiteForecast()
        silverkite.forecast(
            df=train_df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            train_test_thresh=datetime.datetime(2019, 6, 1),
            origin_for_time_vars=2018,
            fs_components_df=pd.DataFrame({
                "name": ["tod", "tow", "conti_year"],
                "period": [24.0, 7.0, 1.0],
                "order": [3, 0, 5]}),
            extra_pred_cols=["ct_sqrt", "dow_hr", "changepoint1:ct_sqrt"],
            changepoints_dict={
                "method": "custom",
                "dates": ["2048-07-01-23"],
                "continuous_time_col": "ct1",
                "growth_func": lambda x: x
            })
        assert "The following features in extra_pred_cols are removed for this training set:" \
               " {'changepoint1:ct_sqrt'}." in record[0].message.args[0]


def test_forecast_silverkite_with_autoreg(hourly_data):
    """Tests forecast_silverkite autoregression"""
    train_df = hourly_data["train_df"]
    test_df = hourly_data["test_df"][:168].reset_index(drop=True)  # one week of data for testing
    test_past_df = train_df.copy()
    fut_time_num = test_df.shape[0]

    # we define a local function to apply `forecast_silverkite`
    # with and without autoregression
    def fit_forecast(
            autoreg_dict=None,
            test_past_df=None,
            simulation_based=False):
        silverkite = SilverkiteForecast()
        trained_model = silverkite.forecast(
            df=train_df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            train_test_thresh=None,
            origin_for_time_vars=2018,
            fs_components_df=pd.DataFrame({
                "name": ["tod", "tow"],
                "period": [24.0, 7.0],
                "order": [1, 1],
                "seas_names": ["daily", "weekly"]}),
            autoreg_dict=autoreg_dict,
            simulation_based=simulation_based)

        fut_df = silverkite.predict_n_no_sim(
            fut_time_num=fut_time_num,
            trained_model=trained_model,
            freq="H",
            new_external_regressor_df=None)

        return {
            "fut_df": fut_df,
            "trained_model": trained_model}

    # without autoregression
    fut_df = fit_forecast(
        autoreg_dict=None,
        test_past_df=None)["fut_df"]
    # with autoregression
    autoreg_dict = {
        "lag_dict": {"orders": [168]},
        "agg_lag_dict": {
            "orders_list": [[168, 168 * 2, 168 * 3]],
            "interval_list": [(168, 168 * 2)]},
        "series_na_fill_func": lambda s: s.bfill().ffill()}

    fut_df_with_autoreg = fit_forecast(
        autoreg_dict=autoreg_dict,
        test_past_df=test_past_df)["fut_df"]

    # without autoregression
    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 1) == 5.7
    # with autoregression
    err = calc_pred_err(test_df[VALUE_COL], fut_df_with_autoreg[VALUE_COL])
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 1) == 1.9

    """
    figs_path = "~/figs/"
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df,
        test_df=test_df,
        file_name=figs_path + "forecast_without_autoreg.png",
        plt_title="without auto-regression")
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df_with_autoreg,
        test_df=test_df,
        file_name=figs_path + "forecast_with_autoreg.png",
        plt_title="with auto-regression")
    """

    # with autoregression option of "auto"
    forecast = fit_forecast(
        autoreg_dict="auto",
        test_past_df=test_past_df)
    fut_df_with_autoreg = forecast["fut_df"]
    trained_model = forecast["trained_model"]
    autoreg_dict = trained_model["autoreg_dict"]
    assert autoreg_dict["lag_dict"] == {"orders": [24, 25, 26]}
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[168, 336, 504]]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(24, 191), (192, 359)]
    assert trained_model["forecast_horizon"] == 24

    err = calc_pred_err(test_df[VALUE_COL], fut_df_with_autoreg[VALUE_COL])
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 1) == 2.5

    # with autoregression option of "auto" and simulation based
    forecast = fit_forecast(
        autoreg_dict="auto",
        test_past_df=test_past_df,
        simulation_based=True)
    fut_df_with_autoreg = forecast["fut_df"]
    trained_model = forecast["trained_model"]
    autoreg_dict = trained_model["autoreg_dict"]
    assert autoreg_dict["lag_dict"] == {"orders": [1, 2, 3]}
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[168, 336, 504]]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(1, 168), (169, 336)]
    assert trained_model["forecast_horizon"] == 24

    err = calc_pred_err(test_df[VALUE_COL], fut_df_with_autoreg[VALUE_COL])
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 1) == 2.8

    expected_match = "is not implemented"
    with pytest.raises(ValueError, match=expected_match):
        fit_forecast(
            autoreg_dict="non-existing-method",
            test_past_df=test_past_df,
            simulation_based=True)


def test_forecast_silverkite_2min():
    """Tests silverkite on 2min data"""
    data = generate_df_for_tests(
        freq="2min",
        periods=24 * 30 * 20,
        train_frac=0.9,
        train_end_date=None,
        noise_std=0.1)
    train_df = data["train_df"]
    test_df = data["test_df"]
    fut_time_num = data["fut_time_num"]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5],
            "seas_names": [None, "weekly", "yearly"]}),
        extra_pred_cols=["ct_sqrt", "dow_hr"])

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="2min",
        new_external_regressor_df=None)
    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 1.2

    """
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df,
        test_df=test_df,
        file_name=None)
    """


def test_forecast_silverkite_with_weighted_col():
    """Tests silverkite on 2min data"""
    data = generate_df_for_tests(
        freq="1D",
        periods=400,
        train_frac=0.9,
        train_end_date=None,
        noise_std=0.1)
    train_df = data["train_df"]
    test_df = data["test_df"]
    fut_time_num = data["fut_time_num"]

    silverkite = SilverkiteForecast()
    # Tests without weighted regression
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tow", "conti_year"],
            "period": [7.0, 1.0],
            "order": [5, 5],
            "seas_names": ["weekly", "yearly"]}),
        extra_pred_cols=["ct_sqrt", "dow_hr"],
        fit_algorithm="ridge",
        regression_weight_col=None)

    assert trained_model["regression_weight_col"] is None

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="1D",
        new_external_regressor_df=None)
    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 2) == 0.21

    # Tests with weighted regression
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tow", "conti_year"],
            "period": [7.0, 1.0],
            "order": [5, 5],
            "seas_names": ["weekly", "yearly"]}),
        extra_pred_cols=["ct_sqrt", "dow_hr"],
        fit_algorithm="ridge",
        regression_weight_col="ct1")

    assert trained_model["regression_weight_col"] == "ct1"

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="1D",
        new_external_regressor_df=None)
    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.RootMeanSquaredError
    # The error is slightly smaller than before
    assert round(err[enum.get_metric_name()], 2) == 0.20


def test_forecast_silverkite_2min_with_uncertainty():
    """Tests silverkite on 2min data"""
    res = generate_df_for_tests(
        freq="2min",
        periods=24 * 50 * 30,
        train_frac=0.8,
        train_end_date=None,
        noise_std=0.1)
    train_df = res["train_df"]
    test_df = res["test_df"][:24 * 30 * 7]
    fut_time_num = test_df.shape[0]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 3],
            "seas_names": [None, "weekly", "yearly"]}),
        extra_pred_cols=["ct_sqrt", "dow_hr"],
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "conditional_cols": ["dow_hr"],
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 20,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.98}}
    )

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="2min",
        new_external_regressor_df=None)

    fut_df["y_true"] = test_df["y"]
    fut_df["inside_95_ci"] = fut_df.apply(
        lambda row: (
                (row["y_true"] <= row["y_quantile_summary"][1])
                and (row["y_true"] >= row["y_quantile_summary"][0])),
        axis=1)

    ci_coverage = 100.0 * fut_df["inside_95_ci"].mean()
    assert round(ci_coverage) == 91, (
        "95 percent CI coverage is not as expected (91%)")

    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 1.2


def test_forecast_silverkite_simulator():
    """Tests silverkite simulator on hourly data with linear model fit"""
    data = generate_df_for_tests(
        freq="H",
        periods=100 * 30,
        train_frac=0.8,
        train_end_date=None,
        noise_std=0.3)
    train_df = data["train_df"]
    test_df = data["test_df"][:30 * 7]
    fut_df = test_df.copy()
    fut_df[VALUE_COL] = None

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=None,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5]}),
        extra_pred_cols=["ct_sqrt", "dow_hr", "ct1"],
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "conditional_cols": ["dow_hr"],
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 20,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.98}})

    past_df = train_df[[TIME_COL, VALUE_COL]].copy()

    # simulations with error
    sim_df = silverkite.simulate(
        fut_df=fut_df,
        trained_model=trained_model,
        past_df=past_df,
        new_external_regressor_df=None,
        include_err=True)

    np.random.seed(123)
    assert sim_df[VALUE_COL].dtype == "float64"
    err = calc_pred_err(test_df[VALUE_COL], sim_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert round(err[enum.get_metric_name()], 2) == 0.97
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 2) == 0.55

    # simulations without errors
    sim_df = silverkite.simulate(
        fut_df=fut_df,
        trained_model=trained_model,
        past_df=past_df,
        new_external_regressor_df=None,
        include_err=False)

    np.random.seed(123)
    assert sim_df[VALUE_COL].dtype == "float64"
    err = calc_pred_err(test_df[VALUE_COL], sim_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert round(err[enum.get_metric_name()], 2) == 0.98
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 2) == 0.47

    # multiple simulations
    sim_df = silverkite.simulate_multi(
        fut_df=fut_df,
        trained_model=trained_model,
        sim_num=2,
        past_df=past_df,
        new_external_regressor_df=None,
        include_err=False)

    assert sim_df[VALUE_COL].dtype == "float64"
    assert sim_df.shape[0] == fut_df.shape[0] * 2
    assert list(sim_df.columns) == [TIME_COL, VALUE_COL, "sim_label"]

    """
    # making a plot of comparison between 10 simulations and observed
    sim_num = 10
    sim_labels = [f"sim{i}" for i in range(sim_num)]
    colors_dict = {label: "grey" for label in sim_labels}

    df_dict = {}
    np.random.seed(123)
    for sim_label in sim_labels:
        sim_df = silverkite.simulate(
            fut_df=fut_df,
            trained_model=trained_model,
            past_df=train_df[[TIME_COL, VALUE_COL]].copy(),
            new_external_regressor_df=None,
            include_err=True)
        df_dict[sim_label] = sim_df

    df_dict.update({"obs": test_df})
    colors_dict.update({"obs": "red"})
    legends_dict = {"sim1": "sim", "obs": "obs"}
    from greykite.common.viz.timeseries_plotting import plt_compare_timeseries
    plt_compare_timeseries(
        df_dict=df_dict,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        colors_dict=colors_dict,
        legends_dict=legends_dict,
        plt_title="",
        linewidth=1)
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    directory = Path(__file__).parents[6]
    file_name = os.path.join(
        directory,
        "simulated_timeseries_vs_observed.png")
    plt.savefig(file_name)
    plt.close()
    """


def test_forecast_silverkite_simulator_exception():
    """Tests silverkite simulator exception catch"""
    data = generate_df_for_tests(
        freq="H",
        periods=24 * 30,
        train_frac=0.8,
        train_end_date=None,
        noise_std=0.3)
    train_df = data["train_df"]
    test_df = data["test_df"][:7]
    fut_df = test_df.copy()
    fut_df[VALUE_COL] = None

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=None,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5]}),
        extra_pred_cols=["ct_sqrt", "dow_hr", "ct1"],
        uncertainty_dict=None)

    past_df = train_df[[TIME_COL, VALUE_COL]].copy()

    # testing for Exception
    expected_match = (
        "Error is requested via ")
    # `uncertainty_dict` is not passed to model.
    # Therefore raising exception is expected.
    with pytest.raises(ValueError, match=expected_match):
        silverkite.simulate(
            fut_df=fut_df,
            trained_model=trained_model,
            past_df=past_df,
            new_external_regressor_df=None,
            include_err=True)


def test_forecast_silverkite_predict_via_sim():
    """Tests silverkite simulator on hourly data with linear model fit"""
    data = generate_df_for_tests(
        freq="H",
        periods=100 * 30,
        train_frac=0.8,
        train_end_date=None,
        noise_std=0.3)
    train_df = data["train_df"]
    test_df = data["test_df"][:30 * 7]
    fut_df = test_df.copy()
    fut_df[VALUE_COL] = None

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=None,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5]}),
        extra_pred_cols=["ct_sqrt", "dow_hr", "ct1"],
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "conditional_cols": ["dow_hr"],
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 20,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.98}})

    past_df = train_df[[TIME_COL, VALUE_COL]].copy()

    # predict via sim
    np.random.seed(123)
    fut_df = silverkite.predict_via_sim(
        fut_df=fut_df,
        trained_model=trained_model,
        past_df=past_df,
        new_external_regressor_df=None,
        sim_num=10,
        include_err=True)

    assert list(fut_df.columns) == [
        TIME_COL,
        VALUE_COL,
        f"{VALUE_COL}_quantile_summary",
        ERR_STD_COL]

    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert round(err[enum.get_metric_name()], 2) == 0.98
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 2) == 0.48

    """
    import os
    from pathlib import Path
    directory = Path(__file__).parents[6]
    file_name = os.path.join(
        directory,
        "predict_silverkite_via_sim.png")
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df,
        test_df=test_df,
        file_name=file_name)
    # to plot CIs
    plt_check_ci(fut_df=fut_df, test_df=test_df)
    """


def test_silverkite_predict():
    """Testing ``predict_silverkite``"""
    data = generate_df_for_tests(
        freq="D",
        periods=300,
        train_frac=0.8,
        train_end_date=None,
        noise_std=3,
        remove_extra_cols=True,
        autoreg_coefs=[10] * 24,
        fs_coefs=[0.1, 1, 0.1],
        growth_coef=2.0)

    train_df = data["train_df"]
    test_df = data["test_df"]
    fut_df = test_df[:5].copy()
    fut_df[VALUE_COL] = None
    fut_df_with_gap = test_df[5:10].copy()
    fut_df_with_gap[VALUE_COL] = None
    fut_df_including_training = pd.concat(
        [train_df, fut_df],
        axis=0,
        ignore_index=True)
    fut_df_including_training[VALUE_COL] = None

    autoreg_dict = {
        "lag_dict": {"orders": list(range(7, 14))},
        "agg_lag_dict": None,
        "series_na_fill_func": lambda s: s.bfill().ffill()}

    # These are the columns we expect to get from the predictions
    expected_fut_df_cols = [
        TIME_COL, VALUE_COL, f"{VALUE_COL}_quantile_summary", ERR_STD_COL]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=None,
        fit_algorithm="statsmodels_ols",
        fs_components_df=pd.DataFrame({
            "name": ["tod", "conti_year"],
            "period": [24.0, 1.0],
            "order": [3, 5]}),
        extra_pred_cols=["ct1", "dow_hr"],
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "conditional_cols": ["dow_hr"],
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 20,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.98}},
        autoreg_dict=autoreg_dict)

    # ``fut_df`` does not include training data
    np.random.seed(123)
    predict_info = silverkite.predict(
        fut_df=fut_df,
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == 5
    assert predict_info["min_lag_order"] == 7
    assert predict_info["fut_df_info"]["forecast_partition_summary"] == {
        "len_before_training": 0,
        "len_within_training": 0,
        "len_after_training": 5,
        "len_gap": 0}
    assert predict_info["fut_df"].shape[0] == fut_df.shape[0]
    assert list(predict_info["fut_df"].columns) == expected_fut_df_cols

    # ``fut_df`` includes training data
    predict_info = silverkite.predict(
        fut_df=fut_df_including_training,
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == 5
    assert predict_info["min_lag_order"] == 7
    assert predict_info["fut_df_info"]["forecast_partition_summary"] == {
        "len_before_training": 0,
        "len_within_training": train_df.shape[0],
        "len_after_training": 5,
        "len_gap": 0}
    assert predict_info["fut_df"].shape[0] == fut_df_including_training.shape[0]
    assert list(predict_info["fut_df"].columns) == expected_fut_df_cols

    # ``fut_df`` has a gap
    # In this case simulations will be invoked
    # This is because ``min_lag_order < forecast_horizon``
    predict_info = silverkite.predict(
        fut_df=fut_df_with_gap,
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert not predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == 10
    assert predict_info["min_lag_order"] == 7
    assert predict_info["fut_df_info"]["forecast_partition_summary"] == {
        "len_before_training": 0,
        "len_within_training": 0,
        "len_after_training": 5,
        "len_gap": 5}
    assert predict_info["fut_df"].shape[0] == fut_df_with_gap.shape[0]
    assert list(predict_info["fut_df"].columns) == expected_fut_df_cols


def test_predict_silverkite_with_regressors():
    """Testing ``predict_silverkite`` in presence of regressors"""
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=500,
        train_start_date=datetime.datetime(2018, 7, 1),
        conti_year_origin=2018)

    fut_time_num = 5
    len_gap = 4
    train_df = data["train_df"]
    test_df = data["test_df"]
    fut_df = test_df[:fut_time_num].reset_index(drop=True)
    fut_df[VALUE_COL] = None
    fut_df_with_gap = test_df[len_gap:(len_gap + fut_time_num)].copy()
    fut_df_with_gap[VALUE_COL] = None
    fut_df_including_training = pd.concat(
        [train_df, fut_df],
        axis=0,
        ignore_index=True)
    fut_df_including_training[VALUE_COL] = None
    regressor_cols = ["regressor1", "regressor_bool", "regressor_categ"]

    autoreg_dict = {
        "lag_dict": {"orders": list(range(7, 14))},
        "agg_lag_dict": None,
        "series_na_fill_func": lambda s: s.bfill().ffill()}

    # These are the columns we expect to get from the predictions
    expected_fut_df_cols = [
        TIME_COL, VALUE_COL, f"{VALUE_COL}_quantile_summary", ERR_STD_COL]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=None,
        fit_algorithm="statsmodels_ols",
        fs_components_df=pd.DataFrame({
            "name": ["tod", "conti_year"],
            "period": [24.0, 1.0],
            "order": [3, 5]}),
        extra_pred_cols=["ct1", "dow_hr"],
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "conditional_cols": ["dow_hr"],
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 20,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.98}},
        autoreg_dict=autoreg_dict)

    # (Case 1.a) ``fut_df`` does not include training data
    # regressors passed through ``fut_df``
    np.random.seed(123)
    predict_info = silverkite.predict(
        fut_df=fut_df,
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == fut_time_num
    assert predict_info["min_lag_order"] == 7
    assert predict_info["fut_df_info"]["forecast_partition_summary"] == {
        "len_before_training": 0,
        "len_within_training": 0,
        "len_after_training": fut_time_num,
        "len_gap": 0}
    assert predict_info["fut_df"].shape[0] == fut_df.shape[0]
    assert list(predict_info["fut_df"].columns) == expected_fut_df_cols

    # (Case 1.b) ``fut_df`` does not include training data
    # regressors passed separately
    np.random.seed(123)
    predict_info = silverkite.predict(
        fut_df=fut_df[[TIME_COL]],
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=fut_df[regressor_cols].copy(),
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == fut_time_num
    assert predict_info["min_lag_order"] == 7
    assert predict_info["fut_df_info"]["forecast_partition_summary"] == {
        "len_before_training": 0,
        "len_within_training": 0,
        "len_after_training": fut_time_num,
        "len_gap": 0}
    assert predict_info["fut_df"].shape[0] == fut_df.shape[0]
    assert list(predict_info["fut_df"].columns) == expected_fut_df_cols

    # (Case 2.a) ``fut_df`` includes training data.
    # Regressors passed through ``fut_df``
    predict_info = silverkite.predict(
        fut_df=fut_df_including_training,
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == fut_time_num
    assert predict_info["min_lag_order"] == 7
    assert predict_info["fut_df_info"]["forecast_partition_summary"] == {
        "len_before_training": 0,
        "len_within_training": train_df.shape[0],
        "len_after_training": fut_time_num,
        "len_gap": 0}
    assert predict_info["fut_df"].shape[0] == fut_df_including_training.shape[0]
    assert list(predict_info["fut_df"].columns) == expected_fut_df_cols

    # (Case 2.b) ``fut_df`` includes training data.
    # Regressors passed directly.
    predict_info = silverkite.predict(
        fut_df=fut_df_including_training[[TIME_COL]],
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=fut_df_including_training[regressor_cols].copy(),
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == fut_time_num
    assert predict_info["min_lag_order"] == 7
    assert predict_info["fut_df_info"]["forecast_partition_summary"] == {
        "len_before_training": 0,
        "len_within_training": train_df.shape[0],
        "len_after_training": fut_time_num,
        "len_gap": 0}
    assert predict_info["fut_df"].shape[0] == fut_df_including_training.shape[0]
    assert list(predict_info["fut_df"].columns) == expected_fut_df_cols

    # (Case 3.a) ``fut_df`` has a gap.
    # Regressors passed through ``fut_df``.
    # In this case simulations will be invoked.
    # This is because ``min_lag_order < forecast_horizon``.
    predict_info = silverkite.predict(
        fut_df=fut_df_with_gap,
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert not predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == len_gap + fut_time_num
    assert predict_info["min_lag_order"] == 7
    assert predict_info["fut_df_info"]["forecast_partition_summary"] == {
        "len_before_training": 0,
        "len_within_training": 0,
        "len_after_training": fut_time_num,
        "len_gap": len_gap}
    assert predict_info["fut_df"].shape[0] == fut_df_with_gap.shape[0]
    assert list(predict_info["fut_df"].columns) == expected_fut_df_cols

    # (Case 3.b) ``fut_df`` has a gap.
    # Regressors passed directly
    # In this case simulations will be invoked.
    # This is because ``min_lag_order < forecast_horizon``.
    predict_info = silverkite.predict(
        fut_df=fut_df_with_gap[[TIME_COL]],
        trained_model=trained_model,
        past_df=None,
        new_external_regressor_df=fut_df_with_gap[regressor_cols].copy(),
        sim_num=5,
        include_err=None,
        force_no_sim=False,
        na_fill_func=lambda s: s.interpolate().bfill())  # Simple NA fill is used for easy to track testing

    assert not predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == len_gap + fut_time_num
    assert predict_info["min_lag_order"] == 7
    assert predict_info["fut_df_info"]["forecast_partition_summary"] == {
        "len_before_training": 0,
        "len_within_training": 0,
        "len_after_training": fut_time_num,
        "len_gap": len_gap}
    assert predict_info["fut_df"].shape[0] == fut_df_with_gap.shape[0]
    assert list(predict_info["fut_df"].columns) == expected_fut_df_cols

    fut_df_gap = predict_info["fut_df_info"]["fut_df_gap"]

    expected_time_gaps = pd.date_range(
        start=train_df.tail(1)[TIME_COL].values[0] + pd.to_timedelta("1D"),
        periods=len_gap,
        freq="1D")

    expected_fut_df_gap = pd.DataFrame({
        TIME_COL: expected_time_gaps,
        "regressor1": [test_df.iloc[len_gap]["regressor1"]]*len_gap,
        "regressor_bool": [test_df.iloc[len_gap]["regressor_bool"]]*len_gap,
        "regressor_categ": [test_df.iloc[len_gap]["regressor_categ"]]*len_gap
        })

    expected_fut_df_gap[TIME_COL] = pd.to_datetime(expected_fut_df_gap[TIME_COL])
    assert_frame_equal(fut_df_gap, expected_fut_df_gap)


def test_predict_silverkite_exceptions():
    """Testing ``predict_silverkite``"""
    data = generate_df_for_tests(
        freq="D",
        periods=300,
        train_frac=0.8,
        train_end_date=None,
        noise_std=3,
        remove_extra_cols=True,
        autoreg_coefs=[10] * 24,
        fs_coefs=[0.1, 1, 0.1],
        growth_coef=2.0)

    train_df = data["train_df"]
    test_df = data["test_df"]
    fut_df = test_df.copy()
    fut_df[VALUE_COL] = None
    fut_df_with_before_training = train_df[[TIME_COL]]
    fut_df_with_before_training[TIME_COL] = (
        fut_df_with_before_training[TIME_COL] - datetime.timedelta(days=1))

    autoreg_dict = {
        "lag_dict": {"orders": list(range(7, 14))},
        "agg_lag_dict": None,
        "series_na_fill_func": lambda s: s.bfill().ffill()}

    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 20,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=None,
        fit_algorithm="statsmodels_ols",
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5]}),
        extra_pred_cols=["ct1", "dow_hr"],
        uncertainty_dict=uncertainty_dict,
        autoreg_dict=autoreg_dict)

    # Trains a model with uncertainty
    trained_model_no_uncertainty = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=None,
        fit_algorithm="statsmodels_ols",
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5]}),
        extra_pred_cols=["ct1", "dow_hr"],
        uncertainty_dict=None,
        autoreg_dict=autoreg_dict)

    # Checks Exception for ``include_err = True`` while no uncertainty in the model
    expected_match = "However model does not support uncertainty. "
    with pytest.raises(ValueError, match=expected_match):
        silverkite.predict(
            fut_df=fut_df,
            trained_model=trained_model_no_uncertainty,
            past_df=None,
            new_external_regressor_df=None,
            sim_num=5,
            include_err=True,
            force_no_sim=False)

    expected_match = "cannot have timestamps occurring before the training data"
    with pytest.raises(ValueError, match=expected_match):
        silverkite.predict(
            fut_df=fut_df_with_before_training,
            trained_model=trained_model,
            past_df=None,
            new_external_regressor_df=None,
            sim_num=5,
            include_err=None,
            force_no_sim=False)

    expected_match = "must be a dataframe of non-zero size"
    with pytest.raises(ValueError, match=expected_match):
        silverkite.predict(
            fut_df=fut_df.iloc[0:0],
            trained_model=trained_model,
            past_df=None,
            new_external_regressor_df=None,
            sim_num=5,
            include_err=None,
            force_no_sim=False)

    expected_match = "which is what ``trained_model`` considers to be the time column"
    with pytest.raises(ValueError, match=expected_match):
        fut_df0 = fut_df[[TIME_COL]]
        fut_df0.columns = ["dummy_ts"]
        silverkite.predict(
            fut_df=fut_df0,
            trained_model=trained_model,
            past_df=None,
            new_external_regressor_df=None,
            sim_num=5,
            include_err=None,
            force_no_sim=False)


def test_predict_silverkite_compare_various_ways():
    """Testing various ways to perform prediction using silverkite model.
    Make sure predictions match when expected."""
    data = generate_df_for_tests(
        freq="H",
        periods=24 * 300,
        train_frac=0.8,
        train_end_date=None,
        noise_std=3,
        remove_extra_cols=True,
        autoreg_coefs=[10] * 24,
        fs_coefs=[0.1, 1, 0.1],
        growth_coef=2.0)

    train_df = data["train_df"]
    test_df = data["test_df"][:5]
    fut_df = test_df.copy()
    fut_df[VALUE_COL] = None

    # With autoregression with min lag = 2
    autoreg_dict_recent_lag = {
        "lag_dict": {"orders": list(range(1, 3))},
        "agg_lag_dict": None,
        "series_na_fill_func": lambda s: s.bfill().ffill()}

    # With autoregression with min lag = 168
    autoreg_dict_old_lag_only = {
        "lag_dict": None,
        "agg_lag_dict": {
            "orders_list": [[168, 168 * 2]],
            "interval_list": [(168, 168 * 2)]},
        "series_na_fill_func": lambda s: s.bfill().ffill()}

    silverkite = SilverkiteForecast()

    def fit_silverkite(autoreg_dict):
        return silverkite.forecast(
            df=train_df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            train_test_thresh=None,
            origin_for_time_vars=None,
            fit_algorithm="statsmodels_ols",
            fs_components_df=pd.DataFrame({
                "name": ["tod", "conti_year"],
                "period": [24.0, 1.0],
                "order": [3, 5]}),
            extra_pred_cols=["ct1", "dow_hr"],
            uncertainty_dict={
                "uncertainty_method": "simple_conditional_residuals",
                "params": {
                    "conditional_cols": ["dow_hr"],
                    "quantiles": [0.025, 0.975],
                    "quantile_estimation_method": "normal_fit",
                    "sample_size_thresh": 20,
                    "small_sample_size_method": "std_quantiles",
                    "small_sample_size_quantile": 0.98}},
            autoreg_dict=autoreg_dict)

    trained_model_old_lag_only = fit_silverkite(
        autoreg_dict=autoreg_dict_old_lag_only)

    trained_model_with_recent_lag = fit_silverkite(
        autoreg_dict=autoreg_dict_recent_lag)

    trained_model_no_autoreg = fit_silverkite(autoreg_dict=None)

    # (Case 1) First the case with autoregression with old lag only
    # In this case we expect that no sim approach will be triggered
    # by ``silverkite.predict_n(``, and ``predict_silverkite``,
    # because ``min_lag_order`` is 2 while forecast horizon is 24
    np.random.seed(123)
    fut_df_with_ar = silverkite.predict_n_no_sim(
        fut_time_num=test_df.shape[0],
        trained_model=trained_model_old_lag_only,
        freq="1H",
        new_external_regressor_df=None)

    # Directly using `silverkite.predict_n(` which will use simulations.
    # We expect the same result as above.
    np.random.seed(123)
    predict_info = silverkite.predict_n(
        fut_time_num=test_df.shape[0],
        trained_model=trained_model_old_lag_only,
        freq="1H",
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == 5
    assert predict_info["min_lag_order"] == 168

    fut_df_with_ar_2 = predict_info["fut_df"]

    # Uses ``predict_silverkite``
    np.random.seed(123)
    predict_info = silverkite.predict(
        fut_df=fut_df,
        trained_model=trained_model_old_lag_only,
        past_df=train_df[[TIME_COL, VALUE_COL]].copy(),
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == 5
    assert predict_info["min_lag_order"] == 168

    fut_df_with_ar_3 = predict_info["fut_df"]

    # Checks the case where `past_df` is not passed
    np.random.seed(123)
    predict_info = silverkite.predict(
            fut_df=fut_df.copy(),
            trained_model=trained_model_old_lag_only,
            past_df=None,
            new_external_regressor_df=None,
            sim_num=5,
            include_err=None,
            force_no_sim=False)

    assert predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == 5
    assert predict_info["min_lag_order"] == 168

    fut_df_with_ar_4 = predict_info["fut_df"]

    # We expect to get the exact same future predictions using three above calls
    assert_frame_equal(
        fut_df_with_ar[[TIME_COL, VALUE_COL]],
        fut_df_with_ar_2[[TIME_COL, VALUE_COL]])
    assert_frame_equal(
        fut_df_with_ar[[TIME_COL, VALUE_COL]],
        fut_df_with_ar_3[[TIME_COL, VALUE_COL]])

    assert_frame_equal(
        fut_df_with_ar[[TIME_COL, VALUE_COL]],
        fut_df_with_ar_4[[TIME_COL, VALUE_COL]])

    assert list(fut_df_with_ar.columns) == [
        TIME_COL,
        VALUE_COL,
        f"{VALUE_COL}_quantile_summary",
        ERR_STD_COL]

    # (Case 2) The case with short autoregression
    # In this case we expect that via_sim approach will be triggered
    # by ``silverkite.predict_n(``, and ``predict_silverkite``
    # because ``min_lag_order`` is 168*2 while forecast horizon is 24
    np.random.seed(123)
    fut_df_with_ar = silverkite.predict_n_via_sim(
        fut_time_num=test_df.shape[0],
        trained_model=trained_model_with_recent_lag,
        freq="1H",
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None)

    # Directly uses ``silverkite.predict_n(`` which will use simulations.
    # We expect the same result as above.
    np.random.seed(123)
    predict_info = silverkite.predict_n(
        fut_time_num=test_df.shape[0],
        trained_model=trained_model_with_recent_lag,
        freq="1H",
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert not predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == 5
    assert predict_info["min_lag_order"] == 1
    fut_df_with_ar_2 = predict_info["fut_df"]

    # Uses ``predict_silverkite``
    np.random.seed(123)
    predict_info = silverkite.predict(
        fut_df=fut_df.copy(),
        trained_model=trained_model_with_recent_lag,
        past_df=train_df[[TIME_COL, VALUE_COL]].copy(),
        new_external_regressor_df=None,
        sim_num=5,
        include_err=None,
        force_no_sim=False)

    assert not predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == 5
    assert predict_info["min_lag_order"] == 1

    fut_df_with_ar_3 = predict_info["fut_df"]

    # Checks the case when`past_df` is not passed.
    np.random.seed(123)
    predict_info = silverkite.predict(
            fut_df=fut_df,
            trained_model=trained_model_with_recent_lag,
            past_df=None,
            new_external_regressor_df=None,
            sim_num=5,
            include_err=None,
            force_no_sim=False)

    assert not predict_info["simulations_not_used"]
    assert predict_info["fut_df_info"]["inferred_forecast_horizon"] == 5
    assert predict_info["min_lag_order"] == 1

    fut_df_with_ar_4 = predict_info["fut_df"]

    # We expect to get the exact same future predictions using three above calls
    assert_frame_equal(
        fut_df_with_ar[[TIME_COL, VALUE_COL]],
        fut_df_with_ar_2[[TIME_COL, VALUE_COL]])
    assert_frame_equal(
        fut_df_with_ar[[TIME_COL, VALUE_COL]],
        fut_df_with_ar_3[[TIME_COL, VALUE_COL]])

    assert_frame_equal(
        fut_df_with_ar[[TIME_COL, VALUE_COL]],
        fut_df_with_ar_4[[TIME_COL, VALUE_COL]])

    assert list(fut_df_with_ar.columns) == [
        TIME_COL,
        VALUE_COL,
        f"{VALUE_COL}_quantile_summary",
        ERR_STD_COL]

    # (Case 3) Tests the cases with no AR
    fut_df_no_ar = silverkite.predict_n_no_sim(
        fut_time_num=test_df.shape[0],
        trained_model=trained_model_no_autoreg,
        freq="1H",
        new_external_regressor_df=None)

    # Directly calculated via ``silverkite.predict_n(``
    predict_info = silverkite.predict_n(
        fut_time_num=test_df.shape[0],
        trained_model=trained_model_no_autoreg,
        freq="1H",
        new_external_regressor_df=None)
    fut_df_no_ar2 = predict_info["fut_df"]

    # Directly calculated via ``predict_silverkite``
    predict_info = silverkite.predict(
        fut_df=fut_df.copy(),
        trained_model=trained_model_no_autoreg,
        past_df=train_df[[TIME_COL, VALUE_COL]].copy(),
        new_external_regressor_df=None,
        sim_num=10,
        include_err=None,
        force_no_sim=False)
    fut_df_no_ar3 = predict_info["fut_df"]

    # We expect to get the exact same future predictions using three above calls
    assert_frame_equal(
        fut_df_no_ar[[TIME_COL, VALUE_COL]],
        fut_df_no_ar2[[TIME_COL, VALUE_COL]])
    assert_frame_equal(
        fut_df_no_ar[[TIME_COL, VALUE_COL]],
        fut_df_no_ar3[[TIME_COL, VALUE_COL]])

    err = calc_pred_err(test_df[VALUE_COL], fut_df_with_ar[VALUE_COL])
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] == pytest.approx(114.2, rel=1e-2)

    err = calc_pred_err(test_df[VALUE_COL], fut_df_no_ar[VALUE_COL])
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] == pytest.approx(59.8, rel=1e-2)

    """
    import os
    from pathlib import Path
    directory = Path(__file__).parents[6]
    file_name = os.path.join(
        directory,
        "silverkite.predict_n(_via_sim.png")
    plt_compare_timeseries(
        df_dict={
            "train data last part": train_df[-(24*60):],
            "test data": test_df,
            "forecast AR/sim": fut_df_with_ar_sim,
            "forecast no AR": fut_df_no_ar},
        time_col=TIME_COL,
        value_col=VALUE_COL,
        colors_dict={
            "train data last part": "orange",
            "test data": "red",
            "forecast AR/sim": "green",
            "forecast no AR": "olive"},
        plt_title="")
    if file_name is not None:
        plt.savefig(file_name)
        plt.close()

    # to plot CIs
    plt_check_ci(fut_df=fut_df_with_ar_sim, test_df=test_df)
    plt_check_ci(fut_df=fut_df_no_ar, test_df=test_df)
    """


def test_silverkite_predict_n_include_err_exception():
    """Testing for exception for `include_err=True` while
    uncertainty is not passsed"""

    data = generate_df_for_tests(
        freq="H",
        periods=24 * 300,
        train_frac=0.8,
        train_end_date=None,
        noise_std=3,
        remove_extra_cols=True,
        autoreg_coefs=[10] * 24,
        fs_coefs=[0.1, 1, 0.1],
        growth_coef=2.0)

    train_df = data["train_df"]
    test_df = data["test_df"]
    fut_df = test_df.copy()
    fut_df[VALUE_COL] = None

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=None,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "conti_year"],
            "period": [24.0, 1.0],
            "order": [0, 1]}),
        extra_pred_cols=[],
        uncertainty_dict=None,
        autoreg_dict=None)

    expected_match = "However model does not support uncertainty. "

    with pytest.raises(ValueError, match=expected_match):
        silverkite.predict_n(
            fut_time_num=test_df.shape[0],
            trained_model=trained_model,
            freq="1H",
            new_external_regressor_df=None,
            sim_num=5,
            include_err=True,
            force_no_sim=False)

    with pytest.raises(ValueError, match=expected_match):
        silverkite.predict(
            fut_df=fut_df,
            trained_model=trained_model,
            new_external_regressor_df=None,
            sim_num=5,
            include_err=True,
            force_no_sim=False)


def test_forecast_silverkite_simulator_regressor():
    """Tests silverkite simulator with regressors"""
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=500,
        train_start_date=datetime.datetime(2018, 7, 1),
        conti_year_origin=2018)
    regressor_cols = ["regressor1", "regressor_bool", "regressor_categ"]
    train_df = data["train_df"].reset_index(drop=True)
    test_df = data["test_df"].reset_index(drop=True)

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        training_fraction=None,
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "conti_year"],
            "period": [24.0, 1.0],
            "order": [3, 5],
            "seas_names": None}),
        extra_pred_cols=["ct_sqrt", "dow_hr", "ct1"] + regressor_cols,
        fit_algorithm="linear",
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "conditional_cols": ["dow_hr"],
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 20,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.98}})

    past_df = train_df[[TIME_COL, VALUE_COL]].copy()

    sim_df = silverkite.simulate(
        fut_df=test_df[[TIME_COL, VALUE_COL]],
        trained_model=trained_model,
        past_df=past_df,
        new_external_regressor_df=test_df[regressor_cols],
        include_err=True)

    assert sim_df[VALUE_COL].dtype == "float64"
    err = calc_pred_err(test_df[VALUE_COL], sim_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert round(err[enum.get_metric_name()], 2) == 0.56
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 2) == 2.83

    # predict via sim
    np.random.seed(123)
    fut_df = silverkite.predict_via_sim(
        fut_df=test_df[[TIME_COL, VALUE_COL]],
        trained_model=trained_model,
        past_df=past_df,
        new_external_regressor_df=test_df[regressor_cols],
        sim_num=10,
        include_err=True)

    assert list(fut_df.columns) == [
        TIME_COL,
        VALUE_COL,
        f"{VALUE_COL}_quantile_summary",
        ERR_STD_COL]

    assert sim_df[VALUE_COL].dtype == "float64"
    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert round(err[enum.get_metric_name()], 2) == 0.65
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()], 2) == 2.35

    """
    plt_comparison_forecast_vs_observed(
        fut_df=sim_df,
        test_df=test_df,
        file_name=None)
    """


def test_forecast_silverkite_with_holidays_hourly():
    """Tests silverkite with holidays and seasonality interactions"""
    res = generate_df_with_holidays(freq="H", periods=24 * 700)
    train_df = res["train_df"]
    test_df = res["test_df"]
    fut_time_num = res["fut_time_num"]

    # generate holidays
    countries = ["US", "India"]
    event_df_dict = get_holidays(countries, year_start=2015, year_end=2025)

    for country in countries:
        event_df_dict[country][EVENT_DF_LABEL_COL] = country + "_holiday"

    # custom seasonality names
    fourier_col1 = get_fourier_col_name(
        k=1,
        col_name="tod",
        function_name="sin",
        seas_name="daily")
    fourier_col2 = get_fourier_col_name(
        k=1,
        col_name="tod",
        function_name="cos",
        seas_name="daily")
    fourier_col3 = get_fourier_col_name(1, "conti_year", function_name="cos")

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col="ts",
        value_col=VALUE_COL,
        train_test_thresh=datetime.datetime(2019, 6, 1),
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5],
            "seas_names": ["daily", "weekly", None]}),
        extra_pred_cols=["ct_sqrt", "dow_hr", f"events_US*{fourier_col1}",
                         f"events_US*{fourier_col2}",
                         f"events_US*{fourier_col3}"],
        daily_event_df_dict=event_df_dict)

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="H",
        new_external_regressor_df=None)
    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.3
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 6.0
    """
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df,
        test_df=test_df,
        file_name=None)
    """


def test_forecast_silverkite_with_holidays_effect():
    """Tests silverkite, modeling a separate effect per holiday
        (instead of per holiday+country as in
        test_forecast_silverkite_with_holidays_hourly)
    """
    res = generate_df_with_holidays(freq="H", periods=24 * 700)
    train_df = res["train_df"]
    test_df = res["test_df"]
    fut_time_num = res["fut_time_num"]

    # generate holidays
    countries = ["US", "India"]
    holidays_to_model_separately = [
        "New Year's Day",
        "Christmas Day",
        "Independence Day",
        "Thanksgiving",
        "Labor Day",
        "Memorial Day",
        "Veterans Day"]

    event_df_dict = generate_holiday_events(
        countries=countries,
        holidays_to_model_separately=holidays_to_model_separately,
        year_start=2015,
        year_end=2025,
        pre_num=0,
        post_num=0)

    # constant event effect at daily level
    event_cols = [f"Q('events_{key}')" for key in event_df_dict.keys()]
    # different hourly seasonality on weekends.
    # fs_* matches the specification to "fs_components_df"
    interaction_cols = cols_interact(
        static_col="is_weekend",
        fs_name="tod",
        fs_order=3,
        fs_seas_name="daily")
    extra_pred_cols = ["ct_sqrt", "dow_hr"] + event_cols + interaction_cols
    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col="ts",
        value_col=VALUE_COL,
        train_test_thresh=datetime.datetime(2019, 6, 1),
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 0, 5],
            "seas_names": ["daily", "weekly", None]}),
        extra_pred_cols=extra_pred_cols,
        daily_event_df_dict=event_df_dict)

    fut_df = silverkite.predict_n_no_sim(
        fut_time_num=fut_time_num,
        trained_model=trained_model,
        freq="H",
        new_external_regressor_df=None)
    err = calc_pred_err(test_df[VALUE_COL], fut_df[VALUE_COL])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.3
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 6.0
    """
    plt_comparison_forecast_vs_observed(
        fut_df=fut_df,
        test_df=test_df,
        file_name=None)
    """


def test_forecast_silverkite_train_test_thresh_error(hourly_data):
    df = hourly_data["df"]
    last_time_available = max(df[TIME_COL])
    train_test_thresh = datetime.datetime(2020, 7, 1)
    with pytest.raises(ValueError) as record:
        silverkite = SilverkiteForecast()
        silverkite.forecast(
            df=df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            train_test_thresh=datetime.datetime(2020, 7, 1),
            origin_for_time_vars=None,
            fs_components_df=pd.DataFrame({
                "name": ["tod", "tow", "conti_year"],
                "period": [24.0, 7.0, 1.0],
                "order": [3, 0, 5]})
        )
        assert f"Input timestamp for the parameter 'train_test_threshold' " \
               f"({train_test_thresh}) exceeds the maximum available " \
               f"timestamp of the time series ({last_time_available})." \
               f"Please pass a value within the range." in record[0].message.args[0]


def test_forecast_silverkite_with_imputation():
    """Tests ``forecast_silverkite`` with imputations"""
    df = pd.DataFrame({
        "ts": len(pd.date_range(start="1/1/2018", end="3/14/2018")),
        "y": list(range(70)) + [np.nan]*3})

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        training_fraction=None,
        origin_for_time_vars=None,
        fs_components_df=None,
        impute_dict={
            "func": impute_with_lags,
            "params": {"orders": [7]}})

    impute_info = trained_model["impute_info"]

    assert impute_info["initial_missing_num"] == 3
    assert impute_info["final_missing_num"] == 0
    imputed_df = impute_info["df"]

    assert list(imputed_df["y"].values) == (
        list(range(70)) + [63, 64, 65])


def test_forecast_silverkite_with_adjust_anomalous():
    """Tests ``forecast_silverkite`` with anomalous_data``"""
    anomalous_data = generate_anomalous_data()
    anomaly_df = anomalous_data["anomaly_df"]
    df = anomalous_data["df"]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        training_fraction=None,
        origin_for_time_vars=None,
        fs_components_df=None,
        adjust_anomalous_dict={
            "func": adjust_anomalous_data,
            "params": {
                "anomaly_df": anomaly_df,
                "start_date_col": START_DATE_COL,
                "end_date_col": END_DATE_COL,
                "adjustment_delta_col": ADJUSTMENT_DELTA_COL,
                "filter_by_dict": {"platform": "MOBILE"}}})

    adj_df_info = trained_model["adjust_anomalous_info"]

    adj_values = pd.Series([np.nan, np.nan, 2., 6., 7., 8., 6., 7., 8., 9.])
    generic_test_adjust_anomalous_data(
        value_col=VALUE_COL,
        adj_df_info=adj_df_info,
        adj_values=adj_values)


def test_silverkite_partition_fut_df():
    """Tests ``partition_fut_df``"""
    freq = "1D"
    data = generate_df_for_tests(
        freq=freq,
        periods=500,
        train_frac=0.8,
        train_end_date=None,
        noise_std=0.1)
    train_df = data["train_df"]
    test_df = data["test_df"]
    all_df = data["df"]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "conti_year"],
            "period": [24.0, 1.0],
            "order": [3, 3],
            "seas_names": [None, "yearly"]}),
        extra_pred_cols=["ct_sqrt", "dow_hr"],
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "conditional_cols": ["dow_hr"],
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 20,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.98}}
    )

    # The case where ``fut_df`` is only future data and with no gaps
    fut_df = test_df[[TIME_COL, VALUE_COL]]
    fut_df_stats = silverkite.partition_fut_df(
        fut_df=fut_df,
        trained_model=trained_model,
        freq=freq)

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_after_training"] == [True] * fut_df.shape[0])
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == 0
    assert fut_df_stats["fut_df_after_training"].shape[0] == fut_df.shape[0]
    assert fut_df_stats["fut_df_gap"] is None
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == fut_df.shape[0]
    assert np.all(fut_df_stats["index_after_training_original"] == [True] * fut_df.shape[0])
    assert fut_df_stats["missing_periods_num"] == 0
    assert fut_df_stats["inferred_forecast_horizon"] == fut_df.shape[0]

    # The case where ``fut_df`` is only future data and with gaps
    fut_df = test_df[[TIME_COL, VALUE_COL]][2:]
    fut_df_stats = silverkite.partition_fut_df(
        fut_df=fut_df,
        trained_model=trained_model,
        freq=freq)

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_after_training"] == [True] * fut_df.shape[0])
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == 0
    assert fut_df_stats["fut_df_after_training"].shape[0] == fut_df.shape[0]
    assert fut_df_stats["fut_df_gap"].shape[0] == 2
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == fut_df.shape[0] + 2
    assert np.all(fut_df_stats["index_after_training_original"] == [False] * 2 + [True] * fut_df.shape[0])
    assert fut_df_stats["missing_periods_num"] == 2
    assert fut_df_stats["inferred_forecast_horizon"] == fut_df.shape[0] + 2

    # The case where ``fut_df`` is only part of the training data (no gaps as a result)
    fut_df = train_df[[TIME_COL, VALUE_COL]][2:]
    fut_df_stats = silverkite.partition_fut_df(
        fut_df=fut_df,
        trained_model=trained_model,
        freq=freq)

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [True] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_after_training"] == [False] * fut_df.shape[0])
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == fut_df.shape[0]
    assert fut_df_stats["fut_df_after_training"].shape[0] == 0
    assert fut_df_stats["fut_df_gap"] is None
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == 0
    assert fut_df_stats["index_after_training_original"] == []
    assert fut_df_stats["missing_periods_num"] == 0
    assert fut_df_stats["inferred_forecast_horizon"] == 0

    # The case where ``fut_df`` has both training and future timestamps
    # and the data has regular time increments
    fut_df = all_df.copy()
    fut_df_stats = silverkite.partition_fut_df(
        fut_df=fut_df,
        trained_model=trained_model,
        freq=freq)

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [True] * train_df.shape[0] + [False] * test_df.shape[0])
    assert np.all(fut_df_stats["index_after_training"] == [False] * train_df.shape[0] + [True] * test_df.shape[0])
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == train_df.shape[0]
    assert fut_df_stats["fut_df_after_training"].shape[0] == test_df.shape[0]
    assert fut_df_stats["fut_df_gap"] is None
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == test_df.shape[0]
    assert fut_df_stats["index_after_training_original"] == [True] * test_df.shape[0]
    assert fut_df_stats["missing_periods_num"] == 0
    assert fut_df_stats["inferred_forecast_horizon"] == test_df.shape[0]

    # The case where both training and future timestamps appear and we have a gap
    # Therefore ``fut_df`` is not a regular increment series
    fut_df = pd.concat(
        [train_df, test_df[5:]],
        axis=0,
        ignore_index=True)

    # The original length of the future timestamps
    fut_length = test_df.shape[0] - 5

    with pytest.warns(Warning) as record:
        fut_df_stats = silverkite.partition_fut_df(
            fut_df=fut_df,
            trained_model=trained_model,
            freq=freq)
        assert "does not have regular time increments" in record[0].message.args[0]

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [True] * train_df.shape[0] + [False] * fut_length)
    assert np.all(fut_df_stats["index_after_training"] == [False] * train_df.shape[0] + [True] * fut_length)
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == train_df.shape[0]
    assert fut_df_stats["fut_df_after_training"].shape[0] == fut_length
    assert fut_df_stats["fut_df_gap"].shape[0] == 5
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == test_df.shape[0]
    assert fut_df_stats["index_after_training_original"] == [False] * 5 + [True] * fut_length
    assert fut_df_stats["missing_periods_num"] == 5
    assert fut_df_stats["inferred_forecast_horizon"] == test_df.shape[0]


def test_partition_fut_df_monthly():
    """Tests ``partition_fut_df`` with monthly data"""
    freq = "MS"
    data = generate_df_for_tests(
        freq=freq,
        periods=60,
        train_frac=0.8,
        train_end_date=None,
        noise_std=0.1)
    train_df = data["train_df"]
    test_df = data["test_df"]
    all_df = data["df"]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["conti_year"],
            "period": [1.0],
            "order": [3],
            "seas_names": ["yearly"]}),
        extra_pred_cols=["ct_sqrt"],
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "conditional_cols": ["dow_hr"],
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 20,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.98}}
    )

    # The case where ``fut_df`` is only future data and with no gaps
    fut_df = test_df[[TIME_COL, VALUE_COL]]
    fut_df_stats = silverkite.partition_fut_df(
        fut_df=fut_df,
        trained_model=trained_model,
        freq=freq)

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600 * 31
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600 * 31
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_after_training"] == [True] * fut_df.shape[0])
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == 0
    assert fut_df_stats["fut_df_after_training"].shape[0] == fut_df.shape[0]
    assert fut_df_stats["fut_df_gap"] is None
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == fut_df.shape[0]
    assert np.all(fut_df_stats["index_after_training_original"] == [True] * fut_df.shape[0])
    assert fut_df_stats["missing_periods_num"] == 0
    assert fut_df_stats["inferred_forecast_horizon"] == fut_df.shape[0]

    # The case where ``fut_df`` is only future data and with gaps
    fut_df = test_df[[TIME_COL, VALUE_COL]][2:]
    fut_df_stats = silverkite.partition_fut_df(
        fut_df=fut_df,
        trained_model=trained_model,
        freq=freq)

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600 * 31
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600 * 31
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_after_training"] == [True] * fut_df.shape[0])
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == 0
    assert fut_df_stats["fut_df_after_training"].shape[0] == fut_df.shape[0]
    assert fut_df_stats["fut_df_gap"].shape[0] == 2
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == fut_df.shape[0] + 2
    assert np.all(fut_df_stats["index_after_training_original"] == [False] * 2 + [True] * fut_df.shape[0])
    assert fut_df_stats["missing_periods_num"] == 2
    assert fut_df_stats["inferred_forecast_horizon"] == fut_df.shape[0] + 2

    # The case where ``fut_df`` is only part of the training data (no gaps as a result)
    fut_df = train_df[[TIME_COL, VALUE_COL]][2:]
    fut_df_stats = silverkite.partition_fut_df(
        fut_df=fut_df,
        trained_model=trained_model,
        freq=freq)

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600 * 31
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600 * 31
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [True] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_after_training"] == [False] * fut_df.shape[0])
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == fut_df.shape[0]
    assert fut_df_stats["fut_df_after_training"].shape[0] == 0
    assert fut_df_stats["fut_df_gap"] is None
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == 0
    assert fut_df_stats["index_after_training_original"] == []
    assert fut_df_stats["missing_periods_num"] == 0
    assert fut_df_stats["inferred_forecast_horizon"] == 0

    # The case where ``fut_df`` has both training and future timestamps
    # and the data has consistent freq
    fut_df = all_df.copy()
    fut_df_stats = silverkite.partition_fut_df(
        fut_df=fut_df,
        trained_model=trained_model,
        freq=freq)

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600 * 31
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600 * 31
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [True] * train_df.shape[0] + [False] * test_df.shape[0])
    assert np.all(fut_df_stats["index_after_training"] == [False] * train_df.shape[0] + [True] * test_df.shape[0])
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == train_df.shape[0]
    assert fut_df_stats["fut_df_after_training"].shape[0] == test_df.shape[0]
    assert fut_df_stats["fut_df_gap"] is None
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == test_df.shape[0]
    assert fut_df_stats["index_after_training_original"] == [True] * test_df.shape[0]
    assert fut_df_stats["missing_periods_num"] == 0
    assert fut_df_stats["inferred_forecast_horizon"] == test_df.shape[0]

    # The case where both training and future timestamps appear and we have a gap
    # Therefore ``fut_df`` has a gap
    fut_df = pd.concat(
        [train_df, test_df[5:]],
        axis=0,
        ignore_index=True)

    # The original length of the future timestamps
    fut_length = test_df.shape[0] - 5

    with pytest.warns(Warning) as record:
        fut_df_stats = silverkite.partition_fut_df(
            fut_df=fut_df,
            trained_model=trained_model,
            freq=freq)
        assert "does not have regular time increments" in record[0].message.args[0]

    assert fut_df_stats["fut_freq_in_secs"] == 24 * 3600 * 31
    assert fut_df_stats["training_freq_in_secs"] == 24 * 3600 * 31
    assert np.all(fut_df_stats["index_before_training"] == [False] * fut_df.shape[0])
    assert np.all(fut_df_stats["index_within_training"] == [True] * train_df.shape[0] + [False] * fut_length)
    assert np.all(fut_df_stats["index_after_training"] == [False] * train_df.shape[0] + [True] * fut_length)
    assert fut_df_stats["fut_df_before_training"].shape[0] == 0
    assert fut_df_stats["fut_df_within_training"].shape[0] == train_df.shape[0]
    assert fut_df_stats["fut_df_after_training"].shape[0] == fut_length
    assert fut_df_stats["fut_df_gap"].shape[0] == 5
    assert fut_df_stats["fut_df_after_training_expanded"].shape[0] == test_df.shape[0]
    assert fut_df_stats["index_after_training_original"] == [False] * 5 + [True] * fut_length
    assert fut_df_stats["missing_periods_num"] == 5
    assert fut_df_stats["inferred_forecast_horizon"] == test_df.shape[0]


def test_partition_fut_df_exceptions():
    """Tests exceptions ``partition_fut_df``"""
    freq = "1D"
    data = generate_df_for_tests(
        freq=freq,
        periods=500,
        train_frac=0.8,
        train_end_date=None,
        noise_std=0.1)
    train_df = data["train_df"]

    silverkite = SilverkiteForecast()
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_test_thresh=None,
        origin_for_time_vars=2018,
        fs_components_df=pd.DataFrame({
            "name": ["tod", "conti_year"],
            "period": [24.0, 1.0],
            "order": [3, 3],
            "seas_names": [None, "yearly"]}),
        extra_pred_cols=["ct_sqrt", "dow_hr"],
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "conditional_cols": ["dow_hr"],
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 20,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.98}}
    )

    expected_match = "must be increasing in time"
    with pytest.raises(ValueError, match=expected_match):
        fut_df = train_df[[TIME_COL, VALUE_COL]].iloc[[3, 2, 1]]
        silverkite.partition_fut_df(
            fut_df=fut_df,
            trained_model=trained_model,
            freq=freq)

    expected_match = "The most immediate time in the future is off"
    with pytest.raises(ValueError, match=expected_match):
        fut_df = train_df[[TIME_COL]].copy()
        last_training_date = max(fut_df[TIME_COL])
        t0 = last_training_date + datetime.timedelta(days=0.5)
        fut_df_after_training = pd.DataFrame({TIME_COL: [t0]})
        fut_df = pd.concat(
            [fut_df, fut_df_after_training],
            axis=0,
            ignore_index=True)
        silverkite.partition_fut_df(
            fut_df=fut_df,
            trained_model=trained_model,
            freq=freq)


def test_predict_silverkite_with_autoreg_horizon_1(hourly_data):
    """Tests forecast_silverkite autoregression"""
    train_df = hourly_data["train_df"]
    silverkite = SilverkiteForecast()
    # Trains model with autoregression.
    trained_model = silverkite.forecast(
        df=train_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        autoreg_dict="auto"
    )
    # Generates future df with horizon 1.
    freq = "H"
    dates = pd.date_range(
        start=trained_model["last_date_for_fit"],
        periods=2,
        freq=freq)
    dates = dates[dates > trained_model["last_date_for_fit"]]  # drops values up to last_date_for_fit
    fut_df = pd.DataFrame({trained_model["time_col"]: dates.tolist()})
    # Makes sure ``partition_fut_df`` handles horizon 1 correctly.
    res = silverkite.partition_fut_df(
        fut_df=fut_df,
        trained_model=trained_model,
        freq=freq
    )
    assert res["fut_freq_in_secs"] is None
    # Runs ``predict_silverkite`` and expects no error.
    silverkite.predict(
        fut_df=fut_df,
        trained_model=trained_model
    )


def test_build_silverkite_features():
    """A basic testing of build_silverkite_features and input validation"""
    silverkite = SilverkiteForecast()
    daily_event_df_dict = get_holidays(["US"], year_start=2015, year_end=2025)

    # simple test
    with pytest.warns(None) as record:
        df = generate_df_for_tests(
            freq="D",
            periods=20)
        explan_df = silverkite._SilverkiteForecast__build_silverkite_features(
            df=df["train_df"],
            time_col=TIME_COL,
            origin_for_time_vars=2017,
            daily_event_df_dict=daily_event_df_dict,
            changepoint_values=None,
            continuous_time_col=None,
            growth_func=None,
            fs_func=None)

        assert list(explan_df[:3]["ct1"].round(4).values) == [1.4959, 1.4986, 1.5014]
        assert list(explan_df[:3]["dow"].values) == [7, 1, 2]
        assert list(explan_df[:3]["hour"].values) == [0, 0, 0]
        assert len(record) == 0  # no warnings

    # warning message for greater than daily data
    with pytest.warns(Warning) as record:
        df = generate_df_for_tests(
            freq="W",
            periods=20)

        explan_df = silverkite._SilverkiteForecast__build_silverkite_features(
            df=df["train_df"],
            time_col=TIME_COL,
            origin_for_time_vars=2017,
            daily_event_df_dict=daily_event_df_dict,
            changepoint_values=None,
            continuous_time_col=None,
            growth_func=None,
            fs_func=None)
        assert list(explan_df[:3]["ct1"].round(4).values) == [1.4959, 1.5151, 1.5342]
        assert list(explan_df[:3]["dow"].values) == [7, 7, 7]
        assert list(explan_df[:3]["hour"].values) == [0, 0, 0]
        assert ("The granularity of data is larger than daily. "
                "Ensure the daily events data match the timestamps" in record[0].message.args[0])

        # works for a single period
        df = generate_df_for_tests(
            freq="W",
            periods=1)
        explan_df = silverkite._SilverkiteForecast__build_silverkite_features(
            df=df["train_df"],
            time_col=TIME_COL,
            origin_for_time_vars=2017,
            daily_event_df_dict=daily_event_df_dict,
            changepoint_values=None,
            continuous_time_col=None,
            growth_func=None,
            fs_func=None)
        assert explan_df.shape[0] == 1


def test_build_silverkite_features2():
    """Detailed testing of build_silverkite_features
    with holidays, changepoints, fourier series, regressor
    """
    silverkite = SilverkiteForecast()
    hourly_data = generate_df_with_reg_for_tests(
        freq="H",
        periods=24 * 500,
        train_start_date=datetime.datetime(2018, 7, 1),
        conti_year_origin=2018)
    df = hourly_data["train_df"]
    regressor_cols = ["regressor1", "regressor_bool", "regressor_categ"]

    time_features_df = build_time_features_df(
        df[TIME_COL],
        conti_year_origin=2017)

    changepoint_values = get_evenly_spaced_changepoints_values(
        time_features_df,
        continuous_time_col="ct1",
        n_changepoints=2)

    fs_func = fourier_series_multi_fcn(
        col_names=["tod", "tow", "toy"],
        periods=[24.0, 7.0, 1.0],
        orders=[3, 0, 5],
        seas_names=None
    )

    # generate holidays
    countries = ["US", "India"]
    daily_event_df_dict = get_holidays(countries, year_start=2015, year_end=2025)

    for country in countries:
        daily_event_df_dict[country][EVENT_DF_LABEL_COL] = country + "_holiday"

    explan_df = silverkite._SilverkiteForecast__build_silverkite_features(
        df=df,
        time_col=TIME_COL,
        origin_for_time_vars=2017,
        daily_event_df_dict=daily_event_df_dict,
        changepoint_values=changepoint_values,
        continuous_time_col="ct1",
        growth_func=lambda x: x,
        fs_func=fs_func)

    assert list(explan_df[:3]["ct1"].round(4).values) == [1.4959, 1.4960, 1.4961]
    assert list(explan_df[:3]["hour"].values) == [0, 1, 2]
    assert list(explan_df[:3]["dow_hr"].values) == ["7_00", "7_01", "7_02"]

    # check regressors
    assert_frame_equal(explan_df[regressor_cols], df[regressor_cols])

    # check change points
    ind = explan_df["ct1"] > changepoint_values[0]
    assert list(explan_df.loc[ind]["changepoint0"][:3].round(6).values) == (
        [0.000114, 0.000228, 0.000342]), "change points data is incorrect"
    # check holidays
    ind = explan_df["conti_year"] >= 2019
    assert list(explan_df.loc[ind][:10]["events_US"].values) == (
            ["US_holiday"] * 10), "holiday data is incorrect"
    # check fourier series
    assert list(explan_df["sin1_tod"][:3].round(6).values) == (
        [0.000000, 0.258819, 0.500000]), "fourier series data is incorrect"


def test_build_autoreg_features(hourly_data):
    """Testing of build_autoreg_features with autoreg_func"""
    silverkite = SilverkiteForecast()
    past_df = hourly_data["train_df"]
    df = hourly_data["test_df"]
    df.index = pd.RangeIndex(start=10, stop=10+df.shape[0], step=1)  # non-default index

    autoreg_info = build_autoreg_df(
        value_col="y",
        lag_dict={"orders": [1, 168]},
        agg_lag_dict={
            "orders_list": [[168, 168 * 2, 168 * 3]],
            "interval_list": [(168, 168 * 2)]},
        series_na_fill_func=None)  # no filling of NAs
    autoreg_func = autoreg_info["build_lags_func"]

    autoreg_df = silverkite._SilverkiteForecast__build_autoreg_features(
        df=df,
        value_col=VALUE_COL,
        autoreg_func=autoreg_func,
        phase="fit",
        past_df=past_df)

    assert_equal(autoreg_df.index, df.index)
    assert None not in df.columns
    expected_cols = [
        "y_lag1",
        "y_lag168",
        "y_avglag_168_336_504",
        "y_avglag_168_to_336"]
    assert list(autoreg_df.columns) == expected_cols, (
        "expected column names for lag data do not appear in obtained feature df")
    expected_autoreg_df = pd.DataFrame({
        "y_lag1": [6.0, 4.3],
        "y_lag168": [0.9, 1.8],
        "y_avglag_168_336_504": [3.3, 3.4],
        "y_avglag_168_to_336": [3.4, 3.4]}, index=df.index[:2])

    obtained_autoreg_df = autoreg_df[expected_cols][:2].round(1)
    # Expected lag data must appear in the result dataframe
    assert_frame_equal(expected_autoreg_df, obtained_autoreg_df)

    # Expected lag1 value must come from last element of `past_df`.
    # Last value in `past_df` should appear as lag1 for first value in `df`.
    expected_lag1_value = round(past_df.tail(1)["y"].values[0], 1)
    assert obtained_autoreg_df["y_lag1"].values[0] == expected_lag1_value

    # Testing for Exception
    expected_match = (
        "At 'predict' phase, if autoreg_func is not None,"
        " 'past_df' and 'value_col' must be provided to `build_autoreg_features`")
    # value_col is None
    with pytest.raises(ValueError, match=expected_match):
        silverkite._SilverkiteForecast__build_autoreg_features(
            df=df,
            value_col=None,
            autoreg_func=autoreg_func,
            phase="predict",
            past_df=past_df)

    # past_df is None
    with pytest.raises(ValueError, match=expected_match):
        silverkite._SilverkiteForecast__build_autoreg_features(
            df=df,
            value_col=VALUE_COL,
            autoreg_func=autoreg_func,
            phase="predict",
            past_df=None)


def test_get_default_autoreg_dict():
    """Testing ``get_default_autoreg_dict``."""
    # Daily, horizon 1 days
    silverkite = SilverkiteForecast()
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1,
        forecast_horizon=1)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 7
    assert autoreg_dict["lag_dict"]["orders"] == [1, 2, 3]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(1, 7), (8, 7*2)]
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[7, 7*2, 7*3]]

    # Daily, horizon 3 days
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1,
        forecast_horizon=3)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 7
    assert autoreg_dict["lag_dict"]["orders"] == [3, 4, 5]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(3, 9), (10, 16)]
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[7, 7*2, 7*3]]

    # Daily, horizon 7
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1,
        forecast_horizon=7)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 7
    assert autoreg_dict["lag_dict"]["orders"] == [7, 8, 9]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(7, 13), (14, 20)]
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[7, 7*2, 7*3]]

    # Daily, horizon 30
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1,
        forecast_horizon=30)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 35
    assert autoreg_dict["lag_dict"]["orders"] == [30, 31, 32]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(30, 36), (37, 43)]
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[7*5, 7*6, 7*7]]

    # Daily, horizon 90
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1,
        forecast_horizon=90)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 91
    assert autoreg_dict is None

    # Daily, horizon 3 days, simulation based
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1,
        forecast_horizon=3,
        simulation_based=True)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 7
    assert autoreg_dict["lag_dict"]["orders"] == [1, 2, 3]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(1, 7), (8, 14)]
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[7, 7*2, 7*3]]

    # Hourly, horizon 1 hour
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1/24,
        forecast_horizon=1)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 24*7
    assert autoreg_dict["lag_dict"]["orders"] == [1, 2, 3]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(1, 24*7), (24*7+1, 24*7*2)]
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[24*7, 24*7*2, 24*7*3]]

    # Hourly, horizon 24 hours
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1/24,
        forecast_horizon=24)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 24*7
    assert autoreg_dict["lag_dict"]["orders"] == [24, 25, 26]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(24, 24*8-1), (24*8, 24*15-1)]
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[24*7, 24*7*2, 24*7*3]]

    # Hourly, horizon 24 hours, simulation based
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1/24,
        forecast_horizon=24,
        simulation_based=True)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 24*7
    assert autoreg_dict["lag_dict"]["orders"] == [1, 2, 3]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(1, 24*7), (24*7+1, 24*7*2)]
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[24*7, 24*7*2, 24*7*3]]

    # Hourly, horizon 4 hours, simulation based
    autoreg_info = silverkite._SilverkiteForecast__get_default_autoreg_dict(
        freq_in_days=1/24,
        forecast_horizon=4,
        simulation_based=True)
    autoreg_dict = autoreg_info["autoreg_dict"]
    proper_order = autoreg_info["proper_order"]

    assert proper_order == 24*7
    assert autoreg_dict["lag_dict"]["orders"] == [1, 2, 3]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(1, 24*7), (24*7+1, 24*7*2)]
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[24*7, 24*7*2, 24*7*3]]


def test_normalize_changepoint_values():
    silverkite = SilverkiteForecast()
    df = pd.DataFrame({
        "ct1": np.arange(0.01, 2.01, 0.01),
        "some_col1": np.random.randn(200),
        "some_col2": np.random.randn(200)
    })
    changepoint_values = np.array([0.88, 1.52])
    # tests min_max
    normalize_result = normalize_df(
        df=df,
        method="min_max"
    )
    pred_cols = normalize_result["keep_cols"]
    normalize_df_func = normalize_result["normalize_df_func"]
    normalized_changepoint_values = silverkite._SilverkiteForecast__normalize_changepoint_values(
        changepoint_values=changepoint_values,
        pred_cols=pred_cols,
        continuous_time_col="ct1",
        normalize_df_func=normalize_df_func
    )
    assert all(np.round(normalized_changepoint_values, 2) == np.array([0.44, 0.76]))
    # tests statistical
    normalize_result = normalize_df(
        df=df,
        method="statistical"
    )
    pred_cols = normalize_result["keep_cols"]
    normalize_df_func = normalize_result["normalize_df_func"]
    normalized_changepoint_values = silverkite._SilverkiteForecast__normalize_changepoint_values(
        changepoint_values=changepoint_values,
        pred_cols=pred_cols,
        continuous_time_col="ct1",
        normalize_df_func=normalize_df_func
    )
    assert all(np.round(normalized_changepoint_values, 2) == np.array([-0.22, 0.89]))
    # tests None changepoint_values
    normalized_changepoint_values = silverkite._SilverkiteForecast__normalize_changepoint_values(
        changepoint_values=None,
        pred_cols=pred_cols,
        continuous_time_col="ct1",
        normalize_df_func=normalize_df_func
    )
    assert normalized_changepoint_values is None
    # tests None normalize function
    normalized_changepoint_values = silverkite._SilverkiteForecast__normalize_changepoint_values(
        changepoint_values=changepoint_values,
        pred_cols=pred_cols,
        continuous_time_col="ct1",
        normalize_df_func=None
    )
    assert all(normalized_changepoint_values == changepoint_values)


def test_remove_fourier_col_with_collinearity():
    silverkite = SilverkiteForecast()
    fourier_cols = [
        "sin1_tow_weekly",
        "cos1_tow_weekly",
        "sin2_tow_weekly",
        "cos2_tow_weekly",
        "sin3_tow_weekly",
        "cos3_tow_weekly",
        "sin4_tow_weekly",
        "cos4_tow_weekly",     # to be removed because of weekly order 3 cosine
        "sin8_tow_weekly",     # to be removed because weekly period is 7
        "cos8_tow_weekly",     # to be removed because weekly period is 7
        "sin1_tom_monthly",    # to be removed because of quarterly order 3
        "cos1_tom_monthly",    # to be removed because of quarterly order 3
        "sin2_tom_monthly",
        "cos2_tom_monthly",
        "sin1_ct1_quarterly",  # to be removed because of yearly order 4
        "cos1_ct1_quarterly",  # to be removed because of yearly order 4
        "sin2_ct1_quarterly",  # to be removed because of yearly order 8
        "cos2_ct1_quarterly",  # to be removed because of yearly order 8
        "sin3_ct1_quarterly",
        "cos3_ct1_quarterly",
        "sin1_ct1_yearly",
        "cos1_ct1_yearly",
        "sin2_ct1_yearly",
        "cos2_ct1_yearly",
        "sin3_ct1_yearly",
        "cos3_ct1_yearly",
        "sin4_ct1_yearly",
        "cos4_ct1_yearly",
        "sin5_ct1_yearly",
        "cos5_ct1_yearly",
        "sin6_ct1_yearly",
        "cos6_ct1_yearly",
        "sin7_ct1_yearly",
        "cos7_ct1_yearly",
        "sin8_ct1_yearly",
        "cos8_ct1_yearly"
    ]
    expected_cols = [
        "sin1_tow_weekly",
        "cos1_tow_weekly",
        "sin2_tow_weekly",
        "cos2_tow_weekly",
        "sin3_tow_weekly",
        "cos3_tow_weekly",
        "sin4_tow_weekly",
        "sin2_tom_monthly",
        "cos2_tom_monthly",
        "sin3_ct1_quarterly",
        "cos3_ct1_quarterly",
        "sin1_ct1_yearly",
        "cos1_ct1_yearly",
        "sin2_ct1_yearly",
        "cos2_ct1_yearly",
        "sin3_ct1_yearly",
        "cos3_ct1_yearly",
        "sin4_ct1_yearly",
        "cos4_ct1_yearly",
        "sin5_ct1_yearly",
        "cos5_ct1_yearly",
        "sin6_ct1_yearly",
        "cos6_ct1_yearly",
        "sin7_ct1_yearly",
        "cos7_ct1_yearly",
        "sin8_ct1_yearly",
        "cos8_ct1_yearly"
    ]
    removed_cols = [
        "sin1_ct1_quarterly",
        "cos1_ct1_quarterly",
        "sin2_ct1_quarterly",
        "cos2_ct1_quarterly",
        "sin1_tom_monthly",
        "cos1_tom_monthly",
        "cos4_tow_weekly",
        "sin8_tow_weekly",
        "cos8_tow_weekly"
    ]
    with pytest.warns(UserWarning) as record:
        cols = silverkite._SilverkiteForecast__remove_fourier_col_with_collinearity(fourier_cols)
        assert f"The following Fourier series terms are removed due to collinearity:\n{removed_cols}" in record[0].message.args[0]
    assert cols == expected_cols

    # Tests monthly terms removal with yearly seasonality only.
    fourier_cols = [
        "sin1_tom_monthly",    # to be removed because of yearly order 12
        "cos1_tom_monthly",    # to be removed because of yearly order 12
        "sin2_tom_monthly",
        "cos2_tom_monthly",
        "sin1_ct1_yearly",
        "cos1_ct1_yearly",
        "sin2_ct1_yearly",
        "cos2_ct1_yearly",
        "sin3_ct1_yearly",
        "cos3_ct1_yearly",
        "sin4_ct1_yearly",
        "cos4_ct1_yearly",
        "sin5_ct1_yearly",
        "cos5_ct1_yearly",
        "sin6_ct1_yearly",
        "cos6_ct1_yearly",
        "sin7_ct1_yearly",
        "cos7_ct1_yearly",
        "sin8_ct1_yearly",
        "cos8_ct1_yearly",
        "sin9_ct1_yearly",
        "cos9_ct1_yearly",
        "sin10_ct1_yearly",
        "cos10_ct1_yearly",
        "sin11_ct1_yearly",
        "cos11_ct1_yearly",
        "sin12_ct1_yearly",
        "cos12_ct1_yearly"
    ]
    expected_cols = [
        "sin2_tom_monthly",
        "cos2_tom_monthly",
        "sin1_ct1_yearly",
        "cos1_ct1_yearly",
        "sin2_ct1_yearly",
        "cos2_ct1_yearly",
        "sin3_ct1_yearly",
        "cos3_ct1_yearly",
        "sin4_ct1_yearly",
        "cos4_ct1_yearly",
        "sin5_ct1_yearly",
        "cos5_ct1_yearly",
        "sin6_ct1_yearly",
        "cos6_ct1_yearly",
        "sin7_ct1_yearly",
        "cos7_ct1_yearly",
        "sin8_ct1_yearly",
        "cos8_ct1_yearly",
        "sin9_ct1_yearly",
        "cos9_ct1_yearly",
        "sin10_ct1_yearly",
        "cos10_ct1_yearly",
        "sin11_ct1_yearly",
        "cos11_ct1_yearly",
        "sin12_ct1_yearly",
        "cos12_ct1_yearly"
    ]
    removed_cols = [
        "sin1_tom_monthly",
        "cos1_tom_monthly",
    ]
    with pytest.warns(UserWarning) as record:
        cols = silverkite._SilverkiteForecast__remove_fourier_col_with_collinearity(fourier_cols)
        assert f"The following Fourier series terms are removed due to collinearity:\n{removed_cols}" in record[0].message.args[0]
    assert cols == expected_cols


def test_remove_fourier_col_with_collinearity_and_interaction():
    silverkite = SilverkiteForecast()
    extra_pred_cols = [
        "a",
        "b:c"
        "d:cos3_tow_weekly",
        "d:cos4_tow_weekly"
        "cos4_tow_weekly:cos3_tow_weekly"
    ]
    fs_cols = [
        "cos1_tow_weekly",
        "cos2_tow_weekly",
        "cos3_tow_weekly"
    ]
    removed_cols = [
        "d:cos4_tow_weekly"
        "cos4_tow_weekly:cos3_tow_weekly"
    ]
    with pytest.warns(UserWarning) as record:
        output = silverkite._SilverkiteForecast__remove_fourier_col_with_collinearity_and_interaction(
            extra_pred_cols=extra_pred_cols,
            fs_cols=fs_cols
        )
        assert (f"The following interaction terms are removed:\n{removed_cols}\n"
                f"due to the removal of the corresponding Fourier series terms."
                in record[0].message.args[0])
    expected_output = [
        "a",
        "b:c"
        "d:cos3_tow_weekly"
    ]
    assert output == expected_output
