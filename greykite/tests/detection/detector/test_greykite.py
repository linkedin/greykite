import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import pytest

from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import PREDICTED_ANOMALY_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils_anomalies import contaminate_df_with_anomalies
from greykite.common.viz.timeseries_annotate import plot_anomalies_over_forecast_vs_actual
from greykite.common.viz.timeseries_annotate import plot_lines_markers
from greykite.common.viz.timeseries_annotate import plt_compare_series_annotations
from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.common.ad_evaluation import precision_score
from greykite.detection.common.ad_evaluation import recall_score
from greykite.detection.detector.ad_utils import partial_return
from greykite.detection.detector.config import F1
from greykite.detection.detector.config import ADConfig
from greykite.detection.detector.data import DetectorData
from greykite.detection.detector.greykite import DETECTOR_PREDICT_COLS
from greykite.detection.detector.greykite import GreykiteDetector
from greykite.detection.detector.reward import Reward
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam


# Evaluation metrics used in the tests.
# F1 score for the True label:
calc_f1 = partial_return(f1_score, True)
# Precision score, for the True label:
calc_precision = partial_return(precision_score, True)
# Recall score for the True label:
calc_recall = partial_return(recall_score, True)

# Boolean to decide if figures are to be shown or not when this test file is run.
# Turn this on when changes are made and include in code reviews.
# Compare before and after the change to confirm everything is as expected.
FIG_SHOW = False


@pytest.fixture(scope="module")
def hourly_data():
    """Generates data for testing `GreykiteDetector`."""

    df = generate_df_for_tests(
        freq="H",
        train_start_date=datetime.datetime(2020, 1, 1),
        intercept=50,
        train_frac=0.99,
        periods=24*28)["df"]

    anomaly_block_list = [
        np.arange(100, 105),
        np.arange(200, 210),
        np.arange(310, 315),
        np.arange(400, 410),
        np.arange(460, 480),
        np.arange(601, 610),
        np.arange(620, 625),
        np.arange(650, 654),
        np.arange(666, 667)]

    # Contaminates `df` with anomalies at the specified locations,
    # via `anomaly_block_list`.
    # If original value is y, the anomalous value is: (1 +/- delta)*y.
    df = contaminate_df_with_anomalies(
        df=df,
        anomaly_block_list=anomaly_block_list,
        delta_range_lower=0.1,
        delta_range_upper=0.2,
        value_col=VALUE_COL,
        min_admissible_value=None,
        max_admissible_value=None)

    fig = plot_lines_markers(
        df=df,
        x_col=TIME_COL,
        line_cols=["y", "contaminated_y"])

    fig.layout.update(title="Generation of hourly anomalous data")
    fig.update_yaxes()
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    df = df.drop(columns=[VALUE_COL]).rename(
        columns={"contaminated_y": VALUE_COL})

    df[ANOMALY_COL] = (df[ANOMALY_COL] == 1)

    assert len(df) == (28 * 24)
    assert sum(df[ANOMALY_COL]) == 69

    train_size = int(26 * 24)
    df_train = df[:train_size].reset_index(drop=True)
    df_test = df[train_size:].reset_index(drop=True)

    assert len(df_train) == 26 * 24
    assert len(df_test) == 2 * 24

    return {
        "df_train": df_train,
        "df_test": df_test,
        "df": df}


@pytest.fixture(scope="module")
def daily_data():
    """Generates data for testing `GreykiteDetector`."""

    df = generate_df_for_tests(
        freq="D",
        train_start_date=datetime.datetime(2020, 1, 1),
        intercept=50,
        train_frac=0.99,
        periods=200)["df"]

    anomaly_block_list = [
        np.arange(10, 15),
        np.arange(33, 35),
        np.arange(60, 65),
        np.arange(82, 85),
        np.arange(94, 98),
        np.arange(100, 105),
        np.arange(111, 113),
        np.arange(125, 130),
        np.arange(160, 163),
        np.arange(185, 190),
        np.arange(198, 200)]

    # Contaminates `df` with anomalies at the specified locations,
    # via `anomaly_block_list`.
    # If original value is y, the anomalous value is: (1 +/- delta)*y.
    df = contaminate_df_with_anomalies(
        df=df,
        anomaly_block_list=anomaly_block_list,
        delta_range_lower=0.25,
        delta_range_upper=0.5,
        value_col=VALUE_COL,
        min_admissible_value=None,
        max_admissible_value=None)

    fig = plot_lines_markers(
        df=df,
        x_col=TIME_COL,
        line_cols=["y", "contaminated_y"])

    fig.layout.update(title="Generation of daily anomalous data")
    fig.update_yaxes()
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    df = df.drop(columns=[VALUE_COL]).rename(
        columns={"contaminated_y": VALUE_COL})

    df[ANOMALY_COL] = (df[ANOMALY_COL] == 1)

    assert len(df) == 200
    assert sum(df[ANOMALY_COL]) == 41

    train_size = int(100)
    df_train = df[:train_size].reset_index(drop=True)
    df_test = df[train_size:].reset_index(drop=True)

    assert len(df_train) == 100
    assert len(df_test) == 100

    return {
        "df_train": df_train,
        "df_test": df_test,
        "df": df}


@pytest.fixture(scope="module")
def forecast_config_info_hourly():
    """Generates ``forecast_config`` for testing."""
    metadata = MetadataParam(
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_end_date=None,
        anomaly_info=None)

    evaluation_period = EvaluationPeriodParam(
        test_horizon=0,
        cv_max_splits=0)

    model_components = ModelComponentsParam(
        autoregression={
            "autoreg_dict": {
                "lag_dict": {"orders": [24]},
                "agg_lag_dict": None}},
        events={
            "auto_holiday": False,
            "holiday_lookup_countries": ["US"],
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2,
            "daily_event_df_dict": None},
        custom={
            "extra_pred_cols": ["dow_hr"],
            "min_admissible_value": 0,
            "normalize_method": "zero_to_one"})

    return ForecastConfig(
        model_template="SILVERKITE_EMPTY",
        metadata_param=metadata,
        coverage=None,
        evaluation_period_param=evaluation_period,
        forecast_horizon=1,
        model_components_param=model_components)


@pytest.fixture(scope="module")
def forecast_config_info_daily():
    """Generates ``forecast_config`` for testing."""
    metadata = MetadataParam(
        time_col=TIME_COL,
        value_col=VALUE_COL,
        train_end_date=None,
        anomaly_info=None)

    evaluation_period = EvaluationPeriodParam(
        test_horizon=0,
        cv_max_splits=0)

    model_components = ModelComponentsParam(
        autoregression={
            "autoreg_dict": {
                "lag_dict": {"orders": [7]},
                "agg_lag_dict": None}},
        events={
            "auto_holiday": False,
            "holiday_lookup_countries": ["US"],
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2,
            "daily_event_df_dict": None},
        custom={
            "extra_pred_cols": ["dow"],
            "min_admissible_value": 0,
            "normalize_method": "zero_to_one"})

    return ForecastConfig(
        model_template="SILVERKITE_EMPTY",
        metadata_param=metadata,
        coverage=None,
        evaluation_period_param=evaluation_period,
        forecast_horizon=1,
        model_components_param=model_components)


def test_greykite_init():
    """Tests ``GreykiteDetector`` initialization."""
    detector = GreykiteDetector()
    assert detector.forecast_config is not None
    assert detector.ad_config is not None
    assert detector.reward is not None


def test_greykite_detector_hourly_f1(hourly_data, forecast_config_info_hourly):
    """Tests ``GreykiteDetector`` with F1 score as reward on hourly data."""
    df_train = hourly_data["df_train"]
    df_test = hourly_data["df_test"]
    df = hourly_data["df"]

    forecast_config = forecast_config_info_hourly
    ad_config = ADConfig(
        volatility_features_list=[["dow"], ["hour"]],
        coverage_grid=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
        target_anomaly_percent=None,
        variance_scaling=False)

    train_data = DetectorData(df=df_train)

    def reward_func(data):
        return calc_f1(
            y_true=data.y_true,
            y_pred=data.y_pred)

    reward = Reward(reward_func)

    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=reward)

    detector.fit(data=train_data)

    # Checks optimal parameter.
    assert detector.fit_info["param"] == {
        "coverage": 0.99,
        "volatility_features": ["dow"]}
    # Checks parameter grid.
    param_obj_list = detector.fit_info["param_obj_list"]
    param_eval_df = pd.DataFrame.from_records(param_obj_list)
    assert list(param_eval_df.columns) == ["coverage", "volatility_features", "obj_value"]

    param_eval_df["volatility_features"] = param_eval_df["volatility_features"].map(str)
    fig = px.line(
        param_eval_df,
        x="coverage",
        y="obj_value",
        color="volatility_features",
        title="'GreykiteDetector' result of parameter search: f1, hourly data")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    test_data = DetectorData(
        df=df_test,
        y_true=df_test[ANOMALY_COL])

    test_data = detector.predict(test_data)
    test_obj_value = detector.reward.apply(test_data)
    assert test_obj_value == pytest.approx(0.70, 0.01)

    test_recall = calc_recall(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    test_precision = calc_precision(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    assert test_recall == pytest.approx(1.00, 0.001)
    assert test_precision == pytest.approx(0.545, 0.001)

    fit_data = detector.fit_info["best_calc_result"].data
    fit_df = fit_data.pred_df
    pred_df = test_data.pred_df

    # Checks if we get the expected columns in the fit / prediction data.
    assert list(pred_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    assert list(fit_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    fit_pred_df = pd.concat([fit_df, pred_df], axis=0).reset_index(drop=True)
    fit_pred_df[ANOMALY_COL] = df[ANOMALY_COL]

    fig = plt_compare_series_annotations(
        df=fit_pred_df,
        x_col=TIME_COL,
        actual_col=ACTUAL_COL,
        actual_label_col=ANOMALY_COL,
        forecast_label_col=PREDICTED_ANOMALY_COL,
        keep_cols=[PREDICTED_LOWER_COL, PREDICTED_UPPER_COL],
        forecast_col=PREDICTED_COL,
        standardize_col=None,
        title="test_greykite_detector_hourly_f1")

    fig.add_vline(
        x=fit_df[TIME_COL].max(),
        line_width=1,
        line_dash="dash",
        line_color="green")

    fig.add_annotation(
        x=fit_df[TIME_COL].max(),
        y=fit_pred_df[ACTUAL_COL].max(),
        text="end of training")
    assert fig is not None
    if FIG_SHOW:
        fig.show()


def test_greykite_detector_hourly_anomaly_percent(
        hourly_data,
        forecast_config_info_hourly):
    """Tests ``GreykiteDetector`` with user-specified anomaly percent as reward."""
    df_train = hourly_data["df_train"]
    df_test = hourly_data["df_test"]
    df = hourly_data["df"]

    forecast_config = forecast_config_info_hourly
    ad_config = ADConfig(
        volatility_features_list=[["dow"], ["hour"]],
        coverage_grid=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
        target_anomaly_percent=10.0,
        variance_scaling=False)

    train_data = DetectorData(df=df_train)

    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=None)

    detector.fit(data=train_data)

    # Checks optimal parameter.
    assert detector.fit_info["param"] == {
        "coverage": 0.99,
        "volatility_features": ["dow"]}
    assert {TIME_COL, ACTUAL_COL, PREDICTED_COL,
            PREDICTED_LOWER_COL, PREDICTED_UPPER_COL,
            PREDICTED_ANOMALY_COL}.issubset(set(detector.fitted_df.columns))
    # Checks parameter grid.
    param_obj_list = detector.fit_info["param_obj_list"]
    param_eval_df = pd.DataFrame.from_records(param_obj_list)
    assert list(param_eval_df.columns) == ["coverage", "volatility_features", "obj_value"]

    param_eval_df["volatility_features"] = param_eval_df["volatility_features"].map(str)
    fig = px.line(
        param_eval_df,
        x="coverage",
        y="obj_value",
        color="volatility_features",
        title="'GreykiteDetector' res. of param search: reward=anomaly_percent, hourly data")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    test_data = DetectorData(
        df=df_test,
        y_true=df_test[ANOMALY_COL])

    test_data = detector.predict(test_data)
    test_obj_value = detector.reward.apply(test_data)
    assert test_obj_value == pytest.approx(-1.129, 0.001)

    test_recall = calc_recall(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    test_precision = calc_precision(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    assert test_recall == pytest.approx(1.00, 0.001)
    assert test_precision == pytest.approx(0.545, 0.001)

    fit_data = detector.fit_info["best_calc_result"].data
    fit_df = fit_data.pred_df
    pred_df = test_data.pred_df

    # Checks if we get the expected columns in the fit / prediction data.
    assert list(pred_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    assert list(fit_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    fit_pred_df = pd.concat([fit_df, pred_df], axis=0).reset_index(drop=True)
    fit_pred_df[ANOMALY_COL] = df[ANOMALY_COL]

    fig = plt_compare_series_annotations(
        df=fit_pred_df,
        x_col=TIME_COL,
        actual_col=ACTUAL_COL,
        actual_label_col=ANOMALY_COL,
        forecast_label_col=PREDICTED_ANOMALY_COL,
        keep_cols=[PREDICTED_LOWER_COL, PREDICTED_UPPER_COL],
        forecast_col=PREDICTED_COL,
        standardize_col=None,
        title="test_greykite_detector_hourly_anomaly_percent")

    fig.add_vline(
        x=fit_df[TIME_COL].max(),
        line_width=1,
        line_dash="dash",
        line_color="green")

    fig.add_annotation(
        x=fit_df[TIME_COL].max(),
        y=fit_pred_df[ACTUAL_COL].max(),
        text="end of training")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    # Tests plot method
    fig = detector.plot(title="test_greykite_detector_hourly_anomaly_percent")
    assert fig is not None
    if FIG_SHOW:
        fig.show()


def test_greykite_detector_daily_f1(
        daily_data,
        forecast_config_info_daily):
    """Tests ``GreykiteDetector`` with F1 score as reward.
    Also tests if specifying objective through `ADConfig` yields the exact same result."""
    df_train = daily_data["df_train"]
    df_test = daily_data["df_test"]
    df = daily_data["df"]

    forecast_config = forecast_config_info_daily
    ad_config = ADConfig(
        volatility_features_list=[["dow"], ["is_weekend"]],
        coverage_grid=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
        variance_scaling=True)

    train_data = DetectorData(df=df_train)

    def reward_func(data):
        return calc_f1(
            y_true=data.y_true,
            y_pred=data.y_pred)

    reward = Reward(reward_func)

    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=reward)

    detector.fit(data=train_data)
    # Checks optimal parameter.
    assert detector.fit_info["param"] == {
        "coverage": 0.99,
        "volatility_features": ["is_weekend"]}
    # Checks parameter grid.
    param_obj_list = detector.fit_info["param_obj_list"]
    param_eval_df = pd.DataFrame.from_records(param_obj_list)

    assert list(param_eval_df.columns) == [
        "coverage",
        "volatility_features",
        "obj_value"]

    param_eval_df["volatility_features"] = param_eval_df["volatility_features"].map(str)
    fig = px.line(
        param_eval_df,
        x="coverage",
        y="obj_value",
        color="volatility_features",
        title="'GreykiteDetector' result of parameter search: reward=f1, daily data")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    test_data = DetectorData(
        df=df_test.copy(),
        y_true=df_test[ANOMALY_COL])

    test_data = detector.predict(test_data)
    test_obj_value = detector.reward.apply(test_data)
    assert test_obj_value == pytest.approx(0.95454, 0.001)

    test_recall = calc_recall(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    test_precision = calc_precision(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    assert test_recall == pytest.approx(0.95454, 0.001)
    assert test_precision == pytest.approx(0.95454, 0.001)

    fit_data = detector.fit_info["best_calc_result"].data
    fit_df = fit_data.pred_df
    pred_df = test_data.pred_df
    # Checks if we get the expected columns in the fit / prediction data.
    assert list(pred_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    assert list(fit_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]

    fit_obj_value = detector.reward.apply(fit_data)
    assert fit_obj_value == pytest.approx(1.0, 0.001)

    fit_recall = calc_recall(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)

    fit_precision = calc_precision(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)

    assert fit_recall == pytest.approx(1.0, 0.001)
    assert fit_precision == pytest.approx(1.0, 0.001)

    fit_pred_df = pd.concat([fit_df, pred_df], axis=0).reset_index(drop=True)
    fit_pred_df[ANOMALY_COL] = df[ANOMALY_COL]

    fig = plt_compare_series_annotations(
        df=fit_pred_df,
        x_col=TIME_COL,
        actual_col=ACTUAL_COL,
        actual_label_col=ANOMALY_COL,
        forecast_label_col=PREDICTED_ANOMALY_COL,
        keep_cols=[PREDICTED_LOWER_COL, PREDICTED_UPPER_COL],
        forecast_col=PREDICTED_COL,
        standardize_col=None,
        title="test_greykite_detector_detector_daily_f1")

    fig.add_vline(
        x=fit_df[TIME_COL].max(),
        line_width=1,
        line_dash="dash",
        line_color="green")

    fig.add_annotation(
        x=fit_df[TIME_COL].max(),
        y=fit_pred_df[ACTUAL_COL].max(),
        text="end of training")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    fig = plot_anomalies_over_forecast_vs_actual(
        df=fit_pred_df,
        time_col=TIME_COL,
        actual_col=ACTUAL_COL,
        predicted_col=PREDICTED_COL,
        predicted_anomaly_col=PREDICTED_ANOMALY_COL,
        anomaly_col=ANOMALY_COL,
        marker_opacity=0.6,
        predicted_anomaly_marker_color="black",
        anomaly_marker_color="yellow",
        predicted_lower_col=PREDICTED_LOWER_COL,
        predicted_upper_col=PREDICTED_UPPER_COL,
        train_end_date=fit_df[TIME_COL].max())
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    # Now we check if specifying objective through `ADConfig` yields the same results.
    # We do that by over-writing the objective from None to `F1`
    # On the other hand we set the reward to be None.
    # We calculate same quantities as the above and assign to variables with f"{quantity}_new".
    # Then we compare those new quantities to the quantities obtained already.
    assert ad_config.objective is None
    ad_config.objective = F1
    reward = None

    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=reward)

    detector.fit(data=train_data)
    # Checks optimal parameter.
    assert detector.fit_info["param"] == {
        "coverage": 0.99,
        "volatility_features": ["is_weekend"]}
    # Checks parameter grid.
    param_obj_list = detector.fit_info["param_obj_list"]
    param_eval_df = pd.DataFrame.from_records(param_obj_list)

    assert list(param_eval_df.columns) == [
        "coverage",
        "volatility_features",
        "obj_value"]

    test_data = DetectorData(
        df=df_test.copy(),
        y_true=df_test[ANOMALY_COL])

    test_data = detector.predict(test_data)
    test_obj_value_new = detector.reward.apply(test_data)
    assert test_obj_value_new == pytest.approx(test_obj_value, 0.001)

    test_recall_new = calc_recall(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    test_precision_new = calc_precision(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    assert test_recall_new == pytest.approx(test_recall, 0.001)
    assert test_precision_new == pytest.approx(test_precision, 0.001)

    fit_data = detector.fit_info["best_calc_result"].data
    fit_df = fit_data.pred_df
    pred_df = test_data.pred_df
    # Checks if we get the expected columns in the fit / prediction data.
    assert list(pred_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    assert list(fit_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]

    fit_obj_value_new = detector.reward.apply(fit_data)
    assert fit_obj_value_new == pytest.approx(fit_obj_value, 0.001)

    fit_recall_new = calc_recall(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)

    fit_precision_new = calc_precision(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)

    assert fit_recall_new == pytest.approx(fit_recall, 0.001)
    assert fit_precision_new == pytest.approx(fit_precision, 0.001)


def test_greykite_detector_daily_outlier(
        daily_data,
        forecast_config_info_daily):
    """Tests ``GreykiteDetector`` with data injected with large outlier.
        It is worth noting the optimal params and test values have not changed
        dramatically compared to the case without the injected outlier in this test:
        `test_greykite_detector_detector_daily_f1`.
        """
    df_train = daily_data["df_train"].copy()
    df_test = daily_data["df_test"]
    df = daily_data["df"]
    # Creates a very large outlier.
    df_train.loc[1, "y"] = 10 * max(abs(df["y"]))
    fig = plot_lines_markers(df=df_train, x_col=TIME_COL, line_cols=["y"])
    if FIG_SHOW:
        fig.show()

    forecast_config = forecast_config_info_daily
    ad_config = ADConfig(
        volatility_features_list=[["dow"], ["is_weekend"]],
        coverage_grid=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
        target_anomaly_percent=None,
        variance_scaling=True)

    train_data = DetectorData(df=df_train)

    def reward_func(data):
        return calc_f1(
            y_true=data.y_true,
            y_pred=data.y_pred)

    reward = Reward(reward_func)

    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=reward)

    detector.fit(data=train_data)
    # Checks optimal parameter.
    assert detector.fit_info["param"] == {
        "coverage": 0.99,
        "volatility_features": ["is_weekend"]}

    # Checks parameter grid.
    param_obj_list = detector.fit_info["param_obj_list"]
    param_eval_df = pd.DataFrame.from_records(param_obj_list)

    assert list(param_eval_df.columns) == [
        "coverage",
        "volatility_features",
        "obj_value"]

    param_eval_df["volatility_features"] = param_eval_df["volatility_features"].map(str)
    fig = px.line(
        param_eval_df,
        x="coverage",
        y="obj_value",
        color="volatility_features",
        title="'GreykiteDetector' result of parameter search: reward=f1, daily data")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    test_data = DetectorData(
        df=df_test,
        y_true=df_test[ANOMALY_COL])

    test_data = detector.predict(test_data)
    test_obj_value = detector.reward.apply(test_data)

    test_recall = calc_recall(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    test_precision = calc_precision(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    fit_data = detector.fit_info["best_calc_result"].data
    fit_df = fit_data.pred_df
    pred_df = test_data.pred_df
    # Checks if we get the expected columns in the fit / prediction data.
    assert list(pred_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    assert list(fit_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]

    fit_obj_value = detector.reward.apply(fit_data)

    fit_recall = calc_recall(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)

    fit_precision = calc_precision(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)

    assert test_obj_value == pytest.approx(0.95, 0.01)
    assert fit_obj_value == pytest.approx(1.0, 0.01)
    assert test_recall == pytest.approx(0.95, 0.01)
    assert test_precision == pytest.approx(0.95, 0.01)
    assert fit_recall == pytest.approx(1.0, 0.01)
    assert fit_precision == pytest.approx(1.0, 0.01)

    fit_pred_df = pd.concat([fit_df, pred_df], axis=0).reset_index(drop=True)
    fit_pred_df[ANOMALY_COL] = df[ANOMALY_COL]

    fig = plt_compare_series_annotations(
        df=fit_pred_df,
        x_col=TIME_COL,
        actual_col=ACTUAL_COL,
        actual_label_col=ANOMALY_COL,
        forecast_label_col=PREDICTED_ANOMALY_COL,
        keep_cols=[PREDICTED_LOWER_COL, PREDICTED_UPPER_COL],
        forecast_col=PREDICTED_COL,
        standardize_col=None,
        title="test_greykite_detector_detector_daily_f1_outlier")

    fig.add_vline(
        x=fit_df[TIME_COL].max(),
        line_width=1,
        line_dash="dash",
        line_color="green")

    fig.add_annotation(
        x=fit_df[TIME_COL].max(),
        y=fit_pred_df[ACTUAL_COL].max(),
        text="end of training")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    # Tests plot method.
    fig = detector.plot(
        phase="train",
        title="test_greykite_detector_detector_daily_f1_outlier - fit phase")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    fig = detector.plot(title="test_greykite_detector_detector_daily_f1_outlier - predict phase")
    assert fig is not None
    if FIG_SHOW:
        fig.show()


def test_greykite_detector_daily_f1_with_ape_filter(
        daily_data,
        forecast_config_info_daily):
    """Tests ``GreykiteDetector`` on daily data with APE filter and F1 score as reward."""
    df_train = daily_data["df_train"]
    df_test = daily_data["df_test"]
    df = daily_data["df"]

    forecast_config = forecast_config_info_daily
    ad_config = ADConfig(
        coverage_grid=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
        target_anomaly_percent=None,
        ape_grid=[0, 20, 50],
        variance_scaling=True)

    train_data = DetectorData(df=df_train)

    def reward_func(data):
        return calc_f1(
            y_true=data.y_true,
            y_pred=data.y_pred)

    reward = Reward(reward_func)

    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=reward)

    detector.fit(data=train_data)
    # Checks optimal parameter.
    assert detector.fit_info["param"] == {
        "coverage": 0.5,
        "volatility_features": [],
        "absolute_percent_error": 20}
    # Checks parameter grid.
    param_obj_list = detector.fit_info["param_obj_list"]
    param_eval_df = pd.DataFrame.from_records(param_obj_list)

    assert list(param_eval_df.columns) == [
        "coverage",
        "volatility_features",
        "absolute_percent_error",
        "obj_value"]

    param_eval_df["absolute_percent_error"] = param_eval_df["absolute_percent_error"].map(str)
    fig = px.line(
        param_eval_df,
        x="coverage",
        y="obj_value",
        color="absolute_percent_error",
        title="'GreykiteDetector' result of parameter search: reward=f1, filter=ape, daily data.")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    test_data = DetectorData(
        df=df_test,
        y_true=df_test[ANOMALY_COL])

    test_data = detector.predict(test_data)
    test_obj_value = detector.reward.apply(test_data)
    assert test_obj_value == pytest.approx(0.97, 0.01)

    test_recall = calc_recall(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    test_precision = calc_precision(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)

    assert test_recall == pytest.approx(0.95454, 0.001)
    assert test_precision == pytest.approx(1.00, 0.001)

    fit_data = detector.fit_info["best_calc_result"].data
    fit_df = fit_data.pred_df
    pred_df = test_data.pred_df
    # Checks if we get the expected columns in the fit / prediction data.
    assert list(pred_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    assert list(fit_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]

    fit_obj_value = detector.reward.apply(fit_data)
    assert fit_obj_value == pytest.approx(1.00, 0.01)

    fit_recall = calc_recall(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)

    fit_precision = calc_precision(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)

    assert fit_recall == pytest.approx(1.00, 0.01)
    assert fit_precision == pytest.approx(1.00, 0.01)

    fit_pred_df = pd.concat([fit_df, pred_df], axis=0).reset_index(drop=True)
    fit_pred_df[ANOMALY_COL] = df[ANOMALY_COL]

    fig = plt_compare_series_annotations(
        df=fit_pred_df,
        x_col=TIME_COL,
        actual_col=ACTUAL_COL,
        actual_label_col=ANOMALY_COL,
        forecast_label_col=PREDICTED_ANOMALY_COL,
        keep_cols=[PREDICTED_LOWER_COL, PREDICTED_UPPER_COL],
        forecast_col=PREDICTED_COL,
        standardize_col=None,
        title="test_greykite_detector_with_ape_filter_daily_f1")

    fig.add_vline(
        x=fit_df[TIME_COL].max(),
        line_width=1,
        line_dash="dash",
        line_color="green")

    fig.add_annotation(
        x=fit_df[TIME_COL].max(),
        y=fit_pred_df[ACTUAL_COL].max(),
        text="end of training")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    fig = plot_anomalies_over_forecast_vs_actual(
        df=fit_pred_df,
        time_col=TIME_COL,
        actual_col=ACTUAL_COL,
        predicted_col=PREDICTED_COL,
        predicted_anomaly_col=PREDICTED_ANOMALY_COL,
        anomaly_col=ANOMALY_COL,
        marker_opacity=0.6,
        predicted_anomaly_marker_color="black",
        anomaly_marker_color="yellow",
        predicted_lower_col=PREDICTED_LOWER_COL,
        predicted_upper_col=PREDICTED_UPPER_COL,
        train_end_date=fit_df[TIME_COL].max())
    assert fig is not None
    if FIG_SHOW:
        fig.show()


def test_greykite_detector_daily_anomaly_at_df_end(daily_data, forecast_config_info_daily):
    """Tests ``GreykiteDetector`` when anomaly is at the end of the training data."""
    df_train = daily_data["df_train"].copy()
    # Adds anomalies at the end of the training data.
    df_train["y"][-4:] = np.NaN
    fig = plot_lines_markers(df=df_train, x_col=TIME_COL, line_cols=["y"])
    if FIG_SHOW:
        fig.show()

    forecast_config = forecast_config_info_daily
    ad_config = ADConfig(
        volatility_features_list=[["dow"], ["is_weekend"]],
        coverage_grid=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
        target_anomaly_percent=None,
        variance_scaling=True)

    train_data = DetectorData(df=df_train)

    def reward_func(data):
        return calc_f1(
            y_true=data.y_true,
            y_pred=data.y_pred)

    reward = Reward(reward_func)

    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=reward)

    detector.fit(data=train_data)
    # Checks optimal parameter.
    assert detector.fit_info["param"] == {
        "coverage": 0.99,
        "volatility_features": ["is_weekend"]}

    # Checks parameter grid.
    param_obj_list = detector.fit_info["param_obj_list"]
    param_eval_df = pd.DataFrame.from_records(param_obj_list)

    assert list(param_eval_df.columns) == [
        "coverage",
        "volatility_features",
        "obj_value"]

    param_eval_df["volatility_features"] = param_eval_df["volatility_features"].map(str)
    fig = px.line(
        param_eval_df,
        x="coverage",
        y="obj_value",
        color="volatility_features",
        title="'GreykiteDetector' result of parameter search: reward=f1, daily data")
    assert fig is not None
    if FIG_SHOW:
        fig.show()


def test_merge_anomaly_info():
    """Tests ``merge_anomaly_info`` method."""
    periods = 10
    df = pd.DataFrame({
        TIME_COL: pd.date_range(start="2020-01-01", periods=periods),
        VALUE_COL: range(periods),
        ANOMALY_COL: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  # Anomalies on 2020-01-03 and 2020-01-04
    })
    y_true = [0, 0, 1, 1, 0, 0, 0, 0, 0, 1]  # Anomalies on 2020-01-03, 2020-01-04 and 2020-01-10
    anomaly_df = pd.DataFrame({
        START_TIME_COL: ["2020-01-04"],
        END_TIME_COL: ["2020-01-06"]})
    data = DetectorData(
        df=df,
        y_true=y_true,
        anomaly_df=anomaly_df)

    detector = GreykiteDetector()
    fit_data = detector.merge_anomaly_info(data, freq="D")
    # Checks `anomaly_df` in `fit_data`.
    # We expect combined anomalies from 2020-01-03 to 2020-01-06, and on 2020-01-10.
    expected_anomaly_df = pd.DataFrame({
        START_TIME_COL: pd.to_datetime(["2020-01-03", "2020-01-10"]),
        END_TIME_COL: pd.to_datetime(["2020-01-06", "2020-01-10"])})
    assert fit_data.anomaly_df.equals(expected_anomaly_df)
    # Checks `y_true` in `fit_data`.
    expected_y_true = pd.Series([0, 0, 1, 1, 1, 1, 0, 0, 0, 1]).astype(bool)
    assert fit_data.y_true.equals(expected_y_true)
    # Checks `df` in `fit_data`.
    assert fit_data.df[ANOMALY_COL].equals(expected_y_true)
    assert f"adjusted_{VALUE_COL}" in fit_data.df.columns


def test_greykite_detector_hourly_anomaly_pickup():
    """Tests that anomaly data is picked up properly by the ``GreykiteDetector``.

    Anomalies are injected to the training data and the anomaly info is passed to
    ``GreykiteDetector`` via `anomaly_df`.

    We check that the anomaly info is picked up by the detector during training
    and the future forecasts are unaffected by the anomaly values.
    """
    metadata = MetadataParam(freq="H")

    evaluation_period = EvaluationPeriodParam(
        test_horizon=0,
        cv_max_splits=0)

    # This forecast configs includes the median of past three weeks as an important predictor.
    # Therefore, if that median is off due to anomalies, the prediction can be off.
    # In this test, we prove that by labeling those points and passing that information to the model
    # via `anomaly_df`, we can avoid the model to fit to those values.
    model_components = ModelComponentsParam(
        growth={
            "growth_term": "linear"},
        hyperparameter_override={
            "input__response__null__impute_algorithm": "ts_interpolate",
            "input__response__null__impute_params": {
                "orders": [168, 336, 504]}},
        seasonality={
            "monthly_seasonality": 2,
            "yearly_seasonality": 7},
        events={
            "holiday_lookup_countries": ["US"],
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2},
        autoregression={
            "autoreg_dict": {
                "agg_lag_dict": {
                    "orders_list": [[168, 336, 504]],
                    "agg_func": "median"}}},
        custom={
            "fit_algorithm_dict": {"fit_algorithm": "ridge"},
            "min_admissible_value": 0,
            "normalize_method": "zero_to_one",
            "extra_pred_cols": [
                "is_event:is_weekend:C(hour)",
                "dow_hr",
                "y_avglag_168_336_504*dow_hr",
                "y_avglag_168_336_504*sin1_ct1_yearly",
                "y_avglag_168_336_504*cos1_ct1_yearly",
                "y_avglag_168_336_504*sin1_tom_monthly",
                "y_avglag_168_336_504*cos1_tom_monthly",
                "us_dst*dow_hr"]})

    forecast_config = ForecastConfig(
        model_template="SILVERKITE_EMPTY",
        metadata_param=metadata,
        coverage=0.95,
        evaluation_period_param=evaluation_period,
        forecast_horizon=1,
        model_components_param=model_components)

    ad_config = ADConfig(
        target_anomaly_percent=0.35,
        volatility_features_list=[["hour"]],
        coverage_grid=[0.996],
        sape_grid=[5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        variance_scaling=True)

    input_data = generate_df_for_tests(
        freq="H",
        train_start_date=datetime.datetime(2020, 1, 1),
        intercept=50,
        train_frac=0.99,
        periods=24*60)
    df_train = input_data["train_df"]
    df_test = input_data["test_df"]

    # Injects 3 anomalies at the same time of the week (0 values).
    # Note that if model does not have access to these anomaly labels,
    # the prediction can be off.
    df_train.loc[[24*28, 24*35, 24*42], "y"] = 0
    # Passes the anomaly info to the `detector` via `anomaly_df`.
    anomaly_df = pd.DataFrame({
        START_TIME_COL: ["2020-01-29 00:00:00", "2020-02-05 00:00:00", "2020-02-12 00:00:00"],
        END_TIME_COL: ["2020-01-29 00:00:00", "2020-02-05 00:00:00", "2020-02-12 00:00:00"]})

    fig = plot_lines_markers(
        df=df_train,
        x_col=TIME_COL,
        line_cols=["y"])
    fig.layout.update(title="Generation of daily anomalous data")
    fig.update_yaxes()
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    # Trains `GreykiteDetector`.
    train_data = DetectorData(df=df_train, anomaly_df=anomaly_df)
    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=None)
    detector.fit(data=train_data)
    # We expect the anomalies to be picked up by the `detector` during training.
    if FIG_SHOW:
        detector.plot(phase="train")

    # Predicts on test data.
    test_data = DetectorData(
        df=df_test)
    detector.predict(test_data)
    fig = detector.plot()
    assert fig is not None
    # Adds a vertical line at the next hourly datapoint.
    # The forecast at this data point should not be close to 0.
    # This is because even though the median aggregated lag across the previous 3 weeks is 0,
    # the anomaly labels help the model not fit those values.
    fig.add_vline(x="2020-02-19 00:00:00", line_dash="dash")
    if FIG_SHOW:
        fig.show()


def test_summary(daily_data, forecast_config_info_daily):
    """Tests ``summary`` method."""
    df_train = daily_data["df_train"]
    forecast_config = forecast_config_info_daily
    ad_config = ADConfig(
        coverage_grid=[0.2, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
        target_anomaly_percent=None,
        ape_grid=[0, 20, 50],
        variance_scaling=True)

    train_data = DetectorData(df=df_train)

    def reward_func(data):
        return calc_f1(
            y_true=data.y_true,
            y_pred=data.y_pred)
    reward = Reward(reward_func)

    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=reward)
    detector.fit(data=train_data)

    summary = detector.summary()
    assert "Anomaly Detection Model Summary" in summary
    assert "Average Anomaly Duration" in summary
    assert "Precision" in summary
    assert "Recall" in summary
    assert "Optimal Parameters" in summary
    # Checks if the summary contains the forecast model summary.
    assert "Residuals" in summary
    assert "Multiple R-squared" in summary
