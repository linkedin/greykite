import datetime
import inspect
import sys
from collections import OrderedDict

import dill
import numpy as np
import pandas as pd
import pytest
from patsy.desc import Term

from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import PREDICTED_ANOMALY_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils_anomalies import contaminate_df_with_anomalies
from greykite.common.viz.timeseries_annotate import plot_anomalies_over_forecast_vs_actual
from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.common.ad_evaluation import precision_score
from greykite.detection.common.ad_evaluation import recall_score
from greykite.detection.common.pickler import GreykitePickler
from greykite.detection.detector.ad_utils import partial_return
from greykite.detection.detector.config import ADConfig
from greykite.detection.detector.constants import FIG_SHOW
from greykite.detection.detector.data import DetectorData as Data
from greykite.detection.detector.greykite import DETECTOR_PREDICT_COLS
from greykite.detection.detector.greykite import GreykiteDetector
from greykite.detection.detector.reward import Reward
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results


# Evaluation metrics used in the tests.
# F1 score for the True label.
f1_calc = partial_return(f1_score, True)
# Precision score, for the True label.
calc_precision = partial_return(precision_score, True)
# Recall score for the True label.
calc_recall = partial_return(recall_score, True)


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


class X:
    def __init__(self, a):
        self.a = a


class TestClass:
    def __init__(self, a, b):
        self.a = a
        self.b = X(b)
        self.c = {
            Term([]): None,
            Term(["a", "b"]): [3, 4, 5]
        }
        self.d = {
            Term([]): {
                "d_1_1": 1,
                Term([]): Term(["a", "b"])
            },
            "d_2": [1, 2]
        }


def test_init():
    """Tests initialization."""
    pickler = GreykitePickler()
    assert pickler.obj is None


def test_integer():
    """Tests pickling and unpickling of integers."""
    pickler = GreykitePickler()
    obj = 1
    serialized = pickler.dumps(obj)
    assert serialized == {"ROOT.pkl": 'gARLAS4=\n'}
    assert pickler.obj == obj
    deserialized = pickler.loads(serialized)
    assert deserialized == obj


def test_list():
    """Tests pickling and unpickling of lists."""
    # Checks simple lists that can be serialized by dill.
    obj = [1, 2, 3]
    serialized_by_dill = GreykitePickler.dumps_to_str(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    assert serialized == {"ROOT.pkl": serialized_by_dill}
    deserialized = pickler.loads(serialized)
    assert deserialized == obj

    # Checks complex lists that cannot be serialized by dill.
    obj = [Term([]), Term(["a", "b"]), 2]
    with pytest.raises(NotImplementedError):
        dill.dumps(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    deserialized = pickler.loads(serialized)
    assert isinstance(deserialized, list)
    assert deserialized == obj


def test_tuple():
    """Tests pickling and unpickling of tuples."""
    # Checks simple lists that can be serialized by dill.
    obj = (1, 2, 3)
    serialized_by_dill = GreykitePickler.dumps_to_str(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    assert serialized == {"ROOT.pkl": serialized_by_dill}
    deserialized = pickler.loads(serialized)
    assert deserialized == obj

    # Checks complex lists that cannot be serialized by dill.
    obj = (Term([]), Term(["a", "b"]), 2)
    with pytest.raises(NotImplementedError):
        dill.dumps(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    deserialized = pickler.loads(serialized)
    assert isinstance(deserialized, tuple)
    assert deserialized == obj


def test_dict():
    """Tests pickling and unpickling of dictionaries."""
    # Checks simple dictionaries that can be serialized by dill.
    obj = {"key1": X(1)}
    serialized_by_dill = GreykitePickler.dumps_to_str(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    assert serialized == {"ROOT.pkl": serialized_by_dill}
    deserialized = pickler.loads(serialized)
    assert isinstance(deserialized, dict)
    assert deserialized.keys() == obj.keys()
    assert deserialized["key1"].a == obj["key1"].a

    # Checks complex dictionaries that cannot be serialized by dill.
    obj = {"key1": Term([])}
    with pytest.raises(NotImplementedError):
        dill.dumps(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    deserialized = pickler.loads(serialized)
    assert isinstance(deserialized, dict)
    assert deserialized.keys() == obj.keys()
    assert deserialized["key1"].factors == obj["key1"].factors


def test_ordered_dict():
    """Tests pickling and unpickling of ordered dictionaries."""
    # Checks simple ordered dictionaries that can be serialized by dill.
    obj = OrderedDict({"a": 1, X(2): 3, 5: ["b"]})
    serialized_by_dill = GreykitePickler.dumps_to_str(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    assert serialized == {"ROOT.pkl": serialized_by_dill}
    deserialized = pickler.loads(serialized)
    assert isinstance(deserialized, OrderedDict)
    assert deserialized["a"] == obj["a"]
    assert deserialized[5] == obj[5]

    # Checks complex ordered dictionaries that cannot be serialized by dill.
    obj = OrderedDict({"a": 1, Term([]): 3, 5: ["b"]})
    with pytest.raises(NotImplementedError):
        dill.dumps(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    deserialized = pickler.loads(serialized)
    assert isinstance(deserialized, dict)
    assert deserialized["a"] == obj["a"]
    assert deserialized[5] == obj[5]


def test_class():
    """Tests pickling and unpickling of classes."""
    # Checks simple classes that can be serialized by dill.
    obj = X(a=1)
    serialized_by_dill = GreykitePickler.dumps_to_str(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    assert serialized == {"ROOT.pkl": serialized_by_dill}
    deserialized = pickler.loads(serialized)
    assert deserialized.a == obj.a

    # Checks complex classes that cannot be serialized by dill.
    obj = Term([])
    with pytest.raises(NotImplementedError):
        dill.dumps(obj)
    pickler = GreykitePickler()
    serialized = pickler.dumps(obj)
    deserialized = pickler.loads(serialized)
    assert deserialized.__class__ == obj.__class__
    assert deserialized.__dict__ == obj.__dict__


def test_silverkite_forecast_result():
    """Tests pickling and unpickling of Silverkite ForecastResult."""
    df = generate_df_for_tests(
        freq="D",
        periods=365)["df"]
    forecaster = Forecaster()
    # Run the forecast
    result = forecaster.run_forecast_config(
        df=df,  # includes the regressor
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            forecast_horizon=7,
            coverage=0.8,
            metadata_param=MetadataParam(
                time_col="ts",
                value_col="y",
                freq="D"
            ),
            evaluation_period_param=EvaluationPeriodParam(
                cv_max_splits=1,
                cv_horizon=7,
                test_horizon=7,
                cv_min_train_periods=80
            ),
            model_components_param=ModelComponentsParam(
                custom={"fit_algorithm_dict": {"fit_algorithm": "linear"}},
                autoregression={"autoreg_dict": "auto"}
            ),
            computation_param=ComputationParam(n_jobs=-1),
        )
    )
    pickler = GreykitePickler()
    serialized = pickler.dumps(result)
    deserialized = pickler.loads(serialized)

    # Tests loaded results.
    # Grid search cv results.
    assert_equal(
        summarize_grid_search_results(result.grid_search),
        summarize_grid_search_results(deserialized.grid_search)
    )
    # Grid search attributes.
    for key in result.grid_search.__dict__.keys():
        if key not in ["scoring", "estimator", "refit", "cv", "error_score", "cv_results_",
                       "scorer_", "best_estimator_"]:
            assert_equal(
                result.grid_search.__dict__[key],
                deserialized.grid_search.__dict__[key])

    # Model.
    assert_equal(
        result.model[-1].predict(df),
        deserialized.model[-1].predict(df)
    )
    assert result.model[-1].model_dict["x_design_info"] is not None
    # Model: estimator.
    for key in result.model[-1].__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model_dict"]:
            assert_equal(
                result.model[-1].__dict__[key],
                deserialized.model[-1].__dict__[key])
    assert_equal(
        inspect.getsource(result.model[-1].__dict__["score_func"]),
        inspect.getsource(deserialized.model[-1].__dict__["score_func"])
    )
    # Model: estimator/model_dict.
    for key in result.model[-1].model_dict.keys():
        # Functions and classes are not testable.
        if key not in ["x_design_info", "fs_func", "ml_model", "plt_pred",
                       "autoreg_dict", "changepoint_detector", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                result.model[-1].model_dict[key],
                deserialized.model[-1].model_dict[key])
        # Tests function source code.
        elif key in ["fs_func", "plt_pred", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                inspect.getsource(result.model[-1].model_dict[key]),
                inspect.getsource(deserialized.model[-1].model_dict[key]))
    # Model: estimator/model_dict/autoreg_dict.
    for key in result.model[-1].model_dict["autoreg_dict"].keys():
        if key not in ["series_na_fill_func"]:
            assert_equal(
                result.model[-1].model_dict["autoreg_dict"][key],
                deserialized.model[-1].model_dict["autoreg_dict"][key])
    assert_equal(
        inspect.getsource(result.model[-1].model_dict["autoreg_dict"]["series_na_fill_func"]),
        inspect.getsource(deserialized.model[-1].model_dict["autoreg_dict"]["series_na_fill_func"]))

    # Forecast.
    assert_equal(
        result.forecast.estimator.predict(df),
        deserialized.forecast.estimator.predict(df)
    )
    assert result.forecast.estimator.model_dict["x_design_info"] is not None
    # Forecast: attributes.
    for key in result.forecast.__dict__.keys():
        if key not in ["r2_loss_function", "estimator"]:
            assert_equal(
                result.forecast.__dict__[key],
                deserialized.forecast.__dict__[key])
    assert_equal(
        inspect.getsource(result.forecast.__dict__["r2_loss_function"]),
        inspect.getsource(deserialized.forecast.__dict__["r2_loss_function"]))
    # Forecast: estimator.
    for key in result.forecast.estimator.__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model_dict"]:
            assert_equal(
                result.forecast.estimator.__dict__[key],
                deserialized.forecast.estimator.__dict__[key])
    assert_equal(
        inspect.getsource(result.forecast.estimator.__dict__["score_func"]),
        inspect.getsource(deserialized.forecast.estimator.__dict__["score_func"])
    )
    # Model: estimator/model_dict
    for key in result.forecast.estimator.model_dict.keys():
        # Functions and classes are not testable.
        if key not in ["x_design_info", "fs_func", "ml_model", "plt_pred",
                       "autoreg_dict", "changepoint_detector", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                result.forecast.estimator.model_dict[key],
                deserialized.forecast.estimator.model_dict[key])
        # Tests function source code.
        elif key in ["fs_func", "plt_pred", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                inspect.getsource(result.forecast.estimator.model_dict[key]),
                inspect.getsource(deserialized.forecast.estimator.model_dict[key]))
    # Model: estimator/model_dict/autoreg_dict.
    for key in result.forecast.estimator.model_dict["autoreg_dict"].keys():
        if key not in ["series_na_fill_func"]:
            assert_equal(
                result.forecast.estimator.model_dict["autoreg_dict"][key],
                deserialized.forecast.estimator.model_dict["autoreg_dict"][key])
    assert_equal(
        inspect.getsource(result.forecast.estimator.model_dict["autoreg_dict"]["series_na_fill_func"]),
        inspect.getsource(deserialized.forecast.estimator.model_dict["autoreg_dict"]["series_na_fill_func"]))

    # Backtest.
    assert_equal(
        result.backtest.estimator.predict(df),
        deserialized.backtest.estimator.predict(df)
    )
    assert result.backtest.estimator.model_dict["x_design_info"] is not None
    # Backtest: attributes.
    for key in result.backtest.__dict__.keys():
        if key not in ["r2_loss_function", "estimator"]:
            assert_equal(
                result.backtest.__dict__[key],
                deserialized.backtest.__dict__[key])
    assert_equal(
        inspect.getsource(result.backtest.__dict__["r2_loss_function"]),
        inspect.getsource(deserialized.backtest.__dict__["r2_loss_function"]))
    # Backtest: estimator.
    for key in result.backtest.estimator.__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model_dict"]:
            assert_equal(
                result.backtest.estimator.__dict__[key],
                deserialized.backtest.estimator.__dict__[key])
    assert_equal(
        inspect.getsource(result.backtest.estimator.__dict__["score_func"]),
        inspect.getsource(deserialized.backtest.estimator.__dict__["score_func"])
    )
    # Model: estimator/model_dict.
    for key in result.backtest.estimator.model_dict.keys():
        # Functions and classes are not testable.
        if key not in ["x_design_info", "fs_func", "ml_model", "plt_pred",
                       "autoreg_dict", "changepoint_detector", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                result.backtest.estimator.model_dict[key],
                deserialized.backtest.estimator.model_dict[key])
        # Tests function source code.
        elif key in ["fs_func", "plt_pred", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                inspect.getsource(result.backtest.estimator.model_dict[key]),
                inspect.getsource(deserialized.backtest.estimator.model_dict[key]))
    # Model: estimator/model_dict/autoreg_dict.
    for key in result.backtest.estimator.model_dict["autoreg_dict"].keys():
        if key not in ["series_na_fill_func"]:
            assert_equal(
                result.backtest.estimator.model_dict["autoreg_dict"][key],
                deserialized.backtest.estimator.model_dict["autoreg_dict"][key])
    assert_equal(
        inspect.getsource(result.backtest.estimator.model_dict["autoreg_dict"]["series_na_fill_func"]),
        inspect.getsource(deserialized.backtest.estimator.model_dict["autoreg_dict"]["series_na_fill_func"]))

    # Timeseries.
    for key in result.timeseries.__dict__.keys():
        assert_equal(
            result.timeseries.__dict__[key],
            deserialized.timeseries.__dict__[key])

    # Checks the size of the serialized object in megabytes.
    memory_size_mb = sys.getsizeof(serialized)*1e-6
    assert memory_size_mb < 64.0


def test_silverkite_ad_result(daily_data, forecast_config_info_daily):
    """Tests pickling and unpickling of Greykite anomaly detector."""
    df_train = daily_data["df_train"]
    df_test = daily_data["df_test"]
    df = daily_data["df"]

    forecast_config = forecast_config_info_daily
    ad_config = ADConfig(
        volatility_features_list=[["dow"], ["is_weekend"]],
        coverage_grid=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
        target_anomaly_percent=None,
        variance_scaling=True)

    train_data = Data(df=df_train)

    def reward_func(data):
        return f1_calc(
            y_true=data.y_true,
            y_pred=data.y_pred)
    reward = Reward(reward_func)

    # Trains the anomaly detector.
    detector = GreykiteDetector(
        forecast_config=forecast_config,
        ad_config=ad_config,
        reward=reward)
    detector.fit(data=train_data)
    fit_data = detector.fit_info["best_calc_result"].data
    fit_df = fit_data.pred_df

    # Pickles and deserializes the anomaly detector.
    pickler = GreykitePickler()
    serialized = pickler.dumps(detector)
    deserialized = pickler.loads(serialized)

    # Checks that the original and deserialized anomaly detector train results are the same.
    deserialized_fit_data = deserialized.fit_info["best_calc_result"].data
    deserialized_fit_df = deserialized_fit_data.pred_df

    # Checks if we get the expected columns in the fit data.
    assert list(fit_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    assert list(deserialized_fit_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]

    fit_obj_value = detector.reward.apply(fit_data)
    deserialized_fit_obj_value = deserialized.reward.apply(deserialized_fit_data)
    assert fit_obj_value == deserialized_fit_obj_value

    fit_recall = calc_recall(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)
    deserialized_f1_recall = calc_recall(
        y_true=deserialized_fit_data.y_true,
        y_pred=deserialized_fit_data.y_pred)
    assert fit_recall == deserialized_f1_recall

    fit_precision = calc_precision(
        y_true=fit_data.y_true,
        y_pred=fit_data.y_pred)
    deserialized_f1_precision = calc_precision(
        y_true=deserialized_fit_data.y_true,
        y_pred=deserialized_fit_data.y_pred)
    assert fit_precision == deserialized_f1_precision

    # Predicts on the test data with the original and deserialized anomaly detector.
    test_data = Data(
        df=df_test,
        y_true=df_test[ANOMALY_COL])
    test_data = detector.predict(test_data)
    pred_df = test_data.pred_df

    deserialized_test_data = deserialized.predict(test_data)
    deserialized_pred_df = deserialized_test_data.pred_df

    # Checks that the original and deserialized anomaly detector test results are the same.
    assert list(pred_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]
    assert list(deserialized_pred_df.columns) == DETECTOR_PREDICT_COLS + [ANOMALY_COL]

    test_obj_value = detector.reward.apply(test_data)
    deserialized_test_obj_value = deserialized.reward.apply(deserialized_test_data)
    assert test_obj_value == deserialized_test_obj_value

    test_recall = calc_recall(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)
    deserialized_test_recall = calc_recall(
        y_true=deserialized_test_data.y_true,
        y_pred=deserialized_test_data.y_pred)
    assert test_recall == deserialized_test_recall

    test_precision = calc_precision(
        y_true=test_data.y_true,
        y_pred=test_data.y_pred)
    deserialized_test_precision = calc_precision(
        y_true=deserialized_test_data.y_true,
        y_pred=deserialized_test_data.y_pred)
    assert test_precision == deserialized_test_precision

    # Checks the size of the serialized object.
    memory_size_mb = sys.getsizeof(serialized)*1e-6
    assert memory_size_mb < 64.0

    # Plots the fit and test results of the original and deserialized anomaly detector.
    # This provides a visual check that the results are the same.
    fit_pred_df = pd.concat([fit_df, pred_df], axis=0)
    fit_pred_df[ANOMALY_COL] = df[ANOMALY_COL]
    "Test of train and predict of the Greykite Detector."
    fig = plot_anomalies_over_forecast_vs_actual(
        df=fit_pred_df,
        time_col=TIME_COL,
        actual_col=ACTUAL_COL,
        predicted_col=PREDICTED_COL,
        predicted_anomaly_col=PREDICTED_ANOMALY_COL,
        anomaly_col=ANOMALY_COL,
        marker_opacity=0.6,
        predicted_anomaly_marker_color="black",
        anomaly_marker_color="green",
        predicted_lower_col=PREDICTED_LOWER_COL,
        predicted_upper_col=PREDICTED_UPPER_COL,
        train_end_date=fit_df[TIME_COL].max(),
        title="Test of train and predict of the Greykite Detector.")
    assert fig is not None
    if FIG_SHOW:
        fig.show()

    deserialized_pred_df = pd.concat([deserialized_fit_df, deserialized_pred_df], axis=0)
    deserialized_pred_df[ANOMALY_COL] = df[ANOMALY_COL]
    fig = plot_anomalies_over_forecast_vs_actual(
        df=deserialized_pred_df,
        time_col=TIME_COL,
        actual_col=ACTUAL_COL,
        predicted_col=PREDICTED_COL,
        predicted_anomaly_col=PREDICTED_ANOMALY_COL,
        anomaly_col=ANOMALY_COL,
        marker_opacity=0.6,
        predicted_anomaly_marker_color="black",
        anomaly_marker_color="red",
        predicted_lower_col=PREDICTED_LOWER_COL,
        predicted_upper_col=PREDICTED_UPPER_COL,
        train_end_date=fit_df[TIME_COL].max(),
        title="Test of train and predict of the deserialized Greykite Detector.")
    assert fig is not None
    if FIG_SHOW:
        fig.show()
