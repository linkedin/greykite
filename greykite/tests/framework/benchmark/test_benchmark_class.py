import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from plotly.colors import DEFAULT_PLOTLY_COLORS

from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import root_mean_squared_error
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.framework.benchmark.benchmark_class import BenchmarkForecastConfig
from greykite.framework.constants import FORECAST_STEP_COL
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.sklearn.cross_validation import RollingTimeSeriesSplit


try:
    import prophet  # noqa
except ModuleNotFoundError:
    pass


@pytest.fixture(scope="module")
def df():
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=20 * 7,
        train_frac=0.9,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols = ["regressor1", "regressor2", "regressor3"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols
    df = data["df"][keep_cols]
    return df


@pytest.fixture(scope="module")
def valid_configs():
    metadata = MetadataParam(
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="D"
    )
    computation = ComputationParam(
        hyperparameter_budget=10,
        n_jobs=None,
        verbose=1
    )
    forecast_horizon = 2 * 7
    coverage = 0.90
    evaluation_metric = EvaluationMetricParam(
        cv_selection_metric=EvaluationMetricEnum.MeanAbsoluteError.name,
        cv_report_metrics=None,
        agg_periods=7,
        agg_func=np.mean,
        null_model_params=None
    )
    evaluation_period = EvaluationPeriodParam(
        test_horizon=2 * 7,
        periods_between_train_test=2 * 7,
        cv_horizon=1 * 7,
        cv_min_train_periods=8 * 7,
        cv_expanding_window=True,
        cv_periods_between_splits=7,
        cv_periods_between_train_test=3 * 7,
        cv_max_splits=2
    )

    silverkite_components = ModelComponentsParam(
        seasonality={
            "yearly_seasonality": False,
            "weekly_seasonality": True
        },
        growth={
            "growth_term": "quadratic"
        },
        events={
            "holidays_to_model_separately": SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES,
            "holiday_lookup_countries": ["UnitedStates"],
            "holiday_pre_num_days": 3,
        },
        changepoints={
            "changepoints_dict": {
                "method": "uniform",
                "n_changepoints": 20,
            }
        },
        regressors={
            "regressor_cols": ["regressor1", "regressor2", "regressor3"]
        },
        uncertainty={
            "uncertainty_dict": "auto",
        },
        hyperparameter_override={
            "input__response__null__max_frac": 0.1
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": {"normalize": True},
            },
            "feature_sets_enabled": False
        }
    )

    prophet_components = ModelComponentsParam(
        seasonality={
            "seasonality_mode": ["additive"],
            "yearly_seasonality": ["auto"],
            "weekly_seasonality": [True],
            "daily_seasonality": ["auto"],
        },
        growth={
            "growth_term": ["linear"]
        },
        events={
            "holiday_pre_num_days": [1],
            "holiday_post_num_days": [1],
            "holidays_prior_scale": [1.0]
        },
        changepoints={
            "changepoint_prior_scale": [0.05],
            "n_changepoints": [1],
            "changepoint_range": [0.5],
        },
        regressors={
            "add_regressor_dict": [{
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
            }]
        },
        uncertainty={
            "uncertainty_samples": [10]
        }
    )

    valid_prophet = ForecastConfig(
        model_template=ModelTemplateEnum.PROPHET.name,
        metadata_param=metadata,
        computation_param=computation,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        forecast_horizon=forecast_horizon,
        model_components_param=prophet_components
    )

    valid_silverkite = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        metadata_param=metadata,
        computation_param=computation,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        forecast_horizon=forecast_horizon,
        model_components_param=silverkite_components
    )

    configs = {
        "valid_prophet": valid_prophet,
        "valid_silverkite": valid_silverkite
    }
    return configs


@pytest.fixture(scope="module")
def custom_tscv():
    # periods_between_splits is less than forecast_horizon
    # so there are
    return RollingTimeSeriesSplit(
        forecast_horizon=2 * 7,
        min_train_periods=10 * 7,
        expanding_window=True,
        use_most_recent_splits=True,
        periods_between_splits=1 * 7,
        periods_between_train_test=2 * 7,
        max_splits=3)


@pytest.fixture(scope="module")
def metric_dict():
    # MSE and custom_MSE should always produce same output
    return {
        "corr": EvaluationMetricEnum.Correlation,
        "q95_loss": EvaluationMetricEnum.Quantile95,
        "MSE": EvaluationMetricEnum.MeanSquaredError,
        "custom_MSE": lambda y_true, y_pred: np.mean((y_true - y_pred)**2),
        "RMSE": lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)
    }


@pytest.fixture(scope="module")
def valid_bm(df, valid_configs, custom_tscv):
    bm = BenchmarkForecastConfig(df=df, configs=valid_configs, tscv=custom_tscv)
    bm.run()
    return bm


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_benchmark_class_init(df, valid_configs, custom_tscv):
    forecaster = Forecaster()
    bm = BenchmarkForecastConfig(df=df, configs=valid_configs, tscv=custom_tscv, forecaster=forecaster)

    assert_equal(bm.df, df)
    assert_equal(bm.configs, valid_configs)
    assert_equal(bm.forecaster, forecaster)
    assert not bm.is_run
    assert_equal(bm.result, dict.fromkeys(bm.configs.keys()))

    # error due to missing configs and df parameters
    with pytest.raises(TypeError, match=fr"__init__\(\) missing 2 required positional arguments: "
                                        fr"'df' and 'configs'"):
        BenchmarkForecastConfig(tscv=custom_tscv)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_validate(df, valid_configs, custom_tscv):
    bm = BenchmarkForecastConfig(df=df, configs=valid_configs, tscv=custom_tscv)
    bm.validate()

    for config_name in bm.result.keys():
        assert bm.result[config_name]["pipeline_params"] is not None

    def copy(obj):
        return pickle.loads(pickle.dumps(obj))

    # Error due to incompatible model components in config
    with pytest.raises(ValueError, match=fr"Unexpected key\(s\) found:"):
        configs = copy(valid_configs)
        configs["valid_prophet"].model_components_param.regressors = {
            "regressor_cols": ["regressor1", "regressor2", "regressor_categ"]
        }
        bm = BenchmarkForecastConfig(df=df, configs=configs, tscv=custom_tscv)
        bm.validate()

    # Error due to different forecast horizons in configs
    with pytest.raises(ValueError, match=r"valid_silverkite's 'forecast_horizon' \(7\) does not "
                                         r"match that of 'tscv' \(14\)"):
        configs = copy(valid_configs)
        configs["valid_silverkite"].forecast_horizon = 7
        bm = BenchmarkForecastConfig(df=df, configs=configs, tscv=custom_tscv)
        bm.validate()

    # Warning due to different periods_between_train_test
    with pytest.raises(ValueError, match=fr"valid_prophet's 'periods_between_train_test' \(7\) does not match "
                                         fr"that of 'tscv' \({custom_tscv.periods_between_train_test}\)."):
        configs = copy(valid_configs)
        configs["valid_prophet"].evaluation_period_param.periods_between_train_test = 7
        bm = BenchmarkForecastConfig(df=df, configs=configs, tscv=custom_tscv)
        bm.validate()

    # Error due to different coverage
    with pytest.raises(ValueError, match="All forecast configs must have same coverage."):
        configs = copy(valid_configs)
        configs["valid_prophet"].coverage = 0.1
        bm = BenchmarkForecastConfig(df=df, configs=configs, tscv=custom_tscv)
        bm.validate()


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_run(valid_bm, valid_configs):
    bm = valid_bm

    assert list(bm.result.keys()) == list(valid_configs.keys())
    for config_name, config in bm.result.items():
        assert set(config.keys()) == {"pipeline_params", "rolling_evaluation"}

        rolling_evaluation = config["rolling_evaluation"]
        assert set(rolling_evaluation.keys()) == {"split_0", "split_1", "split_2"}

        for split_key, split_value in rolling_evaluation.items():
            assert set(split_value.keys()) == {"runtime_sec", "pipeline_result"}

            runtime_sec = split_value["runtime_sec"]
            assert round(runtime_sec, 3) == runtime_sec

            pipeline_result = split_value["pipeline_result"]
            assert pipeline_result.backtest is None
            assert pipeline_result.forecast is not None


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_extract_forecasts(valid_bm, df, valid_configs, custom_tscv):
    bm = valid_bm
    bm.extract_forecasts()

    for config_name, config in bm.result.items():
        rolling_forecast_df = config["rolling_forecast_df"]
        # Addition of train_end_date, forecast_step & step_num results in 8 columns
        assert rolling_forecast_df.shape == (custom_tscv.forecast_horizon * custom_tscv.max_splits, 8)
        # Checks train_end_dates column
        train_end_date_values = rolling_forecast_df["train_end_date"].values
        expected_train_end_date_values = \
            np.repeat([np.datetime64("2018-10-06"), np.datetime64("2018-10-13"), np.datetime64("2018-10-20")],
                      custom_tscv.forecast_horizon)
        np.testing.assert_array_equal(train_end_date_values, expected_train_end_date_values)
        # Checks forecast_step column
        assert_equal(rolling_forecast_df[FORECAST_STEP_COL].values,
                     np.tile(np.arange(custom_tscv.forecast_horizon) + 1, custom_tscv.max_splits))

        expected_columns = {"train_end_date", FORECAST_STEP_COL, "split_num", TIME_COL, ACTUAL_COL,
                            PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL}
        assert expected_columns == set(rolling_forecast_df.columns)

    expected_columns = {
        "train_end_date", FORECAST_STEP_COL, "split_num", TIME_COL, ACTUAL_COL,
        f"valid_prophet_{PREDICTED_COL}", f"valid_prophet_{PREDICTED_LOWER_COL}",
        f"valid_prophet_{PREDICTED_UPPER_COL}", f"valid_silverkite_{PREDICTED_COL}",
        f"valid_silverkite_{PREDICTED_LOWER_COL}", f"valid_silverkite_{PREDICTED_UPPER_COL}"
    }
    assert set(bm.forecasts.columns) == expected_columns

    # error when `run` method has not been executed yet
    with pytest.raises(ValueError, match="Please execute 'run' method to create forecasts."):
        bm = BenchmarkForecastConfig(df=df, configs=valid_configs, tscv=custom_tscv)
        bm.extract_forecasts()


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_plot_forecasts_by_step(valid_bm):
    bm = valid_bm

    # default value
    fig = bm.plot_forecasts_by_step(forecast_step=1)
    assert fig.layout.showlegend
    assert fig.layout.xaxis.title.text == TIME_COL
    assert fig.layout.yaxis.title.text == VALUE_COL
    assert fig.layout.title.text == "1-step ahead rolling forecasts"
    assert fig.layout.title.x == 0.5

    # len(fig.data) = 1 + len(config_keys)
    assert len(fig.data) == 1 + len(bm.configs)

    assert fig.data[0].name == ACTUAL_COL
    assert fig.data[1].name == "valid_prophet_forecast"
    assert fig.data[2].name == "valid_silverkite_forecast"

    # custom value
    fig = bm.plot_forecasts_by_step(
        forecast_step=5,
        config_names=["valid_silverkite"],
        xlabel="xlab",
        ylabel="ylab",
        title="title",
        showlegend=False)
    assert not fig.layout.showlegend
    assert fig.layout.xaxis.title.text == "xlab"
    assert fig.layout.yaxis.title.text == "ylab"
    assert fig.layout.title.text == "title"
    assert fig.layout.title.x == 0.5

    # len(fig.data) = 1 + len(config_names)
    assert len(fig.data) == 1 + 1
    assert fig.data[0].name == ACTUAL_COL
    assert fig.data[1].name == "valid_silverkite_forecast"
    # # error due to wrong config key
    # # error due to wrong forecast step is checked in test_plot_multivariate_grouped
    missing_keys = {"missing_config"}
    with pytest.raises(ValueError, match=f"The following config keys are missing: {missing_keys}"):
        bm.plot_forecasts_by_step(forecast_step=1, config_names=["missing_config"])

    # Error due to forecast_step > forecast_horizon
    forecast_step = 20
    with pytest.raises(ValueError, match=fr"`forecast_step` \({forecast_step}\) must be less than or equal "
                                         fr"to forecast horizon \({bm.tscv.forecast_horizon}\)."):
        bm.plot_forecasts_by_step(forecast_step=forecast_step)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_plot_forecasts_by_config(valid_bm):
    bm = valid_bm

    # default value
    fig = bm.plot_forecasts_by_config(config_name="valid_prophet")
    assert fig.layout.showlegend
    assert fig.layout.xaxis.title.text == TIME_COL
    assert fig.layout.yaxis.title.text == VALUE_COL
    assert fig.layout.title.text == "Rolling forecast for valid_prophet"
    assert fig.layout.title.x == 0.5
    assert len(fig.data) == 1 + bm.tscv.max_splits  # len(fig.data) = 1 + number of splits

    assert fig.data[0].name == ACTUAL_COL
    assert fig.data[0].line.color == DEFAULT_PLOTLY_COLORS[0]
    for i in np.arange(bm.tscv.max_splits):
        assert fig.data[(i+1)].name == f"{i}_split"
        assert fig.data[(i+1)].line.color == DEFAULT_PLOTLY_COLORS[(i+1)]

    # custom values
    default_color = "rgb(0, 145, 202)"  # blue
    fig = bm.plot_forecasts_by_config(
        config_name="valid_silverkite",
        colors=[default_color],
        xlabel="xlab",
        ylabel="ylab",
        title="title",
        showlegend=False)
    assert not fig.layout.showlegend
    assert fig.layout.xaxis.title.text == "xlab"
    assert fig.layout.yaxis.title.text == "ylab"
    assert fig.layout.title.text == "title"
    assert fig.layout.title.x == 0.5

    assert fig.data[0].name == ACTUAL_COL
    assert fig.data[0].line.color == default_color
    for i in np.arange(bm.tscv.max_splits):
        assert fig.data[(i+1)].name == f"{i}_split"
        assert fig.data[(i+1)].line.color == default_color

    # Error due to wrong config name
    missing_config_name = {"missing_config"}
    with pytest.raises(ValueError, match=f"The following config keys are missing: {missing_config_name}"):
        bm.plot_forecasts_by_config(config_name="missing_config")


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_get_evaluation_metrics(valid_bm, metric_dict, df, valid_configs, custom_tscv):
    bm = valid_bm

    # default value, all configs
    with pytest.warns(UserWarning):
        evaluation_metrics_df = bm.get_evaluation_metrics(
            metric_dict=metric_dict)

        expected_columns = {"config_name", "split_num"}
        for metric_name in metric_dict.keys():
            expected_columns = expected_columns.union(
                {f"train_{metric_name}", f"test_{metric_name}"}
            )
        assert set(evaluation_metrics_df.columns) == expected_columns
        expected_row_num = len(bm.configs) * bm.tscv.max_splits
        assert evaluation_metrics_df.shape[0] == expected_row_num
        # check metric values
        assert_equal(
            evaluation_metrics_df["train_MSE"].values,
            evaluation_metrics_df["train_custom_MSE"].values
        )
        assert_equal(
            evaluation_metrics_df["test_MSE"].values,
            evaluation_metrics_df["test_custom_MSE"].values
        )
        assert evaluation_metrics_df["train_corr"].dropna().between(-1, 1).all()
        assert evaluation_metrics_df["test_corr"].dropna().between(-1, 1).all()

    # custom config value
    with pytest.warns(UserWarning):
        config_names = ["valid_silverkite"]
        evaluation_metrics_df = bm.get_evaluation_metrics(
            metric_dict=metric_dict,
            config_names=config_names
        )
        # columns remain the same
        expected_columns = {"config_name", "split_num"}
        for metric_name in metric_dict.keys():
            expected_columns = expected_columns.union(
                {f"train_{metric_name}", f"test_{metric_name}"}
            )
        assert set(evaluation_metrics_df.columns) == expected_columns
        # number of rows change
        expected_row_num = len(config_names) * bm.tscv.max_splits
        assert evaluation_metrics_df.shape[0] == expected_row_num

    # error when `run` method has not been executed yet
    with pytest.raises(ValueError, match="Please execute the 'run' method "
                                         "before computing evaluation metrics."):
        bm = BenchmarkForecastConfig(df=df, configs=valid_configs, tscv=custom_tscv)
        bm.get_evaluation_metrics(metric_dict=metric_dict)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_plot_evaluation_metrics(valid_bm, metric_dict):
    bm = valid_bm

    # default value, all configs
    with pytest.warns(UserWarning):
        fig = bm.plot_evaluation_metrics(
            metric_dict=metric_dict)
        assert fig.layout.showlegend
        assert fig.layout.barmode == "group"
        assert fig.layout.xaxis.title.text is None
        assert fig.layout.yaxis.title.text == "Metric value"
        assert fig.layout.title.text == "Average evaluation metric across rolling windows"
        assert fig.layout.title.x == 0.5
        assert len(fig.data) == len(bm.configs)

        expected_xaxis = set()
        for metric_name in metric_dict.keys():
            expected_xaxis = expected_xaxis.union(
                {f"train_{metric_name}", f"test_{metric_name}"}
            )

        assert fig.data[0].name == "valid_prophet"
        assert_equal(set(fig.data[0].x), expected_xaxis)
        assert fig.data[1].name == "valid_silverkite"
        assert_equal(set(fig.data[1].x), expected_xaxis)

    # custom value
    with pytest.warns(UserWarning):
        config_names = ["valid_prophet"]
        fig = bm.plot_evaluation_metrics(
            metric_dict=metric_dict,
            config_names=config_names,
            xlabel="xlab",
            ylabel="ylab",
            title="title",
            showlegend=False)
        assert not fig.layout.showlegend
        assert fig.layout.xaxis.title.text == "xlab"
        assert fig.layout.yaxis.title.text == "ylab"
        assert fig.layout.title.text == "title"
        assert fig.layout.title.x == 0.5
        assert len(fig.data) == len(config_names)

        assert fig.data[0].name == "valid_prophet"
        assert_equal(set(fig.data[0].x), expected_xaxis)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_get_grouping_evaluation(valid_bm, metric_dict, df, valid_configs, custom_tscv):
    bm = valid_bm

    # default value, all configs, groupby time feature
    grouped_evaluation_df = bm.get_grouping_evaluation_metrics(
        metric_dict=metric_dict,
        groupby_time_feature="dow"
    )
    expected_columns = set(
        ["config_name", "split_num", "dow"] +
        [f"train {metric_name}" for metric_name in metric_dict.keys()]
    )
    assert set(grouped_evaluation_df.columns) == expected_columns
    assert grouped_evaluation_df.shape[0] == 7 * bm.tscv.max_splits * len(bm.configs)

    # check metric values
    assert_equal(
        grouped_evaluation_df["dow"].values,
        np.tile(np.arange(1, 8), bm.tscv.max_splits * len(bm.configs))
    )
    assert_equal(
        grouped_evaluation_df["train MSE"].values,
        grouped_evaluation_df["train custom_MSE"].values
    )
    assert grouped_evaluation_df["train corr"].dropna().between(-1, 1).all()

    # groupby sliding window size
    config_names = ["valid_prophet"]
    grouped_evaluation_df = bm.get_grouping_evaluation_metrics(
        metric_dict=metric_dict,
        config_names=config_names,
        which="test",
        groupby_sliding_window_size=7)
    expected_columns = set(
        ["config_name", "split_num", "ts_downsample"] +
        [f"test {metric_name}" for metric_name in metric_dict.keys()]
    )
    assert set(grouped_evaluation_df.columns) == expected_columns
    # forecasting 2 weeks into future, so 2 rows for each split, config
    assert grouped_evaluation_df.shape[0] == 2 * bm.tscv.max_splits * len(config_names)

    # check metric values
    assert_equal(
        grouped_evaluation_df["test MSE"].values,
        grouped_evaluation_df["test custom_MSE"].values
    )
    assert grouped_evaluation_df["test corr"].dropna().between(-1, 1).all()

    # groupby custom column
    custom_column = pd.Series(np.repeat(["g1", "g2"], 7), name="week")
    grouped_evaluation_df = bm.get_grouping_evaluation_metrics(
        metric_dict=metric_dict,
        which="test",
        groupby_custom_column=custom_column)
    expected_columns = set(
        ["config_name", "split_num", custom_column.name] +
        [f"test {metric_name}" for metric_name in metric_dict.keys()]
    )
    assert set(grouped_evaluation_df.columns) == expected_columns
    # forecasting 2 weeks into future, so 2 rows for each split, config
    assert grouped_evaluation_df.shape[0] == 2 * bm.tscv.max_splits * len(bm.configs)

    # check metric values
    assert_equal(
        grouped_evaluation_df["test MSE"].values,
        grouped_evaluation_df["test custom_MSE"].values
    )
    assert grouped_evaluation_df["test corr"].dropna().between(-1, 1).all()

    # error when `run` method has not been executed yet
    with pytest.raises(ValueError, match="Please execute the 'run' method before "
                                         "computing grouped evaluation metrics."):
        bm = BenchmarkForecastConfig(df=df, configs=valid_configs, tscv=custom_tscv)
        bm.get_grouping_evaluation_metrics(metric_dict=metric_dict)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_plot_grouping_evaluation(valid_bm, metric_dict):
    bm = valid_bm

    # default value, all configs, groupby time feature
    fig = bm.plot_grouping_evaluation_metrics(
        metric_dict=metric_dict,
        groupby_time_feature="woy"
    )
    assert fig.layout.showlegend
    assert fig.layout.xaxis.title.text == "woy"
    assert fig.layout.yaxis.title.text == "Metric value"
    assert fig.layout.title.text == "train performance by woy across rolling windows"
    assert fig.layout.title.x == 0.5
    assert len(fig.data) == len(bm.configs) * len(metric_dict)

    assert fig.data[0].name == "train MSE_valid_prophet"
    assert fig.data[1].name == "train MSE_valid_silverkite"
    assert fig.data[2].name == "train RMSE_valid_prophet"
    assert fig.data[3].name == "train RMSE_valid_silverkite"

    # groupby sliding window size
    config_names = ["valid_prophet"]
    fig = bm.plot_grouping_evaluation_metrics(
        metric_dict=metric_dict,
        config_names=config_names,
        which="test",
        groupby_sliding_window_size=4,
        xlabel="xlab",
        ylabel="ylab",
        title="title",
        showlegend=False)
    assert not fig.layout.showlegend
    assert fig.layout.xaxis.title.text == "xlab"
    assert fig.layout.yaxis.title.text == "ylab"
    assert fig.layout.title.text == "title"
    assert fig.layout.title.x == 0.5
    assert len(fig.data) == len(config_names) * len(metric_dict)

    assert fig.data[0].name == "test MSE_valid_prophet"
    assert fig.data[1].name == "test RMSE_valid_prophet"
    assert fig.data[2].name == "test corr_valid_prophet"
    assert fig.data[3].name == "test custom_MSE_valid_prophet"
    assert fig.data[4].name == "test q95_loss_valid_prophet"

    # groupby custom column
    custom_column = pd.Series(np.repeat(["g1", "g2"], 7), name="group")
    fig = bm.plot_grouping_evaluation_metrics(
        metric_dict=metric_dict,
        which="test",
        groupby_custom_column=custom_column,
        title="Error by custom group")
    assert fig.layout.showlegend
    assert fig.layout.xaxis.title.text == custom_column.name
    assert fig.layout.yaxis.title.text == "Metric value"
    assert fig.layout.title.text == "Error by custom group"
    assert fig.layout.title.x == 0.5
    assert len(fig.data) == len(bm.configs) * len(metric_dict)

    assert fig.data[0].name == "test MSE_valid_prophet"
    assert fig.data[1].name == "test MSE_valid_silverkite"
    assert fig.data[2].name == "test RMSE_valid_prophet"
    assert fig.data[3].name == "test RMSE_valid_silverkite"


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_get_runtimes(valid_bm, valid_configs, custom_tscv):
    bm = valid_bm

    # default value, all configs
    runtimes_df = bm.get_runtimes()
    expected_columns = {"config_name", "split_num", "runtime_sec"}
    assert set(runtimes_df.columns) == expected_columns
    expected_row_num = len(bm.configs) * bm.tscv.max_splits
    assert runtimes_df.shape[0] == expected_row_num

    # custom config value
    config_names = ["valid_silverkite"]
    runtimes_df = bm.get_runtimes(config_names=config_names)
    # columns remain the same
    expected_columns = {"config_name", "split_num", "runtime_sec"}
    assert set(runtimes_df.columns) == expected_columns
    expected_row_num = len(config_names) * bm.tscv.max_splits
    assert runtimes_df.shape[0] == expected_row_num

    # error when `run` method has not been executed yet
    with pytest.raises(ValueError, match="Please execute the 'run' method "
                                         "to obtain runtimes."):
        bm = BenchmarkForecastConfig(df=df, configs=valid_configs, tscv=custom_tscv)
        bm.get_runtimes()


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_plot_runtimes(valid_bm):
    bm = valid_bm

    # default value, all configs
    fig = bm.plot_runtimes()
    assert fig.layout.showlegend
    assert fig.layout.xaxis.title.text is None
    assert fig.layout.yaxis.title.text == "Mean runtime in seconds"
    assert fig.layout.title.text == "Average runtime across rolling windows"
    assert fig.layout.title.x == 0.5

    expected_xaxis = set(bm.configs)
    assert fig.data[0].name == "Runtime"
    assert_equal(set(fig.data[0].x), expected_xaxis)

    # custom value
    config_names = ["valid_prophet"]
    fig = bm.plot_runtimes(
        config_names=config_names,
        xlabel="xlab",
        ylabel="ylab",
        title="title",
        showlegend=False)
    assert not fig.layout.showlegend
    assert fig.layout.xaxis.title.text == "xlab"
    assert fig.layout.yaxis.title.text == "ylab"
    assert fig.layout.title.text == "title"
    assert fig.layout.title.x == 0.5

    expected_xaxis = set(config_names)
    assert fig.data[0].name == "Runtime"
    assert_equal(set(fig.data[0].x), expected_xaxis)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_get_valid_config_names(valid_bm):
    bm = valid_bm

    # default value
    valid_config_names = bm.get_valid_config_names()
    assert_equal(valid_config_names, list(bm.configs.keys()))

    # custom value
    config_names = ["valid_prophet"]
    valid_config_names = bm.get_valid_config_names(config_names=config_names)
    assert_equal(valid_config_names, config_names)

    # Error due to wrong config names
    missing_config = ["missing_config"]
    with pytest.raises(ValueError) as record:
        bm.get_valid_config_names(config_names=missing_config)
        assert f"Input 'config_name' ({missing_config}) is missing." in str(record.value)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_autocomplete_metric_dict(valid_bm, metric_dict):
    bm = valid_bm
    updated_metric_dict = bm.autocomplete_metric_dict(
        metric_dict=metric_dict,
        enum_class=EvaluationMetricEnum
    )
    assert_equal(list(metric_dict.keys()), list(updated_metric_dict.keys()))

    invalid_metric_dict = {
        "corr": EvaluationMetricEnum.Correlation,
        "invalid_metric": 5
    }
    enum_class = EvaluationMetricEnum
    with pytest.raises(ValueError, match="Value of 'invalid_metric' should be a callable or "
                                         f"a member of {enum_class}."):
        bm.autocomplete_metric_dict(
            metric_dict=invalid_metric_dict,
            enum_class=EvaluationMetricEnum
        )
