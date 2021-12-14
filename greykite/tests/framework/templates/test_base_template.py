import dataclasses
import datetime

import pytest
import sklearn

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.enums import TimeEnum
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.python_utils import assert_equal
from greykite.common.python_utils import unique_elements_in_list
from greykite.common.testing_utils import assert_eval_function_equal
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.base_template import BaseTemplate
from greykite.sklearn.estimator.silverkite_estimator import SilverkiteEstimator


NEW_TIME_COL = "new_time_col"
NEW_VALUE_COL = "new_value_col"


@pytest.fixture
def df():
    data = generate_df_with_reg_for_tests(
        freq="H",
        periods=300*24,
        train_start_date=datetime.datetime(2018, 7, 1),
        remove_extra_cols=True,
        mask_test_actuals=True)
    df = data["df"]
    time_col = NEW_TIME_COL
    value_col = NEW_VALUE_COL
    df.rename({
        TIME_COL: time_col,
        VALUE_COL: value_col
    }, axis=1, inplace=True)
    regressor_cols = ["regressor1", "regressor2", "regressor_categ"]
    lagged_regressor_cols = ["regressor2", "regressor_bool"]
    keep_cols = [time_col, value_col] + unique_elements_in_list(regressor_cols + lagged_regressor_cols)
    return df[keep_cols]


class MyTemplate(BaseTemplate):

    def __init__(self):
        super().__init__(estimator=SilverkiteEstimator())

    @property
    def allow_model_template_list(self):
        return False

    @property
    def allow_model_components_param_list(self):
        return False

    def get_regressor_cols(self):
        return ["regressor1", "regressor2", "regressor_categ"]

    def get_lagged_regressor_info(self):
        return {
            "lagged_regressor_cols": ["regressor2", "regressor_bool"],
            "overall_min_lag_order": 1,
            "overall_max_lag_order": 7
        }

    def get_hyperparameter_grid(self):
        return {}


def test_base_template():
    """Tests BaseTemplate"""
    # Tests __init__
    mt = MyTemplate()
    assert mt.df is None
    assert mt.config is None
    assert mt.pipeline_params is None
    assert mt.score_func is None
    assert mt.score_func_greater_is_better is None
    assert isinstance(mt.estimator, SilverkiteEstimator)
    assert mt.regressor_cols is None
    assert mt.lagged_regressor_cols is None
    assert mt.pipeline is None
    assert mt.time_properties is None
    assert mt.hyperparameter_grid is None


def test_get_regressor_cols():
    mt = MyTemplate()
    assert mt.get_regressor_cols() == ["regressor1", "regressor2", "regressor_categ"]


def test_get_lagged_regressor_info():
    mt = MyTemplate()
    lagged_regressor_info = mt.get_lagged_regressor_info()
    assert lagged_regressor_info["lagged_regressor_cols"] == ["regressor2", "regressor_bool"]
    assert lagged_regressor_info["overall_min_lag_order"] == 1
    assert lagged_regressor_info["overall_max_lag_order"] == 7


def test_get_pipeline(df):
    mt = MyTemplate()
    # Initializes attributes needed by the function
    mt.regressor_cols = mt.get_regressor_cols()
    mt.lagged_regressor_cols = mt.get_lagged_regressor_info()["lagged_regressor_cols"]
    metric = EvaluationMetricEnum.MeanSquaredError
    mt.score_func = metric.name
    mt.score_func_greater_is_better = metric.get_metric_greater_is_better()
    mt.config = ForecastConfig(
        coverage=0.9,
        evaluation_metric_param=EvaluationMetricParam(
            cv_selection_metric=metric.name
        )
    )
    # Checks get_pipeline output
    pipeline = mt.get_pipeline()
    assert isinstance(pipeline, sklearn.pipeline.Pipeline)
    estimator = pipeline.steps[-1][-1]
    assert isinstance(estimator, SilverkiteEstimator)
    assert estimator.coverage == mt.config.coverage
    assert mt.estimator is not estimator
    assert mt.estimator.coverage is None
    expected_col_names = ["regressor1", "regressor2", "regressor_categ", "regressor_bool"]
    assert pipeline.named_steps["input"].transformer_list[2][1].named_steps["select_reg"].column_names == expected_col_names
    assert_eval_function_equal(pipeline.steps[-1][-1].score_func,
                               metric.get_metric_func())


def test_get_forecast_time_properties(df):
    mt = MyTemplate()
    mt.df = df

    # with `train_end_date` (masking applied)
    mt.config = ForecastConfig(
        coverage=0.9,
        forecast_horizon=20,
        metadata_param=MetadataParam(
            time_col=NEW_TIME_COL,
            value_col=NEW_VALUE_COL,
            freq="H",
            date_format="%Y-%m-%d-%H",
            train_end_date=datetime.datetime(2019, 2, 1),
        )
    )
    mt.regressor_cols = mt.get_regressor_cols()
    mt.lagged_regressor_cols = mt.get_lagged_regressor_info()["lagged_regressor_cols"]
    time_properties = mt.get_forecast_time_properties()

    period = 3600  # seconds between observations
    time_delta = (mt.config.metadata_param.train_end_date - df[mt.config.metadata_param.time_col].min())  # train end - train start
    num_training_days = (time_delta.days + (time_delta.seconds + period) / TimeEnum.ONE_DAY_IN_SECONDS.value)
    assert time_properties["num_training_days"] == num_training_days

    # without `train_end_date`
    mt.config.metadata_param.train_end_date = None
    time_properties = mt.get_forecast_time_properties()
    time_delta = (datetime.datetime(2019, 2, 26) - df[mt.config.metadata_param.time_col].min())  # by default, train end is the last date with nonnull value_col
    num_training_days = (time_delta.days + (time_delta.seconds + period) / TimeEnum.ONE_DAY_IN_SECONDS.value)
    assert time_properties["num_training_days"] == num_training_days


def test_get_hyperparameter_grid():
    mt = MyTemplate()
    assert mt.get_hyperparameter_grid() == {}


def test_apply_template_for_pipeline_params(df):
    mt = MyTemplate()
    config = ForecastConfig(
        metadata_param=MetadataParam(
            time_col=NEW_TIME_COL,
            value_col=NEW_VALUE_COL,
        ),
        evaluation_metric_param=EvaluationMetricParam(
            cv_selection_metric="MeanSquaredError"
        ),
        evaluation_period_param=EvaluationPeriodParam(
            cv_use_most_recent_splits=True
        )
    )
    original_config = dataclasses.replace(config)

    # Tests apply_template_for_pipeline_params
    pipeline_params = mt.apply_template_for_pipeline_params(
        df=df,
        config=config
    )

    assert_equal(pipeline_params["df"], df)
    assert pipeline_params["train_end_date"] is None
    estimator = pipeline_params["pipeline"].steps[-1][-1]
    assert isinstance(estimator, SilverkiteEstimator)
    assert estimator.coverage == mt.config.coverage
    assert mt.estimator is not estimator
    assert mt.estimator.coverage is None
    expected_col_names = unique_elements_in_list(mt.get_regressor_cols() + mt.get_lagged_regressor_info()["lagged_regressor_cols"])
    assert pipeline_params["pipeline"].named_steps["input"].transformer_list[2][1].named_steps["select_reg"].column_names\
        == expected_col_names
    assert pipeline_params["cv_use_most_recent_splits"] == config.evaluation_period_param.cv_use_most_recent_splits

    # Tests `apply_template_decorator`
    assert mt.config == mt.apply_forecast_config_defaults(config)
    assert mt.config != config  # `mt.config` has default values added
    assert config == original_config  # `config` is not modified by the function
