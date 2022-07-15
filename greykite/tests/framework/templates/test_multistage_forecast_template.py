import datetime

import numpy as np
import pytest
from testfixtures import LogCapture

from greykite.common.constants import LOGGER_NAME
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.testing_utils import generate_df_for_tests
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.multistage_forecast_template import MultistageForecastTemplate
from greykite.framework.templates.multistage_forecast_template_config import SILVERKITE_TWO_STAGE
from greykite.framework.templates.multistage_forecast_template_config import MultistageForecastTemplateConfig
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator
from greykite.sklearn.uncertainty.uncertainty_methods import UncertaintyMethodEnum


@pytest.fixture
def df():
    df = generate_df_for_tests(
        freq="H",
        periods=24 * 7 * 8,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)["df"]
    df["regressor"] = np.arange(len(df))
    return df


@pytest.fixture
def df_daily():
    df = generate_df_for_tests(
        freq="D",
        periods=1000,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)["df"]
    return df


@pytest.fixture
def multistage_forecast_configs():
    configs = [
        MultistageForecastTemplateConfig(
            train_length="30D",
            fit_length=None,
            agg_func="nanmean",
            agg_freq="D",
            model_template="SILVERKITE",
            model_components=ModelComponentsParam(
                seasonality={
                    "yearly_seasonality": 12,
                    "quarterly_seasonality": 5,
                    "monthly_seasonality": 5,
                    "weekly_seasonality": 4,
                    "daily_seasonality": 0,
                },
                growth={
                    "growth_term": "linear"
                },
                events={
                    "holidays_to_model_separately": "auto",
                    "holiday_lookup_countries": "auto",
                    "holiday_pre_num_days": 1,
                    "holiday_post_num_days": 1,
                    "holiday_pre_post_num_dict": None,
                    "daily_event_df_dict": None,
                },
                changepoints={
                    "changepoints_dict": None,
                    "seasonality_changepoints_dict": None
                },
                autoregression={
                    "autoreg_dict": "auto"
                },
                regressors={
                    "regressor_cols": []
                },
                lagged_regressors={
                    "lagged_regressor_dict": None
                },
                uncertainty={
                    "uncertainty_dict": None
                },
                custom={
                    "fit_algorithm_dict": {
                        "fit_algorithm": "ridge",
                        "fit_algorithm_params": None,
                    },
                    "feature_sets_enabled": "auto",  # "auto" based on data freq and size
                    "max_daily_seas_interaction_order": 0,
                    "max_weekly_seas_interaction_order": 2,
                    "extra_pred_cols": [],
                    "min_admissible_value": None,
                    "max_admissible_value": None,
                    "drop_pred_cols": None,
                    "explicit_pred_cols": None,
                    "regression_weight_col": None,
                    "normalize_method": "zero_to_one"
                }
            )
        ),
        MultistageForecastTemplateConfig(
            train_length="7D",
            fit_length=None,
            agg_func="nanmean",
            agg_freq=None,
            model_template="SILVERKITE",
            model_components=ModelComponentsParam(
                seasonality={
                    "yearly_seasonality": 0,
                    "quarterly_seasonality": 0,
                    "monthly_seasonality": 0,
                    "weekly_seasonality": 0,
                    "daily_seasonality": 12,
                },
                growth={
                    "growth_term": None
                },
                events={
                    "holidays_to_model_separately": [],
                    "holiday_lookup_countries": [],
                    "holiday_pre_num_days": 0,
                    "holiday_post_num_days": 0,
                    "holiday_pre_post_num_dict": None,
                    "daily_event_df_dict": None,
                },
                changepoints={
                    "changepoints_dict": None,
                    "seasonality_changepoints_dict": None
                },
                autoregression={
                    "autoreg_dict": "auto"
                },
                regressors={
                    "regressor_cols": []
                },
                lagged_regressors={
                    "lagged_regressor_dict": None
                },
                uncertainty={
                    "uncertainty_dict": None
                },
                custom={
                    "fit_algorithm_dict": {
                        "fit_algorithm": "ridge",
                        "fit_algorithm_params": None,
                    },
                    "feature_sets_enabled": "auto",  # "auto" based on data freq and size
                    "max_daily_seas_interaction_order": 5,
                    "max_weekly_seas_interaction_order": 2,
                    "extra_pred_cols": [],
                    "min_admissible_value": None,
                    "max_admissible_value": None,
                    "drop_pred_cols": None,
                    "explicit_pred_cols": None,
                    "regression_weight_col": None,
                    "normalize_method": "zero_to_one"
                }
            )
        )
    ]
    return configs


@pytest.fixture
def forecast_config(multistage_forecast_configs):
    forecast_config = ForecastConfig(
        model_template="SILVERKITE_TWO_STAGE",
        forecast_horizon=12,
        metadata_param=MetadataParam(
            time_col=TIME_COL,
            value_col=VALUE_COL,
            freq="H"
        ),
        model_components_param=ModelComponentsParam(
            custom=dict(
                multistage_forecast_configs=multistage_forecast_configs
            )
        ),
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=1,
            cv_horizon=12,
            test_horizon=12
        )
    )
    return forecast_config


def test_get_regressor_cols(df, forecast_config):
    """Tests the `self.get_regressor_cols` method."""
    template = MultistageForecastTemplate()
    df["reg1"] = 1
    df["reg2"] = 2
    template.df = df
    template.config = forecast_config
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][0].model_components.regressors["regressor_cols"] = ["reg1"]
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_components.regressors["regressor_cols"] = ["reg2"]
    regressor_cols = template.get_regressor_cols()
    assert set(regressor_cols) == {"reg1", "reg2"}


def test_get_lagged_regressor_info(df, forecast_config):
    template = MultistageForecastTemplate()
    df["reg1"] = 1
    df["reg2"] = 2
    template.df = df
    template.config = forecast_config
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][0].model_components.lagged_regressors["lagged_regressor_dict"] = [{
            "reg1": {
                "lag_dict": {"orders": [12]},
                "series_na_fill_func": lambda s: s.bfill().ffill()}
            }]
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_components.lagged_regressors["lagged_regressor_dict"] = [{
            "reg2": {
                "lag_dict": {"orders": [12]},
                "series_na_fill_func": lambda s: s.bfill().ffill()}
            }]
    lagged_regressor_info = template.get_lagged_regressor_info()
    assert lagged_regressor_info == dict(
        lagged_regressor_cols=["reg1", "reg2"],
        overall_min_lag_order=12.0,
        overall_max_lag_order=12.0
    )


def test_get_hyperparameter_grid(df, forecast_config):
    template = MultistageForecastTemplate()

    # Error when `self.config` is not available.
    with pytest.raises(
            ValueError,
            match="Forecast config must be provided"):
        template.get_hyperparameter_grid()

    template.df = df
    # Adds a list of length 2 to each submodel.
    # The result hyperparameter grid should have 2 * 2 = 4 grids.
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][0].model_components.seasonality["weekly_seasonality"] = [1, 2]
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_components.seasonality["daily_seasonality"] = [10, 12]
    template.config = forecast_config
    hyperparameter_grid = template.get_hyperparameter_grid()
    assert hyperparameter_grid["estimator__forecast_horizon"] == [12]
    assert hyperparameter_grid["estimator__freq"] == ["H"]
    assert len(hyperparameter_grid["estimator__model_configs"]) == 4
    assert hyperparameter_grid["estimator__model_configs"][0][0].estimator_params["weekly_seasonality"] == 1
    assert hyperparameter_grid["estimator__model_configs"][0][1].estimator_params["daily_seasonality"] == 10
    assert hyperparameter_grid["estimator__model_configs"][1][0].estimator_params["weekly_seasonality"] == 1
    assert hyperparameter_grid["estimator__model_configs"][1][1].estimator_params["daily_seasonality"] == 12
    assert hyperparameter_grid["estimator__model_configs"][2][0].estimator_params["weekly_seasonality"] == 2
    assert hyperparameter_grid["estimator__model_configs"][2][1].estimator_params["daily_seasonality"] == 10
    assert hyperparameter_grid["estimator__model_configs"][3][0].estimator_params["weekly_seasonality"] == 2
    assert hyperparameter_grid["estimator__model_configs"][3][1].estimator_params["daily_seasonality"] == 12


def test_get_hyperparameter_grid_same_template(df, forecast_config):
    # Tests the behavior of using the same ``model_template`` to override.
    template = MultistageForecastTemplate()
    template.df = df
    # Sets weekly seasonality to 5.
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_components.seasonality["weekly_seasonality"] = 5
    # Removes the daily seasonality specification.
    del forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_components.seasonality["daily_seasonality"]
    template.config = forecast_config
    hyperparameter_grid = template.get_hyperparameter_grid()
    # The original template has daily seasonality 12 and no weekly seasonality.
    # The second model was overriden with the same ``model_template``, which is ``SILVERKITE``,
    # so the hyperparameter_grid should have both daily seasonality 12 and weekly seasonality 5.
    assert hyperparameter_grid["estimator__model_configs"][0][1].estimator_params["daily_seasonality"] == 12
    assert hyperparameter_grid["estimator__model_configs"][0][1].estimator_params["weekly_seasonality"] == 5


def test_get_hyperparameter_grid_different_template(df, forecast_config):
    # Tests the behavior of using the different ``model_template`` to override.
    template = MultistageForecastTemplate()
    template.df = df
    # Sets the model template to be ``SILVERKITE_EMPTY``.
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_template = "SILVERKITE_EMPTY"
    # Sets weekly seasonality to 5.
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_components.seasonality["weekly_seasonality"] = 5
    # Removes the daily seasonality specification.
    del forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_components.seasonality["daily_seasonality"]
    template.config = forecast_config
    hyperparameter_grid = template.get_hyperparameter_grid()
    # The original template has daily seasonality 12 and no weekly seasonality.
    # The second model was overriden with a different ``model_template``, which is ``SILVERKITE_EMPTY``,
    # so the hyperparameter_grid should have only weekly seasonality 5 and daily seasonality 0.
    assert hyperparameter_grid["estimator__model_configs"][0][1].estimator_params["daily_seasonality"] == 0
    assert hyperparameter_grid["estimator__model_configs"][0][1].estimator_params["weekly_seasonality"] == 5


def test_get_hyperparameter_grid_extra_configs(df, forecast_config):
    """Tests gets hyperparameter grid when the default and override have different lengths."""
    # The empty template has no configs.
    # The override components has two configs.
    forecast_config.model_template = "MULTISTAGE_EMPTY"
    template = MultistageForecastTemplate()
    template.df = df
    template.config = forecast_config
    # The grid should have exactly two configs which are the same as the override configs.
    hyperparameter_grid = template.get_hyperparameter_grid()
    assert hyperparameter_grid["estimator__model_configs"][0][0].estimator_params == {
        'auto_seasonality': False,
        'yearly_seasonality': 12,
        'quarterly_seasonality': 5,
        'monthly_seasonality': 5,
        'weekly_seasonality': 4,
        'daily_seasonality': 0,
        'auto_growth': False,
        'growth_term': 'linear',
        'changepoints_dict': None,
        'seasonality_changepoints_dict': None,
        'auto_holiday': False,
        'holidays_to_model_separately': 'auto',
        'holiday_lookup_countries': 'auto',
        'holiday_pre_num_days': 1,
        'holiday_post_num_days': 1,
        'holiday_pre_post_num_dict': None,
        'daily_event_df_dict': None,
        'feature_sets_enabled': 'auto',
        'fit_algorithm_dict': {
            'fit_algorithm': 'ridge',
            'fit_algorithm_params': None},
        'max_daily_seas_interaction_order': 0,
        'max_weekly_seas_interaction_order': 2,
        'extra_pred_cols': [],
        'drop_pred_cols': None,
        'explicit_pred_cols': None,
        'min_admissible_value': None,
        'max_admissible_value': None,
        'autoreg_dict': 'auto',
        'simulation_num': 10,
        'fast_simulation': False,
        'normalize_method': "zero_to_one",
        'regressor_cols': [],
        'lagged_regressor_dict': None,
        'regression_weight_col': None,
        'uncertainty_dict': None,
        'origin_for_time_vars': None,
        'train_test_thresh': None,
        'training_fraction': None}
    assert hyperparameter_grid["estimator__model_configs"][0][1].estimator_params == {
        'auto_seasonality': False,
        'yearly_seasonality': 0,
        'quarterly_seasonality': 0,
        'monthly_seasonality': 0,
        'weekly_seasonality': 0,
        'daily_seasonality': 12,
        'auto_growth': False,
        'growth_term': None,
        'changepoints_dict': None,
        'seasonality_changepoints_dict': None,
        'auto_holiday': False,
        'holidays_to_model_separately': [],
        'holiday_lookup_countries': [],
        'holiday_pre_num_days': 0,
        'holiday_post_num_days': 0,
        'holiday_pre_post_num_dict': None,
        'daily_event_df_dict': None,
        'feature_sets_enabled': 'auto',
        'fit_algorithm_dict': {
            'fit_algorithm': 'ridge',
            'fit_algorithm_params': None},
        'max_daily_seas_interaction_order': 5,
        'max_weekly_seas_interaction_order': 2,
        'extra_pred_cols': [],
        'drop_pred_cols': None,
        'explicit_pred_cols': None,
        'min_admissible_value': None,
        'max_admissible_value': None,
        'normalize_method': "zero_to_one",
        'autoreg_dict': 'auto',
        'simulation_num': 10,
        'fast_simulation': False,
        'regressor_cols': [],
        'lagged_regressor_dict': None,
        'regression_weight_col': None,
        'uncertainty_dict': None,
        'origin_for_time_vars': None,
        'train_test_thresh': None,
        'training_fraction': None}


def test_get_multistage_forecast_configs_override(df, forecast_config):
    template = MultistageForecastTemplate()
    template.df = df
    # Adds a list of length 2 to each submodel.
    # The result hyperparameter grid should have 2 * 2 = 4 grids.
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][0].model_components.seasonality["weekly_seasonality"] = [1, 2]
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_components.seasonality["daily_seasonality"] = [10, 12]
    template.config = forecast_config

    default_model_components = template._MultistageForecastTemplate__get_default_model_components(
        forecast_config.model_template)
    default_multistage_forecast_configs = default_model_components.custom.get("multistage_forecast_configs")

    new_configs = template._MultistageForecastTemplate__get_multistage_forecast_configs_override(
        custom=forecast_config.model_components_param.custom,
        model_template="SILVERKITE_TWO_STAGE",
        default_multistage_forecast_configs=default_multistage_forecast_configs
    )

    assert new_configs == [
        MultistageForecastTemplateConfig(
            train_length='30D',
            fit_length=None,
            agg_func='nanmean',
            agg_freq='D',
            model_template='SILVERKITE',
            model_components=ModelComponentsParam(
                autoregression={
                    'autoreg_dict': 'auto'
                },
                changepoints={
                    'changepoints_dict': None,
                    'seasonality_changepoints_dict': None
                },
                custom={
                    'fit_algorithm_dict': {
                        'fit_algorithm': 'ridge',
                        'fit_algorithm_params': None
                    },
                    'feature_sets_enabled': 'auto',
                    'max_daily_seas_interaction_order': 0,
                    'max_weekly_seas_interaction_order': 2,
                    'extra_pred_cols': [],
                    'min_admissible_value': None,
                    'max_admissible_value': None,
                    'drop_pred_cols': None,
                    'explicit_pred_cols': None,
                    'regression_weight_col': None,
                    'normalize_method': 'zero_to_one'
                },
                events={
                    'holidays_to_model_separately': 'auto',
                    'holiday_lookup_countries': 'auto',
                    'holiday_pre_num_days': 1,
                    'holiday_post_num_days': 1,
                    'holiday_pre_post_num_dict': None,
                    'daily_event_df_dict': None
                },
                growth={
                    'growth_term': 'linear'
                },
                hyperparameter_override={},
                regressors={
                    'regressor_cols': []
                },
                lagged_regressors={
                    'lagged_regressor_dict': None
                },
                seasonality={
                    'yearly_seasonality': 12,
                    'quarterly_seasonality': 5,
                    'monthly_seasonality': 5,
                    'weekly_seasonality': [1, 2],
                    'daily_seasonality': 0},
                uncertainty={
                    'uncertainty_dict': None
                })),
        MultistageForecastTemplateConfig(
            train_length='7D',
            fit_length=None,
            agg_func='nanmean',
            agg_freq=None,
            model_template='SILVERKITE',
            model_components=ModelComponentsParam(
                autoregression={
                    'autoreg_dict': 'auto'
                },
                changepoints={
                    'changepoints_dict': None,
                    'seasonality_changepoints_dict': None
                },
                custom={
                    'fit_algorithm_dict': {
                        'fit_algorithm': 'ridge',
                        'fit_algorithm_params': None
                    },
                    'feature_sets_enabled': 'auto',
                    'max_daily_seas_interaction_order': 5,
                    'max_weekly_seas_interaction_order': 2,
                    'extra_pred_cols': [],
                    'min_admissible_value': None,
                    'max_admissible_value': None,
                    'drop_pred_cols': None,
                    'explicit_pred_cols': None,
                    'regression_weight_col': None,
                    'normalize_method': 'zero_to_one'
                },
                events={
                    'holidays_to_model_separately': [],
                    'holiday_lookup_countries': [],
                    'holiday_pre_num_days': 0,
                    'holiday_post_num_days': 0,
                    'holiday_pre_post_num_dict': None,
                    'daily_event_df_dict': None
                },
                growth={
                    'growth_term': None
                },
                hyperparameter_override={},
                regressors={
                    'regressor_cols': []
                },
                lagged_regressors={
                    'lagged_regressor_dict': None
                },
                seasonality={
                    'yearly_seasonality': 0,
                    'quarterly_seasonality': 0,
                    'monthly_seasonality': 0,
                    'weekly_seasonality': 0,
                    'daily_seasonality': [10, 12]
                },
                uncertainty={
                    'uncertainty_dict': None
                }))]


def test_get_estimators_and_params_from_template_configs(df, forecast_config):
    template = MultistageForecastTemplate()
    template.df = df
    # Adds a list of length 2 to each submodel.
    # The result hyperparameter grid should have 2 * 2 = 4 grids.
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][0].model_components.seasonality["weekly_seasonality"] = [1, 2]
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][1].model_components.seasonality["daily_seasonality"] = [10, 12]
    template.config = forecast_config

    default_model_components = template._MultistageForecastTemplate__get_default_model_components(
        forecast_config.model_template)
    default_multistage_forecast_configs = default_model_components.custom.get("multistage_forecast_configs")

    new_configs = template._MultistageForecastTemplate__get_multistage_forecast_configs_override(
        custom=forecast_config.model_components_param.custom,
        model_template="SILVERKITE_TWO_STAGE",
        default_multistage_forecast_configs=default_multistage_forecast_configs
    )

    estimator_list, estimator_params_list = template._MultistageForecastTemplate__get_estimators_and_params_from_template_configs(
        new_configs=new_configs
    )

    # We can't test ``time_properties``
    for d in estimator_params_list:
        del d["estimator__time_properties"]

    assert estimator_list == [SimpleSilverkiteEstimator, SimpleSilverkiteEstimator]
    assert estimator_params_list == [
        {
            'estimator__auto_seasonality': [False],
            'estimator__yearly_seasonality': [12],
            'estimator__quarterly_seasonality': [5],
            'estimator__monthly_seasonality': [5],
            'estimator__weekly_seasonality': [1, 2],
            'estimator__daily_seasonality': [0],
            'estimator__auto_growth': [False],
            'estimator__growth_term': ['linear'],
            'estimator__changepoints_dict': [None],
            'estimator__seasonality_changepoints_dict': [None],
            'estimator__auto_holiday': [False],
            'estimator__holidays_to_model_separately': ['auto'],
            'estimator__holiday_lookup_countries': ['auto'],
            'estimator__holiday_pre_num_days': [1],
            'estimator__holiday_post_num_days': [1],
            'estimator__holiday_pre_post_num_dict': [None],
            'estimator__daily_event_df_dict': [None],
            'estimator__feature_sets_enabled': ['auto'],
            'estimator__fit_algorithm_dict': [{
                'fit_algorithm': 'ridge',
                'fit_algorithm_params': None}],
            'estimator__max_daily_seas_interaction_order': [0],
            'estimator__max_weekly_seas_interaction_order': [2],
            'estimator__extra_pred_cols': [[]],
            'estimator__drop_pred_cols': [None],
            'estimator__explicit_pred_cols': [None],
            'estimator__min_admissible_value': [None],
            'estimator__max_admissible_value': [None],
            'estimator__normalize_method': ["zero_to_one"],
            'estimator__autoreg_dict': ['auto'],
            'estimator__simulation_num': [10],
            'estimator__fast_simulation': [False],
            'estimator__regressor_cols': [[]],
            'estimator__lagged_regressor_dict': [None],
            'estimator__regression_weight_col': [None],
            'estimator__uncertainty_dict': [None],
            'estimator__origin_for_time_vars': [None],
            'estimator__train_test_thresh': [None],
            'estimator__training_fraction': [None]
        },
        {
            'estimator__auto_seasonality': [False],
            'estimator__yearly_seasonality': [0],
            'estimator__quarterly_seasonality': [0],
            'estimator__monthly_seasonality': [0],
            'estimator__weekly_seasonality': [0],
            'estimator__daily_seasonality': [10, 12],
            'estimator__auto_growth': [False],
            'estimator__growth_term': [None],
            'estimator__changepoints_dict': [None],
            'estimator__seasonality_changepoints_dict': [None],
            'estimator__auto_holiday': [False],
            'estimator__holidays_to_model_separately': [[]],
            'estimator__holiday_lookup_countries': [[]],
            'estimator__holiday_pre_num_days': [0],
            'estimator__holiday_post_num_days': [0],
            'estimator__holiday_pre_post_num_dict': [None],
            'estimator__daily_event_df_dict': [None],
            'estimator__feature_sets_enabled': ['auto'],
            'estimator__fit_algorithm_dict': [{
                'fit_algorithm': 'ridge',
                'fit_algorithm_params': None}],
            'estimator__max_daily_seas_interaction_order': [5],
            'estimator__max_weekly_seas_interaction_order': [2],
            'estimator__extra_pred_cols': [[]],
            'estimator__drop_pred_cols': [None],
            'estimator__explicit_pred_cols': [None],
            'estimator__min_admissible_value': [None],
            'estimator__max_admissible_value': [None],
            'estimator__normalize_method': ["zero_to_one"],
            'estimator__autoreg_dict': ['auto'],
            'estimator__simulation_num': [10],
            'estimator__fast_simulation': [False],
            'estimator__regressor_cols': [[]],
            'estimator__lagged_regressor_dict': [None],
            'estimator__regression_weight_col': [None],
            'estimator__uncertainty_dict': [None],
            'estimator__origin_for_time_vars': [None],
            'estimator__train_test_thresh': [None],
            'estimator__training_fraction': [None]
        }]


def test_flatten_estimator_params_list():
    template = MultistageForecastTemplate()
    x = [{
        "estimator__a": [1],
        "estimator__b": [2, 3]
    }, {
        "estimator__c": [4, 5]
    }]
    flattened_params = template._MultistageForecastTemplate__flatten_estimator_params_list(
        estimator_params_list=x
    )
    assert flattened_params == [
        [{'a': 1, 'b': 2}, {'c': 4}],
        [{'a': 1, 'b': 2}, {'c': 5}],
        [{'a': 1, 'b': 3}, {'c': 4}],
        [{'a': 1, 'b': 3}, {'c': 5}]
    ]


def test_multistage_forecast_model_template(df, forecast_config):
    forecaster = Forecaster()
    forecast_result = forecaster.run_forecast_config(
        df=df,
        config=forecast_config
    )
    assert forecast_result.backtest is not None
    assert forecast_result.grid_search is not None
    assert forecast_result.forecast is not None

    assert len(forecast_result.model[-1].models) == 2
    # Checks the forecast horizons in each model.
    assert forecast_result.model[-1].models[0].forecast_horizon == 1  # daily model
    assert forecast_result.model[-1].models[1].forecast_horizon == 12  # hourly model

    # Checks the autoregression orders are as expected.
    assert "y_lag1" in forecast_result.model[-1].models[0].model_dict["x_mat"].columns
    assert "y_lag12" in forecast_result.model[-1].models[1].model_dict["x_mat"].columns

    # Checks the forecast is not NAN
    assert len(forecast_result.forecast.df_test[PREDICTED_COL].dropna()) == len(forecast_result.forecast.df_test)
    assert len(forecast_result.backtest.df_test[PREDICTED_COL].dropna()) == len(forecast_result.backtest.df_test)


def test_multistage_forecast_model_template_with_regressor(df, forecast_config):
    forecaster = Forecaster()
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][0].model_components.regressors["regressor_cols"] = ["regressor"]
    df.iloc[-12:, 1] = np.nan
    forecast_result = forecaster.run_forecast_config(
        df=df,
        config=forecast_config
    )
    assert forecast_result.backtest is not None
    assert forecast_result.grid_search is not None
    assert forecast_result.forecast is not None

    assert len(forecast_result.model[-1].models) == 2
    # Checks the forecast horizons in each model.
    assert forecast_result.model[-1].models[0].forecast_horizon == 1  # daily model
    assert forecast_result.model[-1].models[1].forecast_horizon == 12  # hourly model

    # Checks the autoregression orders are as expected.
    assert "y_lag1" in forecast_result.model[-1].models[0].model_dict["x_mat"].columns
    assert "y_lag12" in forecast_result.model[-1].models[1].model_dict["x_mat"].columns

    # Checks that the regressor column is included.
    assert "regressor" in forecast_result.model[-1].models[0].model_dict["x_mat"].columns

    # Checks the forecast is not NAN
    assert len(forecast_result.forecast.df_test[PREDICTED_COL].dropna()) == len(forecast_result.forecast.df_test)
    assert len(forecast_result.backtest.df_test[PREDICTED_COL].dropna()) == len(forecast_result.backtest.df_test)


def test_multistage_forecast_model_template_with_lagged_regressor(df, forecast_config):
    forecaster = Forecaster()
    forecast_config.model_components_param.custom[
        "multistage_forecast_configs"][0].model_components.lagged_regressors["lagged_regressor_dict"] = [{
            "regressor": {
                "lag_dict": {"orders": [12]},
                "series_na_fill_func": lambda s: s.bfill().ffill()}
            }]
    forecast_result = forecaster.run_forecast_config(
        df=df,
        config=forecast_config
    )
    assert forecast_result.backtest is not None
    assert forecast_result.grid_search is not None
    assert forecast_result.forecast is not None

    assert len(forecast_result.model[-1].models) == 2
    # Checks the forecast horizons in each model.
    assert forecast_result.model[-1].models[0].forecast_horizon == 1  # daily model
    assert forecast_result.model[-1].models[1].forecast_horizon == 12  # hourly model

    # Checks the autoregression orders are as expected.
    assert "y_lag1" in forecast_result.model[-1].models[0].model_dict["x_mat"].columns
    assert "y_lag12" in forecast_result.model[-1].models[1].model_dict["x_mat"].columns

    # Checks that the regressor column is included.
    assert "regressor_lag12" in forecast_result.model[-1].models[0].model_dict["x_mat"].columns

    # Checks the forecast is not NAN
    assert len(forecast_result.forecast.df_test[PREDICTED_COL].dropna()) == len(forecast_result.forecast.df_test)
    assert len(forecast_result.backtest.df_test[PREDICTED_COL].dropna()) == len(forecast_result.backtest.df_test)


def test_errors(df, forecast_config):
    # No configs with MULTISTAGE_EMPTY.
    template = MultistageForecastTemplate()
    template.df = df
    forecast_config.model_components_param.custom["multistage_forecast_configs"] = None
    forecast_config.model_template = "MULTISTAGE_EMPTY"
    template.config = forecast_config
    with pytest.raises(
            ValueError,
            match="``MULTISTAGE_EMPTY`` can not be used without over"):
        template.get_hyperparameter_grid()

    # The config has wrong type.
    template = MultistageForecastTemplate()
    template.df = df
    forecast_config.model_components_param.custom["multistage_forecast_configs"] = 5
    forecast_config.model_template = "SILVERKITE_TWO_STAGE"
    template.config = forecast_config
    with pytest.raises(
            ValueError,
            match="The ``multistage_forecast_configs`` parameter must be a list of"):
        template.get_hyperparameter_grid()


def test_get_default_model_components():
    template = MultistageForecastTemplate()
    assert template._MultistageForecastTemplate__get_default_model_components(
        "SILVERKITE_TWO_STAGE") == SILVERKITE_TWO_STAGE
    with pytest.raises(
            ValueError,
            match="The template name "):
        template._MultistageForecastTemplate__get_default_model_components("some_template")


def test_get_template_class():
    template = MultistageForecastTemplate()
    assert template._MultistageForecastTemplate__get_template_class(
        ForecastConfig(model_template="SILVERKITE")
    ) == SimpleSilverkiteTemplate
    with pytest.raises(
            ValueError,
            match="Currently Multistage Forecast only supports"):
        template._MultistageForecastTemplate__get_template_class(
            ForecastConfig(model_template="DAILY_CP_NONE")
        )


def test_uncertainty(df, forecast_config):
    """Tests the uncertainty methods."""

    # Tests no coverage and no uncertainty, there is no uncertainty.

    forecaster = Forecaster()
    forecast_result = forecaster.run_forecast_config(
        df=df,
        config=forecast_config
    )
    assert PREDICTED_LOWER_COL not in forecast_result.backtest.df_test
    assert PREDICTED_LOWER_COL not in forecast_result.forecast.df_test

    # Tests coverage and no uncertainty, there is uncertainty.
    forecast_config.coverage = 0.99
    forecaster = Forecaster()
    forecast_result = forecaster.run_forecast_config(
        df=df,
        config=forecast_config
    )
    assert PREDICTED_LOWER_COL in forecast_result.backtest.df_test
    assert PREDICTED_LOWER_COL in forecast_result.forecast.df_test
    assert forecast_result.model[-1].coverage == 0.99
    # Default method is used when coverage is given but ``uncertainty_dict`` is not given.
    assert (forecast_result.model[-1].uncertainty_model.UNCERTAINTY_METHOD
            == UncertaintyMethodEnum.simple_conditional_residuals.name)
    last_interval_width_99 = (forecast_result.forecast.df[PREDICTED_UPPER_COL].iloc[-1]
                              - forecast_result.forecast.df[PREDICTED_LOWER_COL].iloc[-1])

    # Tests coverage and uncertainty, there is uncertainty.
    forecast_config.model_components_param.uncertainty = dict(
        uncertainty_dict=dict(
            uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
            params=dict(
                conditional_cols=["dow"]
            )
        )
    )
    forecaster = Forecaster()
    forecast_result = forecaster.run_forecast_config(
        df=df,
        config=forecast_config
    )
    assert PREDICTED_LOWER_COL in forecast_result.backtest.df_test
    assert PREDICTED_LOWER_COL in forecast_result.forecast.df_test
    assert forecast_result.model[-1].coverage == 0.99
    # The last 2 days intervals should have different lengths due to conditioning on "dow".
    last_day_interval_width_99 = (forecast_result.forecast.df[PREDICTED_UPPER_COL].iloc[-1]
                                  - forecast_result.forecast.df[PREDICTED_LOWER_COL].iloc[-1])
    second_last_day_interval_width_99 = (forecast_result.forecast.df[PREDICTED_UPPER_COL].iloc[-25]
                                         - forecast_result.forecast.df[PREDICTED_LOWER_COL].iloc[-25])
    assert last_day_interval_width_99 != second_last_day_interval_width_99

    # Tests 95% coverage has narrower interval.
    forecast_config.coverage = 0.95
    forecast_config.model_components_param.uncertainty = dict(
        uncertainty_dict=dict(
            uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
            params=dict()
        )
    )
    forecaster = Forecaster()
    forecast_result = forecaster.run_forecast_config(
        df=df,
        config=forecast_config
    )
    assert PREDICTED_LOWER_COL in forecast_result.backtest.df_test
    assert PREDICTED_LOWER_COL in forecast_result.forecast.df_test
    assert forecast_result.model[-1].coverage == 0.95
    # 95 interval is narrower than 99 interval.
    last_interval_width_95 = (forecast_result.forecast.df[PREDICTED_UPPER_COL].iloc[-1]
                              - forecast_result.forecast.df[PREDICTED_LOWER_COL].iloc[-1])
    assert last_interval_width_99 > last_interval_width_95


def test_uncertainty_fail(df, forecast_config):
    """Tests the pipeline won't fail when uncertainty fails."""
    with LogCapture(LOGGER_NAME) as log_capture:
        forecast_config.coverage = 0.95
        forecast_config.model_components_param.uncertainty = dict(
            uncertainty_dict=dict(
                uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
                params=dict(
                    conditional_cols=["dowww"]
                )
            )
        )
        forecaster = Forecaster()
        forecast_result = forecaster.run_forecast_config(
            df=df,
            config=forecast_config
        )
        # The forecast is still generated.
        assert forecast_result.forecast is not None
        assert (LOGGER_NAME,
                "WARNING",
                "The following errors occurred during fitting the uncertainty model, "
                "the uncertainty model is skipped. "
                "The following conditional columns are not found in `train_df`: ['dowww'].") in log_capture.actual()


def test_silverkite_wow(df_daily):
    """Tests the SILVERKITE_WOW model template."""
    forecaster = Forecaster()
    config = ForecastConfig(
        model_template="SILVERKITE_WOW",
        forecast_horizon=7,
        metadata_param=MetadataParam(
            freq="D"
        ),
        model_components_param=ModelComponentsParam(
            custom=dict(
                multistage_forecast_configs=[
                    MultistageForecastTemplateConfig(
                        train_length="1000D",
                        model_template="SILVERKITE_EMPTY",
                        model_components=ModelComponentsParam(
                            seasonality=dict(quarterly_seasonality=False)
                        )
                    ),
                    MultistageForecastTemplateConfig(
                        train_length="500D",
                        model_template="LAG_BASED",
                        model_components=ModelComponentsParam(
                            custom=dict(lags=[1, 2, 3])
                        )
                    )
                ]
            )
        ),
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=1,
            test_horizon=7
        )
    )
    forecast_result = forecaster.run_forecast_config(
        df=df_daily,
        config=config
    )
    assert all(["quarterly" not in col for col in forecast_result.model[-1].models[0].model_dict["x_mat"].columns])
    with LogCapture(LOGGER_NAME) as log_capture:
        forecast_result.model[-1].models[1].summary()
        log_capture.check_present((
            LOGGER_NAME,
            "INFO",
            f"This is a lag based forecast model that uses lags '[1, 2, 3]', "
            f"with unit 'week' and aggregation function 'mean'."
        ))
