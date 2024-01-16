# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Sayan Patra

"""This file contains forecast configs and corresponding json strings
to be used for testing.
"""

from greykite.common.evaluation import EvaluationMetricEnum
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.model_templates import ModelTemplateEnum


FORECAST_CONFIG_JSON_DEFAULT = dict(
    forecast_config=ForecastConfig(),
    forecast_json="{}"
)

FORECAST_CONFIG_JSON_COMPLETE = dict(
    forecast_config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=24,
        coverage=0.7,
        metadata_param=MetadataParam(
            time_col="time",
            value_col="value",
            freq="H",
            date_format="%Y-%m-%d-%H",
            train_end_date="2021-07-01-10",
        ),
        evaluation_period_param=EvaluationPeriodParam(
            test_horizon=1,
            periods_between_train_test=2,
            cv_horizon=3,
            cv_min_train_periods=4,
            cv_expanding_window=True,
            cv_use_most_recent_splits=True,
            cv_periods_between_splits=5,
            cv_periods_between_train_test=6,
            cv_max_splits=2
        ),
        evaluation_metric_param=EvaluationMetricParam(
            cv_selection_metric=EvaluationMetricEnum.MeanSquaredError.name,
            cv_report_metrics=[EvaluationMetricEnum.MeanAbsoluteError.name,
                               EvaluationMetricEnum.MeanAbsolutePercentError.name],
            null_model_params={
                "strategy": "quantile",
                "constant": None,
                "quantile": 0.8
            },
            relative_error_tolerance=0.02
        ),
        model_components_param=ModelComponentsParam(
            seasonality={
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "monthly_seasonality": "auto",
                "daily_seasonality": 10
            },
            growth={
                "growth_term": "quadratic"
            },
            events={
                "holidays_to_model_separately": [
                    "New Year's Day",
                    "Chinese New Year",
                    "Christmas Day",
                    "Independence Day",
                    "Thanksgiving",
                    "Labor Day",
                    "Good Friday",
                    "Easter Monday",
                    "Memorial Day",
                    "Veterans Day",
                    "Independence Day",
                ],
                "holiday_lookup_countries": ["UnitedStates"],
                "holiday_pre_num_days": 3,
                "holiday_post_num_days": 2
            },
            changepoints={
                "changepoints_dict": {
                    "method": "uniform",
                    "n_changepoints": 20,
                }
            },
            autoregression={
                "autoreg_dict": {
                    "lag_dict": {
                        "orders": [1, 2, 3]
                    },
                    "agg_lag_dict": {
                        "orders_list": [[7, 14, 21]]
                    }
                }
            },
            regressors={
                "regressor_cols": []
            },
            lagged_regressors={
                "lagged_regressor_dict": None
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
        ),
        computation_param=ComputationParam(
            hyperparameter_budget=10,
            n_jobs=None,
            verbose=1
        ),
    ),
    forecast_json="""{
        "model_template": "SILVERKITE",
        "forecast_horizon": 24,
        "coverage": 0.7,
        "metadata_param": {
            "time_col": "time",
            "value_col": "value",
            "freq": "H",
            "date_format": "%Y-%m-%d-%H",
            "train_end_date": "2021-07-01-10"
        },
        "evaluation_period_param": {
            "test_horizon": 1,
            "periods_between_train_test": 2,
            "cv_horizon": 3,
            "cv_min_train_periods": 4,
            "cv_expanding_window": true,
            "cv_use_most_recent_splits": true,
            "cv_periods_between_splits": 5,
            "cv_periods_between_train_test": 6,
            "cv_max_splits": 2
        },
        "evaluation_metric_param": {
            "cv_selection_metric": "MeanSquaredError",
            "cv_report_metrics": ["MeanAbsoluteError", "MeanAbsolutePercentError"],
            "null_model_params": {
                "strategy": "quantile",
                "constant": null,
                "quantile": 0.8
            },
            "relative_error_tolerance": 0.02
        },
        "model_components_param": {
            "seasonality":{
                "yearly_seasonality": true,
                "weekly_seasonality": false,
                "monthly_seasonality": "auto",
                "daily_seasonality": 10
            },
            "growth": {
                "growth_term": "quadratic"
            },
            "events": {
                "holidays_to_model_separately": [
                    "New Year's Day",
                    "Chinese New Year",
                    "Christmas Day",
                    "Independence Day",
                    "Thanksgiving",
                    "Labor Day",
                    "Good Friday",
                    "Easter Monday",
                    "Memorial Day",
                    "Veterans Day",
                    "Independence Day"
                ],
                "holiday_lookup_countries": ["UnitedStates"],
                "holiday_pre_num_days": 3,
                "holiday_post_num_days": 2
            },
            "changepoints": {
                "changepoints_dict": {
                    "method": "uniform",
                    "n_changepoints": 20
                }
            },
            "autoregression": {
                "autoreg_dict": {
                    "lag_dict": {
                        "orders": [1, 2, 3]
                    },
                    "agg_lag_dict": {
                        "orders_list": [[7, 14, 21]]
                    }
                }
            },
            "regressors": {
                "regressor_cols": []
            },
            "custom": {
                "custom_param": 1
            },
            "lagged_regressors": {
                "lagged_regressor_dict": null
            },
            "uncertainty": {
                "uncertainty_dict": "auto"
            },
            "hyperparameter_override": {
                "input__response__null__max_frac": 0.1
            },
            "custom": {
                "fit_algorithm_dict": {
                    "fit_algorithm": "ridge",
                    "fit_algorithm_params": {"normalize": true}
                },
                "feature_sets_enabled": false
            }
        },
        "computation_param": {
            "hyperparameter_budget": 10,
            "n_jobs": null,
            "verbose": 1
        }
    }"""
)
