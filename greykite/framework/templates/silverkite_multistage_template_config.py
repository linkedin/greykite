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
# original author: Kaixu Yang
"""Provides templates for SilverkiteMultistageEstimator that are pre-tuned to fit
specific use cases.

These templates are recognized by
`~greykite.framework.templates.silverkite_multistage_model_templates.SilverkiteMultistageModelTemplateEnum`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable
from typing import Optional
from typing import Type
from typing import Union

from greykite.common.python_utils import mutable_field
from greykite.framework.templates.auto_arima_template import AutoArimaTemplate
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.prophet_template import ProphetTemplate
from greykite.framework.templates.silverkite_template import SilverkiteTemplate
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate
from greykite.sklearn.estimator.silverkite_multistage_estimator import AggregationFunctionEnum


@dataclass
class SilverkiteMultistageTemplateConfig:
    """The dataclass to store SilverkiteMultistage model config for a single model.

    Attributes
    ----------
    train_length : `str`, default "392D"
        The length of data used for training. For example, "56D".
    fit_length : `str` or None, default None
        The length of data where fitted values to be calculated.
        Specify if ``fit_length`` is to be longer than ``train_length``.
    agg_func : str or Callable, default
    `~greykite.sklearn.estimator.silverkite_multistage_estimator.AggregationFunctionEnum.nanmean.name`
        The aggregation function.
    agg_freq : `str` or None, default None
        The aggregation period. If None, no aggregation will be used.
    model_template : `str`, default "SILVERKITE"
        The mode template to be used.
    model_commponents : `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` or None,
        default None
        The parameters used to override the defaults in ``model_template``.
    """
    train_length: str = "392D"  # 56 weeks
    fit_length: Optional[str] = None
    agg_func: Union[str, Callable] = AggregationFunctionEnum.nanmean.name
    agg_freq: Optional[str] = None
    model_template: str = "SILVERKITE"
    model_components: Optional[ModelComponentsParam] = None


# Defines the SILVERKITE_TWO_STAGE template here.
SILVERKITE_TWO_STAGE = ModelComponentsParam(
    custom=dict(
        silverkite_multistage_configs=[
            # Defines the long model.
            # A daily model with 56 weeks training length.
            # Learns the long-term trend, seasonality, events, etc.
            SilverkiteMultistageTemplateConfig(
                train_length="392D",  # 56 weeks
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
                        "changepoints_dict": {
                            "method": "auto",
                            "resample_freq": "D",
                            "regularization_strength": 0.5,
                            "potential_changepoint_distance": "15D",
                            "no_changepoint_distance_from_end": "30D",
                            "yearly_seasonality_order": 15,
                            "yearly_seasonality_change_freq": "365D"
                        },
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
                    }
                )
            ),
            # Defines the short model.
            # Uses the original frequency with 4 weeks training length.
            # Learns daily seasonality with autoregression.
            SilverkiteMultistageTemplateConfig(
                train_length="28D",  # 4 weeks
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
                    }
                )
            )
        ]
    )
)
"""Two stage model for small frequency data.
The first stage uses 56 weeks data with daily frequency and trains the yearly/monthly/weekly seasonality,
trend, holidays effects.
The second stage uses the last 28 days data to train weekly/daily and autoregression effects.
This template is intended to be used with small granularity data (sub-daily) with long history to capture
long-term effects. The template is usually a few times faster than the full Silverkite model,
but still maintains a high level of accuracy.
The template was originally experimented on 5-minute granularity data and worked well.
"""


SILVERKITE_MULTISTAGE_EMPTY = ModelComponentsParam(
    custom=dict(
        silverkite_multistage_configs=[]
    )
)
"""Empty configuration for Silverkite Multistage.
All parameters will be exactly what user inputs.
Not to be used without overriding.
"""


class SilverkiteMultistageModelTemplateEnum(Enum):
    """Templates that can be used with the Silverkite Multistage algorithm.

    The Silverkite Multistage algorithm is defined through
    `~greykite.framework.templates.silverkite_multistage_template.SilverkiteMultistageTemplate`.
    The algorithm includes multiple stages, where each stage can be one of the existing model templates
    such as `SimpleSilverkiteTemplate` via "SILVERKITE".

    This Enum enumerates the model templates that are allowed to use in the Silverkite
    Multistage algorithm, which include common single model templates defined in
    `~greykite.framework.templates.model_templates.ModelTemplateEnum`.
    """
    SILVERKITE = SimpleSilverkiteTemplate
    """Default model template for `SimpleSilverkiteTemplate`."""
    SILVERKITE_WITH_AR = SimpleSilverkiteTemplate
    """Default model template for `SimpleSilverkiteTemplate` with autoregression."""
    SILVERKITE_EMPTY = SimpleSilverkiteTemplate
    """Null model template for `SimpleSilverkiteTemplate`."""
    SK = SilverkiteTemplate
    """Default model template for `SilverkiteTemplate`."""
    PROPHET = ProphetTemplate
    """Default model template for `ProphetTemplate`."""
    AUTO_ARIMA = AutoArimaTemplate
    """Default model template for `AutoArimaTemplate`."""


@dataclass
class SilverkiteMultistageTemplateConstants:
    """Constants used by
    `~greykite.framework.templates.silverkite_multistage_template.SilverkiteMultistageTemplate`.
    Include the model templates and their default values.
    """
    SILVERKITE_TWO_STAGE: ModelComponentsParam = mutable_field(SILVERKITE_TWO_STAGE)
    """Defines the ``"SILVERKITE_TWO_STAGE"`` template.
    Includes a 2-stage model. The first stage uses daily aggregation to learn long term effects.
    The second stage uses the original frequency to learn short term effects from the residuals.
    """
    SILVERKITE_MULTISTAGE_EMPTY: ModelComponentsParam = mutable_field(SILVERKITE_MULTISTAGE_EMPTY)
    """Defines the ``"SILVERKITE_EMPTY"`` template.
    The model config is empty. Uses exactly what user chooses to override.
    Can not be used without overriding.
    """
    SilverkiteMultistageModelTemplateEnum: Type[Enum] = SilverkiteMultistageModelTemplateEnum
    """Defines the model templates that are supported by the Silverkite Multistage algorithm.
    These are common single model templates defined in
    `~greykite.framework.templates.model_templates.ModelTemplateEnum`,
    and can be recognized in each stage of models in Silverkite Multistage.
    """
