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
"""Provides templates for SimpleSilverkiteEstimator that are pre-tuned to fit
specific use cases.

A subset of these templates are recognized by
`~greykite.framework.templates.ModelTemplateEnum`.

`~greykite.framework.templates.simple_silverkite_template`
also accepts any ``model_template`` name that follows
the naming convention in this file. For details, see
the ``model_template`` parameter in
`~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict
from typing import List
from typing import Type
from typing import Union

from greykite.common.python_utils import mutable_field
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam


# Defines the keywords used in default templates.
# The first item in each value is the default value.
# None of the keys can be any of the keywords, otherwise the algorithm will fail to parse the template name.
class SILVERKITE_FREQ(Enum):
    """Valid values for simple silverkite template string name frequency."""
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    DEFAULT = "DAILY"


VALID_FREQ = [name.value for name in SILVERKITE_FREQ if name.name != "DEFAULT"]
"""Valid non-default values for simple silverkite template string name frequency.
These are the non-default frequencies recognized by
`~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`.
"""


class SILVERKITE_SEAS(Enum):
    """Valid values for simple silverkite template string name seasonality.
    """
    LT = "LT"
    NM = "NM"
    HV = "HV"
    LTQM = "LTQM"
    NMQM = "NMQM"
    HVQM = "HVQM"
    NONE = "NONE"
    DEFAULT = "LT"


class SILVERKITE_GR(Enum):
    """Valid values for simple silverkite template string name growth_term.
    """
    LINEAR = "LINEAR"
    NONE = "NONE"
    DEFAULT = "LINEAR"


class SILVERKITE_CP(Enum):
    """Valid values for simple silverkite template string name changepoints_dict.
    """
    NONE = "NONE"
    LT = "LT"
    NM = "NM"
    HV = "HV"
    DEFAULT = "NONE"


class SILVERKITE_HOL(Enum):
    """Valid values for simple silverkite template string name events.
    """
    NONE = "NONE"
    SP1 = "SP1"
    SP2 = "SP2"
    SP4 = "SP4"
    TG = "TG"
    DEFAULT = "NONE"


class SILVERKITE_FEASET(Enum):
    """Valid values for simple silverkite template string name feature_sets_enabled.
    """
    OFF = "OFF"
    AUTO = "AUTO"
    ON = "ON"
    DEFAULT = "OFF"


class SILVERKITE_ALGO(Enum):
    """Valid values for simple silverkite template string name fit_algorithm.
    """
    LINEAR = "LINEAR"
    RIDGE = "RIDGE"
    SGD = "SGD"
    LASSO = "LASSO"
    DEFAULT = "LINEAR"


class SILVERKITE_AR(Enum):
    """Valid values for simple silverkite template string name autoregression.
    """
    OFF = "OFF"
    AUTO = "AUTO"
    DEFAULT = "OFF"


class SILVERKITE_DSI(Enum):
    """Valid values for simple silverkite template string name daily seasonality max interaction order.
    """
    AUTO = "AUTO"
    OFF = "OFF"
    DEFAULT = "AUTO"


class SILVERKITE_WSI(Enum):
    """Valid values for simple silverkite template string name weekly seasonality max interaction order.
    """
    AUTO = "AUTO"
    OFF = "OFF"
    DEFAULT = "AUTO"


# Defines the enum that has all keywords.
class SILVERKITE_COMPONENT_KEYWORDS(Enum):
    """Valid values for simple silverkite template string name keywords.
    The names are the keywords and the values are the corresponding value enum.
    Can be used to create an instance of
    `~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`.
    """
    FREQ = SILVERKITE_FREQ
    SEAS = SILVERKITE_SEAS
    GR = SILVERKITE_GR
    CP = SILVERKITE_CP
    HOL = SILVERKITE_HOL
    FEASET = SILVERKITE_FEASET
    ALGO = SILVERKITE_ALGO
    AR = SILVERKITE_AR
    DSI = SILVERKITE_DSI
    WSI = SILVERKITE_WSI


# Defines the data class of all keywords.
@dataclass
class SimpleSilverkiteTemplateOptions:
    """Defines generic simple silverkite template options.

    Attributes can be set to different values using
    `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_COMPONENT_KEYWORDS`
    for high level tuning.

    ``freq`` represents data frequency.

    The other attributes stand for seasonality,
    growth, changepoints_dict, events, feature_sets_enabled, fit_algorithm and autoregression in
    `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`, which are used in
    `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`.
    """
    freq: SILVERKITE_FREQ = SILVERKITE_FREQ.DEFAULT
    """Valid values for simple silverkite template string name frequency.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_FREQ`.
    """
    seas: SILVERKITE_SEAS = SILVERKITE_SEAS.DEFAULT
    """Valid values for simple silverkite template string name seasonality.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_SEAS`.
    """
    gr: SILVERKITE_GR = SILVERKITE_GR.DEFAULT
    """Valid values for simple silverkite template string name growth.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_GR`.
    """
    cp: SILVERKITE_CP = SILVERKITE_CP.DEFAULT
    """Valid values for simple silverkite template string name changepoints.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_CP`.
    """
    hol: SILVERKITE_HOL = SILVERKITE_HOL.DEFAULT
    """Valid values for simple silverkite template string name holiday.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_HOL`.
    """
    feaset: SILVERKITE_FEASET = SILVERKITE_FEASET.DEFAULT
    """Valid values for simple silverkite template string name feature sets enabled.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_FEASET`.
    """
    algo: SILVERKITE_ALGO = SILVERKITE_ALGO.DEFAULT
    """Valid values for simple silverkite template string name fit algorithm.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_ALGO`.
    """
    ar: SILVERKITE_AR = SILVERKITE_AR.DEFAULT
    """Valid values for simple silverkite template string name autoregression.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_AR`.
    """
    dsi: SILVERKITE_DSI = SILVERKITE_DSI.DEFAULT
    """Valid values for simple silverkite template string name max daily seasonality interaction order.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_DSI`.
    """
    wsi: SILVERKITE_WSI = SILVERKITE_WSI.DEFAULT
    """Valid values for simple silverkite template string name max weekly seasonality interaction order.
    See `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_WSI`.
    """


# Defines the common single parameters in `ModelComponentsParam`.
COMMON_MODELCOMPONENTPARAM_PARAMETERS = dict(
    # Seasonality components and orders depending on frequency HOURLY/DAILY/WEEKLY.
    SEAS=dict(
        HOURLY=dict(
            LT={
                "yearly_seasonality": 8,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 3,
                "daily_seasonality": 5
            },
            NM={
                "yearly_seasonality": 15,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 4,
                "daily_seasonality": 8
            },
            HV={
                "yearly_seasonality": 25,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 6,
                "daily_seasonality": 12
            },
            LTQM={
                "yearly_seasonality": 8,
                "quarterly_seasonality": 2,
                "monthly_seasonality": 2,
                "weekly_seasonality": 3,
                "daily_seasonality": 5
            },
            NMQM={
                "yearly_seasonality": 15,
                "quarterly_seasonality": 3,
                "monthly_seasonality": 3,
                "weekly_seasonality": 4,
                "daily_seasonality": 8
            },
            HVQM={
                "yearly_seasonality": 25,
                "quarterly_seasonality": 4,
                "monthly_seasonality": 4,
                "weekly_seasonality": 6,
                "daily_seasonality": 12
            },
            NONE={
                "yearly_seasonality": 0,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            }
        ),
        DAILY=dict(
            LT={
                "yearly_seasonality": 8,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 3,
                "daily_seasonality": 0
            },
            NM={
                "yearly_seasonality": 15,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 3,
                "daily_seasonality": 0
            },
            HV={
                "yearly_seasonality": 25,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 4,
                "daily_seasonality": 0
            },
            LTQM={
                "yearly_seasonality": 8,
                "quarterly_seasonality": 3,
                "monthly_seasonality": 2,
                "weekly_seasonality": 3,
                "daily_seasonality": 0
            },
            NMQM={
                "yearly_seasonality": 15,
                "quarterly_seasonality": 4,
                "monthly_seasonality": 4,
                "weekly_seasonality": 3,
                "daily_seasonality": 0
            },
            HVQM={
                "yearly_seasonality": 25,
                "quarterly_seasonality": 6,
                "monthly_seasonality": 4,
                "weekly_seasonality": 4,
                "daily_seasonality": 0
            },
            NONE={
                "yearly_seasonality": 0,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            }
        ),
        WEEKLY=dict(
            LT={
                "yearly_seasonality": 8,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            NM={
                "yearly_seasonality": 15,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            HV={
                "yearly_seasonality": 25,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            LTQM={
                "yearly_seasonality": 8,
                "quarterly_seasonality": 2,
                "monthly_seasonality": 2,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            NMQM={
                "yearly_seasonality": 15,
                "quarterly_seasonality": 3,
                "monthly_seasonality": 3,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            HVQM={
                "yearly_seasonality": 25,
                "quarterly_seasonality": 4,
                "monthly_seasonality": 4,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            NONE={
                "yearly_seasonality": 0,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            }
        )),
    # Growth function.
    GR=dict(
        LINEAR={
            "growth_term": "linear"
        },
        NONE={
            "growth_term": None
        }),
    # Trend changepoints depending on frequency HOURLY/DAILY/WEEKLY.
    CP=dict(
        HOURLY=dict(
            LT={
                "method": "auto",
                "resample_freq": "D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "7D",
                "no_changepoint_distance_from_end": "30D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": None
            },
            NM={
                "method": "auto",
                "resample_freq": "D",
                "regularization_strength": 0.5,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "30D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": "365D"
            },
            HV={
                "method": "auto",
                "resample_freq": "D",
                "regularization_strength": 0.3,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "30D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": "365D"
            },
            NONE=None
        ),
        DAILY=dict(
            LT={
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "90D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": None
            },
            NM={
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.5,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "180D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": "365D"
            },
            HV={
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.3,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "180D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": "365D"
            },
            NONE=None
        ),
        WEEKLY=dict(
            LT={
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "14D",
                "no_changepoint_distance_from_end": "180D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": None
            },
            NM={
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.5,
                "potential_changepoint_distance": "14D",
                "no_changepoint_distance_from_end": "180D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": "365D"
            },
            HV={
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.3,
                "potential_changepoint_distance": "14D",
                "no_changepoint_distance_from_end": "180D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": "365D"
            },
            NONE=None
        )),
    # Holiday effect.
    HOL=dict(
        SP1={
            "holidays_to_model_separately": "auto",
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 1,
            "holiday_post_num_days": 1,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        SP2={
            "holidays_to_model_separately": "auto",
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        SP4={
            "holidays_to_model_separately": "auto",
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 4,
            "holiday_post_num_days": 4,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        TG={
            "holidays_to_model_separately": [],
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 3,
            "holiday_post_num_days": 3,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        NONE={
            "holidays_to_model_separately": [],
            "holiday_lookup_countries": [],
            "holiday_pre_num_days": 0,
            "holiday_post_num_days": 0,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        }),
    # Feature sets enabled.
    FEASET=dict(
        AUTO="auto",
        ON=True,
        OFF=False),
    # Fit algorithm.
    ALGO=dict(
        LINEAR={
            "fit_algorithm": "linear",
            "fit_algorithm_params": None
        },
        RIDGE={
            "fit_algorithm": "ridge",
            "fit_algorithm_params": None
        },
        SGD={
            "fit_algorithm": "sgd",
            "fit_algorithm_params": None
        },
        LASSO={
            "fit_algorithm": "lasso",
            "fit_algorithm_params": None
        }),
    # Autoregression.
    AR=dict(
        AUTO={
            "autoreg_dict": "auto"
        },
        OFF={
            "autoreg_dict": None
        }),
    # Max daily/weekly seasonality interaction orders.
    DSI=dict(
        HOURLY=dict(
            AUTO=5,
            OFF=0
        ),
        DAILY=dict(
            AUTO=0,
            OFF=0
        ),
        WEEKLY=dict(
            AUTO=0,
            OFF=0
        )
    ),
    WSI=dict(
        HOURLY=dict(
            AUTO=2,
            OFF=0
        ),
        DAILY=dict(
            AUTO=2,
            OFF=0
        ),
        WEEKLY=dict(
            AUTO=0,
            OFF=0
        )
    )
)
"""Defines the default component values for
`~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`.
The components include seasonality, growth, holiday, trend changepoints,
feature sets, autoregression, fit algorithm, etc. These are used when
`config.model_template` provides the
`~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`.
"""


# Defines the SILVERKITE template here.
SILVERKITE = ModelComponentsParam(
    seasonality={
        "yearly_seasonality": "auto",
        "quarterly_seasonality": "auto",
        "monthly_seasonality": "auto",
        "weekly_seasonality": "auto",
        "daily_seasonality": "auto",
    },
    growth={
        "growth_term": "linear"
    },
    events={
        "holidays_to_model_separately": "auto",
        "holiday_lookup_countries": "auto",
        "holiday_pre_num_days": 2,
        "holiday_post_num_days": 2,
        "holiday_pre_post_num_dict": None,
        "daily_event_df_dict": None,
    },
    changepoints={
        "changepoints_dict": None,
        "seasonality_changepoints_dict": None
    },
    autoregression={
        "autoreg_dict": None
    },
    regressors={
        "regressor_cols": []
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
"""Defines the ``SILVERKITE`` template. Contains automatic growth,
seasonality, holidays, and interactions. Does not include autoregression.
Best for hourly and daily frequencies. Uses `SimpleSilverkiteEstimator`.
"""

# Defines the `SILVERKITE_EMPTY` template. Everything here is None or off.
# The "DAILY" here does not make any difference from "HOURLY" or "WEEKLY" when everything is None or off.
SILVERKITE_EMPTY = "DAILY_SEAS_NONE_GR_NONE_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_OFF_WSI_OFF"


# Defines pre-tuned multi-templates.
SILVERKITE_DAILY_90 = [
    # For daily data, light seasonality up to weekly, light trend changepoints,
    # separate holidays +- 2 days, default feature sets and linear fit algorithm.
    "DAILY_SEAS_LTQM_GR_LINEAR_CP_LT_HOL_SP2_FEASET_AUTO_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO",
    # For daily data, light seasonality up to weekly, no trend changepoints,
    # separate holidays +- 2 days, default feature sets and linear fit algorithm.
    "DAILY_SEAS_LTQM_GR_LINEAR_CP_NONE_HOL_SP2_FEASET_AUTO_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO",
    # For daily data, light seasonality up to weekly, light trend changepoints,
    # separate holidays +- 2 days, default feature sets and ridge fit algorithm.
    "DAILY_SEAS_LTQM_GR_LINEAR_CP_LT_HOL_SP2_FEASET_AUTO_ALGO_RIDGE_AR_OFF_DSI_AUTO_WSI_AUTO",
    # For daily data, normal seasonality up to weekly, light trend changepoints,
    # separate holidays +- 4 days, default feature sets and ridge fit algorithm.
    "DAILY_SEAS_NM_GR_LINEAR_CP_LT_HOL_SP4_FEASET_AUTO_ALGO_RIDGE_AR_OFF_DSI_AUTO_WSI_AUTO"
]
SILVERKITE_WEEKLY = [
    # For weekly data, normal seasonality up to yearly, no trend changepoints,
    # no holiday, no feature sets and linear fit algorithm.
    "WEEKLY_SEAS_NM_GR_LINEAR_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO",
    # For weekly data, normal seasonality up to yearly, light trend changepoints,
    # no holiday, no feature sets and linear fit algorithm.
    "WEEKLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO",
    # For weekly data, heavy seasonality up to yearly, normal trend changepoints,
    # no holiday, no feature sets and ridge fit algorithm.
    "WEEKLY_SEAS_HV_GR_LINEAR_CP_NM_HOL_NONE_FEASET_OFF_ALGO_RIDGE_AR_OFF_DSI_AUTO_WSI_AUTO",
    # For weekly data, heavy seasonality up to yearly, light trend changepoints,
    # no holiday, no feature sets and ridge fit algorithm.
    "WEEKLY_SEAS_HV_GR_LINEAR_CP_LT_HOL_NONE_FEASET_OFF_ALGO_RIDGE_AR_OFF_DSI_AUTO_WSI_AUTO"
]
SILVERKITE_HOURLY_1 = [
    # For hourly data, light seasonality up to daily, no trend changepoints,
    # together holiday, default feature sets, automatic autoregression and linear fit algorithm.
    "HOURLY_SEAS_LT_GR_LINEAR_CP_NONE_HOL_TG_FEASET_AUTO_ALGO_LINEAR_AR_AUTO",
    # For hourly data, normal seasonality up to daily, light trend changepoints,
    # separate holidays +- 4 days, default feature sets, automatic autoregression and linear fit algorithm.
    "HOURLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_SP4_FEASET_AUTO_ALGO_LINEAR_AR_AUTO",
    # For hourly data, light seasonality up to daily, normal trend changepoints,
    # separate holidays +- 4 days, no feature sets, automatic autoregression and ridge fit algorithm.
    "HOURLY_SEAS_LT_GR_LINEAR_CP_NM_HOL_SP4_FEASET_OFF_ALGO_RIDGE_AR_AUTO",
    # For hourly data, normal seasonality up to daily, normal trend changepoints,
    # separate holidays +- 1 day, default feature sets, automatic autoregression and ridge fit algorithm.
    "HOURLY_SEAS_NM_GR_LINEAR_CP_NM_HOL_SP1_FEASET_AUTO_ALGO_RIDGE_AR_AUTO"
]
SILVERKITE_HOURLY_24 = [
    # For hourly data, light seasonality up to daily, normal trend changepoints,
    # separate holidays +- 4 days, default feature sets, automatic autoregression and ridge fit algorithm.
    "HOURLY_SEAS_LT_GR_LINEAR_CP_NM_HOL_SP4_FEASET_AUTO_ALGO_RIDGE_AR_AUTO",
    # For hourly data, light seasonality up to daily, no trend changepoints,
    # separate holidays +- 4 days, default feature sets, automatic autoregression and ridge fit algorithm.
    "HOURLY_SEAS_LT_GR_LINEAR_CP_NONE_HOL_SP4_FEASET_AUTO_ALGO_RIDGE_AR_AUTO",
    # For hourly data, normal seasonality up to daily, light trend changepoints,
    # separate holidays +- 1 days, no feature sets, automatic autoregression and linear fit algorithm.
    "HOURLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_SP1_FEASET_OFF_ALGO_LINEAR_AR_AUTO",
    # For hourly data, normal seasonality up to daily, normal trend changepoints,
    # separate holidays +- 4 day, default feature sets, automatic autoregression and ridge fit algorithm.
    "HOURLY_SEAS_NM_GR_LINEAR_CP_NM_HOL_SP4_FEASET_AUTO_ALGO_RIDGE_AR_AUTO"
]
SILVERKITE_HOURLY_168 = [
    # For hourly data, light seasonality up to daily, light trend changepoints,
    # separate holidays +- 4 days, default feature sets, no autoregression and ridge fit algorithm.
    "HOURLY_SEAS_LT_GR_LINEAR_CP_LT_HOL_SP4_FEASET_AUTO_ALGO_RIDGE_AR_OFF",
    # For hourly data, light seasonality up to daily, light trend changepoints,
    # separate holidays +- 2 days, default feature sets, no autoregression and ridge fit algorithm.
    "HOURLY_SEAS_LT_GR_LINEAR_CP_LT_HOL_SP2_FEASET_AUTO_ALGO_RIDGE_AR_OFF",
    # For hourly data, normal seasonality up to daily, no trend changepoints,
    # separate holidays +- 4 days, no feature sets, automatic autoregression and linear fit algorithm.
    "HOURLY_SEAS_NM_GR_LINEAR_CP_NONE_HOL_SP4_FEASET_OFF_ALGO_LINEAR_AR_AUTO",
    # For hourly data, normal seasonality up to daily, normal trend changepoints,
    # separate holidays +- 1 day, default feature sets, no autoregression and ridge fit algorithm.
    "HOURLY_SEAS_NM_GR_LINEAR_CP_NM_HOL_SP1_FEASET_AUTO_ALGO_RIDGE_AR_OFF"
]
SILVERKITE_HOURLY_336 = [
    # For hourly data, light seasonality up to daily, light trend changepoints,
    # separate holidays +- 2 days, default feature sets, no autoregression and ridge fit algorithm.
    "HOURLY_SEAS_LT_GR_LINEAR_CP_LT_HOL_SP2_FEASET_AUTO_ALGO_RIDGE_AR_OFF",
    # For hourly data, light seasonality up to daily, light trend changepoints,
    # separate holidays +- 4 days, default feature sets, no autoregression and ridge fit algorithm.
    "HOURLY_SEAS_LT_GR_LINEAR_CP_LT_HOL_SP4_FEASET_AUTO_ALGO_RIDGE_AR_OFF",
    # For hourly data, normal seasonality up to daily, light trend changepoints,
    # separate holidays +- 2 days, default feature sets, no autoregression and linear fit algorithm.
    "HOURLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_SP1_FEASET_AUTO_ALGO_LINEAR_AR_OFF",
    # For hourly data, normal seasonality up to daily, normal trend changepoints,
    # separate holidays +- 1 day, default feature sets, automatic autoregression and linear fit algorithm.
    "HOURLY_SEAS_NM_GR_LINEAR_CP_NM_HOL_SP1_FEASET_AUTO_ALGO_LINEAR_AR_AUTO"
]


# Defines pre-defined multi templates.

MULTI_TEMPLATES = {
    "SILVERKITE_DAILY_90": SILVERKITE_DAILY_90,
    "SILVERKITE_WEEKLY": SILVERKITE_WEEKLY,
    "SILVERKITE_HOURLY_1": SILVERKITE_HOURLY_1,
    "SILVERKITE_HOURLY_24": SILVERKITE_HOURLY_24,
    "SILVERKITE_HOURLY_168": SILVERKITE_HOURLY_168,
    "SILVERKITE_HOURLY_336": SILVERKITE_HOURLY_336
}
"""A dictionary of multi templates.

    - Keys are the available multi templates names (valid strings for `config.model_template`).
    - Values correspond to a list of
      `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`.
"""

SINGLE_MODEL_TEMPLATE_TYPE = Union[str, ModelComponentsParam, SimpleSilverkiteTemplateOptions]
"""Types accepted by SimpleSilverkiteTemplate for ``config.model_template``
for a single template.
"""


@dataclass
class SimpleSilverkiteTemplateConstants:
    """Constants used by
    `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`.
    Includes the model templates and their default values.

    `~greykite.common.python_utils.mutable_field` is used when the default value
    is a mutable type like dict and list. Dataclass requires mutable default values
    to be wrapped in 'default_factory', so that instances of this dataclass cannot
    accidentally modify the default value.
    `~greykite.common.python_utils.mutable_field` wraps the constant accordingly.
    """
    COMMON_MODELCOMPONENTPARAM_PARAMETERS: Dict = mutable_field(COMMON_MODELCOMPONENTPARAM_PARAMETERS)
    """Defines the default component values for
    `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`.
    The components include seasonality, growth, holiday, trend changepoints,
    feature sets, autoregression, fit algorithm, etc. These are used when
    `config.model_template` provides the
    `~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`.
    """
    MULTI_TEMPLATES: Dict = mutable_field(MULTI_TEMPLATES)
    """A dictionary of multi templates.

        - Keys are the available multi templates names (valid strings for `config.model_template`).
        - Values correspond to a list of
          `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`.
    """
    SILVERKITE: SINGLE_MODEL_TEMPLATE_TYPE = SILVERKITE
    """Defines the ``"SILVERKITE"`` template. Contains automatic growth,
    seasonality, holidays, and interactions. Does not include autoregression.
    Best for hourly and daily frequencies. Uses `SimpleSilverkiteEstimator`.
    """
    SILVERKITE_COMPONENT_KEYWORDS: Type[Enum] = SILVERKITE_COMPONENT_KEYWORDS
    """Valid values for simple silverkite template string name keywords.
    The names are the keywords and the values are the corresponding value enum.
    Can be used to create an instance of
    `~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`.
    """
    SILVERKITE_EMPTY: SINGLE_MODEL_TEMPLATE_TYPE = SILVERKITE_EMPTY
    """Defines the ``"SILVERKITE_EMPTY"`` template. Everything here is None or off."""
    VALID_FREQ: List = mutable_field(VALID_FREQ)
    """Valid non-default values for simple silverkite template string name frequency.
    `~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`.
    """
    SimpleSilverkiteTemplateOptions: dataclass = SimpleSilverkiteTemplateOptions
    """Defines generic simple silverkite template options.
    Attributes can be set to different values using
    `~greykite.framework.templates.simple_silverkite_template_config.SILVERKITE_COMPONENT_KEYWORDS`
    for high level tuning.
    """
