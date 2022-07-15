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

from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.common.constants import GrowthColEnum
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
                "auto_seasonality": False,
                "yearly_seasonality": 8,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 3,
                "daily_seasonality": 5
            },
            NM={
                "auto_seasonality": False,
                "yearly_seasonality": 15,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 4,
                "daily_seasonality": 8
            },
            HV={
                "auto_seasonality": False,
                "yearly_seasonality": 25,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 6,
                "daily_seasonality": 12
            },
            LTQM={
                "auto_seasonality": False,
                "yearly_seasonality": 8,
                "quarterly_seasonality": 2,
                "monthly_seasonality": 2,
                "weekly_seasonality": 3,
                "daily_seasonality": 5
            },
            NMQM={
                "auto_seasonality": False,
                "yearly_seasonality": 15,
                "quarterly_seasonality": 3,
                "monthly_seasonality": 3,
                "weekly_seasonality": 4,
                "daily_seasonality": 8
            },
            HVQM={
                "auto_seasonality": False,
                "yearly_seasonality": 25,
                "quarterly_seasonality": 4,
                "monthly_seasonality": 4,
                "weekly_seasonality": 6,
                "daily_seasonality": 12
            },
            NONE={
                "auto_seasonality": False,
                "yearly_seasonality": 0,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            }
        ),
        DAILY=dict(
            LT={
                "auto_seasonality": False,
                "yearly_seasonality": 8,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 3,
                "daily_seasonality": 0
            },
            NM={
                "auto_seasonality": False,
                "yearly_seasonality": 15,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 3,
                "daily_seasonality": 0
            },
            HV={
                "auto_seasonality": False,
                "yearly_seasonality": 25,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 4,
                "daily_seasonality": 0
            },
            LTQM={
                "auto_seasonality": False,
                "yearly_seasonality": 8,
                "quarterly_seasonality": 3,
                "monthly_seasonality": 2,
                "weekly_seasonality": 3,
                "daily_seasonality": 0
            },
            NMQM={
                "auto_seasonality": False,
                "yearly_seasonality": 15,
                "quarterly_seasonality": 4,
                "monthly_seasonality": 4,
                "weekly_seasonality": 3,
                "daily_seasonality": 0
            },
            HVQM={
                "auto_seasonality": False,
                "yearly_seasonality": 25,
                "quarterly_seasonality": 6,
                "monthly_seasonality": 4,
                "weekly_seasonality": 4,
                "daily_seasonality": 0
            },
            NONE={
                "auto_seasonality": False,
                "yearly_seasonality": 0,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            }
        ),
        WEEKLY=dict(
            LT={
                "auto_seasonality": False,
                "yearly_seasonality": 8,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            NM={
                "auto_seasonality": False,
                "yearly_seasonality": 15,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            HV={
                "auto_seasonality": False,
                "yearly_seasonality": 25,
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            LTQM={
                "auto_seasonality": False,
                "yearly_seasonality": 8,
                "quarterly_seasonality": 2,
                "monthly_seasonality": 2,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            NMQM={
                "auto_seasonality": False,
                "yearly_seasonality": 15,
                "quarterly_seasonality": 3,
                "monthly_seasonality": 3,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            HVQM={
                "auto_seasonality": False,
                "yearly_seasonality": 25,
                "quarterly_seasonality": 4,
                "monthly_seasonality": 4,
                "weekly_seasonality": 0,
                "daily_seasonality": 0
            },
            NONE={
                "auto_seasonality": False,
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
            "growth_term": GrowthColEnum.linear.name
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
            "auto_holiday": False,
            "holidays_to_model_separately": "auto",
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 1,
            "holiday_post_num_days": 1,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        SP2={
            "auto_holiday": False,
            "holidays_to_model_separately": "auto",
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        SP4={
            "auto_holiday": False,
            "holidays_to_model_separately": "auto",
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 4,
            "holiday_post_num_days": 4,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        TG={
            "auto_holiday": False,
            "holidays_to_model_separately": [],
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 3,
            "holiday_post_num_days": 3,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        NONE={
            "auto_holiday": False,
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
            "autoreg_dict": "auto",
            "simulation_num": 10,  # simulation is not triggered with ``autoreg_dict="auto"``
            "fast_simulation": False
        },
        OFF={
            "autoreg_dict": None,
            "simulation_num": 10,  # simulation is not triggered with ``autoreg_dict=None``
            "fast_simulation": False
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
        "auto_seasonality": False,
        "yearly_seasonality": "auto",
        "quarterly_seasonality": "auto",
        "monthly_seasonality": "auto",
        "weekly_seasonality": "auto",
        "daily_seasonality": "auto",
    },
    growth={
        "growth_term": GrowthColEnum.linear.name
    },
    events={
        "auto_holiday": False,
        "holidays_to_model_separately": "auto",
        "holiday_lookup_countries": "auto",
        "holiday_pre_num_days": 2,
        "holiday_post_num_days": 2,
        "holiday_pre_post_num_dict": None,
        "daily_event_df_dict": None,
    },
    changepoints={
        "auto_growth": False,
        "changepoints_dict": {
            "method": "auto",
            "yearly_seasonality_order": 15,
            "resample_freq": "3D",
            "regularization_strength": 0.6,
            "actual_changepoint_min_distance": "30D",
            "potential_changepoint_distance": "15D",
            "no_changepoint_distance_from_end": "90D"
        },
        "seasonality_changepoints_dict": None
    },
    autoregression={
        "autoreg_dict": "auto",
        "simulation_num": 10,  # simulation is not triggered with ``autoreg_dict="auto"``
        "fast_simulation": False
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
        "drop_pred_cols": None,
        "explicit_pred_cols": None,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "regression_weight_col": None,
        "normalize_method": "zero_to_one"
    }
)
"""Defines the ``SILVERKITE`` template. Contains automatic growth,
seasonality, holidays, autoregression and interactions.
Uses "zero_to_one" normalization method.
Best for hourly and daily frequencies. Uses `SimpleSilverkiteEstimator`.
"""


# Defines the `SILVERKITE_EMPTY` template. Everything here is None or off.
# The "DAILY" here does not make any difference from "HOURLY" or "WEEKLY" when everything is None or off.
SILVERKITE_EMPTY = "DAILY_SEAS_NONE_GR_NONE_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_OFF_WSI_OFF"


# Defines pre-tuned multi-templates.
# The following 3 configs are defined as a single template in order to be used as candidates in the multi-template ``SILVERKITE_DAILY_1``.
# In these 3 configs, yearly_seasonality, monthly_seasonality, weekly_seasonality, regularization_strength, and no_changepoint_distance_from_end
# are tuned specifically for 1-day forecast.
SILVERKITE_DAILY_1_CONFIG_1 = ModelComponentsParam(
    seasonality={
        "auto_seasonality": False,
        "yearly_seasonality": 8,
        "quarterly_seasonality": 0,
        "monthly_seasonality": 7,
        "weekly_seasonality": 1,
        "daily_seasonality": 0,
    },
    growth={
        "growth_term": GrowthColEnum.linear.name
    },
    events={
        "auto_holiday": False,
        "holidays_to_model_separately": SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO,
        "holiday_lookup_countries": SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO,
        "holiday_pre_num_days": 2,
        "holiday_post_num_days": 2,
        "holiday_pre_post_num_dict": None,
        "daily_event_df_dict": None,
    },
    changepoints={
        "auto_growth": False,
        "changepoints_dict": {
            "method": "auto",
            "resample_freq": "7D",
            "regularization_strength": 0.809,
            "potential_changepoint_distance": "7D",
            "no_changepoint_distance_from_end": "7D",
            "yearly_seasonality_order": 8,
            "yearly_seasonality_change_freq": None,
        },
        "seasonality_changepoints_dict": None
    },
    autoregression={
        "autoreg_dict": "auto",
        "simulation_num": 10,  # simulation is not triggered with ``autoreg_dict="auto"``
        "fast_simulation": False
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
        "drop_pred_cols": None,
        "explicit_pred_cols": None,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "regression_weight_col": None,
        "normalize_method": "zero_to_one"
    }
)

SILVERKITE_DAILY_1_CONFIG_2 = ModelComponentsParam(
    seasonality={
        "auto_seasonality": False,
        "yearly_seasonality": 1,
        "quarterly_seasonality": 0,
        "monthly_seasonality": 4,
        "weekly_seasonality": 6,
        "daily_seasonality": 0,
    },
    growth={
        "growth_term": GrowthColEnum.linear.name
    },
    events={
        "auto_holiday": False,
        "holidays_to_model_separately": SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO,
        "holiday_lookup_countries": SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO,
        "holiday_pre_num_days": 2,
        "holiday_post_num_days": 2,
        "holiday_pre_post_num_dict": None,
        "daily_event_df_dict": None,
    },
    changepoints={
        "auto_growth": False,
        "changepoints_dict": {
            "method": "auto",
            "resample_freq": "7D",
            "regularization_strength": 0.624,
            "potential_changepoint_distance": "7D",
            "no_changepoint_distance_from_end": "17D",
            "yearly_seasonality_order": 1,
            "yearly_seasonality_change_freq": None,
        },
        "seasonality_changepoints_dict": None
    },
    autoregression={
        "autoreg_dict": "auto",
        "simulation_num": 10,  # simulation is not triggered with ``autoreg_dict="auto"``
        "fast_simulation": False
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
        "drop_pred_cols": None,
        "explicit_pred_cols": None,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "regression_weight_col": None,
        "normalize_method": "zero_to_one"
    }
)

SILVERKITE_DAILY_1_CONFIG_3 = ModelComponentsParam(
    seasonality={
        "auto_seasonality": False,
        "yearly_seasonality": 40,
        "quarterly_seasonality": 0,
        "monthly_seasonality": 0,
        "weekly_seasonality": 2,
        "daily_seasonality": 0,
    },
    growth={
        "growth_term": GrowthColEnum.linear.name
    },
    events={
        "auto_holiday": False,
        "holidays_to_model_separately": SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO,
        "holiday_lookup_countries": SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO,
        "holiday_pre_num_days": 2,
        "holiday_post_num_days": 2,
        "holiday_pre_post_num_dict": None,
        "daily_event_df_dict": None,
    },
    changepoints={
        "auto_growth": False,
        "changepoints_dict": {
            "method": "auto",
            "resample_freq": "7D",
            "regularization_strength": 0.590,
            "potential_changepoint_distance": "7D",
            "no_changepoint_distance_from_end": "8D",
            "yearly_seasonality_order": 40,
            "yearly_seasonality_change_freq": None,
        },
        "seasonality_changepoints_dict": None
    },
    autoregression={
        "autoreg_dict": "auto",
        "simulation_num": 10,  # simulation is not triggered with ``autoreg_dict="auto"``
        "fast_simulation": False
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
        "drop_pred_cols": None,
        "explicit_pred_cols": None,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "regression_weight_col": None,
        "normalize_method": "zero_to_one"
    }
)

# Defines the SILVERKITE monthly template here.
SILVERKITE_MONTHLY = ModelComponentsParam(
    seasonality={
        "auto_seasonality": False,
        "yearly_seasonality": False,
        "quarterly_seasonality": False,
        "monthly_seasonality": False,
        "weekly_seasonality": False,
        "daily_seasonality": False,
    },
    growth={
        "growth_term": GrowthColEnum.linear.name
    },
    events={
        "auto_holiday": False,
        "holidays_to_model_separately": [],
        "holiday_lookup_countries": [],
        "holiday_pre_num_days": 0,
        "holiday_post_num_days": 0,
        "holiday_pre_post_num_dict": None,
        "daily_event_df_dict": None,
    },
    changepoints={
        "auto_growth": False,
        "changepoints_dict": {
            "method": "auto",
            "regularization_strength": 0.6,
            "resample_freq": "28D",  # no effect for monthly data if less than or equal to 28 days
            "potential_changepoint_distance": "180D",
            "potential_changepoint_n_max": 100,
            "actual_changepoint_min_distance": "730D",
            "no_changepoint_distance_from_end": "180D",
            "yearly_seasonality_order": 6
        },
        "seasonality_changepoints_dict": None
    },
    autoregression={
        "autoreg_dict": {
            "lag_dict": None,
            "agg_lag_dict": {
                "orders_list": [[1, 2, 3]]  # uses aggregated lags
            }
        },
        "simulation_num": 50,  # `simulation_num` is not used when `fast_simulation` is True
        "fast_simulation": True
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
        "feature_sets_enabled": False,
        "max_daily_seas_interaction_order": 0,
        "max_weekly_seas_interaction_order": 0,
        "extra_pred_cols": [
            "y_avglag_1_2_3*C(month, levels=list(range(1, 13)))", "C(month, levels=list(range(1, 13)))"],
        "drop_pred_cols": None,
        "explicit_pred_cols": None,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "regression_weight_col": None,
        "normalize_method": "zero_to_one"
    }
)
"""Defines the ``SILVERKITE_MONTHLY`` template. Contains automatic growth.
Seasonality is modeled via categorical variable "month".
Includes aggregated autoregression.
Simulation is needed when forecast horizon is greater than 1.
Uses statistical normalization method. Uses `SimpleSilverkiteEstimator`.
"""


SILVERKITE_DAILY_1 = ["SILVERKITE_DAILY_1_CONFIG_1", "SILVERKITE_DAILY_1_CONFIG_2", "SILVERKITE_DAILY_1_CONFIG_3"]
"""Defines the ``SILVERKITE_DAILY_1`` template, which contains 3 candidate configs for grid search,
optimized for the seasonality and changepoint parameters.
Best for 1-day forecast for daily time series. Uses `SimpleSilverkiteEstimator`.
"""

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
    # For hourly data, the first template is the same as the "SILVERKITE" template defined above.
    "SILVERKITE",
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
    "SILVERKITE_DAILY_1": SILVERKITE_DAILY_1,
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
    seasonality, holidays, autoregression and interactions.
    Uses "zero_to_one" normalization method.
    Best for hourly and daily frequencies. Uses `SimpleSilverkiteEstimator`.
    """
    SILVERKITE_MONTHLY: SINGLE_MODEL_TEMPLATE_TYPE = SILVERKITE_MONTHLY
    """Defines the ``SILVERKITE_MONTHLY`` template.
    Best for monthly forecasts. Uses `SimpleSilverkiteEstimator`.
    """
    SILVERKITE_DAILY_1_CONFIG_1: SINGLE_MODEL_TEMPLATE_TYPE = SILVERKITE_DAILY_1_CONFIG_1
    """Config 1 in template ``SILVERKITE_DAILY_1``.
    Compared to ``SILVERKITE``, it adds change points and uses parameters
    specifically tuned for daily data and 1-day forecast.
    """
    SILVERKITE_DAILY_1_CONFIG_2: SINGLE_MODEL_TEMPLATE_TYPE = SILVERKITE_DAILY_1_CONFIG_2
    """Config 2 in template ``SILVERKITE_DAILY_1``.
    Compared to ``SILVERKITE``, it adds change points and uses parameters
    specifically tuned for daily data and 1-day forecast.
    """
    SILVERKITE_DAILY_1_CONFIG_3: SINGLE_MODEL_TEMPLATE_TYPE = SILVERKITE_DAILY_1_CONFIG_3
    """Config 3 in template ``SILVERKITE_DAILY_1``.
    Compared to ``SILVERKITE``, it adds change points and uses parameters
    specifically tuned for daily data and 1-day forecast.
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
