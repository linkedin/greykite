"""
Enhanced week over week models
==============================

Week over week model is a useful tool in business applications,
where time series exhibits strong weekly seasonality.
It's fast and somewhat accurate.
Typical drawbacks of week over week models include
not adapting to seasonality (e.g. year-end), fast growth
and holiday effects.
Also, week over week model is vulnerable to corrupted data
such as outliers on last week.

Using aggregated lags such like week over 3 weeks median is more robust to data corruption,
but the growth/seasonality/holiday issue is not resolved.

The enhanced version of week over week model fits a two-step
model with the ``MultistageForecast`` method in Greykite.
It first uses a ``Silverkite`` model to learn the growth,
yearly seasonality and holiday effects.
Then it uses a week over week or other lag-based models to model the residual
weekly patterns.

In this example, we will learn how to do the original
week over week type models and how to use the enhanced versions.

The regular week over week models
---------------------------------

Greykite supports the regular lag-based models through ``LagBasedTemplate``.
To see a general introduction of how to use model templates,
see `model templates <../templates/0100_template_overview>`_.

Lag-based methods are invoked by specifying the ``LAG_BASED`` model template.
"""

import warnings

import pandas as pd
from greykite.common.data_loader import DataLoader
from greykite.common.aggregation_function_enum import AggregationFunctionEnum
from greykite.common import constants as cst
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.multistage_forecast_template import MultistageForecastTemplateConfig
from greykite.sklearn.estimator.lag_based_estimator import LagUnitEnum

warnings.filterwarnings("ignore")

df = DataLoader().load_peyton_manning()
df[cst.TIME_COL] = pd.to_datetime(df[cst.TIME_COL])

# %%
# We specify the data set and evaluation parameters below.
# First, we don't specify model components.
# In this case, the default behavior for ``LAG_BASED`` model template
# is the week over week model.
# If the forecast horizon is longer than a week,
# the model will use the forecasted value to generate further forecasts.

metadata = MetadataParam(
    time_col=cst.TIME_COL,
    value_col=cst.VALUE_COL,
    freq="D"
)

# Turn off cv and test for faster run.
evaluation = EvaluationPeriodParam(
    cv_max_splits=0,
    test_horizon=0
)

config = ForecastConfig(
    forecast_horizon=7,
    model_template=ModelTemplateEnum.LAG_BASED.name,
    metadata_param=metadata,
    evaluation_period_param=evaluation
)

forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=config
)

# %%
# This is the simple week over week estimation.
# If we print the results, we can see that the predictions
# are exactly the same as the last week's observations.

result.forecast.df_train.tail(14)

# %%
# In general, the lag-based method supports any
# aggregation of any lag combinations.
# Now let's use an example to demonstrate how to do a
# week-over-3-week median estimation.
# We override the parameters in ``ModelComponentsParam.custom`` dictionary.
# The parameters that can be customized are
#
#   * ``lag_unit``: the unit of the lags. Options are in
#     `~greykite.sklearn.estimator.lag_based_estimator.LagUnitEnum`.
#   * ``lags``: a list of integers indicating the lags in ``lag_unit``.
#   * ``agg_func``: the aggregation function name. Options are in
#     `~greykite.common.aggregation_function_enum.AggregationFunctionEnum`.
#   * ``agg_func_params``: a dictionary of parameters to be passed to the aggregation function.
#
# Specifying the following, the forecasts will become week-over-3-week median.

model_components = ModelComponentsParam(
    custom=dict(
        lag_unit=LagUnitEnum.week.name,                 # unit is "week"
        lags=[1, 2, 3],                                 # lags are 1 week, 2 weeks and 3 weeks
        agg_func=AggregationFunctionEnum.median.name    # aggregation function is "median"
    )
)

config = ForecastConfig(
    forecast_horizon=7,
    model_template=ModelTemplateEnum.LAG_BASED.name,
    metadata_param=metadata,
    evaluation_period_param=evaluation,
    model_components_param=model_components
)

forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=config
)

result.forecast.df_train.tail(14)

# %%
# The enhanced week over week model
# ---------------------------------
#
# The enhanced week over week model consists of a two-stage model:
#
#   * ``"Silverkite model"``: the first stage uses a Silverkite model to learn the
#     yearly seasonality, growth and holiday effects.
#   * ``"Lag-based model"``: the second stage uses a lag-based model to learn the
#     residual effects including weekly seasonality.
#
# The model is available through the ``MultistageForecastTemplate``.
# For details about the multistage forecast model, see
# `multistage forecast <../tutorials/0200_multistage_forecast>`_.
#
# To use this two-stage enhanced lag model,
# specify the model template as ``SILVERKITE_WOW``.
# The default behavior is to model growth, yearly seasonality and holidays
# with the automatically inferred parameters from the time series.
# Then it models the residual with a week over week model.

config = ForecastConfig(
    forecast_horizon=7,
    model_template=ModelTemplateEnum.SILVERKITE_WOW.name,
    metadata_param=metadata,
    evaluation_period_param=evaluation
)

forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=config
)

result.forecast.df_train.tail(14)

# %%
# You may notice that the forecast is not exactly the observations a week ago,
# because the Silverkite model did some adjustments on the growth, yearly seasonality
# and holidays.
#
# To override the model parameters, we will follow the rules mentioned in
# `multistage forecast <../tutorials/0200_multistage_forecast>`_.
# For each stage of model, if you would like to just change one parameter
# and keep the other parameters the same,
# you can specify the same model template for the stage as in ``SILVERKITE_WOW``
# (they are ``SILVERKITE_EMPTY`` and ``LAG_BASED``),
# and specify a model components object to override the specific parameter.
# Otherwise, you can specify a new model template.
# The code below overrides both the Silverkite model and the lag model.
# In the first stage, it keeps the original configuration but forces turning yearly seasonality off.
# In the second stage, it uses week-over-3-week median instead of wow model.

model_components = ModelComponentsParam(
    custom=dict(
        multistage_forecast_configs=[
            MultistageForecastTemplateConfig(
                train_length="1096D",
                fit_length=None,
                agg_func="nanmean",
                agg_freq="D",
                # Keeps it the same as the model template in `SILVERKITE_WOW` to override selected parameters below
                model_template=ModelTemplateEnum.SILVERKITE_EMPTY.name,
                # Since the model template in this stage is the same as the model template in `SILVERKITE_WOW`,
                # the parameter below will be applied on top of the existing parameters.
                model_components=ModelComponentsParam(
                    seasonality={
                        "yearly_seasonality": False  # force turning off yearly seasonality
                    }
                )
            ),
            MultistageForecastTemplateConfig(
                train_length="28D",  # any value longer than the lags (21D here)
                fit_length=None,  # keep as None
                agg_func="nanmean",
                agg_freq=None,
                # Keeps it the same as the model template in `SILVERKITE_WOW` to override selected parameters below
                model_template=ModelTemplateEnum.LAG_BASED.name,
                # Since the model template in this stage is the same as the model template in `SILVERKITE_WOW`,
                # the parameter below will be applied on top of the existing parameters.
                model_components=ModelComponentsParam(
                    custom={
                        "lags": [1, 2, 3],  # changes to 3 weeks' median, default unit is "week",
                        "lag_unit": LagUnitEnum.week.name,
                        "agg_func": AggregationFunctionEnum.median.name,  # changes to 3 weeks' median
                    }
                )
            )
        ]
    )
)

config = ForecastConfig(
    forecast_horizon=7,
    model_template=ModelTemplateEnum.LAG_BASED.name,
    metadata_param=metadata,
    evaluation_period_param=evaluation,
    model_components_param=model_components
)

forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=config
)

result.forecast.df_train.tail(14)