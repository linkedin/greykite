"""
The Multistage Forecast Model
=============================

This is a tutorial for the Multistage Forecast model.
Multistage Forecast is a fast solution designed for more granular time series
(for example, minute-level), where a long history is needed to train a good model.

For example, suppose we want to train a model on 2 years of 5-minute frequency data.
That's 210,240 observations.
If we directly fit a model to large input data,
training time and resource demand can be high (15+ minutes on i9 CPU).
If we use a shorter period to train the model,
the model will not be able to capture long term effects
such as holidays, monthly/quarterly seasonalities, year-end drops, etc.
There is a trade-off between speed and accuracy.

On the other hand, if due to data retention policy,
we only have data in the original frequency for a short history,
but we have aggregated data for a longer history,
could we utilize both datasets to make the prediction more accurate?

Multistage Forecast is designed to close this gap.
It's easy to observe the following facts:

    - Trend can be learned with data at a weekly/daily granularity.
    - Yearly seasonality, weekly seasonality and holiday effects can be learned with daily data.
    - Daily seasonality and autoregression effects can be learned with most recent data if the forecast horizon
      is small (which is usually the case in minute-level data).

Then it's natural to think of the idea: not all components in the forecast model need
to be learned from minute-level granularity. Training each component with the least granularity data needed
can greatly save time while keeping the desired accuracy.

Here we introduce the Multistage Forecast algorithm, which is built upon the idea above:

    - Multistage Forecast trains multiple models to fit a time series.
    - Each stage of the model trains on the residuals of the previous stages,
      takes an appropriate length of data, does an optional aggregation,
      and learns the appropriate components for the granularity.
    - The final predictions will be the sum of the predictions from all stages of models.

In practice, we’ve found Multistage Forecast to reduce training time by up to 10X while maintaining accuracy,
compared to a Silverkite model trained on the full dataset.

A diagram of the Multistage Forecast model flow is shown below.

.. image:: /figures/multistage_forecast.png
  :width: 600
  :alt: Multistage Forecast training flow

Next, we will see examples of how to configure Multistage Forecast models.
"""

# import libraries
import plotly
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.autogen.forecast_config import ForecastConfig,\
    MetadataParam, ModelComponentsParam, EvaluationPeriodParam
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import cols_interact
from greykite.framework.templates.multistage_forecast_template_config import MultistageForecastTemplateConfig

# %%
# Configuring the Multistage Forecast model
# -----------------------------------------
#
# We take an hourly dataset as an example.
# We will use the hourly Washington D.C. bikesharing dataset
# (`source <https://www.capitalbikeshare.com/system-data>`_).

# loads the dataset
ts = DataLoaderTS().load_bikesharing_ts()
print(ts.df.head())

# plot the data
plotly.io.show(ts.plot())

# %%
# The data contains a few years of hourly data.
# Directly training on the entire dataset may take a couple of minutes.
# Now let's consider a two-stage model with the following configuration:
#
#   - **Daily model**: a model trained on 2 years of data with daily aggregation.
#     The model will learn the trend, yearly seasonality, weekly seasonality and holidays.
#     For an explanation of the configuration below, see the `paper <https://arxiv.org/abs/2105.01098>`_.
#   - **Hourly model**: a model trained on the residuals to learn short term patterns.
#     The model will learn daily seasonality, its interaction with the ``is_weekend`` indicator,
#     and some autoregression effects.
#
# From :doc:`/gallery/tutorials/0100_forecast_tutorial` we know how to specify
# each single model above. The core configuration is specified via
# `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`.
# We can specify the two models as follows.

# the daily model
daily_model_components = ModelComponentsParam(
    growth=dict(
        growth_term="linear"
    ),
    seasonality=dict(
        yearly_seasonality=12,
        quarterly_seasonality=0,
        monthly_seasonality=0,
        weekly_seasonality=5,
        daily_seasonality=0  # daily model does not have daily seasonality
    ),
    changepoints=dict(
        changepoints_dict=dict(
            method="auto",
            regularization_strength=0.5,
            yearly_seasonality_order=12,
            resample_freq="3D",
            potential_changepoint_distance="30D",
            no_changepoint_distance_from_end="30D"
        ),
        seasonality_changepoints_dict=None
    ),
    autoregression=dict(
        autoreg_dict="auto"
    ),
    events=dict(
        holidays_to_model_separately=["Christmas Day", "New Year's Day", "Independence Day", "Thanksgiving"],
        holiday_lookup_countries=["UnitedStates"],
        holiday_pre_num_days=1,
        holiday_post_num_days=1
    ),
    custom=dict(
        fit_algorithm_dict=dict(
            fit_algorithm="ridge"
        ),
        feature_sets_enabled="auto",
        min_admissible_value=0
    )
)

# creates daily seasonality interaction with is_weekend
daily_interaction = cols_interact(
    static_col="is_weekend",
    fs_name="tod_daily",
    fs_order=5
)

# the hourly model
hourly_model_components = ModelComponentsParam(
    growth=dict(
        growth_term=None  # growth is already modeled in daily model
    ),
    seasonality=dict(
        yearly_seasonality=0,
        quarterly_seasonality=0,
        monthly_seasonality=0,
        weekly_seasonality=0,
        daily_seasonality=12  # hourly model has daily seasonality
    ),
    changepoints=dict(
        changepoints_dict=None,
        seasonality_changepoints_dict=None
    ),
    events=dict(
        holidays_to_model_separately=None,
        holiday_lookup_countries=[],
        holiday_pre_num_days=0,
        holiday_post_num_days=0
    ),
    autoregression=dict(
      autoreg_dict="auto"
    ),
    custom=dict(
        fit_algorithm_dict=dict(
            fit_algorithm="ridge"
        ),
        feature_sets_enabled="auto",
        extra_pred_cols=daily_interaction
    )
)

# %%
# Now to use Multistage Forecast,
# just like specifying the model components of the Simple Silverkite model,
# we need to specify the model components for Multistage Forecast.
# The Multistage Forecast configuration is specified via
# ``ModelComponentsParam.custom["multistage_forecast_configs"]``,
# which takes a list of
# `~greykite.framework.templates.multistage_forecast_template_config.MultistageForecastTemplateConfig`
# objects, each of which represents a stage of the model.
#
# The ``MultistageForecastTemplateConfig`` object for a single stage takes the following parameters:
#
#   - ``train_length``: the length of training data, for example ``"365D"``.
#     Looks back from the end of the training data and takes observations up to this limit.
#   - ``fit_length``: the length of data where fitted values are calculated.
#     Even if the training data is not the entire period, the fitted values can still be calculated
#     on the entire period. The default will be the same as the training length.
#   - ``agg_freq``: the aggregation frequency in string representation.
#     For example, "D", "H", etc. If not specified, the original frequency will be kept.
#   - ``agg_func``: the aggregation function name, default is ``"nanmean"``.
#   - ``model_template``: the model template name. This together with the ``model_components`` below
#     specify the full model, just as when using the Simple Silverkite model.
#   - ``model_components``: the model components. This together with the ``model_template`` above
#     specify the full model for a stage, just as when using the Simple Silverkite model.
#
# ``MultistageForecastTemplateConfig`` represents the flow of each stage of the model:
# taking the time series / residual,
# taking the appropriate length of training data, doing an optional aggregation,
# then training the model with the given parameters.
# Now let's define the ``MultistageForecastTemplateConfig`` object one by one.

# the daily model
daily_config = MultistageForecastTemplateConfig(
    train_length="730D",                               # use 2 years of data to train
    fit_length=None,                                   # fit on the same period as training
    agg_func="nanmean",                                # aggregation function is nanmean
    agg_freq="D",                                      # aggregation frequency is daily
    model_template=ModelTemplateEnum.SILVERKITE.name,  # the model template
    model_components=daily_model_components            # the daily model components specified above
)

# the hourly model
hourly_config = MultistageForecastTemplateConfig(
    train_length="30D",                                # use 30 days data to train
    fit_length=None,                                   # fit on the same period as training
    agg_func="nanmean",                                # aggregation function is nanmean
    agg_freq=None,                                     # None means no aggregation
    model_template=ModelTemplateEnum.SILVERKITE.name,  # the model template
    model_components=hourly_model_components           # the daily model components specified above
)

# %%
# The configurations simply go to ``ModelComponentsParam.custom["multistage_forecast_configs"]``
# as a list. We can specify the model components for Multistage Forecast as below.
# Note that all keys other than ``"custom"`` and ``"uncertainty"`` will be ignored.

model_components = ModelComponentsParam(
    custom=dict(
        multistage_forecast_configs=[daily_config, hourly_config]
    ),
    uncertainty=dict()
)

# %%
# Now we can fill in other parameters needed by
# `~greykite.framework.templates.autogen.forecast_config.ForecastConfig`.

# metadata
metadata = MetadataParam(
    time_col="ts",
    value_col="y",
    freq="H"  # the frequency should match the original data frequency
)

# evaluation period
evaluation_period = EvaluationPeriodParam(
    cv_max_splits=0,  # turn off cv for speeding up
    test_horizon=0,  # turn off test for speeding up
)

# forecast config
config = ForecastConfig(
    model_template=ModelTemplateEnum.MULTISTAGE_EMPTY.name,
    forecast_horizon=24,  # forecast 1 day ahead
    coverage=0.95,  # prediction interval is supported
    metadata_param=metadata,
    model_components_param=model_components,
    evaluation_period_param=evaluation_period
)
forecaster = Forecaster()
forecast_result = forecaster.run_forecast_config(
    df=ts.df,
    config=config
)

print(forecast_result.forecast.df_test.head())

# plot the predictions
fig = forecast_result.forecast.plot()
# interactive plot, click to zoom in
plotly.io.show(fig)

# %%
# This model is 3X times faster than training with Silverkite on the entire hourly data
# (23.5 seconds vs 79.4 seconds).
# If speed is a concern due to high frequency data with long history,
# Multistage Forecast is worth trying.
#
# .. note::
#   The order of specifying the ``MultistageForecastTemplateConfig`` objects
#   does not matter. The models will be automatically sorted with respect to
#   ``train_length`` from long to short. This is to ensure that we have enough
#   residuals from the previous model when we fit the next model.
#
# .. note::
#   The estimator expects different stage models to have different aggregation
#   frequencies. If two stages have the same aggregation frequency, an error will
#   be raised.
#
# .. note::
#   Since the models in each stage may not fit on the entire training data,
#   there could be periods at the beginning of the training period where
#   fitted values are not calculated.
#   These NA fitted values are ignored when computing evaluation metrics on the training set.
#
# The uncertainty configuration
# -----------------------------
#
# If you would like to include the uncertainty intervals,
# you can specify the ``"uncertainty"`` parameter in model components.
#
# The ``"uncertainty"`` key in ``ModelComponentsParam`` takes one key:
# ``"uncertainty_dict"``, which is a dictionary taking the following keys:
#
#   - ``"uncertainty_method"``: a string representing the uncertainty method,
#     for example, ``"simple_conditional_residuals"``.
#   - ``"params"``: a dictionary of additional parameter needed by the uncertainty method.
#
# Now let's specify a configuration of uncertainty method via the
# ``uncertainty_dict`` parameter on the ``"simple_conditional_residuals"`` model.

# specifies the ``uncertainty`` parameter
uncertainty = dict(
    uncertainty_dict=dict(
        uncertainty_method="simple_conditional_residuals",
        params=dict(
            conditional_cols=["dow"]  # conditioning on day of week
        )
    )
)

# adds to the ``ModelComponentsParam``
model_components = ModelComponentsParam(
    custom=dict(
        multistage_forecast_configs=[daily_config, hourly_config]
    ),
    uncertainty=uncertainty
)

# %%
# The Multistage Forecast Model templates
# -----------------------------------------
#
# In the example above we have seen an model template named
# ``MULTISTAGE_EMPTY``. The template is an empty template
# that must be used with specified model components.
# Any model components (``multistage_forecast_configs``) specified
# will be exactly the model parameters to be used.
# :doc:`/gallery/templates/0100_template_overview` explains how model templates
# work and how they are overridden by model components.
#
# The Multistage Forecast model also comes with the following model template:
#
#   - ``SILVERKITE_TWO_STAGE``: a two-stage model similar to the model we present above.
#     The first stage is a daily model trained on 56 * 7 days of data learning the long term effects
#     including yearly/quarterly/monthly/weekly seasonality, holidays, etc. The second stage is
#     a short term model in the original data frequency learning the daily seasonality
#     and autoregression effects. Both stages' ``model_templates`` are ``SILVERKITE``.
#     Note that this template assumes the data to be sub-daily.
#
# When you choose to use the Multistage Forecast model templates,
# you can override default values by specifying the model components.
# The overriding in Multistage Forecast works as follows:
#
#   - For each ``MultistageForecastTemplateConfig``'s overridden, there are two situations.
#
#     If the customized ``model_template`` is the same as the ``model_template`` in the default model,
#     for example, both are ``SILVERKITE``, then the customized ``model_components``
#     in the ``MultistageForecastTemplateConfig`` will be used to override the
#     ``model_components`` in the default ``MultistageForecastTemplateConfig``,
#     as overriding is done in the Silverkite template.
#
#     If the model templates are different, say ``SILVERKITE`` in the default and ``SILVERKITE_EMPTY``
#     in the customized, then both the new ``model_template`` and the new entire ``model_components``
#     will be used to replace the original ``model_template`` and ``model_components`` in the default model.
#
#     In both cases, the ``train_length``, ``fit_length``, ``agg_func`` and ``agg_freq`` will be overridden.
#
# For example, in ``SILVERKITE_TWO_STAGE``, both stages of default templates are ``SILVERKITE``.
# Consider the following example.

model_template = "SILVERKITE_TWO_STAGE"
model_components_override = ModelComponentsParam(
    custom=dict(
        multistage_forecast_configs=[
            MultistageForecastTemplateConfig(
                train_length="730D",
                fit_length=None,
                agg_func="nanmean",
                agg_freq="D",
                model_template=ModelTemplateEnum.SILVERKITE.name,
                model_components=ModelComponentsParam(
                    seasonality=dict(
                        weekly_seasonality=7
                    )
                )
            ),
            MultistageForecastTemplateConfig(
                train_length="30D",
                fit_length=None,
                agg_func="nanmean",
                agg_freq=None,
                model_template=ModelTemplateEnum.SILVERKITE_EMPTY.name,
                model_components=ModelComponentsParam(
                    seasonality=dict(
                        daily_seasonality=10
                    )
                )
            )
        ]
    )
)


# %%
# The first model has the same model template ``SILVERKITE`` as the default model template,
# so in ``model_components``, only the weekly seasonality parameter will be used to override
# the default weekly seasonality in ``SILVERKITE`` model template.
# The second model has a different model template ``SILVERKITE_EMPTY``.
# Then the second model will use exactly the model template and model components specified in
# the customized parameters.
#
# This design is to maximize the flexibility to override an existing Multistage Forecast model template.
# However, if you fully know what your configuration will be for each stage of the model,
# the suggestion is to use ``MULTISTAGE_EMPTY`` and specify your own configurations.
#
# .. note::
#   If the customized model components contains fewer models than provided by the model template,
#   for example, only 1 stage model is customized when using ``SILVERKITE_TWO_STAGE``.
#   The ` customized ``MultistageForecastTemplateConfig`` will
#   be used to override the first model in the ``SILVERKITE_TWO_STAGE``,
#   and the 2nd model in ``SILVERKITE_TWO_STAGE`` will be appended to the end of the first overridden model.
#   Oppositely, if the number of customized models is 3, the extra customized model will be appended to the
#   end of the 2 models in ``SILVERKITE_TWO_STAGE``.
#
# Grid search
# -----------
#
# See :doc:`/gallery/quickstart/03_benchmark/0100_grid_search` for an introduction to the grid search functionality
# of Greykite. Grid search is also supported in Multistage Forecast.
# The way to specify grid search is similar to specifying grid search in Simple Silverkite model:
# please specify the grids in each stage of the model's ``model_components``.
# The Multistage Forecast model will automatically recognize the grids
# and formulate the full grids across all models.
# This design is to keep the behavior the same as using grid search in Silverkite models.
#
# For example, the following model components specifies two stage models.
# The first model has a grid on weekly seasonality with candidates 3 and 5.
# The second model has a grid on daily seasonality with candidates 10 and 12.
# The Multistage Forecast model will automatically combine the grids from the two models,
# and generate a grid of size 4.

model_components_grid = ModelComponentsParam(
    custom=dict(
        multistage_forecast_configs=[
            MultistageForecastTemplateConfig(
                train_length="730D",
                fit_length=None,
                agg_func="nanmean",
                agg_freq="D",
                model_template=ModelTemplateEnum.SILVERKITE.name,
                model_components=ModelComponentsParam(
                    seasonality=dict(
                        weekly_seasonality=[3, 5]
                    )
                )
            ),
            MultistageForecastTemplateConfig(
                train_length="30D",
                fit_length=None,
                agg_func="nanmean",
                agg_freq=None,
                model_template=ModelTemplateEnum.SILVERKITE.name,
                model_components=ModelComponentsParam(
                    seasonality=dict(
                        daily_seasonality=[10, 12]
                    )
                )
            )
        ]
    )
)
