"""
Benchmarking
============

You can easily compare predictive performance of multiple algorithms such as
``Silverkite`` and ``Prophet`` using the
`~greykite.framework.benchmark.benchmark_class.BenchmarkForecastConfig` class.
In this tutorial we describe the step-by-step process of defining, running and monitoring a benchmark.
We also demonstrate how to use the class functions to compute and plot errors for multiple models.
"""

from dataclasses import replace

import plotly
import plotly.graph_objects as go

from greykite.common.evaluation import EvaluationMetricEnum
from greykite.framework.benchmark.benchmark_class import BenchmarkForecastConfig
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.sklearn.cross_validation import RollingTimeSeriesSplit

# %%
# Load the data
# -------------
# First load your dataset into a pandas dataframe.
# We will use the peyton-manning dataset as a running example.

# Loads dataset into UnivariateTimeSeries
dl = DataLoaderTS()
ts = dl.load_peyton_manning_ts()
df = ts.df  # cleaned pandas.DataFrame

# %%
# Define the Configs
# ------------------
# We specify the models we want to benchmark via the ``configs`` parameter.
# In this example we will benchmark 1 ``Prophet`` and 2 different ``Silverkite`` models.
# We first define the common components of the models
# such as ``MetadataParam`` and ``EvaluationMetricParam``, and then update the configuration to specify
# individual models.

## Define common components  of the configs
# Specifies dataset information
metadata = MetadataParam(
    time_col="ts",   # name of the time column
    value_col="y",   # name of the value column
    freq="D"         # "H" for hourly, "D" for daily, "W" for weekly, etc.
)

# Defines number of periods to forecast into the future
forecast_horizon = 7

# Specifies intended coverage of the prediction interval
coverage = 0.95

# Defines the metrics to evaluate the forecasts
# We use Mean Absolute Percent Error (MAPE) in this tutorial
evaluation_metric = EvaluationMetricParam(
    cv_selection_metric=EvaluationMetricEnum.MeanAbsolutePercentError.name,
    cv_report_metrics=None
)

# Defines the cross-validation config within pipeline
evaluation_period = EvaluationPeriodParam(
    cv_max_splits=1,  # Benchmarking n_splits is defined in tscv, here we don't need split to choose parameter sets
    periods_between_train_test=0,
)

# Defines parameters related to grid-search computation
computation = ComputationParam(
    hyperparameter_budget=None,
    n_jobs=-1,  # to debug, change to 1 for more informative error messages
    verbose=3)

# Defines common components across all the configs
# ``model_template`` and ``model_components_param`` changes between configs
common_config = ForecastConfig(
    metadata_param=metadata,
    forecast_horizon=forecast_horizon,
    coverage=coverage,
    evaluation_metric_param=evaluation_metric,
    evaluation_period_param=evaluation_period,
    computation_param=computation,
)

# %%
# Now we update ``common_config`` to specify the individual models.

# Defines ``Prophet`` model template with custom seasonality
model_components = ModelComponentsParam(
    seasonality={
            "seasonality_mode": ["additive"],
            "yearly_seasonality": ["auto"],
            "weekly_seasonality": [True],
        },
        growth={
            "growth_term": ["linear"]
        }
)
param_update = dict(
    model_template=ModelTemplateEnum.PROPHET.name,
    model_components_param=model_components
)
Prophet = replace(common_config, **param_update)

# Defines ``Silverkite`` model template with automatic autoregression
# and changepoint detection
model_components = ModelComponentsParam(
    changepoints={
        "changepoints_dict": {
            "method": "auto",
        }
    },
    autoregression={
        "autoreg_dict": "auto"
    }
)
param_update = dict(
    model_template=ModelTemplateEnum.SILVERKITE.name,
    model_components_param=model_components
)
Silverkite_1 = replace(common_config, **param_update)

# Defines ``Silverkite`` model template via string encoding
param_update = dict(
    model_template="DAILY_SEAS_NMQM_GR_LINEAR_CP_NM_HOL_SP2_FEASET_AUTO_ALGO_RIDGE_AR_AUTO_DSI_AUTO_WSI_AUTO",
    model_components_param=None
)
Silverkite_2 = replace(common_config, **param_update)

# Define the list of configs to benchmark
# The dictionary keys will be used to store the benchmark results
configs = {
    "Prophet": Prophet,
    "SK_1": Silverkite_1,
    "SK_2": Silverkite_2,
}

# %%
# Define the Cross-Validation (CV)
# --------------------------------
# In time-series forecasting we use a Rolling Window CV.
# You can easily define it by using
# `~greykite.sklearn.cross_validation.RollingTimeSeriesSplit` class.
# The CV parameters depend on the data frequency,
# forecast horizon as well as the speed of the models.
# See ``Benchmarking documentation`` for guidance on how
# to choose CV parameters for your use case.

# Define the benchmark folds
# CV parameters are changed for illustration purpose
tscv = RollingTimeSeriesSplit(
    forecast_horizon=forecast_horizon,
    min_train_periods=2 * 365,
    expanding_window=True,
    use_most_recent_splits=True,
    periods_between_splits=5,
    periods_between_train_test=0,
    max_splits=4)  # reduced to 4 from 16 for faster runtime

# Print the train, test split for benchmark folds
for split_num, (train, test) in enumerate(tscv.split(X=df)):
    print(split_num, train, test)

# %%
# Run the Benchmark
# -----------------
# To start the benchmarking procedure execute its ``run`` method.
#
# If you get an error message at this point, then there is a compatibility issue between your
# benchmark inputs. Check :ref:`Debugging the Benchmark` section for instructions on how to derive valid inputs.

bm = BenchmarkForecastConfig(df=df, configs=configs, tscv=tscv)
bm.run()

# %%
# Monitor the Benchmark
# ---------------------
# During benchmarking a couple of color coded progress bars are displayed to inform the user of the
# advancement of the entire process. The first bar displays ``config`` level information, while
# the second bar displays split level information for the current ``config``.
# See example in `Benchmarking documentation`.
#
# On the left side of the progress bar, it shows which ``config``/ split is currently being
# benchmarked and progress within that level as a percentage.
#
# On the right side, the user can see how many ``configs``/ splits have been benchmarked
# and how many are remaining. Additionally, this bar also displays elapsed time and remaining runtime
# for the corresponding level.

# %%
# Benchmark Output
# ----------------
# The output of a successful benchmark procedure is stored as a nested dictionary under the class attribute
# ``result``. For details on the structure of this tree check
# ``Benchmarking documentation``.
#
# You can extract any specific information by navigating this tree. For example, you can
# check the summary and component plot of any ``config``.

# Check summary of SK_1 model on first fold
model = bm.result["SK_2"]["rolling_evaluation"]["split_0"]["pipeline_result"].model
model[-1].summary(max_colwidth=30)

# %%

# Check component plot of SK_2 on second fold
model = bm.result["SK_2"]["rolling_evaluation"]["split_1"]["pipeline_result"].model
fig = model[-1].plot_components()
plotly.io.show(fig)


# %%
# Compare forecasts
# ^^^^^^^^^^^^^^^^^
# To obtain forecasts run the ``extract_forecasts`` method. You only need to run this once.

bm.extract_forecasts()

# %%
# This method does two things.
#
# * For every ``config``, it gathers forecast results across rolling windows and stores it
#   as a dataframe in ``rolling_forecast_df`` under the ``config`` key. This helps in comparing forecasts
#   and prediction accuracy across splits for the ``config``.

# Forecast across rolling windows for SK_1
forecast_sk_1 = bm.result["SK_1"]["rolling_forecast_df"]
forecast_sk_1.head()

# %%
# * Concatenates ``rolling_forecast_df`` for all the ``configs`` and stores it as a dataframe in the
#   class attribute ``forecasts``. This helps in comparing forecasts and prediction accuracies across ``configs``.

# Forecasts across configs
bm.forecasts.head()

# %%
# For any ``config`` you can plot forecasts across splits. This allows you to quickly check if there is
# any particular time window where the test performance drops. The forecasts for adjacent folds will
# overlap if the time windows of the corresponding folds overlap.

fig = bm.plot_forecasts_by_config(config_name="SK_1")
plotly.io.show(fig)

# %%
# The importance of this function becomes more significant when assessing a models performance over a
# longer period e.g. a year or multiple years. You can quickly catch if models test performance drops
# during weekends, specific months or holiday seasons.
#
# You can also compare forecasts from multiple ``configs`` by ``forecast_step`` which is
# defined as any number between 1 and ``forecast_horizon``. This is useful in forecasts with longer
# forecast horizons to check if the forecast volatility changes over time.

fig = bm.plot_forecasts_by_step(forecast_step=3)
plotly.io.show(fig)

# %%
# Compare Errors
# ^^^^^^^^^^^^^^
# You can compare the predictive performance of your models via multiple evaluation metrics.
# In this example we will use MAPE and RMSE, but you can use any metric from ``EvaluationMetricEnum``.

metric_dict = {
    "MAPE": EvaluationMetricEnum.MeanAbsolutePercentError,
    "RMSE": EvaluationMetricEnum.RootMeanSquaredError
}

# %%
# Non Grouping Errors
# ^^^^^^^^^^^^^^^^^^^
# To compare evaluation metrics without any grouping use ``get_evaluation_metrics``.
# The output shows metric values by ``config`` and ``split``. We can group by ``config_name`` to get
# metric values aggregated across all folds.

# Compute evaluation metrics
evaluation_metrics_df = bm.get_evaluation_metrics(metric_dict=metric_dict)
# Aggregate by model across splits
error_df = evaluation_metrics_df.drop(columns=["split_num"]).groupby("config_name").mean()
error_df

# %%

# Visualize
fig = bm.plot_evaluation_metrics(metric_dict)
plotly.io.show(fig)

# %%
# Train MAPE is high because some values in training dataset are close to 0.
#
# You can also compare the predictive accuracy across splits for any model from ``configs``.
# This allows you to check if the model performance varies significantly across time periods.

# Compute evaluation metrics for a single config
evaluation_metrics_df = bm.get_evaluation_metrics(metric_dict=metric_dict, config_names=["SK_1"])
# Aggregate by split number
error_df = evaluation_metrics_df.groupby("split_num").mean()
error_df.head()

# %%

# Visualize
title = "Average evaluation metric across rolling windows"
data = []
# Each row (index) is a config. Adds each row to the bar plot.
for index in error_df.index:
    data.append(
        go.Bar(
            name=index,
            x=error_df.columns,
            y=error_df.loc[index].values
        )
    )
layout = go.Layout(
    xaxis=dict(title=None),
    yaxis=dict(title="Metric Value"),
    title=title,
    title_x=0.5,
    showlegend=True,
    barmode="group",
)
fig = go.Figure(data=data, layout=layout)
plotly.io.show(fig)

# %%
# Grouping Errors
# ^^^^^^^^^^^^^^^
# To compare evaluation metrics with grouping use ``get_grouping_evaluation_metrics``.
# This allows you to group the error values by time features such as day of week, month etc.

# Compute grouped evaluation metrics
grouped_evaluation_df = bm.get_grouping_evaluation_metrics(
    metric_dict=metric_dict,
    which="test",
    groupby_time_feature="str_dow")
# Aggregate by split number
error_df = grouped_evaluation_df.groupby(["str_dow", "config_name"]).mean()
error_df

# %%

# Visualize
fig = bm.plot_grouping_evaluation_metrics(
    metric_dict=metric_dict,
    which="test",
    groupby_time_feature="str_dow")
plotly.io.show(fig)

# %%
# As you can see all the models have higher MAPE and RMSE during weekends. That means adding
# ``is_weekend`` indicator to the models will help.
#
# Compare runtimes
# ^^^^^^^^^^^^^^^^
# You can compare and visualize runtimes of the models using the following codes.

# Compute runtimes
runtime_df = bm.get_runtimes()
# Aggregate across splits
runtimes_df = runtime_df.drop(columns=["split_num"]).groupby("config_name").mean()
runtimes_df

# %%

# Visualize
fig = bm.plot_runtimes()
plotly.io.show(fig)

# %%
# You can see ``Silverkite`` models run almost 3 times faster compared to ``Prophet``.
#
# Debugging the Benchmark
# -----------------------
# When the `run` method is called, the input ``configs`` are first assessed of
# their suitability for a cohesive benchmarking procedure via the ``validate`` method.
# This is done prior to passing the ``configs`` to the forecasting pipeline to save wasted
# computing time for the user.
# Though not necessary, the user is encouraged to use ``validate`` for debugging.
#
# The ``validate`` method runs a series of checks to ensure that
#
# * The ``configs`` are compatible among themselves. For example, it checks if all the ``configs``
#   have the same ``forecast horizon``.
# * The ``configs`` are compatible with the CV schema. For example, ``forecast_horizon`` and
#   ``periods_between_train_test`` parameters of ``configs`` are
#   matched against that of the ``tscv``.
#
# Note that the ``validate`` method does not guarantee that the models will execute properly
# while in the pipeline. It is a good idea to do a test run on a smaller data and/ or smaller
# number of splits before running the full procedure.
#
# In the event of a mismatch a ``ValueError`` is raised with informative error messages
# to help the user in debugging. Some examples are provided below.
#
# Error due to incompatible model components in config
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# regressor_cols is not part of Prophet's model components
model_components=ModelComponentsParam(
    regressors={
        "regressor_cols": ["regressor1", "regressor2", "regressor_categ"]
    }
)
invalid_prophet = replace(Prophet, model_components_param=model_components)
invalid_configs = {"invalid_prophet": invalid_prophet}
bm = BenchmarkForecastConfig(df=df, configs=invalid_configs, tscv=tscv)
try:
    bm.validate()
except ValueError as err:
    print(err)

# %%
# Error due to wrong template name
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# model template name is not part of TemplateEnum, thus invalid
unknown_template = replace(Prophet, model_template="SOME_TEMPLATE")
invalid_configs = {"unknown_template": unknown_template}
bm = BenchmarkForecastConfig(df=df, configs=invalid_configs, tscv=tscv)
try:
    bm.validate()
except ValueError as err:
    print(err)

# %%
# Error due to different forecast horizons in configs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# the configs are valid by themselves, however incompatible for
# benchmarking as these have different forecast horizons
Prophet_forecast_horizon_30 = replace(Prophet, forecast_horizon=30)
invalid_configs = {
    "Prophet": Prophet,
    "Prophet_30": Prophet_forecast_horizon_30
}
bm = BenchmarkForecastConfig(df=df, configs=invalid_configs, tscv=tscv)
try:
    bm.validate()
except ValueError as err:
    print(err)

# %%
# Error due to different forecast horizons in config and tscv
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Error due to different forecast horizons in config and tscv
tscv = RollingTimeSeriesSplit(forecast_horizon=15)
bm = BenchmarkForecastConfig(df=df, configs=configs, tscv=tscv)
try:
    bm.validate()
except ValueError as err:
    print(err)
