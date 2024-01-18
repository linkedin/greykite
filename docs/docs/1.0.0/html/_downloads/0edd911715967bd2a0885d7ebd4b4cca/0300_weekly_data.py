"""
Example for weekly data
=======================

This is a basic example for weekly data using Silverkite.
Note that here we are fitting a few simple models and the goal is not to optimize
the results as much as possible.
"""

import warnings
from collections import defaultdict

import plotly
import pandas as pd

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.framework.benchmark.data_loader_ts import DataLoader
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.utils.result_summary import summarize_grid_search_results

warnings.filterwarnings("ignore")

# %%
# Loads weekly dataset into ``UnivariateTimeSeries``.
dl = DataLoader()
agg_func = {"count": "sum"}
df = dl.load_bikesharing(agg_freq="weekly", agg_func=agg_func)
# In this dataset the first week and last week's data are incomplete, therefore we drop it
df.drop(df.head(1).index,inplace=True)
df.drop(df.tail(1).index,inplace=True)
df.reset_index(drop=True)
ts = UnivariateTimeSeries()
ts.load_data(
    df=df,
    time_col="ts",
    value_col="count",
    freq="W-MON")
print(ts.df.head())

# %%
# Exploratory Data Analysis (EDA)
# -------------------------------
# After reading in a time series, we could first do some exploratory data analysis.
# The `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries` class is
# used to store a timeseries and perform EDA.

# %%
# A quick description of the data can be obtained as follows.
print(ts.describe_time_col())
print(ts.describe_value_col())

# %%
# Let's plot the original timeseries.
# (The interactive plot is generated by ``plotly``: **click to zoom!**)
fig = ts.plot()
plotly.io.show(fig)

# %%
# Exploratory plots can be plotted to reveal the time series's properties.
# Monthly overlay plot can be used to inspect the annual patterns.
# This plot overlays various years on top of each other.
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="month",
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    center_values=True,
    overlay_label_time_feature="year",  # splits overlays by year
    overlay_style={"line": {"width": 1}, "opacity": 0.5},
    xlabel="Month",
    ylabel=ts.original_value_col,
    title="Yearly seasonality by year (centered)",
)
plotly.io.show(fig)

# %%
# Weekly overlay plot.
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="woy",
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    center_values=True,
    overlay_label_time_feature="year",  # splits overlays by year
    overlay_style={"line": {"width": 1}, "opacity": 0.5},
    xlabel="Week of year",
    ylabel=ts.original_value_col,
    title="Yearly seasonality by year (centered)",
)
plotly.io.show(fig)

# %%
# Fit Greykite Models
# -------------------
# After some exploratory data analysis, let's specify the model parameters and fit a Greykite model.

# %%
# Specify common metadata.
forecast_horizon = 4  # Forecast 4 weeks
time_col = TIME_COL  # "ts"
value_col = VALUE_COL  # "y"
metadata = MetadataParam(
    time_col=time_col,
    value_col=value_col,
    freq="W-MON",  # Optional, the model will infer the data frequency
)

# %%
# Specify common evaluation parameters.
# Set minimum input data for training.
cv_min_train_periods = 52 * 2
# Let CV use most recent splits for cross-validation.
cv_use_most_recent_splits = True
# Determine the maximum number of validations.
cv_max_splits = 6
evaluation_period = EvaluationPeriodParam(
    test_horizon=forecast_horizon,
    cv_horizon=forecast_horizon,
    periods_between_train_test=0,
    cv_min_train_periods=cv_min_train_periods,
    cv_expanding_window=True,
    cv_use_most_recent_splits=cv_use_most_recent_splits,
    cv_periods_between_splits=None,
    cv_periods_between_train_test=0,
    cv_max_splits=cv_max_splits,
)

# %%
# Let's also define a helper function that generates the model results summary and plots.
def get_model_results_summary(result):
    """Generates model results summary.

    Parameters
    ----------
    result : `ForecastResult`
        See :class:`~greykite.framework.pipeline.pipeline.ForecastResult` for documentation.

    Returns
    -------
    Prints out model coefficients, cross-validation results, overall train/test evalautions.
    """
    # Get the useful fields from the forecast result
    model = result.model[-1]
    backtest = result.backtest
    grid_search = result.grid_search

    # Check model coefficients / variables
    # Get model summary with p-values
    print(model.summary())

    # Get cross-validation results
    cv_results = summarize_grid_search_results(
        grid_search=grid_search,
        decimals=2,
        cv_report_metrics=None,
        column_order=[
            "rank", "mean_test", "split_test", "mean_train", "split_train",
            "mean_fit_time", "mean_score_time", "params"])
    # Transposes to save space in the printed output
    print("================================= CV Results ==================================")
    print(cv_results.transpose())

    # Check historical evaluation metrics (on the historical training/test set).
    backtest_eval = defaultdict(list)
    for metric, value in backtest.train_evaluation.items():
        backtest_eval[metric].append(value)
        backtest_eval[metric].append(backtest.test_evaluation[metric])
    metrics = pd.DataFrame(backtest_eval, index=["train", "test"]).T
    print("=========================== Train/Test Evaluation =============================")
    print(metrics)

# %%
# Fit a simple model without autoregression.
# The the most important model parameters are specified through ``ModelComponentsParam``.
# The ``extra_pred_cols`` is used to specify growth and annual seasonality
# Growth is modelled with both "ct_sqrt", "ct1" for extra flexibility as we have
# longterm data and ridge regularization will avoid over-fitting the trend.
# The yearly seasonality is modelled using Fourier series. In the ``ModelComponentsParam``,
# we can specify the order of that - the higher the order is, the more flexible pattern
# the model could capture. Usually one can try integers between 10 and 50.

autoregression = None
extra_pred_cols = ["ct1", "ct_sqrt", "ct1:C(month, levels=list(range(1, 13)))"]

# Specify the model parameters
model_components = ModelComponentsParam(
    autoregression=autoregression,
    seasonality={
        "yearly_seasonality": 25,
        "quarterly_seasonality": 0,
        "monthly_seasonality": 0,
        "weekly_seasonality": 0,
        "daily_seasonality": 0
    },
    changepoints={
        'changepoints_dict': {
            "method": "auto",
            "resample_freq": "7D",
            "regularization_strength": 0.5,
            "potential_changepoint_distance": "14D",
            "no_changepoint_distance_from_end": "60D",
            "yearly_seasonality_order": 25,
            "yearly_seasonality_change_freq": None,
        },
        "seasonality_changepoints_dict": None
    },
    events={
        "holiday_lookup_countries": []
    },
    growth={
        "growth_term": None
    },
    custom={
        'feature_sets_enabled': False,
        'fit_algorithm_dict': dict(fit_algorithm='ridge'),
        'extra_pred_cols': extra_pred_cols,
    }
)

forecast_config = ForecastConfig(
    metadata_param=metadata,
    forecast_horizon=forecast_horizon,
    coverage=0.95,
    evaluation_period_param=evaluation_period,
    model_components_param=model_components
)

# Run the forecast model
forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=ts.df,
    config=forecast_config
)

# %%
# Let's check the model results summary and plots.
get_model_results_summary(result)

# %%
# Fit/backtest plot:
fig = result.backtest.plot()
plotly.io.show(fig)

# %%
# Forecast plot:
fig = result.forecast.plot()
plotly.io.show(fig)

# %%
# The components plot:
fig = result.forecast.plot_components()
plotly.io.show(fig)

# %%
# Fit a simple model with autoregression.
# This is done by specifying the ``autoregression`` parameter in ``ModelComponentsParam``.
# Note that the auto-regressive structure can be customized further depending on your data.
autoregression = {
    "autoreg_dict": {
        "lag_dict": {"orders": [1]},  # Only use lag-1
        "agg_lag_dict": None
    }
}
extra_pred_cols = ["ct1", "ct_sqrt", "ct1:C(month, levels=list(range(1, 13)))"]

# Specify the model parameters
model_components = ModelComponentsParam(
    autoregression=autoregression,
    seasonality={
        "yearly_seasonality": 25,
        "quarterly_seasonality": 0,
        "monthly_seasonality": 0,
        "weekly_seasonality": 0,
        "daily_seasonality": 0
    },
    changepoints={
        'changepoints_dict': {
            "method": "auto",
            "resample_freq": "7D",
            "regularization_strength": 0.5,
            "potential_changepoint_distance": "14D",
            "no_changepoint_distance_from_end": "60D",
            "yearly_seasonality_order": 25,
            "yearly_seasonality_change_freq": None,
        },
        "seasonality_changepoints_dict": None
    },
    events={
        "holiday_lookup_countries": []
    },
    growth={
        "growth_term": None
    },
    custom={
        'feature_sets_enabled': False,
        'fit_algorithm_dict': dict(fit_algorithm='ridge'),
        'extra_pred_cols': extra_pred_cols,
    }
)

forecast_config = ForecastConfig(
    metadata_param=metadata,
    forecast_horizon=forecast_horizon,
    coverage=0.95,
    evaluation_period_param=evaluation_period,
    model_components_param=model_components
)

# Run the forecast model
forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=ts.df,
    config=forecast_config
)

# %%
# Let's check the model results summary and plots.
get_model_results_summary(result)

# %%
# Fit/backtest plot:
fig = result.backtest.plot()
plotly.io.show(fig)

# %%
# Forecast plot:
fig = result.forecast.plot()
plotly.io.show(fig)

# %%
# The components plot:
fig = result.forecast.plot_components()
plotly.io.show(fig)

# %%
# Fit a greykite model with autoregression and forecast one-by-one. Forecast one-by-one is only
# used when autoregression is set to "auto", and it can be enable by setting ``forecast_one_by_one=True``
# in
# Without forecast one-by-one, the lag order in autoregression has to be greater
# than the forecast horizon in order to avoid simulation (which leads to less accuracy).
# The advantage of turning on forecast_one_by_one is to improve the forecast accuracy by breaking
# the forecast horizon to smaller steps, fitting multiple models using immediate lags.
# Note that the forecast one-by-one option may slow down the training.
autoregression = {
    "autoreg_dict": "auto"
}
extra_pred_cols = ["ct1", "ct_sqrt", "ct1:C(month, levels=list(range(1, 13)))"]
forecast_one_by_one = True

# Specify the model parameters
model_components = ModelComponentsParam(
    autoregression=autoregression,
    seasonality={
        "yearly_seasonality": 25,
        "quarterly_seasonality": 0,
        "monthly_seasonality": 0,
        "weekly_seasonality": 0,
        "daily_seasonality": 0
    },
    changepoints={
        'changepoints_dict': {
            "method": "auto",
            "resample_freq": "7D",
            "regularization_strength": 0.5,
            "potential_changepoint_distance": "14D",
            "no_changepoint_distance_from_end": "60D",
            "yearly_seasonality_order": 25,
            "yearly_seasonality_change_freq": None,
        },
        "seasonality_changepoints_dict": None
    },
    events={
        "holiday_lookup_countries": []
    },
    growth={
        "growth_term": None
    },
    custom={
        'feature_sets_enabled': False,
        'fit_algorithm_dict': dict(fit_algorithm='ridge'),
        'extra_pred_cols': extra_pred_cols,
    }
)

forecast_config = ForecastConfig(
    metadata_param=metadata,
    forecast_horizon=forecast_horizon,
    coverage=0.95,
    evaluation_period_param=evaluation_period,
    model_components_param=model_components,
    forecast_one_by_one=forecast_one_by_one
)

# Run the forecast model
forecaster = Forecaster()
result =  forecaster.run_forecast_config(
    df=ts.df,
    config=forecast_config
)

# %%
# Let's check the model results summary and plots. Here the forecast_one_by_one option fits 4 models
# for each step, hence 4 model summaries are printed, and 4 components plots are generated.
get_model_results_summary(result)

# %%
# Fit/backtest plot:
fig = result.backtest.plot()
plotly.io.show(fig)

# %%
# Forecast plot:
fig = result.forecast.plot()
plotly.io.show(fig)

# %%
# The components plot:
figs = result.forecast.plot_components()
for fig in figs:
    plotly.io.show(fig)
