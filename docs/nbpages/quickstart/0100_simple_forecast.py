"""
Simple Forecast
===============

You can create and evaluate a forecast with just a few lines of code.

Provide your timeseries as a pandas dataframe with timestamp and value.

For example, to forecast daily sessions data, your dataframe could look like this:

.. code-block:: python

    import pandas as pd
    df = pd.DataFrame({
        "date": ["2020-01-08-00", "2020-01-09-00", "2020-01-10-00"],
        "sessions": [10231.0, 12309.0, 12104.0]
    })

The time column can be any format recognized by `pandas.to_datetime`.

In this example, we'll load a dataset representing ``log(daily page views)``
on the Wikipedia page for Peyton Manning.
It contains values from 2007-12-10 to 2016-01-20. More dataset info
`here <https://facebook.github.io/prophet/docs/quick_start.html>`_.
"""

from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly

from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results

# Loads dataset into pandas DataFrame
dl = DataLoader()
df = dl.load_peyton_manning()

# specify dataset information
metadata = MetadataParam(
    time_col="ts",  # name of the time column ("date" in example above)
    value_col="y",  # name of the value column ("sessions" in example above)
    freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
              # Any format accepted by `pandas.date_range`
)

# %%
# Create a forecast
# -----------------
# You can choose from many available
# models (see :doc:`/pages/stepbystep/0100_choose_model`).
#
# In this example, we choose the "AUTO" model template,
# which uses the Silverkite algorithm with automatic parameter configuration
# given the input data frequency, forecast horizon and evaluation configs.
# We recommend starting with the "AUTO" template for most use cases.
forecaster = Forecaster()  # Creates forecasts and stores the result
result = forecaster.run_forecast_config(  # result is also stored as `forecaster.forecast_result`.
    df=df,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.AUTO.name,
        forecast_horizon=365,  # forecasts 365 steps ahead
        coverage=0.95,         # 95% prediction intervals
        metadata_param=metadata
    )
)

# %%
# Check results
# -------------
# The output of ``run_forecast_config`` is a dictionary that contains
# the future forecast, historical forecast performance, and
# the original timeseries.

# %%
# Timeseries
# ^^^^^^^^^^
# Let's plot the original timeseries.
# ``run_forecast_config`` returns this as ``ts``.
#
# (The interactive plot is generated by ``plotly``: **click to zoom!**)
ts = result.timeseries
fig = ts.plot()
plotly.io.show(fig)

# %%
# Cross-validation
# ^^^^^^^^^^^^^^^^
# By default, ``run_forecast_config`` provides historical evaluation,
# so you can see how the forecast performs on past data.
# This is stored in ``grid_search`` (cross-validation splits)
# and ``backtest`` (holdout test set).
#
# Let's check the cross-validation results.
# By default, all metrics in `~greykite.common.evaluation.ElementwiseEvaluationMetricEnum`
# are computed on each CV train/test split.
# The configuration of CV evaluation metrics can be found at
# `Evaluation Metric <../../pages/stepbystep/0400_configuration.html#evaluation-metric>`_.
# Below, we show the Mean Absolute Percentage Error (MAPE)
# across splits (see `~greykite.framework.utils.result_summary.summarize_grid_search_results`
# to control what to show and for details on the output columns).
grid_search = result.grid_search
cv_results = summarize_grid_search_results(
    grid_search=grid_search,
    decimals=2,
    # The below saves space in the printed output. Remove to show all available metrics and columns.
    cv_report_metrics=None,
    column_order=["rank", "mean_test", "split_test", "mean_train", "split_train", "mean_fit_time", "mean_score_time", "params"])
# Transposes to save space in the printed output
cv_results["params"] = cv_results["params"].astype(str)
cv_results.set_index("params", drop=True, inplace=True)
cv_results.transpose()

# %%
# Backtest
# ^^^^^^^^
# Let's plot the historical forecast on the holdout test set.
# You can zoom in to see how it performed in any given period.
backtest = result.backtest
fig = backtest.plot()
plotly.io.show(fig)

# %%
# You can also check historical evaluation metrics (on the historical training/test set).
backtest_eval = defaultdict(list)
for metric, value in backtest.train_evaluation.items():
    backtest_eval[metric].append(value)
    backtest_eval[metric].append(backtest.test_evaluation[metric])
metrics = pd.DataFrame(backtest_eval, index=["train", "test"]).T
metrics

# %%
# Forecast
# ^^^^^^^^
# The ``forecast`` attribute contains the forecasted result.
# Just as for ``backtest``, you can plot the result or
# see the evaluation metrics.
#
# Let's plot the forecast (trained on all data):
forecast = result.forecast
fig = forecast.plot()
plotly.io.show(fig)

# %%
# The forecasted values are available in ``df``.
forecast.df.head().round(2)

# %%
# Model Diagnostics
# ^^^^^^^^^^^^^^^^^
# The component plot shows how your dataset's trend,
# seasonality, event / holiday and other patterns are handled in the model.
# When called, with defaults, function displays three plots: 1) components of the model,
# 2) linear trend and changepoints, and 3) the residuals of the model and
# smoothed estimates of the residuals.  By clicking different legend entries, the visibility of
# lines in each plot can be toggled on or off.
fig = forecast.plot_components()
plotly.io.show(fig)     # fig.show() if you are using "PROPHET" template

# %%
# Model summary allows inspection of individual model terms.
# Check parameter estimates and their significance for insights
# on how the model works and what can be further improved.
summary = result.model[-1].summary()  # -1 retrieves the estimator from the pipeline
print(summary)

# %%
# Apply the model
# ^^^^^^^^^^^^^^^^^
# The trained model is available as a fitted `sklearn.pipeline.Pipeline`.
model = result.model
model

# %%
# You can take this model and forecast on any date range
# by passing a new dataframe to predict on. The
# `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.make_future_dataframe`
# convenience function can be used to create this dataframe.
# Here, we predict the next 4 periods after the model's train end date.
#
# .. note::
#   The dataframe passed to .predict() must have the same columns
#   as the ``df`` passed to ``run_forecast_config`` above, including
#   any regressors needed for prediction. The ``value_col`` column
#   should be included with values set to `np.nan`.
future_df = result.timeseries.make_future_dataframe(
    periods=4,
    include_history=False)
future_df

# %%
# Call .predict() to compute predictions
model.predict(future_df)

# %%
# What's next?
# ------------
# If you're satisfied with the forecast performance, you're done!
#
# For a complete example of how to tune this forecast, see
# :doc:`/gallery/tutorials/0100_forecast_tutorial`.
#
# Besides the component plot, we offer additional tools to
# help you improve your forecast and understand the result.
#
# See the following guides:
#
# * :doc:`/gallery/quickstart/01_exploration/0100_changepoint_detection`
# * :doc:`/gallery/quickstart/01_exploration/0300_seasonality_plots`
# * :doc:`/gallery/quickstart/02_interpretability/0100_model_summary`
# * :doc:`/gallery/quickstart/03_benchmark/0100_grid_search`
#
# For example, for this dataset, you could add changepoints to
# handle the change in trend around 2014 and avoid the overprediction
# issue seen in the backtest plot.
#
# Or you might want to try a different model template.
# Model templates bundle an algorithm with recommended
# hyperparameters. The template that works best for you depends on
# the data characteristics and forecast requirements
# (e.g. short / long forecast horizon). We recommend trying
# a few and tuning the ones that look promising.
# All model templates are available through the same forecasting
# and tuning interface shown here.
#
# For details about the model templates and how to set model
# components, see the following guides:
#
# * :doc:`/gallery/templates/0100_template_overview`
# * :doc:`/pages/stepbystep/0000_stepbystep`
