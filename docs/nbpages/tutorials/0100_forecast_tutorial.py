"""
Tune your first forecast model
==============================

This is a basic tutorial for creating and tuning a forecast model.
It is intended to provide a basic sense of a forecast process without
assuming background knowledge in forecasting.

You can use the ``PROPHET`` or ``SILVERKITE`` model.
In this tutorial, we focus on ``SILVERKITE``.
However, the basic ideas of tuning are similar to both models.
You may see detailed information about ``PROPHET`` at
`Prophet <../../pages/model_components/0100_introduction.html#prophet>`_.


``SILVERKITE`` decomposes time series into various components, and it
creates time-based features, autoregressive features,
together with user-provided features such as macro-economic features
and their interactions, then performs a machine learning regression
model to learn the relationship between the time series and these
features. The forecast is based on the learned relationship and
the future values of these features. Therefore, including the correct
features is the key to success.

Common features include:

    Datetime derivatives:
        Including features derived from datetime such as ``day of year``,
        ``hour of day``, ``weekday``, ``is_weekend`` and etc.
        These features are useful in capturing special patterns.
        For example, the patterns of weekdays and weekends are different
        for most business related time series, and this can be modeled with ``is_weekend``.
    Growth:
        First defines the basic feature ``ct1`` that counts how
        long has passed in terms of years (could be fraction)
        since the first day of training data.
        For example, if the training data starts with "2018-01-01",
        then the date has ``ct1=0.0``, and "2018-01-02" has ``ct1=1/365``.
        "2019-01-01" has ``ct1=1.0``. This ``ct1`` can be as granular
        as needed. A separate growth function can be applied to ``ct1``
        to support different types of growth model. For example, ``ct2``
        is defined as the square of ``ct1`` to model quadratic growth.
    Trend:
        Trend describes the average tendency of the time series.
        It is defined through the growth term with possible changepoints.
        At every changepoint, the growth rate could change (faster or slower).
        For example, if ``ct1`` (linear growth) is used with changepoints,
        the trend is modeled as piece-wise linear.
    Seasonality:
        Seasonality describes the periodical pattern of the time series.
        It contains multiple levels including daily seasonality, weekly seasonality,
        monthly seasonality, quarterly seasonality and yearly seasonality.
        Seasonality are defined through Fourier series with different orders.
        The greater the order, the more detailed periodical pattern the model
        can learn. However, an order that is too large can lead to overfitting.
    Events:
        Events include holidays and other short-term occurrences that could
        temporarily affect the time series, such as Thanksgiving long weekend.
        Typically, events are regular and repeat at know times in the future.
        These features made of indicators that covers the event day and their neighbor days.
    Autoregression:
        Autoregressive features include the time series observations
        in the past and their aggregations. For example, the past day's observation,
        the same weekday on the past week, or the average of the past 7 days, etc.
        can be used. Note that autoregression features are very useful in short term
        forecasts, however, this should be avoided in long term forecast.
        The reason is that long-term forecast focuses more on the correctness
        of trend, seasonality and events. The lags and autoregressive terms in
        a long-term forecast are calculated based on the forecasted values.
        The further we forecast into the future, the more forecasted values we
        need to create the autoregressive terms, making the forecast less stable.
    Custom:
        Extra features that are relevant to the time series such as macro-ecomonic
        features that are expected to affect the time series.
        Note that these features need to be manually provided for both
        the training and forecasting periods.
    Interactions:
        Any interaction between the features above.

Now let's use an example to go through the full forecasting and tuning process.
In this example, we'll load a dataset representing ``log(daily page views)``
on the Wikipedia page for Peyton Manning.
It contains values from 2007-12-10 to 2016-01-20. More dataset info
`here <https://facebook.github.io/prophet/docs/quick_start.html>`_.
"""

import datetime

import numpy as np
import pandas as pd
import plotly

from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector
from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonalityEnum
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import cols_interact
from greykite.common import constants as cst
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.features.timeseries_features import convert_date_to_continuous_time
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results


# Loads dataset into UnivariateTimeSeries
dl = DataLoaderTS()
ts = dl.load_peyton_manning_ts()
df = ts.df  # cleaned pandas.DataFrame

# %%
# Exploratory data analysis (EDA)
# --------------------------------
# After reading in a time series, we could first do some exploratory data analysis.
# The `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries` class is
# used to store a timeseries and perform EDA.

# describe
print(ts.describe_time_col())
print(ts.describe_value_col())

# %%
# The df has two columns, time column "ts" and value column "y".
# The data is daily that ranges from 2007-12-10 to 2016-01-20.
# The data value ranges from 5.26 to 12.84
#
# Let's plot the original timeseries.
# (The interactive plot is generated by ``plotly``: **click to zoom!**)

fig = ts.plot()
plotly.io.show(fig)

# %%
# A few exploratory plots can be plotted to reveal the time series's properties.
# The `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries` class
# has a very powerful plotting tool
# `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`.
# A tutorial of using the function can be found at :doc:`/gallery/quickstart/01_exploration/0300_seasonality_plots`.

# %%
# Baseline model
# --------------------
# A simple forecast can be created on the data set,
# see details in :doc:`/gallery/quickstart/0100_simple_forecast`.
# Note that if you do not provide any extra parameters, all model parameters are by default.
# The default parameters are chosen conservatively, so consider this a baseline
# model to assess forecast difficulty and make further improvements if necessary.

# Specifies dataset information
metadata = MetadataParam(
    time_col="ts",  # name of the time column
    value_col="y",  # name of the value column
    freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
)

forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=365,  # forecasts 365 steps ahead
        coverage=0.95,  # 95% prediction intervals
        metadata_param=metadata
    )
)

# %%
# For a detailed documentation about the output from
# :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`,
# see :doc:`/pages/stepbystep/0500_output`. Here we could plot the forecast.

forecast = result.forecast
fig = forecast.plot()
plotly.io.show(fig)

# %%
# Model performance evaluation
# ----------------------------
# We can see the forecast fits the existing data well; however, we do not
# have a good ground truth to assess how well it predicts into the future.
#
# Train-test-split
# ^^^^^^^^^^^^^^^^
# The typical way to evaluate model performance is to reserve part of the training
# data and use it to measure the model performance.
# Because we always predict the future in a time series forecasting problem,
# we reserve data from the end of training set to measure the performance
# of our forecasts. This is called a time series train test split.
#
# By default, the results returned by :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`
# creates a time series train test split and stores the test result in ``result.backtest``.
# The reserved testing data by default has the
# same length as the forecast horizon. We can access the evaluation results:

pd.DataFrame(result.backtest.test_evaluation, index=["Value"]).transpose()  # formats dictionary as a pd.DataFrame

# %%
# Evaluation metrics
# ^^^^^^^^^^^^^^^^^^
# From here we can see a list of metrics that measure the model performance on the test data.
# You may choose one or a few metrics to focus on. Typical metrics include:
#
#   MSE:
#       Mean squared error, the average squared error. Could be affected by extreme values.
#
#   RMSE:
#       Root mean squared error, the square root of MSE.
#
#   MAE:
#       Mean absolute error, the average of absolute error. Could be affected by extreme values.
#
#   MedAE:
#       Median absolute error, the median of absolute error. Less affected by extreme values.
#
#   MAPE:
#       Mean absolute percent error, measures the error percent with respective to the true values.
#       This is useful when you would like to consider the relative error instead of the absolute error.
#       For example, an error of 1 is considered as 10% for a true observation of 10, but as 1% for a true
#       observation of 100. This is the default metric we like.
#
#   MedAPE:
#       Median absolute percent error, the median version of MAPE, less affected by extreme values.
#
# Let's use MAPE as our metric in this example. Looking at these results,
# you may have a basic sense of how the model is performing on the unseen test data.
# On average, the baseline model's prediction is 11.3% away from the true values.
#
# Time series cross-validation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Forecast quality depends a lot of the evaluation time window.
# The evaluation window selected above might happen to be a relatively easy/hard period to predict.
# Thus, it is more robust to evaluate over a longer time window when dataset size allows.
# Let's consider a more general way of evaluating a forecast model: time series cross-validation.
#
# Time series cross-validation is based on a time series rolling split.
# Let's say we would like to perform an evaluation with a 3-fold cross-validation,
# The whole training data is split in 3 different ways.
# Since our forecast horizon is 365 days, we do:
#
#     First fold:
#       Train from 2007-12-10 to 2013-01-20, forecast from
#       2013-01-21 to 2014-01-20, and compare the forecast with the actual.
#     Second fold:
#       Train from 2007-12-10 to 2014-01-20, forecast from
#       2014-01-21 to 2015-01-20, and compare the forecast with the actual.
#     Third fold:
#       Train from 2007-12-10 to 2015-01-20, forecast from
#       2015-01-21 to 2016-01-20, and compare the forecast with the actual.
#
# The split could be more flexible, for example, the testing periods could have gaps.
# For more details about evaluation period configuration, see
# `Evaluation Period <../../pages/stepbystep/0400_configuration.html#evaluation-period>`_.
# The forecast model's performance will be the average of the three evaluations
# on the forecasts.
#
# By default, the results returned by :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`
# also runs time series cross-validation internally.
# You are allowed to configure the cross-validation splits, as shown below.
# Here note that the ``test_horizon`` are reserved from the back of
# the data and not used for cross-validation.
# This part of testing data can further evaluate the model performance
# besides the cross-validation result, and is available for plotting.

# Defines the cross-validation config
evaluation_period = EvaluationPeriodParam(
    test_horizon=365,             # leaves 365 days as testing data
    cv_horizon=365,               # each cv test size is 365 days (same as forecast horizon)
    cv_max_splits=3,              # 3 folds cv
    cv_min_train_periods=365 * 4  # uses at least 4 years for training because we have 8 years data
)

# Runs the forecast
result = forecaster.run_forecast_config(
    df=df,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=365,  # forecasts 365 steps ahead
        coverage=0.95,  # 95% prediction intervals
        metadata_param=metadata,
        evaluation_period_param=evaluation_period
    )
)

# Summarizes the cv result
cv_results = summarize_grid_search_results(
    grid_search=result.grid_search,
    decimals=1,
    # The below saves space in the printed output. Remove to show all available metrics and columns.
    cv_report_metrics=None,
    column_order=["rank", "mean_test", "split_test", "mean_train", "split_train", "mean_fit_time", "mean_score_time", "params"])
# Transposes to save space in the printed output
cv_results["params"] = cv_results["params"].astype(str)
cv_results.set_index("params", drop=True, inplace=True)
cv_results.transpose()

# %%
# By default, all metrics in `~greykite.common.evaluation.ElementwiseEvaluationMetricEnum`
# are computed on each CV train/test split.
# The configuration of CV evaluation metrics can be found at
# `Evaluation Metric <../../pages/stepbystep/0400_configuration.html#evaluation-metric>`_.
# Here, we show the Mean Absolute Percentage Error (MAPE)
# across splits (see `~greykite.framework.utils.result_summary.summarize_grid_search_results`
# to control what to show and for details on the output columns).
# From the result, we see that the cross-validation ``mean_test_MAPE`` is 7.3%, which
# means the prediction is 7.3% away from the ground truth on average. We also see the
# 3 cv folds have ``split_test_MAPE`` 5.1%, 8.5% and 8.4%, respectively.
#
# When we have different sets of model parameters, a good way to compare them is
# to run a time series cross-validation on each set of parameters, and pick the
# set of parameters that has the best cross-validated performance.
#
# Start tuning
# ------------
# Now that you know how to evaluate model performance,
# let's see if we can improve the model by tuning its parameters.
#
# Anomaly
# ^^^^^^^
# An anomaly is a deviation in the metric that is not expected to occur again
# in the future. Including anomaly points will lead the model to fit the
# anomaly as an intrinsic property of the time series, resulting in inaccurate forecasts.
# These anomalies could be identified through overlay plots, see
# :doc:`/gallery/quickstart/01_exploration/0300_seasonality_plots`.

fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="month_dom",
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    overlay_label_time_feature="year",
    overlay_style={"line": {"width": 1}, "opacity": 0.5},
    center_values=True,
    xlabel="day of year",
    ylabel=ts.original_value_col,
    title="yearly seasonality for each year (centered)",
)
plotly.io.show(fig)

# %%
# From the yearly overlay plot above, we could see two big anomalies:
# one in March of 2012, and one in June of 2010. Other small anomalies
# could be identified as well, however, they have less influence.
# The ``SILVERKITE`` template currently supports masking anomaly points
# by supplying the ``anomaly_info`` as a dictionary. You could
# either assign adjusted values to them, or simply mask them as NA
# (in which case these dates will not be used in fitting).
# For a detailed introduction about the ``anomaly_info`` configuration,
# see :doc:`/pages/stepbystep/0300_input`.
# Here we define an ``anomaly_df`` dataframe to mask them as NA,
# and wrap it into the ``anomaly_info`` dictionary.

anomaly_df = pd.DataFrame({
    # start and end date are inclusive
    # each row is an anomaly interval
    cst.START_TIME_COL: ["2010-06-05", "2012-03-01"],  # inclusive
    cst.END_TIME_COL: ["2010-06-20", "2012-03-20"],  # inclusive
    cst.ADJUSTMENT_DELTA_COL: [np.nan, np.nan],  # mask as NA
})
# Creates anomaly_info dictionary.
# This will be fed into the template.
anomaly_info = {
    "value_col": "y",
    "anomaly_df": anomaly_df,
    "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL,
}

# %%
# Adding relevant features
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Growth and trend
# """"""""""""""""
# First we look at the growth and trend. Detailed growth configuration can be found
# at :doc:`/pages/model_components/0200_growth`.
# In these two features, we care less about the short-term fluctuations but rather long-term tendency.
# From the original plot we see there is no obvious growth pattern, thus we
# could use a linear growth to fit the model. On the other hand, there could be
# potential trend changepoints, at which time the linear growth changes its rate.
# Detailed changepoint configuration can be found at :doc:`/pages/model_components/0500_changepoints`.
# These points can be detected with the ``ChangepointDetector`` class. For a quickstart example,
# see :doc:`/gallery/quickstart/01_exploration/0100_changepoint_detection`.
# Here we explore the automatic changepoint detection.
# The parameters in this automatic changepoint detection is customized for this data set.
# We keep the ``yearly_seasonality_order`` the same as the model's yearly seasonality order.
# The ``regularization_strength`` controls how many changepoints are detected.
# 0.5 is a good choice, while you may try other numbers such as 0.4 or 0.6 to see the difference.
# The ``resample_freq`` is set to 7 days, because we have a long training history, thus we should
# keep this relatively long (the intuition is that shorter changes will be ignored).
# We put 25 potential changepoints to be the candidates, because we do not expect too many changes.
# However, this could be higher.
# The ``yearly_seasonality_change_freq`` is set to 365 days, which means we refit the yearly seasonality
# every year, because it can be see from the time series plot that the yearly seasonality varies every year.
# The ``no_changepoint_distance_from_end`` is set to 365 days, which means we do not allow any changepoints
# at the last 365 days of training data. This avoids fitting the final trend with too little data.
# For long-term forecast, this is typically the same as the forecast horizon, while for short-term forecast,
# this could be a multiple of the forecast horizon.

model = ChangepointDetector()
res = model.find_trend_changepoints(
    df=df,  # data df
    time_col="ts",  # time column name
    value_col="y",  # value column name
    yearly_seasonality_order=10,  # yearly seasonality order, fit along with trend
    regularization_strength=0.5,  # between 0.0 and 1.0, greater values imply fewer changepoints, and 1.0 implies no changepoints
    resample_freq="7D",  # data aggregation frequency, eliminate small fluctuation/seasonality
    potential_changepoint_n=25,  # the number of potential changepoints
    yearly_seasonality_change_freq="365D",  # varying yearly seasonality for every year
    no_changepoint_distance_from_end="365D")  # the proportion of data from end where changepoints are not allowed
fig = model.plot(
    observation=True,
    trend_estimate=False,
    trend_change=True,
    yearly_seasonality_estimate=False,
    adaptive_lasso_estimate=True,
    plot=False)
plotly.io.show(fig)

# %%
# From the plot we see the automatically detected trend changepoints.
# The results shows that the time series is generally increasing until 2012,
# then generally decreasing. One possible explanation is that 2011 is
# the last year Peyton Manning was at the Indianapolis Colts before joining the
# Denver Broncos. If we feed the trend changepoint detection parameter to the template,
# these trend changepoint features will be automatically included in the model.

# The following specifies the growth and trend changepoint configurations.
growth = {
    "growth_term": "linear"
}
changepoints = {
    "changepoints_dict": dict(
        method="auto",
        yearly_seasonality_order=10,
        regularization_strength=0.5,
        resample_freq="7D",
        potential_changepoint_n=25,
        yearly_seasonality_change_freq="365D",
        no_changepoint_distance_from_end="365D"
    )
}

# %%
# Seasonality
# """""""""""
# The next features we will look into are the seasonality features.
# Detailed seasonality configurations can be found at
# :doc:`/pages/model_components/0300_seasonality`.
# A detailed seasonality detection quickstart example on the same data set is
# available at :doc:`/gallery/quickstart/01_exploration/0300_seasonality_plots`.
# The conclusions about seasonality terms are:
#
#   - daily seasonality is not available (because frequency is daily);
#   - weekly and yearly patterns are evident (weekly will also interact with football season);
#   - monthly or quarterly seasonality is not evident.
#
# Therefore, for pure seasonality terms, we include weekly and yearly
# seasonality. The seasonality orders are something to be tuned; here
# let's take weekly seasonality order to be 5 and yearly seasonality order to be 10.
# For tuning info, see :doc:`/pages/model_components/0300_seasonality`.

# Includes yearly seasonality with order 10 and weekly seasonality with order 5.
# Set the other seasonality to False to disable them.
yearly_seasonality_order = 10
weekly_seasonality_order = 5
seasonality = {
    "yearly_seasonality": yearly_seasonality_order,
    "quarterly_seasonality": False,
    "monthly_seasonality": False,
    "weekly_seasonality": weekly_seasonality_order,
    "daily_seasonality": False
}

# %%
# We will add the interaction between weekly seasonality and the football season
# later in this tutorial.
# The ``SILVERKITE`` template also supports seasonality changepoints. A seasonality
# changepoint is a time point after which the periodic effect behaves
# differently. For ``SILVERKITE``, this means the Fourier series coefficients are allowed
# to change. We could decide to add this feature if cross-validation performance is poor
# and seasonality changepoints are detected in exploratory analysis.
# For details, see :doc:`/gallery/quickstart/01_exploration/0100_changepoint_detection`.

# %%
# Holidays and events
# """""""""""""""""""
# Then let's look at holidays and events. Detailed holiday and event configurations
# can be found at :doc:`/pages/model_components/0400_events`.
# Ask yourself which holidays are likely to affect the time series' values.
# We expect that major United States holidays may affect wikipedia pageviews,
# since most football fans are in the United States.
# Events such as superbowl could potentially increase the pageviews.
# Therefore, we add United States holidays and superbowls dates as custom events.
# Other important events that affect the time series can also be found
# through the yearly seasonality plots in :doc:`/gallery/quickstart/01_exploration/0300_seasonality_plots`.

# Includes major holidays and the superbowl date.
events = {
    # These holidays as well as their pre/post dates are modeled as individual events.
    "holidays_to_model_separately": SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES,  # all holidays in "holiday_lookup_countries"
    "holiday_lookup_countries": ["UnitedStates"],  # only look up holidays in the United States
    "holiday_pre_num_days": 2,  # also mark the 2 days before a holiday as holiday
    "holiday_post_num_days": 2,  # also mark the 2 days after a holiday as holiday
    "daily_event_df_dict": {
        "superbowl": pd.DataFrame({
            "date": ["2008-02-03", "2009-02-01", "2010-02-07", "2011-02-06",
                     "2012-02-05", "2013-02-03", "2014-02-02", "2015-02-01", "2016-02-07"],  # dates must cover training and forecast period.
            "event_name": ["event"] * 9  # labels
        })
    },
}

# %%
# Autoregression
# """"""""""""""
# The autoregressive features are very useful in short-term forecasting, but
# could be risky to use in long-term forecasting. Detailed autoregression
# configurations can be found at :doc:`/pages/model_components/0800_autoregression`.
#
# Custom
# """"""
# Now we consider some custom features that could relate to the pageviews. The documentation for
# extra regressors can be found at :Doc:`/pages/model_components/0700_regressors`. As mentioned
# in :doc:`/gallery/quickstart/01_exploration/0300_seasonality_plots`, we observe that the football
# season heavily affects the pageviews, therefore we need to use regressors to identify the football season.
# There are multiple ways to include this feature: adding indicator for the whole season;
# adding number of days till season start (end) and number of days since season start (end).
# The former puts a uniform effect over all in-season dates, while the latter quantify
# the on-ramp and down-ramp. If you are not sure which effect to include, it's ok to include both
# effects. ``SILVERKITE`` has the option to use Ridge regression as the fit algorithm to avoid
# over-fitting too many features. Note that many datetime features could also be added to
# the model as features. ``SILVERKITE`` calculates some of these features, which can be added to
# ``extra_pred_cols`` as an arbitrary patsy expression.
# For a full list of such features, see `~greykite.common.features.timeseries_features.build_time_features_df`.
#
# If a feature is not automatically created by ``SILVERKITE``, we need to create it
# beforehand and append it to the data df.
# Here we create the "is_football_season" feature.
# Note that we also need to provide the customized column for the forecast horizon period as well.
# The way we do it is to first create the df with timestamps covering the forecast horizon.
# This can be done with the `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.make_future_dataframe`
# function within the `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries` class.
# Then we create a new column of our customized regressor for this augmented df.

# Makes augmented df with forecast horizon 365 days
df_full = ts.make_future_dataframe(periods=365)
# Builds "df_features" that contains datetime information of the "df"
df_features = build_time_features_df(
    dt=df_full["ts"],
    conti_year_origin=convert_date_to_continuous_time(df_full["ts"][0])
)

# Roughly approximates the football season.
# "woy" is short for "week of year", created above.
# Football season is roughly the first 6 weeks and last 17 weeks in a year.
is_football_season = (df_features["woy"] <= 6) | (df_features["woy"] >= 36)
# Adds the new feature to the dataframe.
df_full["is_football_season"] = is_football_season.astype(int).tolist()
df_full.reset_index(drop=True, inplace=True)

# Configures regressor column.
regressors = {
    "regressor_cols": ["is_football_season"]
}

# %%
# Interactions
# """"""""""""
# Finally, let's consider what possible interactions are relevant to the forecast problem.
# Generally speaking, if a feature behaves differently on different values of another feature,
# these two features could have potential interaction effects.
# As in :doc:`/gallery/quickstart/01_exploration/0300_seasonality_plots`, the weekly seasonality
# is different through football season and non-football season, therefore, the multiplicative
# term ``is_football_season x weekly_seasonality`` is able to capture this pattern.

fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="str_dow",
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    center_values=True,
    overlay_label_time_feature="month",  # splits overlays by month
    overlay_style={"line": {"width": 1}, "opacity": 0.5},
    xlabel="day of week",
    ylabel=ts.original_value_col,
    title="weekly seasonality by month",
)
plotly.io.show(fig)

# %%
# Now let's create the interaction terms: interaction between ``is_football_season`` and ``weekly seasonality``.
# The interaction terms between a feature and a seasonality feature
# can be created with the `~greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper.cols_interact` function.

football_week = cols_interact(
    static_col="is_football_season",
    fs_name=SilverkiteSeasonalityEnum.WEEKLY_SEASONALITY.value.name,
    fs_order=weekly_seasonality_order,
    fs_seas_name=SilverkiteSeasonalityEnum.WEEKLY_SEASONALITY.value.seas_names
)

extra_pred_cols = football_week

# %%
# Moreover, the multiplicative term ``month x weekly_seasonality`` and the ``dow_woy`` features also
# account for the varying weekly seasonality through the year. One could added these features, too.
# Here we just leave them out. You may use ``cols_interact`` again to create the ``month x weekly_seasonality``
# similar to ``is_football_season x weekly_seasonality``. ``dow_woy`` is automatically calcuated by ``SILVERKITE``,
# you may simply append the name to ``extra_pred_cols`` to include it in the model.
#
# Putting things together
# ^^^^^^^^^^^^^^^^^^^^^^^
# Now let's put everything together and produce a new forecast.
# A detailed template documentation can be found at
# :doc:`/pages/stepbystep/0400_configuration`.
# We first configure the ``MetadataParam`` class.
# The ``MetadataParam`` class includes basic proporties of the time series itself.

metadata = MetadataParam(
    time_col="ts",              # column name of timestamps in the time series df
    value_col="y",              # column name of the time series values
    freq="D",                   # data frequency, here we have daily data
    anomaly_info=anomaly_info,  # this is the anomaly information we defined above,
    train_end_date=datetime.datetime(2016, 1, 20)
)

# %%
# Next we define the ``ModelComponentsParam`` class based on the discussion on relevant features.
# The ``ModelComponentsParam`` include properties related to the model itself.

model_components = ModelComponentsParam(
    seasonality=seasonality,
    growth=growth,
    events=events,
    changepoints=changepoints,
    autoregression=None,
    regressors=regressors,  # is_football_season defined above
    uncertainty={
        "uncertainty_dict": "auto",
    },
    custom={
        # What algorithm is used to learn the relationship between the time series and the features.
        # Regularized fitting algorithms are recommended to mitigate high correlations and over-fitting.
        # If you are not sure what algorithm to use, "ridge" is a good choice.
        "fit_algorithm_dict": {
            "fit_algorithm": "ridge",
        },
        "extra_pred_cols": extra_pred_cols  # the interaction between is_football_season and weekly seasonality defined above
    }
)

# %%
# Now let's run the model with the new configuration.
# The evaluation config is kept the same as the previous case;
# this is important for a fair comparison of parameter sets.

# Runs the forecast
result = forecaster.run_forecast_config(
    df=df_full,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=365,  # forecasts 365 steps ahead
        coverage=0.95,  # 95% prediction intervals
        metadata_param=metadata,
        model_components_param=model_components,
        evaluation_period_param=evaluation_period
    )
)

# Summarizes the cv result
cv_results = summarize_grid_search_results(
    grid_search=result.grid_search,
    decimals=1,
    # The below saves space in the printed output. Remove to show all available metrics and columns.
    cv_report_metrics=None,
    column_order=["rank", "mean_test", "split_test", "mean_train", "split_train", "mean_fit_time", "mean_score_time", "params"])
# Transposes to save space in the printed output
cv_results["params"] = cv_results["params"].astype(str)
cv_results.set_index("params", drop=True, inplace=True)
cv_results.transpose()

# %%
# Now we see that after analyzing the problem and adding appropriate features,
# the cross-validation test MAPE is 5.4%, which is improved compared with the baseline (7.3%).
# The 3 cv folds also have their MAPE reduced to 3.9%, 8.7% and 3.8%, respectively.
# The first and third fold improved significantly. With some investigation, we can see that
# the second fold did not improve because there is a trend changepoint right at the the start
# of its test period.
#
# It would be hard to know this situation until we see it. In the cross-validation step, one
# way to avoid this is to set a different evaluation period. However, leaving this period
# also makes sense because it could happen again in the future.
# In the forecast period, we could monitor the forecast and actual, and re-train the model
# to adapt to the most recent pattern if we see a deviation. In the changepoints dictionary,
# tune ``regularization_strength`` or ``no_changepoint_distance_from_end`` accordingly, or
# add manually specified changepoints to the automatically detected ones. For details, see
# :doc:`/pages/model_components/0500_changepoints`.
#
# We could also plot the forecast.

forecast = result.forecast
fig = forecast.plot()
plotly.io.show(fig)

# %%
# Check model summary
# ^^^^^^^^^^^^^^^^^^^
# To further investigate the model mechanism, it's also helpful
# to see the model summary.
# The `~greykite.algo.common.model_summary.ModelSummary` module
# provides model results such as estimations, significance, p-values,
# confidence intervals, etc.
# that can help the user understand how the model works and
# what can be further improved.
#
# The model summary is a class method of the estimator and can be used as follows.

summary = result.model[-1].summary()  # -1 retrieves the estimator from the pipeline
print(summary)

# %%
# The model summary shows the model information, the coefficients and their significance,
# and a few summary statistics. For example,
# we can see the changepoints and how much the growth rate
# changes at each changepoint.
# We can see that some of the holidays have significant
# effect in the model, such as Christmas, Labor day, Thanksgiving, etc.
# We can see the significance of the interaction between football season and weekly seasonality
# etc.
#
# For a more detailed guide on model summary, see
# :doc:`/gallery/quickstart/02_interpretability/0100_model_summary`.

# %%
# Summary in model tuning
# -----------------------
# After the example, you may have some sense about how to select parameters and tune the model.
# Here we list a few steps and tricks that might help select the best models.
# What you may do:
#
#   #. Detect anomaly points with the overlay plots
#      (`~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`).
#      Mask these points with NA. Do not specify the adjustment unless you are confident about how to correct the anomalies.
#
#   #. Choose an appropriate way to model the growth (linear, quadratic, square root, etc.)
#      If none of the typical growth shape fits the time series, you might consider linear
#      growth with trend changepoints. Try different changepoint detection configurations.
#      You may also plot the detected changepoints and see if it makes sense to you.
#      The template also supports custom changepoints. If the automatic changepoint detection result
#      does not make sense to you, you might supply your own changepoints.
#
#   #. Choose the appropriate seasonality orders. The higher the order, the more details the model can learn.
#      However, too large orders could overfit the training data. These can also be detected from the
#      overlay plots (`~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`).
#      There isn't a unified way to choose seasonality, so explore different seasonality orders and compare the results.
#
#   #. Consider what events and holidays to model. Are there any custom events we need to add?
#      If you add a custom event, remember also adding the dates for the event in the forecast period.
#
#   #. Add external regressors that could be related to the time series. Note that you will need to provide the
#      values of the regressors in the forecast period as well. You may use another time series as a regressor,
#      as long as you have a ground truth/good forecast for it that covers your forecast period.
#
#   #. Adding interaction terms. Let's mention again here that there could be interaction between two features
#      if the behaviors of one feature are different when the other feature have different values.
#      Try to detect this through the overlay plot
#      (`~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`), too.
#      By default, we have a few pre-defined interaction terms, see
#      `feature_sets_enabled <../../pages/model_components/0600_custom.html#interactions>`_.
#
#   #. Choose an appropriate fit algorithm. This is the algorithm that models the relationship between the features
#      and the time series. See a full list of available algorithms at
#      `fit_algorithm <../../pages/model_components/0600_custom.html#fit-algorithm>`_.
#      If you are unsure about their difference, try some of them and compare the results. If you don't want to, choosing "ridge"
#      is a safe option.
#
# It is worth noting that the template supports automatic grid search with different sets of parameters.
# For each parameter, if you provide the configuration in a list, it will automatically run each combination
# and choose the one with the best cross-validation performance. This will save a lot of time.
# For details, see :doc:`/gallery/quickstart/03_benchmark/0100_grid_search`.

# %%
# Follow your insights and intuitions, and play with the parameters, you will get good forecasts!
