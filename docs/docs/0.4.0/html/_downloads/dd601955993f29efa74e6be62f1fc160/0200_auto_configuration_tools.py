"""
Auto Configuration Tools
========================

The Silverkite model has many hyperparameters to tune.
Besides domain knowledge, we also have tools that can help
find good choices for certain hyperparameters.
In this tutorial, we will present

  * seasonality inferrer
  * holiday inferrer

.. note::
  If you use the model templates, you can specify the "auto" option for certain model components
  (growth, seasonality and holiday),
  and the auto configuration tool will be activated automatically.
  See `auto seasonality <../../../pages/model_components/0300_seasonality.html#silverkite>`_,
  `auto growth <../../../pages/model_components/0500_changepoints.html#auto-growth>`_ and
  `auto holidays <../../../pages/model_components/0400_events.html#auto-holiday>`_ for the way to activate them.
  This doc explains how the "auto" options work behind the code.
  You can replay the "auto" options with the Seasonality Inferrer and Holiday Inferrer below.
  Please remember that if you are doing train-test split,
  running the inferrers on training data only is closer to the reality.

Seasonality Inferrer
--------------------

The Silverkite model uses Fourier series to model seasonalities.
It's sometimes difficult to decide what orders we should use
for each Fourier series.
Larger orders tend to fit more closely to the curves, while having
the risk of overfitting.
Small orders tend to underfit the curve and may not learn the exact seasonality patterns.

`~greykite.algo.common.seasonality_inferrer.SeasonalityInferrer`
is a tool that can help you decide what order to use for a seasonality's Fourier series.
Note that there are many ways to decide the orders,
and you don't have to strictly stick to the results from Seasonality Inferrer.

How it works
~~~~~~~~~~~~

The seasonality inferrer utilizes criteria including AIC and BIC to find the most
appropriate Fourier series orders.
For a specific seasonality, e.g. yearly seasonality, the steps are as follows:

* Trend removal: seasonality inferrer provides 4 options for trend removal.
  They are listed in `~greykite.algo.common.seasonality_inferrer.TrendAdjustMethodEnum`.
  Specifically:

    * ``"seasonal_average"``: given an indicator of seasonal period, the method subtracts
      the average within each seasonal period from the original time series.
      For example, given the column ``year``, the average is calculated on each different year.
    * ``"overall_average"``: subtracts the overall average from the original time series.
    * ``"spline_fit"``: fits a polynomial up to a given degree and subtract from the original time series.
    * ``"none"``: does not adjust the trend.

  Typically "seasonal_average" is a good choice with appropriate columns.
  For example, we can use ``year_quarter`` for quarterly seasonality, ``year_month`` for monthly seasonality,
  ``year_woy_iso`` for weekly seasonality and ``year_woy_dow_iso`` for daily seasonality.
* Optional aggregation: sometimes we want to get rid of shorter fluctuations before
  fitting a longer seasonality period. We can do an optional aggregation beforehand.
  For example, when we model yearly seasonality, we can do a ``"7D"`` aggregation to eliminate
  weekly effects to make the result more stable.
* With a pre-specified maximum order ``n``, we fit the de-trended (and aggregated) time series
  with Fourier series from 1 to n, and calculate the AIC/BIC for those fits.
  The most appropriate order is then decided by choosing the order with best AIC or BIC.
  The method also allows to slightly sacrifice the criterion and reduce the order
  for less risk of overfitting using the ``tolerance`` parameter.
* Finally, an optional offset can be applied to any inferred orders to allow manual adjustments.
  For example, if one would like to use less yearly seasonality order, they may specify
  offset for yearly seasonality to be -2, and the final order will subtract 2 from the inferred result.
  This is useful when users tend to use more or less orders to model seasonality,
  and want a knob on top of the inferring results.

Example
~~~~~~~

Now we look at an example with the Peyton-Manning Wiki page view data.
"""

import pandas as pd
import plotly
from greykite.common.data_loader import DataLoader
from greykite.algo.common.seasonality_inferrer import SeasonalityInferConfig
from greykite.algo.common.seasonality_inferrer import SeasonalityInferrer
from greykite.algo.common.seasonality_inferrer import TrendAdjustMethodEnum
from greykite.common import constants as cst

# %%
# The ``SeasonalityInferrer`` class uses
# `~greykite.algo.common.seasonality_inferrer.SeasonalityInferConfig`
# to specify configuration for a single seasonality component,
# and it takes a list of such configurations to infer multiple seasonality
# components together.
# Now we specify seasonality inferring configs for yearly to weekly seasonalities.
# In each of these configs, specify the parameters that are distinct for each component.
# If there are parameters that are the same across all configs,
# you can specify them in the function directly.

yearly_config = SeasonalityInferConfig(
    seas_name="yearly",                     # name for seasonality
    col_name="toy",                         # column to generate Fourier series, fixed for yearly
    period=1.0,                             # seasonal period, fixed for yearly
    max_order=30,                           # max number of orders to model
    adjust_trend_param=dict(
        trend_average_col="year"
    ),                                      # column to adjust trend for method "seasonal_average"
    aggregation_period="W",                 # aggregation period,
    offset=0                                # add this to the inferred result, default 0
)
quarterly_config = SeasonalityInferConfig(
    seas_name="quarterly",                  # name for seasonality
    col_name="toq",                         # column to generate Fourier series, fixed for quarterly
    period=1.0,                             # seasonal period, fixed for quarterly
    max_order=20,                           # max number of orders to model
    adjust_trend_param=dict(
        trend_average_col="year_quarter"
    ),                                      # column to adjust trend for method "seasonal_average"
    aggregation_period="2D",                # aggregation period
)
monthly_config = SeasonalityInferConfig(
    seas_name="monthly",                    # name for seasonality
    col_name="tom",                         # column to generate Fourier series, fixed for monthly
    period=1.0,                             # seasonal period, fixed for monthly
    max_order=20,                           # max number of orders to model
    adjust_trend_param=dict(
        trend_average_col="year_month"
    ),                                      # column to adjust trend for method "seasonal_average"
    aggregation_period="D"                  # aggregation period
)
weekly_config = SeasonalityInferConfig(
    seas_name="weekly",                     # name for seasonality
    col_name="tow",                         # column to generate Fourier series, fixed for weekly
    period=7.0,                             # seasonal period, fixed for weekly
    max_order=10,                           # max number of orders to model
    adjust_trend_param=dict(
        trend_average_col="year_woy_iso"
    ),                                      # column to adjust trend for method "seasonal_average"
    aggregation_period="D",
    tolerance=0.005,                        # allows 0.5% higher criterion for lower orders
)

# %%
# Next, we put everything together to infer seasonality effects.

df = DataLoader().load_peyton_manning()
df[cst.TIME_COL] = pd.to_datetime((df[cst.TIME_COL]))

model = SeasonalityInferrer()
result = model.infer_fourier_series_order(
    df=df,
    time_col=cst.TIME_COL,
    value_col=cst.VALUE_COL,
    configs=[
        yearly_config,
        quarterly_config,
        monthly_config,
        weekly_config
    ],
    adjust_trend_method=TrendAdjustMethodEnum.seasonal_average.name,
    fit_algorithm="linear",
    plotting=True,
    criterion="bic",
)

# %%
# The method runs quickly and we can simply extract the inferred results
# from the output.

result["best_orders"]

# %%
# We can also plot the results to see how different orders vary the criterion.
# Similar to other trade-off plots, the plot first goes down and then goes up,
# reaching the best at some appropriate value in the middle.

# The [0] extracts the first seasonality component from the results.
plotly.io.show(result["result"][0]["fig"])

# %%
# Holiday Inferrer
# ----------------
#
# The Silverkite model supports modeling holidays and their neighboring days
# as indicators. Significant days are modeled separately,
# while similar days can be grouped together as one indicator,
# assuming their effects are the same.
#
# It's sometimes difficult to decide which holidays to include,
# to model separately or to model together.
# `~greykite.algo.common.holiday_inferrer.HolidayInferrer`
# is a tool that can help you decide which holidays to model
# and how to model them.
# It can also automatically generate the holiday configuration parameters.
# Note that there are many ways to decide the holiday configurations,
# and you don't have to strictly stick to the results from Holiday Inferrer.
#
# How it works
# ~~~~~~~~~~~~
#
# The holiday inferrer estimates individual holiday or their
# neighboring days' effects by comparing the observations
# on these days with some baseline prior to or after the holiday period.
# Then it ranks the effects by their magnitude.
# Depending on some thresholds, it decides whether to model
# a day independently, together with others or do not model it.
#
# In detail, the first step is to unify the data frequency.
# For data whose frequency is greater than daily,
# holiday effect is automatically turned off.
# For data whose frequency is less than daily,
# it is aggregated into daily data,
# since holidays are daily events.
# From now on, we have daily data for the next step.
#
# Given a list of countries, the tool automatically pulls candidate
# holidays from the database. With a ``pre_search_days`` and a ``post_search_days``
# parameters, those holidays' neighboring days are included in the candidate pool
# as well.
#
# For every candidate holiday or neighboring day,
# the baseline is the average of a configurable offsets.
# For example, for data that exhibits strong weekly seasonality,
# the offsets can be ``(-7, 7)``, where the baseline will be
# the average of the last same day of week's observation and the
# next same day of week's observation.
# For example, if the holiday is New Year on 1/1 while 12/25 (7 days ago) is Christmas,
# it will look at the value on 12/18 instead of 12/25 as baseline.
#
# The day's effect is the average of the signed difference between
# the true observation and the baseline across all occurrences in the time series.
# The effects are ranked from the highest to the lowest by their absolute effects.
#
# To decide how each holiday is modeled, we rely on two parameters:
# ``independent_holiday_thres`` and ``together_holiday_thres``.
# These parameters are between 0 and 1.
# Starting from the largest effect,
# we calculate the cumulative sum of effect of all candidates.
# Once the cumulative effect reaches ``independend_holiday_thres`` of the total effects,
# these days will be modeled independently (i.e, each day has an individual coefficient).
# We keep accumulating effects until the sum reaches ``together_holiday_thres``,
# the days in the between are grouped into "positive_group" and "negative_group",
# with each group modeled together.
#
# Example
# ~~~~~~~
#
# Now we look at an example with the Peyton-Manning Wiki page view data.

import pandas as pd
import plotly
from greykite.algo.common.holiday_inferrer import HolidayInferrer
from greykite.common.data_loader import DataLoader
from greykite.common import constants as cst

df = DataLoader().load_peyton_manning()
df[cst.TIME_COL] = pd.to_datetime(df[cst.TIME_COL])

# %%
# Let's say we want to infer the holidays in the United States,
# with consideration on +/- 2 days of each holiday as potential candidates too.

hi = HolidayInferrer()
result = hi.infer_holidays(
    df=df,
    countries=["US"],                   # Search holidays in United States
    plot=True,                          # Output a plot
    pre_search_days=2,                  # Considers 2 days before each holiday
    post_search_days=2,                 # Considers 2 days after each holiday
    independent_holiday_thres=0.9,      # The first 90% of effects are modeled separately
    together_holiday_thres=0.99,        # The 90% to 99% of effects are modeled together
    baseline_offsets=[-7, 7]            # The baseline is the average of -7/+7 observations
)

# %%
# We can plot the inferred holiday results.

plotly.io.show(result["fig"])

# %%
# The class also has a method to generate the holiday configuration
# based on the inferred results, that is consumable directly by the Silverkite model.

hi.generate_daily_event_dict()
