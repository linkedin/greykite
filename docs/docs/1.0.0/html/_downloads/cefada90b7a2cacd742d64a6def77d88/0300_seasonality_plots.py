"""
Seasonality Plots
=================

Forecast models learn seasonal (cyclical) patterns and project them into the
future. Understanding the seasonal patterns in your dataset
can help you create a better forecast. Your goal is to identify which
seasonal patterns are most important to capture, and which should
be excluded from the model.

This tutorial explains how to identify the dominant seasonal patterns and
check for interactions (e.g. daily seasonality that depends on day of week,
or weekly seasonality increases in magnitude over time). Such interactions
are important to model if they affect a large number of data points.

We use the Peyton Manning dataset as a running example.
"""

# %%
# Quick reference
# ---------------
# You will learn how to use the function
# `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`
# in `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries` to assess seasonal patterns.
#
# Steps to detect seasonality:
#
# #. Start with the longest seasonal cycles to see the big picture, then proceed to shorter cycles.
#    (yearly -> quarterly -> monthly -> weekly -> daily).
# #. For a given seasonality period:
#
#    a. First, check for seasonal effect over the entire timeseries (main effect).
#       Look for large variation that depends on the location in the cycle.
#       Pick the time feature for your seasonality cycle. See available time features at
#       `~greykite.common.features.timeseries_features.build_time_features_df`.
#
#         - for yearly: ``"doy"``, ``"month_dom"``, ``"woy_dow"``
#         - for quarterly: ``"doq"``
#         - for monthly ``"dom"``
#         - for weekly: ``"str_dow"``, ``"dow_grouped"``, ``"is_weekend"``, ``"woy_dow"``
#         - for daily: ``"hour"``, ``"tod"``
#         - ("do" = "day of", "to" = "time of")
#
#       .. code-block:: python
#
#         fig = ts.plot_quantiles_and_overlays(
#             groupby_time_feature="doy",  # day of year (yearly seasonality)
#             show_mean=True,              # shows mean on the plot
#             show_quantiles=[0.1, 0.9],   # shows quantiles [0.1, 0.9] on the plot
#             xlabel="day of year",
#             ylabel=ts.original_value_col,
#             title="yearly seasonality",
#         )
#
#    b. Then, check for interactions by adding overlays and centering the values.
#       These may be present even when there is no main effect::
#
#         # random sample of individual overlays (check for clusters with similar patterns)
#         fig = ts.plot_quantiles_and_overlays(
#             groupby_time_feature="str_dow",  # day of week (weekly seasonality)
#             show_mean=True,
#             show_quantiles=False,
#             # shows every 5th overlay. (accepts a list of indices/names, a number to sample, or `True` to show all)
#             show_overlays=np.arange(0, ts.df.shape[0], 5),
#             center_values=True,
#             # each overlay contains 28 observations (4 weeks)
#             overlay_label_sliding_window_size=28,
#             xlabel="day of week",
#             ylabel=ts.original_value_col,
#             title="weekly seasonality with selected 28d averages",
#         )
#         # interactions with periodic time feature
#         fig = ts.plot_quantiles_and_overlays(
#             groupby_time_feature="str_dow",
#             show_mean=True,
#             show_quantiles=False,
#             show_overlays=True,
#             center_values=True,
#             # splits overlays by month (try other features too)
#             overlay_label_time_feature="month",
#             # optional overlay styling, passed to `plotly.graph_objects.Scatter`
#             overlay_style={"line": {"width": 1}, "opacity": 0.5},
#             xlabel="day of week",
#             ylabel=ts.original_value_col,
#             title="weekly seasonality by month",
#         )
#         # interactions with an event (holiday, etc.)
#         fig = ts.plot_quantiles_and_overlays(
#             groupby_time_feature="str_dow",
#             show_mean=True,
#             show_quantiles=False,
#             show_overlays=True,
#             center_values=True,
#             # splits overlays by custom pd.Series value
#             overlay_label_custom_column=is_football_season,
#             overlay_style={"line": {"width": 1}, "opacity": 0.5},
#             # optional, how to aggregate values for each overlay (default=mean)
#             aggfunc=np.nanmean,
#             xlabel="day of week",
#             ylabel=ts.original_value_col,
#             title="weekly seasonality:is_football_season interaction",
#         )
#         # seasonality changepoints (option a): overlay against time (good for yearly/quarterly/monthly)
#         fig = ts.plot_quantiles_and_overlays(
#             groupby_time_feature="woy_dow",  # yearly(+weekly) seasonality
#             show_mean=True,
#             show_quantiles=True,
#             show_overlays=True,
#             overlay_label_time_feature="year",  # splits by time
#             overlay_style={"line": {"width": 1}, "opacity": 0.5},
#             center_values=True,
#             xlabel="weekofyear_dayofweek",
#             ylabel=ts.original_value_col,
#             title="yearly and weekly seasonality for each year",
#         )
#         # seasonality changepoints (option b): overlay by seasonality value (good for daily/weekly/monthly)
#         # see advanced version below, where the mean is removed.
#         fig = ts.plot_quantiles_and_overlays(
#             # The number of observations in each sliding window.
#             # Should contain a whole number of complete seasonality cycles, e.g. 24*7*k for k weekly seasonality cycles on hourly data.
#             groupby_sliding_window_size=7*13,  # x-axis, sliding windows with 13 weeks of daily observations.
#             show_mean=True,
#             show_quantiles=False,
#             show_overlays=True,
#             center_values=True,
#             # overlays by the seasonality of interest (e.g. "hour", "str_dow", "dom")
#             overlay_label_time_feature="str_dow",
#             overlay_style={"line": {"width": 1}, "opacity": 0.5},
#             ylabel=ts.original_value_col,
#             title="daily averages over time (centered)",
#         )
#
# #. For additional customization, fetch the dataframe for plotting via
#    `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.get_quantiles_and_overlays`,
#    compute additional stats as needed, and plot with
#    `~greykite.common.viz.timeseries_plotting.plot_multivariate`.
#    For example, to remove the mean effect in seasonality changepoints (option b)::
#
#        grouped_df = ts.get_quantiles_and_overlays(
#            groupby_sliding_window_size=7*13,  # accepts the same parameters as `plot_quantiles_and_overlays`
#            show_mean=True,
#            show_quantiles=False,
#            show_overlays=True,
#            center_values=False,  # note! does not center, to compute raw differences from the mean below
#            overlay_label_time_feature="str_dow",
#        )
#        overlay_minus_mean = grouped_df[OVERLAY_COL_GROUP] - grouped_df[MEAN_COL_GROUP].values  # subtracts the mean
#        x_col = overlay_minus_mean.index.name
#        overlay_minus_mean.reset_index(inplace=True)  # `plot_multivariate` expects the x-value to be a column
#        fig = plot_multivariate(  # plots the deviation from the mean
#            df=overlay_minus_mean,
#            x_col=x_col,
#            ylabel=ts.original_value_col,
#            title="day of week effect over time",
#        )
#
#
# #. The yearly seasonality plot can also be used to check for holiday effects. Click
#    and drag to zoom in on the dates of interest::
#
#       fig = ts.plot_quantiles_and_overlays(
#           groupby_time_feature="month_dom",  # date on x-axis
#           show_mean=True,
#           show_quantiles=False,
#           show_overlays=True,
#           overlay_label_time_feature="year",  # see the value for each year
#           overlay_style={"line": {"width": 1}, "opacity": 0.5},
#           center_values=True,
#           xlabel="day of year",
#           ylabel=ts.original_value_col,
#           title="yearly seasonality for each year (centered)",
#       )
#
# .. tip::
#   #. `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`
#      allows grouping or overlays by (1) a time feature, (2) a sliding window, or (3) a custom column.
#      See available time features at `~greykite.common.features.timeseries_features.build_time_features_df`.
#   #. You can customize the plot style. See
#      `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`
#      for details.
#
# Load data
# ---------
# To start, let's plot the dataset. It contains daily observations between
# ``2007-12-10`` and ``2016-01-20``.

# necessary imports
import numpy as np
import plotly

from greykite.framework.input.univariate_time_series import UnivariateTimeSeries
from greykite.framework.constants import MEAN_COL_GROUP, OVERLAY_COL_GROUP
from greykite.common.constants import TIME_COL
from greykite.common.data_loader import DataLoader
from greykite.common.viz.timeseries_plotting import add_groupby_column
from greykite.common.viz.timeseries_plotting import plot_multivariate

# Loads dataset into pandas DataFrame
dl = DataLoader()
df = dl.load_peyton_manning()
df.rename(columns={"y": "log(pageviews)"}, inplace=True)  # uses a more informative name

# plots dataset
ts = UnivariateTimeSeries()
ts.load_data(
    df=df,
    time_col="ts",
    value_col="log(pageviews)",
    freq="D")
fig = ts.plot()
plotly.io.show(fig)

# %%
# Yearly seasonality
# ------------------
# Because the observations are at daily frequency,
# it is possible to see yearly, quarterly, monthly, and weekly seasonality.
# The name of the seasonality refers to the length of one cycle. For example,
# yearly seasonality is a pattern that repeats once a year.
#
# .. tip::
#   It's helpful to start with the longest cycle to see the big picture.
#
# To examine yearly seasonality, plot the average value by day of year.
#
# Use `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`
# with ``show_mean=True`` and ``groupby_time_feature="doy"`` (day of year).
# ``groupby_time_feature`` accepts any time feature generated by
# `~greykite.common.features.timeseries_features.build_time_features_df`.
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="doy",  # day of year
    show_mean=True,              # shows the mean
    xlabel="day of year",
    ylabel=f"mean of {ts.original_value_col}",
    title="yearly seasonality",
)
plotly.io.show(fig)

# %%
# There is a varying, non-constant pattern over the year, which indicates
# the presence of yearly seasonality. But the mean often does not reveal
# the entire story.
#
# Use `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`
# to see the volatility. Set ``show_mean=True`` and ``show_quantiles=True`` to plot the mean with the 0.1 and
# 0.9 quantiles.
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="doy",
    show_mean=True,       # shows mean on the plot
    show_quantiles=True,  # shows quantiles [0.1, 0.9] on the plot
    xlabel="day of year",
    ylabel=ts.original_value_col,
    title="yearly seasonality",
)
plotly.io.show(fig)

# %%
# The day of year does explain a lot of the variation in ``log(pageviews)``. However, the wide quantiles
# indicate that a lot of variation is not explained by this variable alone. This includes variation from
# trend, events, and other factors.
#
# You can easily request additional quantiles for a better sense of the distribution.
# Pass a list of the desired quantiles via ``show_quantiles``.
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="doy",
    show_mean=True,
    show_quantiles=[0.1, 0.25, 0.75, 0.9],  # specifies quantiles to include
    xlabel="day of year",
    ylabel=ts.original_value_col,
    title="yearly seasonality",
)
plotly.io.show(fig)

# %%
# Surprisingly, the 75th percentile is below the mean between days 67 and 81.
#
# .. tip::
#   Click and drag to zoom in on the plot.
#   Reset the view by double clicking inside the plot.
#
# To better understand what causes the volatility, we can use overlays to see the
# seasonality pattern split by a dimension of interest. Let's plot one line
# for each year to see if the pattern is consistent over time. Specify
# ``overlay_label_time_feature=True`` and
# ``overlay_label_time_feature="year"`` to request overlays, where one line is shown for
# each year.
#
# We also provide plotly styling options for the overlay lines via ``overlay_style`` (optional).
# Finally, we group by "month_dom" instead of "doy" on the x-axis to make it easier
# read the dates in "MM/DD" format.
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="month_dom",   # groups by "MM/DD", e.g. 03/20 for March 20th.
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,                 # shows overlays, as configured by `overlay_label_time_feature`
    overlay_label_time_feature="year",  # splits by "year"
    # optional overlay styling, passed to `plotly.graph_objects.Scatter`
    overlay_style={"line": {"width": 1}, "opacity": 0.5},
    xlabel="day of year",
    ylabel=ts.original_value_col,
    title="yearly seasonality for each year",
)
plotly.io.show(fig)

# %%
# Before we look too carefully, to isolate the effect against the selected groupby
# feature, it can be helpful to center the overlays. This removes the effect of trend and
# longer seasonal cycles from the overlays. Each line is shifted so that the average effect
# over a cycle is zero. Quantiles and mean are shifted together, centering the mean at 0,
# to maintain their relative positions; note that quantiles are still computed on the original
# uncentered distribution.
#
# .. tip::
#   Always start with an uncentered plot with mean and quantiles to check the magnitude
#   of the seasonal effect relative to the timeseries' values. Then, center the plot and
#   use overlays to better understand the effect.
#
# The plot below is the same plot after centering with ``center_values=True``.
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
# This plot reveals some new insights:
#
#   1. Yearly seasonality is actually weak in this dataset; the line is mostly constant
#      above/below 0 depending on whether the date is during the football season, which
#      runs between September and early February.
#   2. The volatility is larger during the football season, and smaller otherwise.
#   3. The volatility in early March can be explained by a single spike in 2012.
#      Similarly, there is an anomaly in June and December.
#
# .. note::
#   The above plot can also be used to assess the effect of yearly holidays. Use
#   `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`
#   with ``overlay_label_time_feature="year"``, and ``groupby_time_feature`` set to:
#
#     - ``"doy"`` (day of year),
#     - ``"month_dom"`` (month + day of month),
#     - or ``"woy_dow"`` (week of year + day of week).
#     - (to align the holiday to the groupby value across years).
#
#   Click and drag to zoom in on a particular date range, to see the holiday's
#   effect in each year.
#
# These insights provide hints for forecasting:
#
#   - A feature indicating whether a particular date is in the football season
#     or off-season (potentially split by regular season vs playoffs), is a simple
#     way to capture most of the yearly variation.
#   - Because the season starts on a different calendar day each year, consider adding
#     add a feature for "days till start of season" and "days since end of season" to capture
#     the on-ramp and down-ramp.
#   - Check the anomalies to see if they should be considered outliers; if so,
#     remove them from the training data to avoid affecting future predictions.
#
# With the insight that the values closely depend on the football season,
# and knowing that football games are played on particular days of the week,
# starting on a particular week of the year, we may expect yearly seasonal patterns
# to depend more on "week of year" + "day of week" than on the calendar date. (The
# same calendar date can fall on a different day of the week, depending on the year.)
# To check this, simply group by ``woy_dow``. This variable is encoded as
# {week of year}_{day of week}, e.g. 04_01 for Monday of 4th week.
#
# This is a different way to label each day of the year that captures both
# yearly and weekly seasonality at the same time.
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="woy_dow",  # week of year and day of week
    show_mean=True,
    show_quantiles=True,
    show_overlays=True,
    overlay_label_time_feature="year",
    overlay_style={"line": {"width": 1}, "opacity": 0.5},
    center_values=True,
    xlabel="weekofyear_dayofweek",
    ylabel=ts.original_value_col,
    title="yearly and weekly seasonality for each year",
)
plotly.io.show(fig)

# %%
# Notice a much stronger relationship than before: the mean varies more
# with the x-axis value, with tigher quantiles, so ``woy_dow`` explains more
# variability in the time series. There is a different weekly pattern during and outside
# the football season, with increasing volatility toward the playoffs (end of season).
# Next, let's explore the weekly patterns in more detail.
#
# Weekly seasonality
# ------------------
# So far, we learned that the main seasonal effects depend on day of week and whether the day
# is during the football season.
#
# To check overall weekly seasonality, group by day of week (``str_dow``). We
# also set ``overlay_label_sliding_window_size=7`` and ``show_overlays=20`` to
# plot the values for 20 randomly selected weeks from the dataset. The "size" parameter indicates
# the number of sequential observations contained in each overlay (7=1 week). In the legend, each
# overlay is labeled by the first date in the overlay's sliding window.
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="str_dow",
    show_mean=True,
    show_quantiles=True,
    show_overlays=20,  # randomly selects up to 20 overlays
    overlay_label_sliding_window_size=7,  # each overlay is a single cycle (week)
    center_values=False,
    xlabel="day of week",
    ylabel=ts.original_value_col,
    title="weekly seasonality with overlays"
)
plotly.io.show(fig)

# %%
# In the above plot, the effect doesn't vary much by day of week,
# but quantiles are large. Such a plot indicates one of two possibilities:
#
#   1) there is no seasonal pattern for this period (cycle length)
#   2) there is a seasonal pattern for this period, but it
#      is not consistent across the entire timeseries.
#
# (2) is possible when seasonality depends on an interaction
# term. It may vary by a time dimension, change during an event,
# or evolve over time. In this case, it could be useful to model
# the seasonality conditional on the parameter when forecasting
# (interaction terms).
#
# For the Peyton Manning dataset, we know there is weekly seasonality
# during the football season. We suspect the effect is washed out in the
# above plot, because it averages weekly seasonality during the season
# and off-season.
#
# Suppose we did not already have this insight. How could we detect the presence
# of weekly seasonality conditional on interactions?
#
#   - Overlays of individual cycles can suggest the presence of an interaction effect.
#   - Look for clusters of overlay lines with similar (and not flat) patterns.
#     Try to identify what they have in common.
#
# The previous plot showed a random sample of 20 overlays. The plot below
# selects every 5th overlay, evenly spaced through time. Each overlay is
# the average of a 28 day sliding window (four cycles) to smooth out volatility
# (``overlay_label_sliding_window_size=28``).
# There is a trade off when setting sliding window size:
#
#   - Smaller window = see unique effects, but adds noise
#   - Larger window = smooths out noise, but values regress toward the mean and may hide effects.
#
# Given this tradeoff, try a few window sizes to see if any patterns emerge.

# Selects every 5th overlay. ``which_overlays`` is a list of
# allowed overlays. Each overlay spans 28 days, so every 5th overlay
# allows selection of different months across years.
which_overlays = np.arange(0, ts.df.shape[0], 5)  # ``ts.df.shape[0]`` is an upper bound on the number of overlays
overlay_style = {  # this is the default style
    "opacity": 0.5,
    "line": dict(
        width=1,
        color="#B3B3B3",  # light gray
        dash="solid"),
    "legendgroup": OVERLAY_COL_GROUP}
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="str_dow",
    show_mean=True,
    show_quantiles=False,
    show_overlays=which_overlays,  # indices to show. Also accepts a list of strings (overlay names).
    center_values=True,
    overlay_label_sliding_window_size=28,  # each overlay contains 28 observations (4 weeks)
    overlay_style=overlay_style,
    xlabel="day of week",
    ylabel=ts.original_value_col,
    title="weekly seasonality with 28d overlays",
)
plotly.io.show(fig)

# %%
# In the above plot, some lines are close together above/below the mean
# on Monday, Saturday, and Sunday, suggesting the presence of
# an interaction pattern. In the next section, we explain how to
# detect such interactions.
#
# Checking for interactions
# -------------------------
# The same function, `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.plot_quantiles_and_overlays`,
# can be used to check the three possible interaction factors:
#
#   1) interaction with time dimension,
#   2) interaction with events,
#   3) seasonality changepoints
#
# 1) Time dimension interaction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# It is common for a seasonality pattern to depend on a time dimension.
# For example, daily seasonality may differ by day of week, or
# weekly seasonality may change around year end. The seasonality
# changes periodically with a time feature.
#
# To check this, use ``overlay_label_time_feature``. We check whether
# weekly seasonality interacts with month, by setting
# ``overlay_label_time_feature="month"``.
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
# There is a clear interaction -- notice two clusters of lines with
# different weekly seasonality patterns.
# (When forecasting, we do need to pay special attention to February,
# whose line is in between the two clusters. This is because it has
# one weekend in the football season and one weekend outside it.
# The month interaction alone is too coarse to reflect this.)
#
# 2) Event/holiday interaction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# It is also common to have yearly seasonality that interacts with
# an event. For example, for hourly traffic data, a holiday can affect
# the daily seasonality as rush hour traffic is reduced.
# In our dataset, the football season may affect the weekly
# seasonality.
#
# .. note::
#   Both ``events`` and ``time dimensions`` occur at known times in the future;
#   the difference is that events require external knowledge about when they
#   occur, whereas time dimensions can be derived directly from
#   the date itself, without any external knowledge.
#
#   Our library contains information about the dates of common holidays,
#   but you will need to supply information about other events if desired.
#
# You can pass a custom `pandas.Series` to the plotting function
# to define overlays. The series assigns a label to each row, and
# must have the same length as your input data.
#
# In the code below, we create two overlays using
# a (rough) indicator for ``is_football_season``.
# We used
# `~greykite.common.viz.timeseries_plotting.add_groupby_column`
# to get the derived time feature used to define this indicator.
# See the function's documentation for details.

# Defines `is_football_season` by "week of year",
# using `add_groupby_column` to get the "week of year" time feature.
df_week_of_year = add_groupby_column(
    df=ts.df,
    time_col=TIME_COL,           # The time column in ts.df is always TIME_COL
    groupby_time_feature="woy")  # Computes "week of year" based on the time column
added_column = df_week_of_year["groupby_col"]
week_of_year = df_week_of_year["df"][added_column]
is_football_season = (week_of_year <= 6) | (week_of_year >= 36)  # rough approximation
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="str_dow",
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    center_values=True,
    overlay_label_custom_column=is_football_season,  # splits overlays by `is_football_season` value
    overlay_style={"line": {"width": 1}, "opacity": 0.5},
    aggfunc=np.nanmean,  # how to aggregate values for each overlay (default=mean)
    xlabel="week of year",
    ylabel=ts.original_value_col,
    title="weekly seasonality:is_football_season interaction",
)
plotly.io.show(fig)

# %%
# ``is_football_season`` is able to distinguish the two weekly seasonality
# patterns identified by previous plots. There is strong weekly seasonality
# during the football season, but not outside it. Forecasts that use weekly
# seasonality should account for this important interaction.
#
# 3) Seasonality changepoint
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Lastly, we check if seasonality changes over time.
# For example, the seasonality may increase or decrease
# in magnitude, or its shape may change.
#
# For this, use seasonality changepoint detection.
# See :doc:`/gallery/quickstart/01_exploration/0100_changepoint_detection`
# for details.
#
# Plots can provide additional understanding to tune
# the parameters for changepoint detection.
#
# Let's plot the mean value by "day of week" over time. There will be one
# line for Mondays, one for Tuesdays, etc. We are looking for a change
# in the distribution of the values around the mean; this could indicate,
# for example, that the value on Mondays becomes a smaller % of the weekly
# total over time.
#
# Unlike before, notice that day of week is now the `overlay` feature,
# and we group by sliding windows of 91 observations each. The x-axis is
# indexed by the start of each window. You can adjust the window size
# as you'd like; as before, larger windows smooth out noise, but if the
# window is too large, it may mask meaningful changes.
fig = ts.plot_quantiles_and_overlays(
    groupby_sliding_window_size=7*13,  # x-axis, sliding windows with 91 days (13 weeks) each
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    center_values=True,
    overlay_label_time_feature="str_dow",  # overlays by the seasonality of interest
    overlay_style={"line": {"width": 1}, "opacity": 0.5},
    ylabel=ts.original_value_col,
    title="daily averages over time (centered)",
)
plotly.io.show(fig)

# %%
# This plot is hard to assess because of mean changes
# over time. It would be more clear to see if relative offset
# from the mean changes over time.
#
# To do this, get the raw daily averages using
# `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.get_quantiles_and_overlays`,
# subtract the mean, and plot the result with
# `~greykite.common.viz.timeseries_plotting.plot_multivariate`.
grouped_df = ts.get_quantiles_and_overlays(
    groupby_sliding_window_size=7*13,  # accepts the same parameters as `plot_quantiles_and_overlays`
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    center_values=False,  # note! does not center, to compute raw differences from the mean below
    overlay_label_time_feature="str_dow",
)
overlay_minus_mean = grouped_df[OVERLAY_COL_GROUP] - grouped_df[MEAN_COL_GROUP].values  # subtracts the mean
x_col = overlay_minus_mean.index.name
overlay_minus_mean.reset_index(inplace=True)  # `plot_multivariate` expects the x-value to be a column
fig = plot_multivariate(  # plots the deviation from the mean
    df=overlay_minus_mean,
    x_col=x_col,
    ylabel=ts.original_value_col,
    title="day of week effect over time")
plotly.io.show(fig)

# %%
# The pattern looks fairly stable until Nov 2013, when Monday
# far surpasses Sunday as the weekly peak. The relative values on Monday
# and Tuesday increase, and the relative values on Saturday and Sunday decline.
# Thus, it may be useful to include a seasonality changepoint around
# that time.
#
# .. tip::
#   You can interact with the plot to focus on a particular day
#   by double clicking its name in the legend. Double click again
#   to unselect, or single click to show/hide a single series.
#
# Quarterly and monthly seasonality
# ---------------------------------
# Finally, let's check quarterly and monthly seasonality.
#
# Quarterly seasonality is weak relative to the size of the quantiles.
# The overlays do not suggest any clear interaction effects.
# It is likely not useful for a forecast model.
# (Remember to check the plot with ``center_values=False``
# as well, to better assess the magnitude of the effect.)
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="doq",  # day of quarter
    show_mean=True,
    show_quantiles=True,
    show_overlays=20,  # randomly selects up to 20 overlays
    # No explicit overlay feature. Each overlay is a single cycle (quarter)
    center_values=True,
    xlabel="day of quarter",
    ylabel=ts.original_value_col,
    title="quarterly seasonality",
)
plotly.io.show(fig)

# %%
# Monthly seasonality is weak relative to the size of the quantiles.
# The overlays do not suggest any clear interaction effects.
# It is likely not useful for a forecast model.
# (Remember to check the plot with ``center_values=False``
# as well, to better assess the magnitude of the effect.)
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature="dom",
    show_mean=True,
    show_quantiles=True,
    show_overlays=20,  # randomly selects up to 20 overlays
    # No explicit overlay feature. Each overlay is a single cycle (month)
    center_values=True,
    xlabel="day of month",
    ylabel=ts.original_value_col,
    title="monthly seasonality",
)
plotly.io.show(fig)

# %%
# How to forecast with this information
# -------------------------------------
# Our goal was to identify seasonal patterns in the dataset to create
# a better forecast.
#
# We learned that a good forecast model must model dates during the
# football season and off-season differently. At a minimum, both the
# mean value and weekly seasonality should be allowed to vary depending
# on this ``is_football_season`` variable.
#
# To accomplish this, the following approaches could be considered, from least
# to most complex:
#
#     1. ``is_football_season*weekly_seasonality`` interaction
#     2. ``month*weekly_seasonality + february_week_num*weekly_seasonality`` interaction
#     3. ``woy_dow`` effect
#
# The first option is the most basic. The second allows capturing month-specific,
# weekly seasonality patterns, with special attention given to February, which
# falls both inside and outside the football season. Each week is allowed to
# have a different weekly seasonality. february_week_num is a categorical variable
# indicating the week of February (1, 2, 3, 4). The last option model every day
# of the year as a separate variable. This is unlikely to work well because it
# has too many parameters for the amount of data.
#
# .. note::
#   Appropriately increasing model complexity can improve the model's ability
#   to capture meaningful variation. However, unnecessary complexity adds variance to
#   the forecast due to estimation noise. A sparser model can better predict the
#   future by making more efficient use of the data, as long as it captures the underlying
#   dynamics. Proper cross validation and backtesting can be used to pick the best model.
#
#   For example, while ``woy_dow`` enables modeling each day of year separately, doing so
#   is likely to overfit the training data. Typically, weekly patterns should be modeled with
#   weekly seasonality, rather than using yearly seasonality to model shorter
#   cyclical patterns.
#
# To capture other seasonal effects, the following model components can be added:
#
#     a) ``yearly_seasonality`` to capture weak yearly seasonality
#     b) ``season_start`` and ``season_end`` events to capture start and end of season effect
#     c) ``weekly seasonality changepoint`` (around Nov 2013) to capture shift in weekly seasonality shape
#
# In the "Silverkite" forecast model, the above components could be specified via
#
# .. code-block:: none
#
#    - weekly seasonality: seasonality->weekly_seasonality, custom->extra_pred_cols->"str_dow"
#    - yearly seasonality: seasonality->yearly_seasonality, custom->extra_pred_cols->"woy" or "woy_dow"
#    - is_football_season: regressors->regressor_cols->"is_football_season" (define custom regressor)
#    - start/end of season: holidays->daily_event_df_dict->"season_start","season_end" (define custom event)
#    - interactions: custom->feature_sets_enabled, custom->extra_pred_cols (define interactions yourself)
#    - changepoint: changepoints->seasonality_changepoints_dict
#
# See :doc:`/pages/model_components/0100_introduction` for details.
#
# Daily seasonality
# -----------------
# The Peyton Manning dataset cannot have daily seasonality
# (variation within one day), because there is only one observation
# each day.
#
# For completeness, we show how to test for daily seasonality
# using an hourly bike sharing dataset.
#
# First, prepare and load your dataset.
df = dl.load_bikesharing()
bikesharing_ts = UnivariateTimeSeries()
bikesharing_ts.load_data(
    df=df,
    time_col="ts",
    value_col="count",
    freq="H",
    regressor_cols=["tmax", "tmin", "pn"]
)
plotly.io.show(bikesharing_ts.plot())

# %%
# We proceed with further exploration for now. Group by
# ``"hour"`` to see the daily seasonality effect.
# There is more bikesharing activity during the day than at night.
fig = bikesharing_ts.plot_quantiles_and_overlays(
    groupby_time_feature="hour",
    show_mean=True,
    show_quantiles=True,
    show_overlays=25,
    overlay_label_sliding_window_size=24,  # each overlay contains 24 observations (1 day)
    center_values=False,
    xlabel="hour of day",
    ylabel="number of shared bikes",
    title="bike sharing activity by hour of day"
)
plotly.io.show(fig)

# %%
# Check for interactions with day of week as follows. In this plot, weekdays
# follow a similar pattern, but Saturday and Sunday are different.
fig = bikesharing_ts.plot_quantiles_and_overlays(
    groupby_time_feature="hour",
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    center_values=True,
    overlay_label_time_feature="str_dow",  # splits overlays by day of week
    overlay_style={"line": {"width": 1}, "opacity": 0.5},
    xlabel="hour of day",
    ylabel="number of shared bikes",
    title="bike sharing daily seasonality, by day of week"
)
plotly.io.show(fig)

# %%
# As an aside, for multivariate datasets, you may set
# ``value_col`` to check the seasonality pattern
# for a different metric in the dataset.
# The bike sharing dataset is a multivariate dataset
# with columns "tmax", "tmin", "pn" for max/min daily
# temperature and precipitation. Let's plot max daily
# temperature by week of year.
print(f"Columns: {bikesharing_ts.df.columns}")
fig = bikesharing_ts.plot_quantiles_and_overlays(
    value_col="tmax",
    groupby_time_feature="woy",
    show_mean=True,
    show_quantiles=True,
    show_overlays=False,
    center_values=False,
    xlabel="week of year",
    title="max daily temperature by week of year"
)
plotly.io.show(fig)
