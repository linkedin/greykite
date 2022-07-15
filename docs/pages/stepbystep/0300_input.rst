Examine Input Data
==================

Expected format
---------------

Your input ``df`` should be pandas DataFrame with a time column and value column.

* The `time column` can have any format recognized by `pandas.to_datetime`
* The `value column` should be numeric and non-negative. Missing values are allowed.

Include regressors as columns in ``df``. Forecasts will start after the date of the last
non-null observation in `value column`. The regressors must be available past this date
to create a forecast.

Your ``df`` will look like this:

.. code-block:: none

  (x) data point
  (-) missing data

  time_col    value_col    regressor1_col    regressor2_col
  x           x            x                 x
  x           -            x                 x       <- missing values okay; let Greykite handle imputation
  x           x            -                 x       <- missing values okay; let Greykite handle imputation
  x           x            x                 -       <- missing values okay; let Greykite handle imputation
  x           x            x                 x
  x           x            x                 x
  -           -            -                 -       <- missing values okay; let Greykite handle imputation
  x           x            x                 x
  x           x            x                 x
  x           -            x                 x       <- forecast start date, continue to provide regressors
  x           -            x                 x
  x           -            -                 x       <- Greykite will impute regressor1
  x           -            x                 x
  x           -            x                 -       <- Greykite will impute regressor2
  x           -            x                 x
  x           -            -                 -       <- Greykite will impute regressor1 and regressor2
  x           -            x                 x
  x           -            x                 x       <- last date for prediction (no regressors after this point)
  x           -            -                 -       <- no prediction
  x           -            -                 -       <- no prediction
  x           -            -                 -       <- no prediction

  note: for clarity, this diagram shows time_col sorted in ascending order. This is
  not required for your input data.

.. note::

    The input data frequency can be whatever you'd like. Hourly, daily, weekly, monthly,
    every 6 hours, etc.

.. note::

    Greykite handles missing values and even missing timestamps. You should *not* impute the missing
    values on your own.

    Greykite's imputation methods prevent leakage of future information into the past during time-series
    cross validation and backtesting. They allow imputation based on future values, but only within each
    training set. See :doc:`/pages/model_components/1000_override`.

.. note::

  As a rule of thumb, provide at least twice as much training data as you intend to forecast.

  A more nuanced answer considers seasonality: you need a few full seasonality cycles to properly model
  the seasonality patterns and distinguish them from other terms (e.g. growth). For example, if yearly
  seasonality is important to your problem, then provide at least 2 years of data.

  You can still create a forecast with fewer data points, e.g. 1 year for yearly seasonality, but
  it may be more challenging to train a good model or do proper historical validation.

.. tip::

  Sometimes you may have a dataframe with additional columns not relevant to the forecast.

  When creating a forecast, subset the ``df`` to the relevant columns:
  ``df[[time_col, value_col, regressor_col1, regressor_col2, ...]].``

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd

    # no regressors
    df = pd.DataFrame({
        "ts": pd.date_range(start="2018-01-03", periods=400, freq="D"),
        "value": np.random.normal(size=400)
    })

    # with regressors
    df = pd.DataFrame({
        "ts": pd.date_range(start="2018-01-03-00", periods=5, freq="H"),
        "value": [1.0, 2.0, 3.0, None, None],
        "regressor1": [0.19, None, 0.14, 0.16, 0.17],
        "regressor2": [1.18, 1.12, 1.14, 1.16, None],
        "regressor3": [2.17, 2.12, 2.14, 2.16, 2.17]
    })


Inspect data
------------

We will be using
`~greykite.framework.input.univariate_time_series.UnivariateTimeSeries`
to inspect the data.

.. note::

  While following steps are not necessary to create a forecast, it's
  always helpful to know what your data looks like.

  Greykite provides functions to visualize your input timeseries
  and examine the trend, seasonality, holidays.


Load data
~~~~~~~~~

Make sure your data loads correctly. First, check the printed logs of ``load_data``.

.. code-block:: python

    from greykite.framework.input.univariate_time_series import UnivariateTimeSeries

    ts = UnivariateTimeSeries()
    ts.load_data(
        df=df,
        time_col="ts",
        value_col="y",
        freq="D")  # optional, but recommended if you have missing data points
                   # W for weekly, D for daily, H for hourly, etc. See ``pd.date_range``


Here is some example logging info for hourly data. The loaded data spans 2017-10-11 to 2020-02-23.
11 missing dates were added.

::

    INFO:root:Added 11 missing dates. There were 20773 values originally.
    INFO:root:Input time stats:
    INFO:root:  data points: 20784
    INFO:root:  avg increment (sec): 3600.00
    INFO:root:  start date: 2017-10-11 00:00:00
    INFO:root:  end date: 2020-02-23 23:00:00
    INFO:root:Input value stats:
    INFO:root:count     20773.000000
    mean     234249.356472
    std       30072.193941
    min        9169.000000
    25%      191494.000000
    50%      234046.000000
    75%      242572.000000
    max      34832.000000
    Name: y, dtype: float64
    INFO:root:  last date for fit: 2020-02-23 23:00:00
    INFO:root:  columns available to use as regressors: []
    INFO:root:  last date for regressors:

Alternatively, if you already have a forecast from
:class:`~greykite.framework.templates.forecaster.Forecaster`,
the time series is included in the result.

.. code-block:: python

    from greykite.framework.templates.autogen.forecast_config import ForecastConfig
    from greykite.framework.templates.autogen.forecast_config import MetadataParam
    from greykite.framework.templates.forecaster import Forecaster
    from greykite.framework.templates.model_templates import ModelTemplateEnum

    metadata = MetadataParam(
        time_col="ts",
        value_col="y",
        freq="D"
    )
    forecaster = Forecaster()
    result = forecaster.run_forecast_config(
        df=df,  # input data
        config=ForecastConfig(
            model_template=ModelTemplateEnum.AUTO.name,
            metadata_param=metadata,
            forecast_horizon=30,
            coverage=0.95
        )
    )
    ts = result.timeseries  # a `UnivariateTimeSeries`

You can also check the information programatically:

.. code-block:: python

    print(ts.time_stats)         # time statistics
    print(ts.value_stats)        # value statistics
    print(ts.freq)               # frequency
    print(ts.regressor_cols)     # available regressors
    print(ts.last_date_for_fit)  # last date with value_col
    print(ts.last_date_for_reg)  # last date for any regressor
    print(ts.df.head())          # the standardized dataset for forecasting
    print(ts.fit_df.head())      # the standardized dataset for fitting and historical evaluation

Simple plot
~~~~~~~~~~~

The best way to check your data is to plot it. You can do this interactively in a Jupyter notebook.

.. code-block:: python

    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode(connected=True)   # for generating offline graphs within Jupyter Notebook

    fig = ts.plot()
    iplot(fig)


Anomalies
~~~~~~~~~

An anomaly is a deviation in the metric that is not expected to occur again
in the future. Check for anomalies using ``ts.plot()`` and label them before
forecasting.

You may label anomalies by passing ``anomaly_info`` to ``load_data()``.
An anomaly in a timeseries is defined by its time period (start, end).
If you are able to estimate the hypothetical value had the
anomaly not occurred, you may specify an adjustment to get this corrected value.
Otherwise, the values during the anomalous period will simply be masked
and properly handled when forecasting. It is important to provide the
anomaly information, rather than correcting the data yourself.

* ``ts.df`` contains the values after adjustments, and ``ts.df_before_adjustment`` contains
  the values before adjustment.
* The plot function has an option to show the anomaly
  adjustment (``show_anomaly_adjustment=True``).
* The same ``anomaly_info`` can be used in the forecast configuration.
  See :ref:`Anomaly Configuration <anomaly-info>`

For example:

.. code-block:: python

    import numpy as np
    import pandas as pd

    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode(connected=True)   # for generating offline graphs within Jupyter Notebook

    import greykite.common.constants as cst
    from greykite.framework.input.univariate_time_series import UnivariateTimeSeries

    # Suppose 30.0 is an anomaly in "value" and we know the value should be lowered by 27.
    # Suppose 20.17 and 20.12 are an anomalies in "regressor3" and we don't know the true value,
    # so they should be replaced with np.nan.
    df = pd.DataFrame({
        "ts": ["2018-07-13", "2018-07-14", "2018-07-15", "2018-07-16", "2018-07-17"],
        "y": [1.0, 2.0, 30.0, None, None],
        "regressor1": [0.19, None, 0.14, 0.16, 0.17],
        "regressor2": [1.18, 1.12, 1.14, 1.16, None],
        "regressor3": [20.17, 20.12, 2.14, 2.16, 2.17]
    })
    # The corrected df should look like this:
    # df_adjusted = pd.DataFrame({
    #     "ts": ["2018-07-13", "2018-07-14", "2018-07-15", "2018-07-16", "2018-07-17"],
    #     "y": [1.0, 2.0, 3.0, None, None],
    #     "regressor1": [0.19, None, 0.14, 0.16, 0.17],
    #     "regressor2": [1.18, 1.12, 1.14, 1.16, None],
    #     "regressor3": [None, None, 2.14, 2.16, 2.17]
    # })

    # Specify anomalies using ``anomaly_df``.
    # Each row corresponds to an anomaly. The start date, end date,
    # and impact (if known) are provided. Extra columns can be
    # used to annotate information such as which metrics the
    # anomaly applies to.
    anomaly_df = pd.DataFrame({
        # start and end date are inclusive
        cst.START_TIME_COL: ["2018-07-15", "2018-07-13"],  # inclusive
        cst.END_TIME_COL: ["2018-07-15", "2018-07-14"],    # inclusive
        cst.ADJUSTMENT_DELTA_COL: [-27, np.nan],
        cst.METRIC_COL: ["y", "regressor3"]
    })
    # ``anomaly_info`` dictates which columns
    # in ``df`` to correct (``value_col`` below), and which rows
    # in ``anomaly_df`` to use to correct them.
    # Rows are filtered using ``filter_by_dict``.
    anomaly_info = [
        {
            "value_col": "y",
            "anomaly_df": anomaly_df,
            "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL,
            "filter_by_dict": {cst.METRIC_COL: "y"},
        },
        {
            "value_col": "regressor3",
            "anomaly_df": anomaly_df,
            "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL,
            "filter_by_dict": {cst.METRIC_COL: "regressor3"},
        },
    ]

    # Pass ``anomaly_info`` to ``load_data``.
    # Since our dataset has regressors, we pass ``regressor_cols`` as well.
    ts = UnivariateTimeSeries()
    ts.load_data(
        df=df,
        time_col="ts",
        value_col="y",
        freq="D",
        regressor_cols=["regressor1", "regressor2", "regressor3"],
        anomaly_info=anomaly_info)

    # Plots the dataset after correction
    fig = ts.plot()
    iplot(fig)
    # Set show_anomaly_adjustment=True to show the dataset before correction
    fig = ts.plot(show_anomaly_adjustment=True)
    iplot(fig)
    # The results are stored as attributes.
    ts.df                    # dataset after correction (same as ``df_adjusted`` above)
    ts.df_before_adjustment  # dataset before correction (same as ``df`` above)

Check trend
~~~~~~~~~~~

Plot your data over time to see how it trends.

If you have daily or hourly data, it helps to aggregate.
For example, look at weekly averages.

.. code-block:: python

    import numpy as np

    # aggregate daily data to weekly
    fig = ts.plot_grouping_evaluation(
        aggregation_func=np.mean,  # any aggregation function you want
        aggregation_func_name="mean",
        groupby_time_feature=None,
        groupby_sliding_window_size=7,  # any aggregation window you want
                                        # (7*24 for weekly aggregation of hourly data)
        groupby_custom_column=None,
        title=f"Weekly average of {value_col}")
    iplot(fig)

For a more detailed examination, including automatic changepoint detection,
see :doc:`/gallery/quickstart/01_exploration/0100_changepoint_detection`.

Check seasonality
~~~~~~~~~~~~~~~~~

Look for cyclical patterns in your data (i.e. seasonality).

For example, daily seasonality is a pattern that repeats once per day.
To check daily seasonality, aggregate by hour of day and plot the average:

.. code-block:: python

    fig = ts.plot_grouping_evaluation(
        aggregation_func=np.mean,
        aggregation_func_name="mean",
        groupby_time_feature="hour",  # hour of day
        groupby_sliding_window_size=None,
        groupby_custom_column=None,
        title=f"daily seasonality: mean of {value_col}")
    iplot(fig)


To check weekly seasonality, group by day of week.

.. code-block:: python

    fig = ts.plot_grouping_evaluation(
        aggregation_func=np.mean,
        aggregation_func_name="mean",
        groupby_time_feature="str_dow",  # day of week
        groupby_sliding_window_size=None,
        groupby_custom_column=None,
        title=f"weekly seasonality: mean of {value_col}")
    iplot(fig)


To check yearly seasonality, group by week of year.

.. code-block:: python

    fig = ts.plot_grouping_evaluation(
        aggregation_func=np.mean,
        aggregation_func_name="mean",
        groupby_time_feature="woy",  # week of year
        groupby_sliding_window_size=None,
        groupby_custom_column=None,
        title=f"yearly seasonality: mean of {value_col}")
    iplot(fig)

To see other features to group by:
see :py:func:`~greykite.common.features.timeseries_features.build_time_features_df`.

For a more detailed examination using a more powerful
plotting function, see :doc:`/gallery/quickstart/01_exploration/0300_seasonality_plots`.
