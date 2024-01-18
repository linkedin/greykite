Configure a Forecast
====================

Use the class :class:`~greykite.framework.templates.forecaster.Forecaster`
to create a forecast. It has a method, ``run_forecast_config`` that accepts the input
data (``df``) and forecast configuration (``config``).

.. code-block:: python

    from greykite.framework.templates.autogen.forecast_config import (
        ComputationParam, EvaluationMetricParam, EvaluationPeriodParam,
        ForecastConfig, MetadataParam, ModelComponentsParam)
    from greykite.framework.templates.forecaster import Forecaster

    # defines forecast configuration
    config=ForecastConfig(
        model_template=model_template,                       # which model template to use
        forecast_horizon=forecast_horizon,                   # how many steps ahead to forecast
        coverage=coverage,                                   # intended coverage of the prediction bands
        metadata_param=MetadataParam(...),                   # input data description
        evaluation_metric_param=EvaluationMetricParam(...),  # what metric to evaluate
        evaluation_period_param=EvaluationPeriodParam(...),  # how to evaluate (train/test splits)
        model_components_param=ModelComponentsParam(...),    # model template tuning parameters
        computation_param=ComputationParam(...),             # parallelization
        forecast_one_by_one=forecast_one_by_one,             # allows training multiple models that span the horizon
    )

    # creates forecast
    forecaster = Forecaster()
    result = forecaster.run_forecast_config(
        df=df,         # input data
        config=config  # forecast configuration
    )

For basic usage, provide ``df`` and specify a ``config`` containing

* ``model_template``,
* ``forecast_horizon``,
* ``coverage``,
* ``metadata_param``.

The function will parse your input data according to ``metadata_param``.
The output contains a forecast for the specified forecast
horizon and coverage using the model template's defaults.

The other parameters can be used to tune the model and define evaluation criteria.
See the sections below for explanations.

df
--
Required. Your input data. See :doc:`/pages/stepbystep/0300_input` for details on data format.

config
------
Optional. The forecast configuration.
An instance of :class:`~greykite.framework.templates.autogen.forecast_config.ForecastConfig`.

The following sections explain each optional attribute.

model_template
--------------
Optional. Name of the model template. A model template is a pre-packaged forecasting configuration. You
can tune the template using the other parameters in the ``config``.

Examples:

.. code-block:: python

    model_template = "SILVERKITE"  # the default
    model_template = "PROPHET"

For the full list of options, see `~greykite.framework.templates.model_templates.ModelTemplateEnum`.

For a high level comparison between Silverkite and Prophet template families,
see :doc:`/pages/stepbystep/0100_choose_model`.

forecast_horizon
----------------
Optional. Number of periods to forecast into the future. Must be > 0.

If not provided, default is determined from input data frequency.

Examples:

.. code-block:: python

    forecast_horizon = 30      # one month ahead, for daily data
    forecast_horizon = 365*24  # one year ahead, for hourly data
    forecast_horizon = 52      # one year ahead, for weekly data

.. _coverage:

coverage
--------
Optional. Intended coverage of the prediction interval. Must be between 0.0 and 1.0.

Prediction intervals quantify the uncertainty of the forecast. They create a band that
goes above/below the forecasted value, to provide an upper/lower prediction.

``coverage`` specifies what % of points you want to fall within the bands.
Larger coverage results in wider bands.

Examples:

.. code-block:: python

    coverage = None  # no prediction interval
    coverage = 0.80  # 80% of actuals should fall within the prediction interval
    coverage = 0.95  # 95% of actuals should fall within the prediction interval

metadata_param
--------------
Optional. Specifies properties of the input ``df``.
An instance of :class:`~greykite.framework.templates.autogen.forecast_config.MetadataParam`.

The attributes are:

.. code-block:: none

    time_col : str, default "ts"
        name of timestamp column in df

    value_col : str, default "y"
        name of value column in df (containing values to forecast)

    freq : str or None, default None
        Frequency of input data. Used to generate future dates for prediction.
        Frequency strings can have multiples, e.g. '5H'.
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        for a list of frequency aliases.
        If None, inferred by pd.infer_freq.
        Provide this parameter if df has missing timepoints.

        Examples:
        "BH" business hour frequency
        "H" hourly frequency
        "B", business day frequency
        "D", calendar day frequency
        "W", weekly frequency

        "M", month end frequency
        "SM", semi-month end frequency (15th and end of month)
        "BM", business month end frequency
        "MS", month start frequency
        "SMS", semi-month start frequency (1st and 15th)
        "BMS", business month start frequency

        "Q", quarter end frequency
        "BQ", business quarter end frequency
        "QS", quarter start frequency
        "BQS", business quarter start frequency

        "A" or "Y" year end frequency
        "BA" or "BY" business year end frequency
        "AS" or "YS" year start frequency
        "BAS" or  "BYS" business year start frequency

    date_format : str or None, default None
        strftime format to parse time column, eg ``%m/%d/%Y``.
        Note that ``%f`` will parse all the way up to nanoseconds.
        If None (recommended), inferred by `pandas.to_datetime`.

    train_end_date : datetime.datetime or None, default None
        Last date to use for fitting the model. Forecasts are generated after this date.
        If None, it is set to the last date with a non-null value in
        ``value_col`` of ``df``.

    anomaly_info : `dict` or `list` [`dict`] or None, default None
        Anomaly adjustment info. Anomalies in ``df``
        are corrected before any forecasting is done.

        If None, no adjustments are made.

        A dictionary containing the parameters to
        `~greykite.common.features.adjust_anomalous_data.adjust_anomalous_data`.
        See that function for details.
        The possible keys are:

            ``"value_col"`` : `str`
                The name of the column in ``df`` to adjust. You may adjust the value
                to forecast as well as any numeric regressors.
            ``"anomaly_df"`` : `pandas.DataFrame`
                Adjustments to correct the anomalies.
            ``"start_time_col"``: `str`, default START_TIME_COL
                Start date column in ``anomaly_df``.
            ``"end_time_col"``: `str`, default END_TIME_COL
                End date column in ``anomaly_df``.
            ``"adjustment_delta_col"``: `str` or None, default None
                Impact column in ``anomaly_df``.
            ``"filter_by_dict"``: `dict` or None, default None
                Used to filter ``anomaly_df`` to the relevant anomalies for
                the ``value_col`` in this dictionary.
                Key specifies the column name, value specifies the filter value.
            ``"filter_by_value_col""``: `str` or None, default None
                Adds ``{filter_by_value_col: value_col}`` to ``filter_by_dict``
                if not None, for the ``value_col`` in this dictionary.
            ``"adjustment_method"`` : `str` ("add" or "subtract"), default "add"
                How to make the adjustment, if ``adjustment_delta_col`` is provided.

        Accepts a list of such dictionaries to adjust multiple columns in ``df``.

Examples:

.. code-block:: python

    from greykite.framework.templates.autogen.forecast_config import MetadataParam

    metadata = MetadataParam(
        time_col="ts",       # this is the default (TIME_COL constant)
        value_col="y",       # this is the default (VALUE_COL constant)
        freq=None,           # infer
        date_format=None,    # infer
        anomaly_info=None,   # no adjustments
    )

    metadata = MetadataParam(
        time_col="date",
        value_col="sessions",
        freq="W",
        date_format="%Y-%m-%d-%H",
        train_end_date=datetime.datetime(2020, 3, 1),
        anomaly_info=None,
    )

.. _anomaly-info:

anomaly_info
^^^^^^^^^^^^

An anomaly is a deviation in the metric that is not expected to occur again
in the future. ``anomaly_info`` can be used to adjust your input data if there
are known anomalies. For example, you can choose to mask anomalies or correct
the value to their hypothetical value, had the anomaly not occurred.
This way, the forecast model will not project the anomalous pattern into the future.
In most cases, you do not know the hypothetical value, so masking is sufficient.

You can correct anomalies in ``df`` using ``anomaly_info``.
For parameter details, see
`~greykite.common.features.adjust_anomalous_data.adjust_anomalous_data`. For an example,
see :doc:`/pages/stepbystep/0300_input`.

.. Uncomment after SILVERKITE uses original values for uncertainty calculation:
   .. tip::
      Anomalies should be removed or corrected when fitting the forecast model, but
      retained when calculating model uncertainty (prediction intervals).
      You should not correct the values yourself, because the model needs both the original
      and corrected values.
      - In SILVERKITE models, the corrected values are used for fitting the forecast,
        and the original values are used to calculate uncertainty intervals.
      - In PROPHET models, the same values are used in both fitting and uncertainty.
        Thus, if you label anomalies, the uncertainty model may be too conservative,
        and if you don't label anomalies, the forecast may be less accurate.

.. note::

   Measurement errors are different from anomalies.

   - ``Measurement error``: the actual value is misreported. Correct the values in ``df``
     before calling ``run_forecast_config``.
     For example, the database is corrupted, or a tracking error causes the actual value
     to be underreported.
   - ``Anomaly``: the measurements are accurate, but the typical pattern is disrupted in a
     one-time event. Correct these via ``anomaly_info``. For example, a site issue causes
     the actual value to drop by 20% for 1 hour.

.. tip::

  It's important to provide ``freq`` if the input data has missing timepoints.
  `pandas.infer_freq` has trouble with missing values.

.. _evaluation-metric:

evaluation_metric_param
-----------------------
Optional. Defines the metrics used to evaluate the forecast.
An instance of :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam`.

The attributes are:

.. code-block:: none

    cv_selection_metric : str or None, default "MeanAbsolutePercentError"
        EvaluationMetricEnum name, e.g. "MeanAbsolutePercentError"
        Used to select the optimal model during cross-validation.
        Defines ``score_func``, ``score_func_greater_is_better`` in ``forecast_pipeline``.

    cv_report_metrics : str, or list [str], or None, default CV_REPORT_METRICS_ALL
        Additional metrics to compute during CV, besides the one specified by ``cv_selection_metric``.

            - If the string constant `greykite.common.constants.CV_REPORT_METRICS_ALL`,
              computes all metrics in ``EvaluationMetricEnum``. Also computes
              ``FRACTION_OUTSIDE_TOLERANCE`` if ``relative_error_tolerance`` is not None.
              The results are reported by the short name (``.get_metric_name()``) for ``EvaluationMetricEnum``
              members and ``FRACTION_OUTSIDE_TOLERANCE_NAME`` for ``FRACTION_OUTSIDE_TOLERANCE``.
              These names appear in the keys of ``forecast_result.grid_search.cv_results_``
              returned by this function.
            - If a list of strings, each of the listed metrics is computed. Valid strings are
              `greykite.common.evaluation.EvaluationMetricEnum` member names
              and `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.

              For example::

                ["MeanSquaredError", "MeanAbsoluteError", "MeanAbsolutePercentError", "MedianAbsolutePercentError", "FractionOutsideTolerance2"]

            - If None, no additional metrics are computed.

    agg_periods : int or None, default None
        Number of periods to aggregate before evaluation.

        Model is fit and forecasted on the dataset's original frequency.

        Before evaluation, the actual and forecasted values are aggregated,
        using rolling windows of size ``agg_periods`` and the function
        ``agg_func``. (e.g. if the dataset is hourly, use ``agg_periods=24, agg_func=np.sum``,
        to evaluate performance on the daily totals).

        If None, does not aggregate before evaluation.

        Currently, this is only used when calculating CV metrics and
        the R2_null_model_score metric in backtest/forecast. No pre-aggregation
        is applied for the other backtest/forecast evaluation metrics.

    agg_func : callable or None, default None
        Takes an array and returns a number, e.g. np.max, np.sum.

        Defines how to aggregate rolling windows of actual and predicted values
        before evaluation.

        Ignored if ``agg_periods`` is None.

        Currently, this is only used when calculating CV metrics and
        the R2_null_model_score metric in backtest/forecast. No pre-aggregation
        is applied for the other backtest/forecast evaluation metrics.

    null_model_params : dict or None, default None
        Defines baseline model to compute ``R2_null_model_score`` evaluation metric.
        ``R2_null_model_score`` is the improvement in the loss function relative
        to a null model. It can be used to evaluate model quality with respect to
        a simple baseline. For details, see
        `~greykite.common.evaluation.r2_null_model_score`.

        The null model is a `~sklearn.dummy.DummyRegressor`,
        which returns constant predictions.

        Valid keys are "strategy", "constant", "quantile".
        See https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html

        For example::

            null_model_params = {
                "strategy": "mean",
            }
            null_model_params = {
                "strategy": "median",
            }
            null_model_params = {
                "strategy": "quantile",
                "quantile": 0.8,
            }
            null_model_params = {
                "strategy": "constant",
                "constant": 2.0,
            }

        If None, ``R2_null_model_score`` is not calculated.

        Note: CV model selection always optimizes ``score_func`, not
        the ``R2_null_model_score``.

    relative_error_tolerance : float or None, default None
        Threshold to compute the ``Outside Tolerance`` metric,
        defined as the fraction of forecasted values whose relative
        error is strictly greater than ``relative_error_tolerance``.
        If `None`, the metric is not computed.

EvaluationMetricEnum names (valid for ``cv_selection_metric`` and
``cv_report_metrics``) are listed below. See their descriptions at:
:py:class:`~greykite.common.evaluation.EvaluationMetricEnum`.

.. code-block:: none

    "MeanSquaredError"
    "RootMeanSquaredError"
    "MeanAbsoluteError"
    "MedianAbsoluteError"
    "MeanAbsolutePercentError"
    "MedianAbsolutePercentError"
    "SymmetricMeanAbsolutePercentError"
    "Quantile80"  # quantile loss, 80th quantile
    "Quantile95"  # quantile loss, 95th quantile
    "Quantile99"  # quantile loss, 99th quantile

    # auxiliary metrics (typically not optimized directly)
    "CoefficientOfDetermination"  # also known as "R2", `1.0 - MeanSquaredError / variance(actuals)`
    "FractionOutsideTolerance1"   # fraction of errors > 1%
    "FractionOutsideTolerance2"   # fraction of errors > 2%
    "FractionOutsideTolerance3"   # fraction of errors > 3%
    "FractionOutsideTolerance4"   # fraction of errors > 4%
    "FractionOutsideTolerance5"   # fraction of errors > 5%
    "Correlation"                 # correlation between forecast and actuals

In most cases, use "MeanAbsolutePercentError" as the selection metric.
Because it is a relative metric, it is comparable across forecasts.

See `~greykite.common.evaluation.r2_null_model_score` for the relationship
between "CoefficientOfDetermination" ("R2") and "R2_null_model_score".

To assess model quality, "CoefficientOfDetermination" ("R2") is preferred over
"Correlation". (They are equivalent for linear regression.)
"CoefficientOfDetermination" accounts for bias whereas "Correlation" does not.

Examples:

.. code-block:: python

    from greykite.common.constants import CV_REPORT_METRICS_ALL
    from greykite.common.evaluation import EvaluationMetricEnum
    from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam

    # Evaluates without aggregating.
    # Calculates R2_null_model_score against null model that predicts 80th quantile.
    # Note that the null model predicts the 0.8 quantile of the
    #   training set, which matches `cv_selection_metric`.
    # Reports all available metrics on each CV split.
    # 5% tolerance level to compute "Outside Tolerance" metric.
    evaluation_metric = EvaluationMetricParam(
        cv_selection_metric=EvaluationMetricEnum.Quantile80.name,
        cv_report_metrics=CV_REPORT_METRICS_ALL,  # the default, recommended
        agg_periods=None,
        agg_func=None,
        null_model_params = {
            "strategy": "quantile",
            "constant": None,
            "quantile": 0.8
        },
        relative_error_tolerance=0.05
    )

    # Creates forecast using daily data, evaluates accuracy of weekly totals.
    # Null model predicts mean of training set.
    # Reports a few extra metrics on each CV split.
    # 1% tolerance level to compute "Outside Tolerance" metric.
    evaluation_metric = EvaluationMetricParam(
        cv_selection_metric=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        cv_report_metrics=[
            EvaluationMetricEnum.MeanSquaredError.name,
            EvaluationMetricEnum.MeanAbsoluteError.name,
            EvaluationMetricEnum.MedianAbsoluteError.name,
            EvaluationMetricEnum.MedianAbsolutePercentError.name,
        ],
        agg_periods=7,
        agg_func=np.sum,
        null_model_params = {
            "strategy": "mean"
        },
        relative_error_tolerance=0.01
    )

.. note::

  If you specify ``agg_periods``, ``agg_func``, we calculate all evaluation metrics
  after aggregation, but the forecast is returned at the same frequency as the input ``df``.

  Currently, these are only used when calculating CV metrics and
  the R2_null_model_score metric in backtest/forecast. No pre-aggregation is applied
  for the other backtest/forecast evaluation metrics.

.. _evaluation-period:

evaluation_period_param
-----------------------
Optional. Defines how to split the data into train/test sets for evaluation.
An instance of :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationPeriodParam`.

Greykite runs the following steps for evaluation:

1. Run ``time-series cross validation`` (CV) to select the best hyperparameters, via grid search
2. Retrain and predict on holdout ``backtest`` period using best model
3. Retrain and predict on ``forecast`` period using best model

To do this, Greykite separates the data into three segments (training, backtest, forecast) as shown
below. Each row corresponds to a train/test split. We record train and test error for each split
(the average and std. are reported for CV).

.. code-block:: none

    x = train period
    - = forecast period
      = not used

    | TRAINING                     | BACKTEST    | FORECAST    |

    xxxxxxxxxxxxx----                                              (cross-validation)
    xxxxxxxxxxxxxxxxx----                                          (cross-validation)
    xxxxxxxxxxxxxxxxxxxxx----                                      (cross-validation)
    xxxxxxxxxxxxxxxxxxxxxxxxx----                                  (cross-validation)

    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx--------------                 (backtest)

    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx--------------   (forecast)

``evaluation_period`` has these attributes:

.. code-block:: none

    test_horizon : int or None, default None
        Numbers of periods held back from end of df for test.
        The rest is used for cross validation.
        If None, default is forecast_horizon. Set to 0 to skip backtest.

    periods_between_train_test : int or None, default None
        Number of periods for the gap between train and test data.
        Applies to both backtest and forecast, however the behaviour is slightly different.
        Check the illustration of test parameters for a visual explanation.
        If None, default is 0.

    cv_horizon : int or None, default None
        Number of periods in each CV test set
        If None, default is forecast_horizon. Set to 0 to skip CV.

    cv_min_train_periods : int or None, default None
        Minimum number of periods for training each CV fold.
        If cv_expanding_window is False, every training period is this size
        If None, default is 2 * cv_horizon

    cv_expanding_window : bool, default determined by template
        If True, training window for each CV split is fixed to the first available date.
        Otherwise, train start date is sliding, determined by cv_min_train_periods

    cv_use_most_recent_splits: `bool`, optional, default False
        If True, splits from the end of the dataset are used.
        Else a sampling strategy is applied. Check
        `~greykite.sklearn.cross_validation.RollingTimeSeriesSplit._sample_splits` for details.

    cv_periods_between_splits : int or None, default None
        Number of periods to slide the test window between CV splits. Has to be greater than or equal to 1.
        If None, default is cv_horizon.

    cv_periods_between_train_test : int, default 0
        Number of periods for the gap between train and test in a CV split.
        If None, default is periods_between_train_test.

    cv_max_splits : int or None, default 3
        Maximum number of CV splits.
        Given the above configuration, samples up to max_splits train/test splits,
        preferring splits toward the end of available data. If None, uses all splits.

To illustrate the test parameters:

.. code-block:: none

    (x) = train period
    (-) = forecast period
    (|) = train_end_date

    backtest
    (train_data)(periods_between_train_test)(test_horizon) |
    xxxxxxxxxxxx                             ------------- |
                                                           |
    forecast                                               |
    (train_data)                                           | (periods_between_train_test)(forecast_horizon)
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|                              -----------------

    etc.

To illustrate the CV parameters:

.. code-block:: none

    (x) = train period
    (-) = forecast period

    SPLIT 1
    (cv_min_train_periods)(cv_periods_between_train_test)(cv_horizon)
    xxxxxxxxxxxxxxxxxxxxxx                               ------------

    SPLIT 2: If cv_expanding_window = False
    (cv_period_between_splits)(cv_min_train_periods)(cv_periods_between_train_test)(cv_horizon)
                              xxxxxxxxxxxxxxxxxxxxxx                               ------------

    SPLIT 2: If cv_expanding_window = True
    (cv_period_between_splits)(cv_min_train_periods)(cv_periods_between_train_test)(cv_horizon)
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                               ------------

    etc.

.. note::

    The defaults are designed for proper evaluation based on your ``forecast_horizon`` and
    ``periods_between_train_test``, by matching ``forecast_horizon=test_horizon=cv_horizon``,
    and ``periods_between_train_test=cv_periods_between_train_test``.

    You can reduce the values if you don't have sufficient data to evaluate.

Examples:

.. code-block:: python

    from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam

    # daily data, 3mo evaluation
    evaluation_period = EvaluationPeriodParam(
        test_horizon=90,
        cv_horizon=90,
        cv_min_train_periods=None,
        cv_expanding_window=False,
        cv_use_most_recent_splits=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=0,
        cv_max_splits=3,
    )

    # Use CV to check 3 step-ahead error (cv_periods_between_train_test + cv_horizon)
    evaluation_period = EvaluationPeriodParam(
        test_horizon=1,
        periods_between_train_test=2,
        cv_horizon=1,
        cv_min_train_periods=90,
        cv_expanding_window=True,
        cv_use_most_recent_splits=False,
        cv_periods_between_splits=1,
        cv_periods_between_train_test=2,
        cv_max_splits=None,
    )


model_components_param
----------------------
Optional. Tuning parameters for the selected ``model_template``.
An instance of :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`.

While the other parameters define input data and
evaluation approach, these parameters allow you to tune the forecast model.

* On how to choose a template, see :doc:`/pages/stepbystep/0200_choose_template`.
* For details about the ``model_components`` for each model template, see
  :doc:`/pages/model_components/0100_introduction`.

computation_param
-----------------
Optional. Parameters related to grid search computation.
An instance of :class:`~greykite.framework.templates.autogen.forecast_config.ComputationParam`.

The attributes are:

.. code-block:: none

    hyperparameter_budget : int or None, default None
        max number of hyperparameter sets to try within the hyperparameter_grid search space

        Runs a full grid search if hyperparameter_budget is sufficient to exhaust full
        hyperparameter_grid, otherwise samples uniformly at random from the space

        If None, uses defaults:
            full grid search if all values are constant
            20 if any value is a distribution to sample from

    n_jobs : int or None, default=-1
        Number of jobs to run in parallel during grid search
        ``None`` is treated as 1. ``-1`` uses all processors

    verbose : int, default 1
        Verbosity level during CV.
        if > 0, prints number of fits
        if > 1, prints fit parameters, total score + fit time
        if > 2, prints train/test scores


Examples:

.. code-block:: python

    from greykite.framework.templates.autogen.forecast_config import ComputationParam

    computation = ComputationParam(
        hyperparameter_budget=3,
        n_jobs=-1,
        verbose=1
    )

    # for error messages/debugging, do not
    # run in parallel, and increase verbosity
    computation = ComputationParam(
        hyperparameter_budget=None,
        n_jobs=1,
        verbose=2
    )

forecast_one_by_one
-------------------
Optional. Whether to multiple models spanning the horizon and combine their predictions.
This may improve forecast quality when forecast horizon > 1
and autoregression or lagged regressors are used.

See :doc:`/gallery/templates/0400_forecast_one_by_one`.
