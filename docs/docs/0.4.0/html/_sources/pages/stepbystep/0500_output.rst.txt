Check Forecast Result
=====================

The output of ``run_forecast_config`` helps you evaluate and interpret the forecast result.

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
            model_template=ModelTemplateEnum.SILVERKITE.name,
            metadata_param=metadata,
            forecast_horizon=30,
            coverage=0.95
        )
    )
    # `result` is also stored as `forecaster.forecast_result`

The function returns a `~greykite.framework.pipeline.pipeline.ForecastResult` object.
This is a dataclass with the forecast results. Below, we explain how to check each attribute
of the class.

Timeseries
----------

``timeseries`` is your original input data, loaded into
a `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries` object.
The time series is represented in a standard format with convenient plotting functions and statistics.
If ``MetadataParam.anomaly_info`` is specified, it contains the values before and after
anomaly adjustment.
For details, see :doc:`/pages/stepbystep/0300_input`.

.. code-block:: python

    ts = result.timeseries  # access the time series

Cross validation
----------------

``grid_search`` contains the cross validation results.
See :ref:`Evaluation Period <evaluation-period>` for cross validation configuration.

Use `~greykite.framework.utils.result_summary.summarize_grid_search_results`
to summarize the results. While ``grid_search.cv_results_`` could be converted into
a `pandas.DataFrame` without this function, the following conveniences
are provided:

    - returns the correct ranks based on each metric's greater_is_better direction.
    - summarizes the hyperparameter space, only showing the parameters that change
    - combines split scores into a tuple to save table width
    - rounds the values to specified decimals
    - orders columns by type (test score, train score, metric, etc.)

.. code-block:: python

    from greykite.common.evaluation import EvaluationMetricEnum
    from greykite.framework.utils.result_summary import summarize_grid_search_results

    grid_search = result.grid_search
    # default parameters
    summarize_grid_search_results(
        grid_search=grid_search,
        only_changing_params=True,
        combine_splits=True,
        decimals=None,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        score_func_greater_is_better=False,
        cv_report_metrics=CV_REPORT_METRICS_ALL,
        column_order=None
    )

    # some customization
    metric = EvaluationMetricEnum[evaluation_metric.cv_selection_metric]
    summarize_grid_search_results(
        grid_search=grid_search,
        decimals=2  # rounds the evaluation metrics
        # List of regex to order the columns in the returned dataframe
        column_order=["rank", "mean_test", "split_test", "mean_train", "split_train",
                      "mean_fit_time", "mean_score_time", "params", ".*"])
    )

``grid_search`` is an instance of `sklearn.model_selection.RandomizedSearchCV`, see
documentation for details on its attributes.

.. warning::

  The ranks (e.g. "rank_test_MAPE") in ``grid_search.cv_results_`` should be ignored.
  They are reversed for metrics where ``greater_is_better=False``, because sklearn
  always assumes higher values are better when calculating the ranks, we report
  metrics with their original sign.

  Instead, use `~greykite.framework.utils.result_summary.summarize_grid_search_results`
  to extract the CV results with proper ranks.

Backtest
--------

``backtest`` checks the forecast accuracy on historical data. It is an instance
of `~greykite.framework.output.univariate_forecast.UnivariateForecast`.

The forecast is computed on a holdout test set toward the end of the input dataset.
The length of the holdout test set is configured by ``test_horizon`` in the template
configuration. See :ref:`Evaluation Period <evaluation-period>`.

You can plot the results:

.. code-block:: python

      from plotly.offline import init_notebook_mode, iplot
      init_notebook_mode(connected=True)   # for generating offline graphs within Jupyter Notebook

      backtest = result.backtest
      fig = backtest.plot()
      iplot(fig)


Show the evaluation metrics:

.. code-block:: python

    print(backtest.train_evaluation)  # backtest training set
    print(backtest.test_evaluation)   # hold out test set


See the component plot to understand how trend, seasonality,
and holidays are handled by the forecast:

.. code-block:: python

    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode(connected=True)

    fig = backtest.plot_components()
    iplot(fig)  # fig.show() if you are using "PROPHET" template

Access backtest forecasted values and prediction intervals:

.. code-block:: python

    backtest.df

You can use ``backtest.plot_grouping_evaluation()`` to examine the train/test error by
various dimensions (e.g. over time, by day of week). See
`~greykite.framework.output.univariate_forecast.UnivariateForecast` for details.

See `~greykite.common.evaluation.EvaluationMetricEnum` for the available error metrics
and their descriptions.

.. code-block:: python

    from plotly.offline import init_notebook_mode, iplot
    from greykite.common.evaluation import EvaluationMetricEnum

    init_notebook_mode(connected=True)   # for generating offline graphs within Jupyter Notebook

    # MAPE by day of week
    fig = backtest.plot_grouping_evaluation(
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_func(),
        score_func_name=EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_name(),
        which="train", # "train" or "test" set
        groupby_time_feature="dow",  # day of week
        groupby_sliding_window_size=None,
        groupby_custom_column=None)
    iplot(fig)

    # RMSE over time
    fig = backtest.plot_grouping_evaluation(
        score_func=EvaluationMetricEnum.RootMeanSquaredError.get_metric_func(),
        score_func_name=EvaluationMetricEnum.RootMeanSquaredError.get_metric_name(),
        which="test", # "train" or "test" set
        groupby_time_feature=None,
        groupby_sliding_window_size=7,  # weekly aggregation of daily data
        groupby_custom_column=None)
    iplot(fig)

See `~greykite.framework.output.univariate_forecast.UnivariateForecast.plot_flexible_grouping_evaluation`
for a more powerful plotting function to plot the quantiles of the error along with the mean.

You can use component plots for a concise visual representation of how the dataset's trend, seasonality
and holiday patterns are estimated by the forecast model. Currently, ``Silverkite`` calculates component
plots based on dataset passed to the ``fit`` method, whereas ``Prophet`` calculates component plots
based on dataset passed to the ``predict``  method.

.. code-block:: python

  backtest.plot_components()

Forecast
--------

``forecast`` contains the forecasted values and the fit on the input dataset.
It is an instance of `~greykite.framework.output.univariate_forecast.UnivariateForecast`.

After creating a forecast, plot it to see
if it looks reasonable.

.. code-block:: python

      forecast = result.forecast
      fig = forecast.plot()
      iplot(fig)


Show the error metrics on the training set.

.. code-block:: python

    print(forecast.train_evaluation)  # fit on the entire input dataset


Access future forecasted values and prediction intervals:

.. code-block:: python

    forecast.df


Just as for backtest, you can use ``forecast.plot_grouping_evaluation()`` to examine the training error by
various dimensions (e.g. over time, by day of week), and ``forecast.plot_components()`` to check the trend,
seasonality and holiday effects. See
`~greykite.framework.output.univariate_forecast.UnivariateForecast` for details.


Model
-----

``model`` is a `sklearn.pipeline.Pipeline` object. It was
fit to the full dataset, with the best parameters selected via CV.
This model was used to generate ``forecast``. You can use it to extract
fitted model information (:doc:`/gallery/quickstart/02_interpretability/0100_model_summary`)
or make another forecast (:doc:`/gallery/quickstart/0100_simple_forecast`).

.. code-block:: python

    model = result.model  # access the model
