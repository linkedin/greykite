Docs
====

.. only include user-facing docstrings in the docs

All Templates
-------------
.. autoclass:: greykite.framework.templates.forecaster.Forecaster

.. currentmodule:: greykite.framework.templates.model_templates
.. autoclass:: ModelTemplate
.. autoclass:: ModelTemplateEnum

.. currentmodule:: greykite.framework.templates.autogen.forecast_config
.. autoclass:: ForecastConfig
.. autoclass:: MetadataParam
.. autoclass:: EvaluationMetricParam
.. autoclass:: EvaluationPeriodParam
.. autoclass:: ModelComponentsParam
.. autoclass:: ComputationParam

Silverkite Template
-------------------
.. currentmodule:: greykite.framework.templates.simple_silverkite_template
.. autoclass:: SimpleSilverkiteTemplate

.. autoclass:: greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator
.. autoclass:: greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator
.. autoclass:: greykite.sklearn.estimator.base_silverkite_estimator.BaseSilverkiteEstimator
.. autoclass:: greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions

.. currentmodule:: greykite.framework.templates.silverkite_template
.. autoclass:: SilverkiteTemplate

Lag Based Template
------------------
.. autoclass:: greykite.framework.templates.lag_based_template.LagBasedTemplate

.. autoclass:: greykite.sklearn.estimator.lag_based_estimator.LagBasedEstimator
.. autoclass:: greykite.sklearn.estimator.lag_based_estimator.LagUnitEnum

Multistage Forecast Template
----------------------------
.. currentmodule:: greykite.framework.templates.multistage_forecast_template
.. autoclass:: MultistageForecastTemplate

.. autoclass:: greykite.sklearn.estimator.multistage_forecast_estimator.MultistageForecastEstimator

Prophet Template
----------------
.. currentmodule:: greykite.framework.templates.prophet_template
.. autoclass:: ProphetTemplate

.. autoclass:: greykite.sklearn.estimator.prophet_estimator.ProphetEstimator

ARIMA Template
--------------
.. autoclass::  greykite.framework.templates.auto_arima_template.AutoArimaTemplate
.. autoclass::  greykite.sklearn.estimator.auto_arima_estimator.AutoArimaEstimator

Forecast Pipeline
-----------------
.. currentmodule:: greykite.framework.pipeline.pipeline
.. autofunction:: forecast_pipeline
.. autoclass:: ForecastResult

Template Output
---------------
.. autoclass:: greykite.framework.input.univariate_time_series.UnivariateTimeSeries
.. autoclass:: greykite.framework.output.univariate_forecast.UnivariateForecast
.. autoclass:: greykite.algo.common.model_summary.ModelSummary
    :members:
    :private-members:

Constants
---------
.. autoclass:: greykite.common.aggregation_function_enum.AggregationFunctionEnum
.. autoclass:: greykite.common.evaluation.EvaluationMetricEnum
.. automodule:: greykite.common.constants
.. automodule:: greykite.framework.constants
.. autoclass:: greykite.algo.forecast.silverkite.constants.silverkite_constant.SilverkiteConstant
.. autoclass:: greykite.algo.forecast.silverkite.constants.silverkite_column.SilverkiteColumn
.. autoclass:: greykite.algo.forecast.silverkite.constants.silverkite_component.SilverkiteComponentsEnum
.. autoclass:: greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday
.. autoclass:: greykite.algo.forecast.silverkite.constants.silverkite_seasonality.SilverkiteSeasonalityEnum
.. autoclass:: greykite.algo.forecast.silverkite.constants.silverkite_time_frequency.SilverkiteTimeFrequencyEnum
.. automodule:: greykite.framework.templates.simple_silverkite_template_config

EasyConfig
----------
.. currentmodule:: greykite.algo.common.seasonality_inferrer
.. autoclass:: SeasonalityInferrer
.. autoclass:: TrendAdjustMethodEnum
.. autoclass:: SeasonalityInferConfig

.. autoclass:: greykite.algo.common.holiday_inferrer.HolidayInferrer

Changepoint Detection
---------------------
.. autoclass:: greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector

Benchmarking
----------------
.. autoclass:: greykite.framework.benchmark.benchmark_class.BenchmarkForecastConfig
    :members:

Cross Validation
----------------
.. autoclass:: greykite.sklearn.cross_validation.RollingTimeSeriesSplit
    :members:
    :private-members:

Transformers
------------
.. automodule:: greykite.sklearn.transform.zscore_outlier_transformer
.. automodule:: greykite.sklearn.transform.normalize_transformer
.. automodule:: greykite.sklearn.transform.null_transformer
.. automodule:: greykite.sklearn.transform.drop_degenerate_transformer

Quantile Regression
-------------------
.. autoclass:: greykite.algo.common.l1_quantile_regression.QuantileRegression

Hierarchical Forecast
---------------------
.. autoclass:: greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts
.. autoclass:: greykite.algo.reconcile.hierarchical_relationship.HierarchicalRelationship

.. the items below are included because they are linked to by the docs

Utility Functions
-----------------
.. currentmodule:: greykite.common.features.timeseries_features
.. autofunction:: get_available_holiday_lookup_countries
.. autofunction:: get_available_holidays_across_countries
.. autofunction:: build_time_features_df
.. autofunction:: get_holidays
.. autofunction:: add_event_window_multi
.. autofunction:: add_daily_events
.. autofunction:: convert_date_to_continuous_time

.. currentmodule:: greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper
.. autofunction:: get_event_pred_cols

.. autofunction:: greykite.framework.pipeline.utils.get_basic_pipeline
.. autofunction:: greykite.framework.utils.result_summary.summarize_grid_search_results
.. autofunction:: greykite.framework.utils.result_summary.get_ranks_and_splits

.. currentmodule:: greykite.common.viz.timeseries_plotting
.. autofunction:: plot_multivariate
.. autofunction:: plot_univariate
.. autofunction:: plot_forecast_vs_actual

.. currentmodule:: greykite.common.features.timeseries_impute
.. autofunction:: impute_with_lags
.. autofunction:: impute_with_lags_multi

.. currentmodule:: greykite.common.features.adjust_anomalous_data
.. autofunction:: adjust_anomalous_data

.. currentmodule:: greykite.common.evaluation
.. autofunction:: r2_null_model_score

.. currentmodule:: greykite.framework.pipeline.utils
.. autofunction:: get_score_func_with_aggregation
.. autofunction:: get_hyperparameter_searcher
.. autofunction:: get_scoring_and_refit
.. autofunction:: get_best_index
.. autofunction:: get_forecast

.. currentmodule:: greykite.framework.templates.pickle_utils
.. autofunction:: dump_obj
.. autofunction:: load_obj

.. autoclass:: greykite.common.data_loader.DataLoader
.. autoclass:: greykite.framework.benchmark.data_loader_ts.DataLoaderTS

.. autoclass:: greykite.algo.reconcile.convex.reconcile_forecasts.TraceInfo

Internal Functions
------------------
.. autoclass:: greykite.algo.forecast.silverkite.constants.silverkite_seasonality.SilverkiteSeasonalityEnum

.. autofunction:: greykite.algo.common.ml_models.fit_ml_model
.. autofunction:: greykite.algo.common.ml_models.fit_ml_model_with_evaluation
.. autofunction:: greykite.algo.forecast.silverkite.forecast_silverkite_helper.get_silverkite_uncertainty_dict

.. currentmodule:: greykite.algo.forecast.silverkite.forecast_simple_silverkite
.. autoclass:: SimpleSilverkiteForecast

.. autofunction:: greykite.algo.uncertainty.conditional.conf_interval.conf_interval
.. autofunction:: greykite.algo.changepoint.adalasso.changepoints_utils.combine_detected_and_custom_trend_changepoints

.. currentmodule:::: greykite.common.features.timeseries_lags
.. autofunction:: build_autoreg_df
.. autofunction:: build_agg_lag_df
.. autofunction:: build_autoreg_df_multi

.. currentmodule:: greykite.algo.forecast.silverkite.forecast_silverkite
.. autoclass:: SilverkiteForecast
