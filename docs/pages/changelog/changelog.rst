Changelog
=========

0.0.77
^^^^^^

Improved short-term forecasts:

* Created templates for short-term daily T+1 forecasts: ``SILVERKITE_DAILY_1``, ``SILVERKITE_WITH_AR``.
  See `~greykite.framework.templates.model_templates.ModelTemplateEnum`.
* Created "forecast one by one" option. This may improve forecast quality when ``forecast_horizon > 1``
  and autoregression or lagged regressors are used.
  See :doc:`/gallery/quickstart/0600_forecast_one_by_one`.

Improved weekly/monthly forecasts:

* Published tutorials for monthly forecast (:doc:`/gallery/tutorials/0400_monthly_data`),
  weekly forecast (:doc:`/gallery/tutorials/0500_weekly_data`).

Silverkite model:

* Support lagged regressors (:doc:`/pages/model_components/0700_regressors`)
* Use ``explicit_pred_cols``, ``drop_pred_cols`` to directly specify or exclude model formula terms
  (see :doc:`/pages/model_components/0600_custom`).
* Use ``simulation_num`` to specify number of simulations to use for generating forecasts and prediction intervals.
  Applies only if any of the lags in ``autoreg_dict`` are smaller than ``forecast_horizon``
  (see :doc:`/pages/model_components/0800_autoregression`).
* Use ``normalize_method`` to normalize the design matrix (see :doc:`/pages/model_components/0600_custom`).
* Removed Silverkite's dependency on fbprophet for fetching holidays.
* Allow calculating fitted values before train start date (even when autoregression is used).

Hierarchical forecast:

* Published tutorial for reconcile forecasts (:doc:`/gallery/tutorials/0600_reconcile_forecasts`)
* Improved `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
  defaults and diagnostics.

Other:

* Allow no CV and no backtest in pipeline
* Bug fix: ``cv_use_most_recent_splits`` in
  `~greykite.framework.templates.autogen.forecast_config.EvaluationPeriodParam`
  was previously ignored
* Updated plotly (4.12.0).

Under development:

* Silverkite multistage algorithm
* Anomaly detection simulation & labeling

0.0.33
^^^^^^

Models:

* Added AutoArima (``AUTO_ARIMA``) template,
  see `~greykite.framework.templates.model_templates.ModelTemplateEnum`
  and `~greykite.framework.templates.auto_arima_template.AutoArimaTemplate`.

Documentation:

* Benchmarking tutorial (:doc:`/gallery/tutorials/0300_benchmark`)
* Autoregression docs (:doc:`/pages/model_components/0800_autoregression`)

Utilities:

* Functions to dump and load forecast result (:doc:`/pages/miscellaneous/store_model`).
* Support aggregation in `~greykite.common.data_loader.DataLoader`
  ``load_data`` function (via ``agg_freq``, ``agg_func``).

Other:

* Updated numpy (1.20), scikit-learn (0.24), matplotlib (3.1.1)
* Bug fix: allow Prophet forecasts when coverage=None

0.0.6
^^^^^
Initial release!

* Removed Brazil, Netherlands, and Australia from the default holiday country list

0.0.0
^^^^^
Code from dsar-forecast 1.0.1 was moved into this MP.