=======
History
=======

0.3.0 (2021-12-14)
------------------

* New tutorials
  * @Reza Hosseini: Monthly time series forecast.
  * @Yi Su: Weekly time series forecast.
  * @Albert Chen: Forecast reconciliation.
  * @Kaixu Yang: Forecast one-by-one method.
* New methods
  * @Yi Su: Lagged regressor (method was released in 0.2.0 but documentation was added in this release).
  * @Kaixu Yang @Saad Eddin Al Orjany: Model storage (method was released in 0.2.0 but documentation was added in this release).
  * @Kaixu Yang: Silverkite Multistage method for fast training on small granularity data (with tutorial).
  * @Albert Chen: Forecast reconciliation with interface and defaults optimized.
* New model templates
  * @Yi Su: `SILVERKITE_WITH_AR`: The `SILVERKITE` template with autoregression.
  * @Yi Su: `SILVERKITE_DAILY_1`: A SimpleSilverkite template designed for daily data with forecast horizon 1.
  * @Kaixu Yang: `SILVERKITE_TWO_STAGE`: A two stage model using the Silverkite Multistage method that is good for sub-daily data with a long history.
  * @Kaixu Yang: `SILVERKITE_MULTISTAGE_EMPTY`: A base template for the Silverkite Multistage method.
* Library enhancements and bug fixes
  * @Yi Su: Updated plotly to v5.
  * @Reza Hosseini: Use `explicit_pred_cols`, `drop_pred_cols` to directly specify or exclude model formula terms (see Custom Parameters).
  * @Reza Hosseini: Use `simulation_num` to specify number of simulations to use for generating forecasts and prediction intervals. Applies only if any of the lags in `autoreg_dict` are smaller than forecast_horizon (see Auto-regression).
  * @Reza Hosseini: Use `normalize_method` to normalize the design matrix (see Custom Parameters).
  * @Yi Su: Allow no CV and no backtest in pipeline.
  * @Albert Chen: Added synthetic hierarchical dataset.
  * Bug fix: `cv_use_most_recent_splits` in EvaluationPeriodParam was previously ignored.
  * @Albert Chen @Kaixu Yang @Reza Hosseini @Saad Eddin Al Orjany @Sayan Patra @Yi Su: Other library enhancements and bug fixes.

0.2.0 (2021-06-30)
------------------

* @Kaixu Yang: Removed the dependency on `fbprophet` and change it to optional.
* @Kaixu Yang @Saad Eddin Al Orjany: Added model dumping and loading for storing (see `Forecaster.dump_forecast_result` and `Forecaster.load_forecast_result`).
* @Kaixu Yang @Reza Hosseini: Added forecast one-by-one method.
* @Sayan Patra: Added the support of AutoArima by `pmdarima`, see the `AUTO_ARIMA` template.

0.1.1 (2021-05-12)
------------------

* First release on PyPI.