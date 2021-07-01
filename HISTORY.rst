=======
History
=======

0.2.0 (2021-06-30)
------------------

* Removes the dependency on `fbprophet` and change it to optional.
* Added model dumping and loading for storing (see `Forecaster.dump_forecast_result` and `Forecaster.load_forecast_result`).
* Added the support of AutoArima by `pmdarima`, see the `AUTO_ARIMA` template.
* (Beta, dev, not officially released): added support for lagged regressors.

0.1.1 (2021-05-12)
------------------

* First release on PyPI.