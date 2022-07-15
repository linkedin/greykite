Choose a Model Template
=======================

Greykite offers model templates for Silverkite, Prophet, ARIMA,
and lag-based forecasts (such as week-over-week).
Model templates provide default parameters for these algorithms.

You can use the same forecaster
(:class:`~greykite.framework.templates.forecaster.Forecaster`)
to run different algorithms and compare the results.

We recommend starting with the ``"AUTO"`` model template, which
automatically selects an appropriate Silverkite model based on
the timeseries, forecast horizon, and evaluation configs.

See `~greykite.framework.templates.model_templates.ModelTemplateEnum`
for valid ``model_template`` names.
The model templates can be classified into a few categories:

* The ``SILVERKITE`` category provides a high-level interface to the
  Silverkite model. This category of model templates includes configurations that are
  tailored to various forecast horizons, data frequencies, and data characteristics.
  It is easy to customize these configurations to try different options.
  The model template names are ``"AUTO"``, strings starting with ``"SILVERKITE"``, or instances of
  `~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`.
  The class that applies these templates is
  `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`.
* The ``"PROPHET"`` model template is used for the Prophet model.
  The class that applies this template is
  `~greykite.framework.templates.prophet_template.ProphetTemplate`.
* The ``"AUTO_ARIMA"`` model template is used for the ARIMA model.
  The class that applies this template is
  `~greykite.framework.templates.auto_arima_template.AutoArimaTemplate`.
* The ``"LAG_BASED"`` model template is used for the lag-based model.
  This is useful for simple baselines like week-over-week.
  The class that applies this template is
  `~greykite.framework.templates.lag_based_template.LagBasedTemplate`.
  See :doc:`/gallery/templates/0300_lag_forecast` for details.
* The ``"SK"`` model template is a low-level interface to the Silverkite model.
  This model template allows you to change lower-level parameters in Silverkite
  and is intended for more advanced users.
  The class that applies this template is
  `~greykite.framework.templates.silverkite_template.SilverkiteTemplate`.

Greykite also offers multistage templates to fit multiple models sequentially,
fitting each each to the remaining residuals.
The prediction is the sum of the models.
The class that applies this template is
`~greykite.framework.templates.multistage_forecast_template.MultistageForecastTemplate`.
See :doc:`/gallery/templates/0200_multistage_forecast` for details.

* ``"SILVERKITE_TWO_STAGE"`` is designed to quickly fit a
  more granular time series (for example, minute-level),
  where a long history is needed to train a good model.
* ``"SILVERKITE_WOW"`` is an enhanced week-over-week estimator that uses Silverkite
  to model seasonality, growth and holiday effects first, and then uses week over week
  to estimate the residuals.
* ``"MULTISTAGE_EMPTY"`` is a blank multistage template that you can customize for
  your own multistage model.

A detailed tutorial of how to use and customize templates can be found at
:doc:`/gallery/templates/0100_template_overview`.

.. note::

  One approach is to try a few templates with default parameters.
  After tuning a few of the main parameters, focus on the one that
  looks most promising for your dataset.

  Silverkite and its extensions are flexible models that support additional
  tuning of interpretable parameters.
