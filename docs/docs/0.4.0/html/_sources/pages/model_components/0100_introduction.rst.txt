Greykite models and components
==============================
See :doc:`/pages/stepbystep/0100_choose_model` for model options and their comparison.

In :doc:`/pages/stepbystep/0400_configuration`,
follow :doc:`/pages/stepbystep/0200_choose_template`
to set ``config.model_template`` to use the proper model.

Silverkite
----------
Silverkite is a forecasting algorithm developed by LinkedIn.

It works by generating basis functions for growth, seasonality, holidays, etc.
These features, along with any regressors you provide, are used to fit the timeseries.

The features can be combined with interaction terms in a flexible and powerful way.

This approach has the following advantages.

  1. Flexible. Supports different kinds of growth, interactions, and fitting algorithms.
  2. Intepretable. The default fitting algorithms are additive so you can identify the contribution
     of each component.
  3. Fast. Runs much faster than Bayesian alternatives.

Prophet
-------
Prophet is a forecasting algorithm developed by Facebook.

Details on `Prophet <https://facebook.github.io/prophet/docs/quick_start.html>`_.

ARIMA
-----
ARIMA is a classic forecasting algorithm. We use the implementation from ``pmdarima``.

Details on `ARIMA <https://alkaline-ml.com/pmdarima/>`_.

model_components
----------------

``config.model_components_param`` is an instance of
:class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`.

The attributes represent different categories of tuning parameters.
See the rest of this section for how to configure each component.

.. code-block:: python

    from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam

    model_components = ModelComponentsParam(
        growth=growth,
        seasonality=seasonality,
        events=events,
        changepoints=changepoints,
        regressors=regressors,
        lagged_regressors=lagged_regressors,
        autoregression=autoregression,
        uncertainty=uncertainty,
        custom=custom,
        hyperparameter_override=hyperparameter_override
    )
