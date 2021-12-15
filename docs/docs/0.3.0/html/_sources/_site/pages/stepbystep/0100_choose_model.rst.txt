Choose a Model
==============

Greykite offers two forecasting models:
the Prophet model and the Silverkite model.
This page explains your options.

.. csv-table:: high-level comparison
   :widths: 20 35 35
   :header: "", "PROPHET", "SILVERKITE"

   "speed", "slower", "**faster**"
   "forecast accuracy (default)", "good", "good"
   "forecast accuracy (*customized*)", "limited", "**high**"
   "prediction interval accuracy", "*TBD*", "*TBD*"
   "interpretability", "good (additive model)", "good (additive model)"
   "ease of use", "good", "good"
   "API", "similar to ``sklearn``", "uses ``sklearn``"
   "fit", "Bayesian", "ridge, elastic net, boosted trees, etc."

Both models have the similar customization options. Differences are **bolded** below.

.. csv-table:: customization options
   :header: "", "PROPHET", "SILVERKITE"

   "automatic defaults", "yes", "yes"
   "growth", "linear, **logistic**", "linear, sqrt, quadratic, any combination, **custom**"
   "seasonality", "daily, weekly, yearly, custom", "daily, weekly, monthly, quarterly, yearly"
   "holidays", "specify countries, with window", "specify by name or country, with window; or custom events"
   "trend changepoints", "yes", "yes"
   "seasonality changepoints", "no", "**yes**"
   "regressors", "yes", "yes"
   "autoregression", "limited, via regressors", "full support, coming soon"
   "interaction terms", "build it yourself (regressor)", "**model formula** terms, or as regressor"
   "extras", "**prior scale** (bayesian)", "**fitting algorithm**"
   "loss function", "MSE", "MSE, **Quantile loss** (with ``gradient_boosting`` fitting algorithm)"
   "prediction intervals", "yes", "yes"

.. note:: When to use the Prophet model?

  * If it works better for your dataset
  * If you like Bayesian models
  * If you need logistic growth with changing capacity over time

.. note:: When to use the Silverkite model?

  * If it works better for your dataset (e.g. b/c of custom growth, interaction
    terms, seasonality changepoints).
  * If speed is important.
  * If you want to forecast a quantile, not the mean.

.. note::

  We use Prophet 0.5 (`Prophet documentation <https://facebook.github.io/prophet/docs/installation.html>`_.)
