Choose a Model
==============

Greykite offers a few forecasting models: Silverkite, Prophet, and ARIMA.
This page provides an overview.

.. csv-table:: high-level comparison
   :widths: 16 28 28 28
   :header: "", "SILVERKITE", "PROPHET", "ARIMA"

   "speed", "**fast**", "slow", "**fast**"
   "forecast accuracy (default)", "decent", "decent", "decent"
   "forecast accuracy (*customized*)", "**very good**", "good", "good"
   "interpretability", "**good**", "**good**", "decent"
   "ease of use", "good", "good", "**very good**"
   "API", "``sklearn``", "similar to ``sklearn``", "similar to ``sklearn``"

Like Prophet, Silverkite includes intepretable terms for growth,
seasonality, holidays, trend changepoints, and regressors.
Silverkite also supports autoregression, seasonality changepoints,
easy-to-use interaction terms, quantile loss, and custom fit algorithms.
This makes Silverkite flexible to capture different time series patterns.

.. we use '|' before all entries in the table to make the font sizes consistent

.. csv-table:: customization options
   :header: "", "SILVERKITE", "PROPHET"

   "automatic defaults", "| yes", "| yes"
   "growth", "| linear, sqrt, quadratic,
   | any combination, **custom**", "| linear, **logistic**"
   "seasonality", "| daily, weekly, monthly,
   | quarterly, yearly", "| daily, weekly, yearly, custom"
   "holidays", "| specify names or countries,
   | with window; or custom events", "| specify countries, with window"
   "regressors", "| yes", "| yes"
   "trend changepoints", "| yes", "| yes"
   "seasonality changepoints", "| **yes**", "| no"
   "autoregression", "| **yes**", "| limited
   | (via regressors)"
   "interaction terms", "| **yes**
   | (via model formula or regressors)", "| limited
   | (via regressors)"
   "loss function", "| MSE, **Quantile loss**", "| MSE"
   "fit algorithm", "| **custom**
   | (ridge, quantile regression, etc.)", "| fixed
   | (Bayesian formulation)"

.. note:: When to use the Silverkite model?

  * If both speed and interpretability are important
  * If you need flexible tuning options to achieve high accuracy
  * If you need to forecast a quantile

.. note:: When to use the Prophet model?

  * If you need logistic growth with changing capacity over time
  * If speed is not as important

.. note:: When to use the ARIMA model?

  * If you want to try a classic algorithm that is different from the other two
  * If you want to quickly establish an accuracy baseline to assess forecast difficulty
  * If interpretability is not as important
