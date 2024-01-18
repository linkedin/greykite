Regressors
==========

Use ``model_components.regressors`` to specify external regressors. Regressors can be numeric
or categorical.

You need to provide historical values of the regressor for training, and future values for prediction.
See :doc:`/pages/stepbystep/0300_input` for the input data format.

For example, to use weather as a regressor to forecast the amount of vehicle traffic, you can
train a model using historical traffic and weather conditions. Then predict future traffic based
on forecasted weather conditions.

Silverkite
----------

Examples for SILVERKITE:

.. code-block:: python

    # For input data with 3 regressors.
    # Input data columns: ["time", "value", "gdp", "weather", "population"]
    regressors=dict(
        regressor_cols=["gdp", "weather", "population"]
    )

    # No regressors (default)
    regressors=None

    # Grid search is possible
    regressors=dict(
        regressor_cols=[
            ["gdp", "weather", "population"],
            ["gdp", "weather"],
            None
        ]
    )

.. note::
  If you use the low-level model template SK, it expects a different way to specify regressors.
  The low-level interface Silverkite does not expect the ``regressors.regressor_cols`` variable.
  Instead, please add any regressor columns to ``custom.extra_pred_cols``.

Examples for SK:

.. code-block:: python

    # For input data with 3 regressors.
    # Input data columns: ["time", "value", "gdp", "weather", "population"]
    custom=dict(
        extra_pred_cols=["gdp", "weather", "population"]
    )

    # No regressors (default)
    custom=dict(
      extra_pred_cols=None
    )

    # Grid search is possible
    custom=dict(
        extra_pred_cols=[
            ["gdp", "weather", "population"],
            ["gdp", "weather"],
            None
        ]
    )

You can specify lagged regressors using ``model_components.lagged_regressors``.
For each regressor column, provide the list of lags and aggregated lags to include, or
use the "auto" setting to have Silverkite choose for you.

Lagged regressor examples:

.. code-block:: python

    lagged_regressors=dict(
        lagged_regressor_dict = {
            "gdp": {
                "lag_dict": {"orders": [1, 2, 3]}, # individual lags: lag 1, lag 2, lag 3
                "agg_lag_dict": {
                    "orders_list": [(7, 14, 21)],  # average of lags 7, 14, 21
                    "interval_list": [(8, 14), (15, 21)]},   # average of lags 8 to 14, lags 15 to 21
            },
            "weather": "auto",  # automatically chooses lags based on data frequency and forecast horizon
        }
    )

``model_components.lagged_regressors`` can be used with or without ``model_components.regressors``.
For details and more options,
see `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`.

``lag_dict`` and ``agg_lag_dict`` work the same way as for autoregression.
See :doc:`/pages/model_components/0800_autoregression`.

Prophet
-------

Options:

.. code-block:: none

    add_regressor_dict: `dict` or None or list of such values for grid search
        Dictionary of extra regressors to be modeled. Predictions will be influenced by these regressors.
        None by default.

Follow the same guidance as Silverkite for input data format.

Examples:

.. code-block:: python

    # For input data with 3 regressors.
    # Input data columns: ["time", "value", "gdp", "weather", "population"]
    regressors=dict(
        add_regressor_dict={  # add as many regressors as you'd like, in the following format
            "gdp": {
                "prior_scale": 10.0,  # default is 10.0, decreasing the prior scale will add additional regularization
                "mode": 'additive'  # this regressor's effect on predictions
            },
            "weather": {
                "prior_scale": 20.0,
                "mode": 'multiplicative'
            },
            "population": {
                "prior_scale": 15.0,
                "mode": 'multiplicative'
            }
        }
    )

    # No regressors (default)
    regressors=None

    # Grid search is possible
    regressors=dict(
        add_regressor_dict=[{  # it is possible to enable different modes for given regressors
            "gdp": {
                "prior_scale": 10.0,
                "mode": 'additive'
            },
            "weather": {
                "prior_scale": 20.0,
                "mode": 'multiplicative'
            },
            "population": {
                "prior_scale": 15.0,
                "mode": 'multiplicative'
            }
        },
        {
            "gdp": {
                "prior_scale": 15.0,
                "mode": 'additive'
            },
            "weather": {
                "prior_scale": 10.0,
                "mode": 'additive'
            },
            "population": {
                "prior_scale": 25.0,
                "mode": 'additive'
            }
        }
    )

.. note::

    ``prior_scale`` and ``mode`` work in similar way as for custom
    seasonality (:doc:`/pages/model_components/0300_seasonality`).
    Fit customization can be done for each regressor.


We do not support lagged regressors with Prophet.