Growth
======

Use ``model_components.growth`` to specify the growth.

Growth is used to model the overall trend of your data over time.

To understand the growth characteristics of your dataset, plot your input training data over time,
smoothing out local variation as necessary. For example, aggregate hourly data to the weekly total,
and plot the overall trend. See :doc:`/pages/stepbystep/0300_input` for details.

By default, silverkite and prophet use ``linear`` growth. Linear growth is commonly seen, which
implies a constant growth rate over time (e.e. constant y/y growth).

.. note::

    In some cases, a metric exhibits ``linear`` growth, but the slope changes at certain points.
    For example, a ramp or external event accelerates the rate of growth.

    In this case, choose ``linear`` growth, and add changepoints to allow the slope to change.
    This fits a piecewise linear trend.
    For more information, see :doc:`/pages/model_components/0500_changepoints`.


Silverkite
----------

Examples:

.. code-block:: python

    growth = dict(growth_term="linear")     # constant growth rate (default)
    growth = dict(growth_term="quadratic")  # growth rate increases over time
    growth = dict(growth_term="sqrt")       # growth rate decreases over time
    growth = dict(growth_term=None)         # no growth term
    growth = None                           # same as "linear"
    growth = dict(growth_term=["linear", "quadratic"])  # grid search over each option

.. note::

    ``quadratic`` and ``sqrt`` may apply to some datasets, but tend not to extrapolate
    well very far into future. If you use these, make sure it is reasonable to expect
    future growth rate to increase/diminish accordingly for the full training period
    and ``forecast_horizon``.


.. note::

    You may need to set the pipeline parameter ``estimator__origin_for_time_vars`` for proper
    historical backtesting. See :doc:`/pages/model_components/0600_custom`.

Custom growth term
^^^^^^^^^^^^^^^^^^

You can use any parametric function of time to specify growth by leveraging the changepoints specification.
See :ref:`Custom Growth <custom-growth>`.

While it's possible to provide any custom growth curve as a regressor (:doc:`/pages/model_components/0700_regressors`),
use changepoints when possible:

* Silverkite does not know if regressors should be considered part of the trend.
* Thus, by default, interactions with trend would not include your regressor. See
  ``feature_sets_enabled`` at :doc:`/pages/model_components/0600_custom`.
* You can add those interactions yourself via ``extra_pred_cols``. See
  :doc:`/pages/model_components/0600_custom`.


Prophet
-------

Examples:

.. code-block:: python

    growth = dict(growth_term=["linear"])     # linear, unbounded growth (default)
    growth = dict(growth_term=["logistic"])   # saturating maximum or minimum growth
    growth = None                             # same as "linear"
    growth = dict(growth_term=["linear", "logistic"])  # grid search over each option

Prophet allows forecasts with saturating maximum or minimum using ``logistic``
trend, with specified ``cap`` and ``floor`` respectively. For example, user growth
bounded by total addressable market. More details at
`Prophet docs <https://facebook.github.io/prophet/docs/saturating_forecasts.html#saturating-minimum>`_.
