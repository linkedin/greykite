Changepoints
============

Use ``model_components.changepoints`` to specify the changepoints.

Silverkite
----------

There are two type of changepoints that are allowed
in Silverkite: (trend) changepoints and seasonality changepoints.
Changepoints allow you to specify changes in the growth rate.
Together, growth and changepoints define the overall trend.
Seasonality changepoints allow you to specify changes in the seasonality shapes.

A quickstart example for automatic trend changepoint detection in Silverkite can be
found at :doc:`/gallery/quickstart/01_exploration/0100_changepoint_detection`

By default, there are no changepoints.

Trend changepoint options:

.. code-block:: none

    changepoints : `dict` [`str`, `dict`] or None
        Specifies the changepoint configuration. Dictionary with the following
        optional key:

        ``"changepoints_dict"`` : `dict` or None or a list of such values for grid search
                changepoints dictionary passed to ``forecast_simple_silverkite``. A dictionary
                with the following optional keys:

                ``"method"`` : `str`
                    The method to locate changepoints. Valid options:
                        "uniform". Places n_changepoints evenly spaced changepoints to allow growth to change.
                        "custom". Places changepoints at the specified dates.
                        "auto". Automatically detects change points.
                    Additional keys to provide parameters for each particular method are described below.
                ``"continuous_time_col"`` : `str` or None
                    Column to apply `growth_func` to, to generate changepoint features
                    Typically, this should match the growth term in the model
                ``"growth_func"`` : callable or None
                    Growth function (`numeric` -> `numeric`). Changepoint features are created
                    by applying `growth_func` to "continuous_time_col" with offsets.
                    If None, uses identity function to use `continuous_time_col` directly
                    as growth term

            If changepoints_dict["method"] == "uniform", this other key is required:

                ``"n_changepoints"`` : `int`
                    number of changepoints to evenly space across training period

            If changepoints_dict["method"] == "custom", this other key is required:

                ``"dates"`` : `list` [`int` or `float` or `str` or `datetime`]
                    Changepoint dates. Must be parsable by pd.to_datetime.
                    Changepoints are set at the closest time on or after these dates
                    in the dataset.

            If changepoints_dict["method"] == "auto", optional keys can be passed that match the parameters in
            `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_trend_changepoints`
            (except ``df``, ``time_col`` and ``value_col``, which are already known).
            To add manually specified changepoints to the automatically detected ones, the keys ``dates``,
            ``combine_changepoint_min_distance`` and ``keep_detected`` can be specified, which correspond to the
            three parameters ``custom_changepoint_dates``, ``min_distance`` and ``keep_detected`` in
            `~greykite.algo.changepoint.adalasso.changepoints_utils.combine_detected_and_custom_trend_changepoints`.


Examples:

.. code-block:: python

    # Places three changepoints uniformly within each training set.
    # Piecewise linear growth is defined by `continuous_time_col` and `growth_func`
    changepoints = dict(
        changepoints_dict=dict(
            method="uniform",
            n_changepoints=3,
            continuous_time_col="ct1",  # the default, no need to specify
            growth_func=lambda x: x  # the default, no need to specify
        )
    )

    # Piecewise linear growth, two changepoints at specific dates
    # If a date is not contained in one of the training splits,
    # it is ignored for that split.
    # `dates`: Iterable[Union[int, float, str, datetime]], interpreted by pd.to_datetime
    changepoints = dict(
        changepoints_dict=dict(
            method="custom",
            dates=["2018-08-01", "2019-08-03"]
        )
    )

    # Grid search is possible
    changepoints = dict(
        changepoints_dict=[
            dict(
                method="custom",
                dates=["2018-08-01", "2019-08-03"]
            ),
            dict(
                method="custom",
                dates=["2019-08-03"]
            ),
        ]
    )

    # Automatic change point detection
    changepoints=dict(
        changepoints_dict=dict(
          method="auto",
          regularization_strength=0.6,
          resample_freq="7D",
          actual_changepoint_min_distance="100D",
          potential_changepoint_distance="50D",
          no_changepoint_proportion_from_end=0.3,
          yearly_seasonality_order=6,
          # Manually specifies two changepoints to add
          dates=["2007-12-01", "2009-06-01"],
          combine_changepoint_min_distance="100D",  # Default is `actual_changepoint_min_distance` if not specified
          keep_detected=False,  # Prefers manual changepoints over detected ones in case of overlap
        )
    )


To place changepoints, plot the overall trend over time, and see if you can identify
places where the trend changes noticeably. If so, use ``method="custom"`` and
add a changepoint at those dates. If not, try a few uniform changepoints, and
see if your backtest error improves.

.. note::

    Do not place a changepoint too close to the end of your dataset, because it may not
    have enough data points to learn the new trend.

    As a rule of thumb, the last changepoint should have enough training data following it
    to validate the forecast accuracy in a backtest. For example, for daily data and a
    a forecast horizon of a few months, reserve 2 months after the last changepoint.

    In automatic change point detection, this can be avoided by specifying
    ``no_changepoint_proportion_from_end`` or ``no_changepoint_distance_from_end``.

.. note::

    Changepoints add flexibility to your model. As a rule of thumb, if you do not use the
    "auto" option for "method", use at most 3 changepoints, fewer if you include interactions
    with changepoints.

    If you have yearly seasonality in your model and just one year of training data, a
    very flexible trend can become conflated with the yearly seasonality.

    Unlike Prophet, the current implementation of Silverkite does not explicitly encourage
    smoothness how changepoints affect the growth rate. So it's best not to have too many.

.. _custom-growth:

Custom growth
^^^^^^^^^^^^^

Changepoints also allow you to customize the growth rate as a function of time.

How it works
~~~~~~~~~~~~
Each changepoint as introduces a regressor whose value is 0 before the changepoint, and
whose value after the changepoint is defined by ``continuous_time_col`` and ``growth_func``.

Let :math:`g(t)` be the value of continuous_time_col at :math:`t`.
A changepoint at date :math:`t_0` adds a regressor defined by:

* :math:`\text{growth_func}(g(t)-g(t_0))` if :math:`t >= t_0`
* :math:`0` otherwise.

For linear growth (``continuous_time_col="ct1"``), this is simply:

* :math:`\text{growth_func}(t-t_0)` if :math:`t >= t_0`
* :math:`0` otherwise.


(:math:`t-t_0`` is the fractional years since :math:`t_0`.)

For most applications, you can set ``continuous_time_col="ct1"`` and define
``growth_func`` to model type of curve you want with time.

.. note::

    Every changepoint date must be a date in your dataset.
    (The changepoint dates are mapped to the first date on or after the requested date
    within each training split, and deduped).


growth_func
~~~~~~~~~~~

By leveraging ``growth_func``, you can specify the growth as any function of time.
For example, if you believe the growth rate should be logistic
(`wikipedia <https://en.wikipedia.org/wiki/Logistic_function>`_), and have some domain
knowledge about the growth rate, capacity, and inflection point:

.. code-block:: python

    from greykite.common.features.timeseries_features import get_logistic_func

    #  Defines f(continuous_time_col) =
    #    floor + capacity / (1 + exp(-growth_rate * (continuous_time_col - inflection_point)))
    logistic_func = get_logistic_func(
        growth_rate=0.5,        # how fast the values go from floor to capacity
        capacity=2000.0,        # in units of the timeseries value
        floor=0.0,              # in units of the timeseries value
        inflection_point=1.0)   # in units of continuous_time_col. How far after the changepoint to place the inflection point
    changepoints=dict(
        changepoints_dict=dict(
            method="custom",
            dates=["2018-09-01"],  # The dates where continuous_time_col=0 for each logistic curve.
                                   # Placing multiple dates results in multiple logistic curves.
            continuous_time_col="ct1",
            growth_func=logistic_func
        )
    )

.. tip::

    If you place a changepoint at your train start date, you can define a custom growth term beyond
    those available in :doc:`/pages/model_components/0200_growth`.

    Make sure to set ``growth=dict(growth_term=None)`` for :doc:`/pages/model_components/0200_growth` so you have a single
    growth term.

.. tip::

    It's best to have ``growth_func(0.0) = 0.0`` for continuity at the changepoints.

    For logistic growth, this means ``floor`` should be 0.0 and ``inflection_point`` should be large.

continuous_time_col
~~~~~~~~~~~~~~~~~~~

``continuous_time_col`` is a numeric representation of time. You can use this parameter to specify
non-linear growth rate without writing your own ``growth_func``.

If you specify ``growth_func``, you will most likely leave this as the
default (linear time ``"ct1"``).

Here are the options:

.. csv-table::
   :widths: 25 25
   :header: "continuous_time_col", "description"

   "ct1", "linear growth, -infinity to infinity"
   "ct2", "signed quadratic growth, -infinity to infinity"
   "ct3", "signed cubic growth, -infinity to infinity"
   "ct_sqrt", "signed square root growth, -infinity to infinity"
   "ct_root3", "signed cubic root growth, -infinity to infinity"

.. note::

    What is signed growth?

    * signed growth at ``x`` is defined by: ``np.sign(x) * np.power(np.abs(x), pow)``.
    * For example, signed square root is this function with ``pow=0.5``.
    * These functions are monotonically increasing with time, making them useful to model growth

For each changepoint, signed growth is calculated on the fractional years
since the changepoint date.

The ``growth_term`` specified at :doc:`/pages/model_components/0200_growth`
maps to ``continuous_time_col`` as follows:

.. csv-table::
   :widths: 25 25
   :header: "growth_term", "continuous_time_col"

   "linear", "ct1"
   "quadratic", "ct2"
   "sqrt", "ct_sqrt"


Other usage
~~~~~~~~~~~

.. note::

    By interacting trend and seasonality, you can use changepoints to
    specify changes in seasonality (e.g increasing over time).


Auto growth
^^^^^^^^^^^

The Silverkite model supports automatically setting the growth configuration.
Although automatic changepoint detection is already set by ``method = "auto"``,
there are still parameters that need to be specified.
The "auto" growth will automatically configure the growth function and
the automatic changepoint detection parameters by checking the data and forecast configurations.

To use the auto growth option, simply specify ``auto_growth = True`` in the ``changepoints`` dictionary.
The auto growth functionality will override the specified growth function and trend changepoint detection
parameters. However, you can specify custom changepoints parameters in ``changepoints_dict``
(``dates``, ``combine_changepoint_min_distance`` and ``keep_detected``),
so that these custom changepoints will be added to any detected changepoints.

.. code-block:: python

    changepoints=dict(
        auto_growth=True
    )

.. _seasonality-changepoints:

Seasonality changepoints
^^^^^^^^^^^^^^^^^^^^^^^^

Seasonality changepoints allow seasonality shapes to change at every component level. For example,
the shape of yearly seasonality may change at a seasonality changepoint for yearly component, but
the shape of seasonality for other components such as weekly or daily will not change at that point.
These changepoints can be automatically detected.

You may specify ``seasonality_changepoints_dict`` by itself or along with ``changepoints_dict`` in
``model_components.changepoints``. Optional keys of ``seasonality_changepoints_dict`` include keys
in `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`,
except ``df``, ``time_col``, ``value_col`` and ``trend_changepoints``, which will be automatically passed
within the algorithm.

Examples:

.. code-block:: python

    # Includes seasonality changepoints with trend changepoints.
    # The detected trend changepoints will be used as partial information
    # when detecting seasonality changepoints.
    # Both are used in the forecast model.
    changepoints = dict(
        changepoints_dict=dict(
            method="auto",
            no_changepoint_distance_from_end="180D"
        ),
        seasonality_changepoints_dict=dict(
            no_changepoint_distance_from_end="180D"
        )
    )

    # Includes seasonality changepoints only.
    # Trend changepoints detection will be triggered and used as partial information
    # when detecting seasonality changepoints, but will not be used in the forecast model.
    changepoints = dict(
        seasonality_changepoints_dict=dict(
            potential_changepoint_distance="30D",
            seasonality_components_df=pd.DataFrame({
            "name": ["tow", "conti_year"],
            "period": [7.0, 1.0],
            "order": [4, 6],
            "seas_names": ["weekly", "yearly"]})
        )
    )

    # Grid search is possible
    changepoints = dict(
        changepoints_dict=[
            dict(
                method="custom",
                dates=["2018-08-01", "2019-08-03"]
            ),
            dict(
                method="custom",
                dates=["2019-08-03"]
            ),
        ],
        seasonality_changepoints_dict=[
            dict(),  # an empty dictionary triggers seasonality changepoints detection with default parameters
            dict(
                regularization_strength=0.4
            )
        ]
    )


.. note::

    Similar to (trend) changepoints, placing seasonality changepoints too close to the end of data is not
    recommended. It's important to specify the parameter ``no_changepoint_distance_from_end`` or
    ``no_changepoint_proportion_from_end``, where the former overrides the latter.


Prophet
-------

By default, there are 25 uniformly spaced changepoints.

Options:

.. code-block:: none

    changepoints : `dict` [`str`, `any`] or None
        Specifies the changepoint configuration. Dictionary with the following optional keys:

        changepoint_prior_scale : `float` or None or list of such values for grid search, default 0.05
            Parameter modulating the flexibility of the automatic changepoint selection.
            0.05 by default.
            Large values will allow many changepoints, small values will allow few changepoints.
        changepoints : `list` [`datetime.datetime`] or None or list of such values for grid search, default None
            List of dates at which to include potential changepoints. None by default, if not specified,
            potential changepoints are selected automatically.
        n_changepoints : `int` or None or list of such values for grid search, default 25
            Number of potential changepoints to include. Not used if input `changepoints` is supplied.
            If `changepoints` is not supplied, then n_changepoints potential changepoints are selected uniformly from
            the first `changepoint_range` proportion of the history.
        changepoint_range : `float` or None or list of such values for grid search, default 0.8
            Proportion of history in which trend changepoints will be estimated. Permitted values: (0,1]
            Not used if input `changepoints` is supplied.


Examples:

.. code-block:: python

    # Places specific changepoints to model piecewise linear growth.
    # If a date is not contained in one of the training splits, it is ignored for that split.
    changepoints = dict(
        changepoints=['2019-01-01', '2019-05-01', '2019-07-01']
    )

    # Specifies number of changepoints and % of training data used to place these points.
    # Grid search example.
    changepoints = dict(
        n_changepoints=[20, 30],
        changepoint_range=[0.7, 0.8]
    )

    # Modulates flexibility of trend fit
    changepoints = dict(
        changepoint_prior_scale=[0.04, 0.1, 0.5],
        n_changepoints=[20, 30],
        changepoint_range=[0.7, 0.8]
    )

.. note::

    Follow the same guidance as Silverkite for changepoints, with one key difference: n_changepoints
    can be higher for Prophet than for Silverkite.
    As a Bayesian model, Prophet uses ``changepoint_prior_scale`` to limit flexibility,
    even if the number of changepoints is high.

    To further improve model fit, you may try different ``n_changepoints`` and
    increasing ``changepoint_range``. However, make sure to reserve enough data after the
    last changepoint to learn the new trend and validate the forecast accuracy,
    as explained for Silverkite above.

    If ``changepoint_range`` is too high, seasonality effects can be mistaken for trend,
    and the forecast will not be accurate. On the other hand, because ``changepoint_range``
    is determined as a fraction of the input dataset,
    it can be safe to increase if you have many training points.
    For example, for 3 years of daily data, ``changepoint_range=0.8`` reserves 7.2 months after the
    last changepoint. If you believe the trend changed in those last 7.2 months,
    then you can try increasing ``changepoint_range`` while reserving data for validation.

.. note::

    It is possible to fine tune changepoint fit using ``changepoint_prior_scale``.
    You can fix overfit (too much flexibility) or underfit (not enough flexibility) using this parameter.
    There is no standard threshold. Optimum value depends on underlying data.

    Try different values in grid search to find a good fit via cross validation.
    As a general rule, choose final model which has lower test MAPE
    than other models to eliminate overfitting risk.
    e.g. you may try [0.01, 0.05, 0.25].

