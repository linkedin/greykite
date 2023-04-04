Seasonality
===========

Use ``model_components.seasonality`` to specify the seasonality.

Seasonality is used to model cyclical patterns in your data. These patterns
are predictable and repeat over a fixed period of time (e.g. yearly seasonality is a
cyclical pattern over 1 year).

You can control each seasonality for different cycle lengths.

To decide which types of seasonality to include, plot your input training data.
Plot the values over each cycle and see if there are repeated patterns.
See :doc:`/pages/stepbystep/0300_input` for details.

Silverkite
----------

Options (defaults shown for ``SILVERKITE`` template):

.. code-block:: none

    seasonality: `dict` [`str`, `any`] or None, optional
        Seasonality configuration dictionary, with the following optional keys.
        (keys are `SilverkiteSeasonalityEnum` members in lower case).

        The keys are parameters of
        `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.forecast_simple_silverkite`.
        Refer to that function for more details.

        auto_seasonality: `bool`, default False
            If set to True, will trigger automatic seasonality inferring.
            The keys below are ignored unless the value is False to force turn the seasonality off.
        yearly_seasonality: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
            Determines the yearly seasonality
            'auto', True, False, or a number for the Fourier order
        quarterly_seasonality: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
            Determines the quarterly seasonality
            'auto', True, False, or a number for the Fourier order
        monthly_seasonality: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
            Determines the monthly seasonality
            'auto', True, False, or a number for the Fourier order
        weekly_seasonality: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
            Determines the weekly seasonality
            'auto', True, False, or a number for the Fourier order
        daily_seasonality: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
            Determines the daily seasonality
            'auto', True, False, or a number for the Fourier order


For each seasonality, you can choose ``'auto', True, False, or a number for the Fourier order``.
Alternatively, you can choose to use automatic seasonality inferrer, which will infer
all seasonality orders for you.
Simply set ``auto_seasonality=True``, and the seasonality orders will be populated automatically.
For more details, see Seasonality Inferrer in :doc:`/gallery/quickstart/01_exploration/0200_auto_configuration_tools`.

.. code-block:: none

    True: model this seasonality with default Fourier order
    False: do not model this seasonality
    "auto": let the template decide, based on input data frequency and
            the amount of training data. As a general principle, it requires
            at least 2 full seasonal cycles and multiple observations per cycle
            (e.g. no weekly/daily seasonality for weekly data)
    int: model this seasonality with specified Fourier order

The Fourier order is a tuning knob for the flexibility. For ``Fourier order=k``,
the model includes ``2k`` seasonality terms. Higher values result in a more flexible model,
but may overfit to past data. Default fourier order is defined in
:py:class:`~greykite.algo.forecast.silverkite.constants.silverkite_seasonality.SilverkiteSeasonalityEnum`,
and you can decide how to increase/decrease order by comparing to the defaults.
The optimal value can be selected via hyperparameter tuning (CV grid search).

Examples:

.. code-block:: python

    # Example for daily data
    seasonality = dict(
        yearly_seasonality=True,      # turn seasonality "on"
        quarterly_seasonality=False,  # turn seasonality "off"
        monthly_seasonality="auto",   # let the template decide
        # weekly_seasonality=False,   # missing key is the same as "auto"
        daily_seasonality=False,      # it's not possible to have daily seasonality for daily observations
    )

    # Example for hourly data with custom yearly seasonality
    seasonality = dict(
        yearly_seasonality=3,
        # Leaves the rest up to the template ("auto")
    )

    # Example for weekly data with strong monthly patterns (e.g. Sales)
    seasonality = dict(
        yearly_seasonality=True,
        quarterly_seasonality=False,
        monthly_seasonality=2,  # there are 4 weeks in a month, so the number of terms must not exceed 4
        weekly_seasonality=False,
        daily_seasonality=False,
    )

    # Grid search is possible
    seasonality = dict(
        yearly_seasonality=True,
        quarterly_seasonality=[True, False],
        monthly_seasonality=False,
        weekly_seasonality=[3, 1],
        daily_seasonality=False,
    )

    # Auto seasonality
    seasonality = dict(
        auto_seasonality=True,  # automatically infers all seasonality orders
        yearly_seasonality=False,  # forces turning yearly seasonality off despite the inferring result
    )


.. note::

    Typical values are <= 4, up to 12 for daily seasonality, and up to 15 for yearly seasonality.
    The forecast will not improve (and may get worse) if Fourier order is too high.

    The Fourier order ``k`` should satisfy ``2k <= n_levels+1``. ``n_levels`` is the
    number of possible values in a cycle. For example, if your input data is at
    ``daily`` frequency:

    * monthly seasonality: ``n_levels=31`` days in a month.
    * weekly seasonality: ``n_levels=7`` days in a week.
    * daily seasonality: ``n_levels=1`` hours in a day.

    Furthermore, plot your timeseries to check how strong each seasonality effect is.
    Usually monthly seasonality has a weak effect, so we would limit its Fourier order to 2.


Prophet
-------

Options:

.. code-block:: none

    seasonality : `dict` [`str`, `any`] or None, optional
        Seasonality config dictionary, with the following optional keys.

        seasonality_mode: `str` or None or list of such values for grid search
            Can be 'additive' (default) or 'multiplicative'.
        seasonality_prior_scale: `float` or None or list of such values for grid search
            Parameter modulating the strength of the seasonality model. 10.0 by default.
            Larger values allow the model to fit larger seasonal fluctuations, smaller values dampen the seasonality.
            Specify for individual seasonalities using add_seasonality_dict.
        yearly_seasonality: `str` or `bool` or `int` or list of such values for grid search, default 'auto'
            Determines the yearly seasonality.
            Can be 'auto' (default), True, False, or a number of Fourier terms to generate.
        weekly_seasonality: `str` or `bool` or `int` or list of such values for grid search, default 'auto'
            Determines the weekly seasonality
            Can be 'auto' (default), True, False, or a number of Fourier terms to generate.
        daily_seasonality: `str` or `bool` or `int` or list of such values for grid search, default 'auto'
            Determines the daily seasonality
            Can be 'auto' (default), True, False, or a number of Fourier terms to generate.
        add_seasonality_dict: `dict` or None or list of such values for grid search
            dict of custom seasonality parameters to be added to the model, default=None
            Key is the seasonality component name e.g. 'monthly'; parameters are specified via dict.


For more information on ``add_seasonality_dict``,
see `~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator`.

.. note::

    Refer to Silverkite section for Fourier order.

    To define other seasonalities (``monthly``, ``quarterly``, etc),
    use ``add_seasonality_dict``.

    To further customize or overwrite built-in seasonalities (``yearly``, ``weekly``, ``daily``),
    set them to ``False`` in ``seasonalities`` and define within ``add_seasonality_dict``.
    See examples below.

.. note::

    If no seasonalities are provided or set as ``"auto"``, Prophet:

    * Turns on yearly seasonality if there is >=2 years of history.
    * Turns on weekly seasonality if there is >=2 weeks of history,
      and the spacing between dates in the history is <7 days.
    * Turns on daily seasonality if there is >=2 days of history,
      and the spacing between dates in the history is <1 day.

    Seasonality values ``True`` and ``False`` result in similar behavior as Silverkite.

Examples:

.. code-block:: python

    # example for daily data
    seasonality = dict(
        seasonality_mode=["additive"],        # seasonality effect is added to the trend to get forecast
        yearly_seasonality=[True],            # turn seasonality "on"
        # weekly_seasonality=[False],         # missing key is the same as "auto"
        daily_seasonality=[False],            # it's not possible to have daily seasonality for daily observations
        seasonality_prior_scale=[10.0, 25.0]  # grid search over seasonality strength options
    )

    # example for hourly data with custom yearly, monthly, and quarterly seasonality
    seasonality = dict(
        seasonality_mode=["multiplicative"],  # seasonality effect is multiplied
        yearly_seasonality=[3],
        add_seasonality_dict=[
        # custom seasonality - specify period, and optionally fourier_order & prior scale
        # it is possible to model different `seasonality_mode`s for custom seasonalities
            {
                'monthly': {
                    'period': 365.25/12,
                    'fourier_order': 10.0,
                    'mode': "additive"
                },
                'quarterly': {
                    'period': 365.25/4,
                    'fourier_order': 15.0,
                    'prior_scale': 15.0,
                    'mode': "multiplicative",
                }
            }]
        # leave the rest up to the template ("auto")
    )

    # example of disabling built-in weekly seasonality and customizing via add_seasonality_dict
    seasonality = dict(
        weekly_seasonality=[False],
        seasonality_prior_scale=[3.0],      # applies to daily and yearly ("auto") seasonalities.
        add_seasonality_dict=[
            {
                'weekly': {
                    'period': 7,
                    'fourier_order': 1.0,
                    'prior_scale': 5.0,     # customized, otherwise defaults to seasonality_prior_scale
                    'mode': "multiplicative"
                }
            }

    # example for weekly data with strong monthly patterns (e.g. Sales)
    seasonality = dict(
        seasonality_mode=["additive", "multiplicative"],  # grid search over both
        yearly_seasonality=[True, False],
        seasonality_prior_scale=[4.0, 8.0], # grid search over multiple seasonality strength options
        add_seasonality_dict=[              # grid search over multiple custom seasonalities
            {
                'monthly': {
                    'period': 365.25/12,
                    'fourier_order': 1.0
                },
                'quarterly': {
                    'period': 365.25/4,
                    'fourier_order': 3.0
                }
            },
            {
                'monthly': {
                    'period': 365.25/12,
                    'fourier_order': 2.0
                },
                'quarterly': {
                    'period': 365.25/4,
                    'fourier_order': 7.0
                }
            }]
    )


.. note::

    Use ``fourier_order`` and ``prior_scale`` to tune strength of seasonality effects.
    To model Seasonality that depends on other factors,
    see more details at `Prophet <https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#seasonalities-that-depend-on-other-factors>`_.
