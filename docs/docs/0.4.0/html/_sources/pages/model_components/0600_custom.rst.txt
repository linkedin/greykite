Custom Parameters
=================

Parameters unique to the Silverkite algorithm are included under ``model_components.custom``.

.. tip::

    Refer to this section for customizing Silverkite model.
    ``model_components.custom`` is currently not supported for Prophet.


.. code-block:: python

    custom = dict(
        fit_algorithm_dict=fit_algorithm_dict,
        feature_sets_enabled=feature_sets_enabled,
        max_daily_seas_interaction_order=max_daily_seas_interaction_order,
        max_weekly_seas_interaction_order=max_weekly_seas_interaction_order,
        extra_pred_cols=extra_pred_cols,
        drop_pred_cols=drop_pred_cols,
        explicit_pred_cols=explicit_pred_cols,
        min_admissible_value=min_admissible_value,
        max_admissible_value=max_admissible_value,
    )


Fit algorithm
-------------

Silverkite abstracts feature generation from model fitting. First, Silverkite generates features
for forecasting using time-series domain knowledge. Then, any standard algorithm can be applied
for fitting.

Silverkite supports many fit algorithms from ``scikit-learn`` and ``statsmodels``, as
well as our own fast quantile regression implementation.
Here are a few important ones:

.. csv-table:: Silverkite fitting algorithms
   :widths: 25 25 50
   :header: "fit_algorithm", "implementation", "notes"

   "ridge", `~sklearn.linear_model.RidgeCV`, "Default alpha = np.logspace(-5, 5, 30)"
   "lasso", `~sklearn.linear_model.LassoCV`, ""
   "elastic_net", `~sklearn.linear_model.ElasticNetCV`, ""
   "linear", `~statsmodels.regression.linear_model.OLS`, ""
   "quantile_regression", `~greykite.algo.common.l1_quantile_regression.QuantileRegression`, "Default quantile=0.5, alpha=0"

.. note::

  For mean forecasts, ``"ridge"`` or ``"linear"`` are recommended.
  ``"ridge"`` is preferred when there are many features.

  For quantile prediction, ``"quantile_regression"`` is recommended.

  ``"sgd"`` is often unstable. Tree-based methods (``"rf"``, ``"gradient_boosting"``)
  are generally not good at projecting growth into the future.

Here is the full list of supported fit algorithms. You can select the algorithm via
``fit_algorithm`` and set its initialization parameters via ``fit_algorithm_params``.

.. code-block:: none

    fit_algorithm_dict : `dict` or a list of such values for grid search
        How to fit the model. A dictionary with the following optional keys.

        ``"fit_algorithm"`` : `str`, default "ridge"
            The type of model used in fitting
            (implemented by sklearn and statsmodels).

            Available models are:

                - "statsmodels_ols"   : `statsmodels.regression.linear_model.OLS`
                - "statsmodels_wls"   : `statsmodels.regression.linear_model.WLS`
                - "statsmodels_gls"   : `statsmodels.regression.linear_model.GLS`
                - "statsmodels_glm"   : `statsmodels.genmod.generalized_linear_model.GLM`
                - "linear"            : `statsmodels.regression.linear_model.OLS`
                - "elastic_net"       : `sklearn.linear_model.ElasticNetCV`
                - "ridge"             : `sklearn.linear_model.RidgeCV`
                - "lasso"             : `sklearn.linear_model.LassoCV`
                - "sgd"               : `sklearn.linear_model.SGDRegressor`
                - "lars"              : `sklearn.linear_model.LarsCV`
                - "lasso_lars"        : `sklearn.linear_model.LassoLarsCV`
                - "rf"                : `sklearn.ensemble.RandomForestRegressor`
                - "gradient_boosting" : `sklearn.ensemble.GradientBoostingRegressor`
                - "quantile_regression" : `greykite.algo.common.l1_quantile_regression.QuantileRegression`

            See `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
            for the sklearn and statsmodels classes that implement these methods, and their default parameters.

            "linear" is the same as "statsmodels_ols", because `statsmodels.regression.linear_model.OLS`
            is more stable than `sklearn.linear_model.LinearRegression`.
        ``"fit_algorithm_params"`` : `dict` or None, default None
            Parameters passed to the requested fit algorithm.
            If None, uses the defaults in `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.

Examples:

.. code-block:: python

    custom = dict(
        fit_algorithm_dict=dict(
            fit_algorithm="linear"
        )
    )

    custom = dict(
        fit_algorithm_dict=dict(
            fit_algorithm="ridge",
            fit_algorithm_params={
                "alphas": np.logspace(-5, 5, 30)
            }
        )
    )

    custom = dict(
        fit_algorithm_dict=dict(
            fit_algorithm="quantile_regression",
            fit_algorithm_params={
                "quantile": 0.5,
                "alpha": 1,
            }
        )
    )

    # Grid search is possible
    custom = dict(
        fit_algorithm_dict=[
            dict(
                fit_algorithm="linear"
            ),
            dict(
                fit_algorithm="ridge",
                fit_algorithm_params={
                    "alphas": np.logspace(-5, 5, 30)
                }
            ),
        ]
    )


Interactions
------------

You can include interactions via the ``feature_sets_enabled`` parameter. Setting ``feature_sets_enabled="auto"``
adds interactions appropriate to the data frequency and amount of training history.

The fourier order of seasonality interaction terms can be capped by
``max_daily_seas_interaction_order`` and ``max_weekly_seas_interaction_order``
for daily and weekly seasonality, respectively.

Options (defaults shown for ``SILVERKITE`` template):

.. code-block:: none

    feature_sets_enabled : `dict` [`str`, `bool` or "auto" or None] or `bool` or "auto" or None; or a list of such values for grid search, default "auto"
        Whether to include interaction terms and categorical variables to increase model flexibility.

        If a `dict`, boolean values indicate whether include various sets of features in the model.
        The following keys are recognized
        (from `~greykite.algo.forecast.silverkite.constants.silverkite_column.SilverkiteColumn`):

            ``"COLS_HOUR_OF_WEEK"`` : `str`
                Constant hour of week effect
            ``"COLS_WEEKEND_SEAS"`` : `str`
                Daily seasonality interaction with is_weekend
            ``"COLS_DAY_OF_WEEK_SEAS"`` : `str`
                Daily seasonality interaction with day of week
            ``"COLS_TREND_DAILY_SEAS"`` : `str`
                Allow daily seasonality to change over time by is_weekend
            ``"COLS_EVENT_SEAS"`` : `str`
                Allow sub-daily event effects
            ``"COLS_EVENT_WEEKEND_SEAS"`` : `str`
                Allow sub-daily event effect to interact with is_weekend
            ``"COLS_DAY_OF_WEEK"`` : `str`
                Constant day of week effect
            ``"COLS_TREND_WEEKEND"`` : `str`
                Allow trend (growth, changepoints) to interact with is_weekend
            ``"COLS_TREND_DAY_OF_WEEK"`` : `str`
                Allow trend to interact with day of week
            ``"COLS_TREND_WEEKLY_SEAS"`` : `str`
                Allow weekly seasonality to change over time

        The following dictionary values are recognized:

            - True: include the feature set in the model
            - False: do not include the feature set in the model
            - None: do not include the feature set in the model
            - "auto" or not provided: use the default setting based on data frequency and size

        If not a `dict`:

            - if a boolean, equivalent to a dictionary with all values set to the boolean.
            - if None, equivalent to a dictionary with all values set to False.
            - if "auto", equivalent to a dictionary with all values set to "auto".

    max_daily_seas_interaction_order : `int` or None or a list of such values for grid search, default 5
        Max fourier order to use for interactions with daily seasonality
        (COLS_EVENT_SEAS, COLS_EVENT_WEEKEND_SEAS, COLS_WEEKEND_SEAS, COLS_DAY_OF_WEEK_SEAS, COLS_TREND_DAILY_SEAS).

        Model includes interactions terms specified by ``feature_sets_enabled``
        up to the order limited by this value and the available order from ``seasonality``.

    max_weekly_seas_interaction_order : `int` or None or a list of such values for grid search, default 2
        Max fourier order to use for interactions with weekly seasonality (COLS_TREND_WEEKLY_SEAS).

        Model includes interactions terms specified by ``feature_sets_enabled``
        up to the order limited by this value and the available order from ``seasonality``.


.. csv-table:: when to use each feature set
   :widths: 25 25 25 25
   :header: "feature set", "max freq", "when to use", "(human-readable) formula"

   "COLS_HOUR_OF_WEEK", "hourly", "hour of week effect to help daily seasonality model", "hour_of_week"
   "COLS_WEEKEND_SEAS", "hourly", "weekend has a different daily seasonality pattern", "is_weekend:daily_seas"
   "COLS_DAY_OF_WEEK_SEAS", "hourly", "each day has a different daily seasonality pattern", "day_of_week:daily_seas"
   "COLS_TREND_DAILY_SEAS", "hourly", "daily seasonality pattern changes over time, by is_weekend", "trend:is_weekend:daily_seas"
   "COLS_EVENT_SEAS", "hourly", "events have a different daily seasonality pattern", "event:daily_seas"
   "COLS_EVENT_WEEKEND_SEAS", "hourly", "events have a different daily event seasonality pattern, by is_weekend", "event:is_weekend:daily_seas"
   "COLS_DAY_OF_WEEK", "daily", "day of week effect to help weekly seasonality model", "day_of_week"
   "COLS_TREND_WEEKEND", "daily", "growth rate differs for weekend/weekday", "trend:is_weekend"
   "COLS_TREND_DAY_OF_WEEK", "daily", "growth rate differs by day of week", "trend:day_of_week"
   "COLS_TREND_WEEKLY_SEAS", "daily", "weekly seasonality pattern changes over time", "trend:weekly_seas"


Examples:

.. code-block:: python

    from greykite.algo.forecast.silverkite.constants.silverkite_column import SilverkiteColumn

    # Uses the default for all feature sets based on data frequency and size (training data)
    custom = dict(
        feature_sets_enabled="auto"
    )
    # Turns off all feature sets
    custom = dict(
        feature_sets_enabled=False
    )
    custom = dict(
        feature_sets_enabled=None  # same as False (prefer False to be explicit)
    )

    # Turns on all feature sets
    # (Not recommended. Use "auto" to enable all relevant feature sets, or
    #  enable specific feature sets as shown below.)
    custom = dict(
        feature_sets_enabled=True
    )
    # Turns on specific feature sets
    custom = dict(
        feature_sets_enabled={
            # Not included in the model.
            SilverkiteColumn.COLS_HOUR_OF_WEEK: False,
            SilverkiteColumn.COLS_WEEKEND_SEAS: False,
            SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: False,
            SilverkiteColumn.COLS_TREND_DAILY_SEAS: False,
            SilverkiteColumn.COLS_EVENT_SEAS: False,
            # None is the same as False (prefer False to be explicit)
            SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: None,
            # Included in the model.
            SilverkiteColumn.COLS_DAY_OF_WEEK: True,
            SilverkiteColumn.COLS_TREND_WEEKEND: True,
            # Auto uses the default based on data frequency and size.
            SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: "auto",
            # Omitted key is treated the same as "auto".
            # SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: "auto"
        },
        # Allows up to fourier order 2 for weekly seasonality interactions
        max_weekly_seas_interaction_order=2
    )

    # Turns on a few feature sets relevant for hourly data
    custom = dict(
        feature_sets_enabled={
            SilverkiteColumn.COLS_HOUR_OF_WEEK: False,
            SilverkiteColumn.COLS_WEEKEND_SEAS: True,
            SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: True,
            SilverkiteColumn.COLS_TREND_DAILY_SEAS: True,
            SilverkiteColumn.COLS_EVENT_SEAS: False,  # unnecessary when COLS_EVENT_WEEKEND_SEAS is used
            SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: True,
            SilverkiteColumn.COLS_DAY_OF_WEEK: False,
            SilverkiteColumn.COLS_TREND_WEEKEND: True,
            SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: True,
            SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: True
        },
        # Allows up to fourier order 2 for daily/weekly seasonality interactions
        max_daily_seas_interaction_order=2,
        max_weekly_seas_interaction_order=2
    )


To check which features sets are enabled by default for your dataset, call
``get_feature_sets_enabled``.

* The parameter ``num_days`` is the number of days in your input timeseries
  (historical data for training, without future dates for regressors). It does not need
  to be exact.

.. code-block:: python

    from greykite.algo.forecast.silverkite.forecast_simple_silverkite import SimpleSilverkiteForecast
    from greykite.common.enums import SimpleTimeFrequencyEnum

    silverkite = SimpleSilverkiteForecast()
    # 60 days of hourly data
    silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=60,
        feature_sets_enabled="auto")
    # 60 days of daily data
    silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        num_days=60,
        feature_sets_enabled="auto")
    # 30 weeks of weekly data
    silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.WEEK.name,
        num_days=7*30,
        feature_sets_enabled="auto")
    # 30 months of monthly data
    silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.MONTH.name,
        num_days=30*30,
        feature_sets_enabled="auto")
    # 20 quarters of quarterly data
    silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.QUARTER.name,
        num_days=90*20,
        feature_sets_enabled="auto")
    # 12 years of yearly data
    silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.YEAR.name,
        num_days=365*12,
        feature_sets_enabled="auto")


Specify model terms
-------------------

For even finer control than ``feature_sets_enabled``, you can specify additional model terms
via ``extra_pred_cols``. Any valid patsy model formula term is accepted. You need to know how
columns are internally coded to use this function. Most users will not need this option.

.. note::

  While it's possible to add many terms in the model, a high degree of complexity may not
  generalize well into the forecast period and should not be necessary for most forecasts.

.. code-block:: none

    extra_pred_cols : `list` [`str`] or None or a list of such values for grid search, default None
        Names of extra predictor columns to pass to ``forecast_silverkite``.
        The standard interactions can be controlled via ``feature_sets_enabled`` parameter.
        Accepts any valid patsy model formula term. Can be used to model complex interactions
        of time features, events, seasonality, changepoints, regressors. Columns should be
        generated by ``build_silverkite_features`` or included with input data.
        These are added to any features already included by ``feature_sets_enabled`` and
        terms specified by ``model``.

Example:

.. code-block:: python

    import greykite.common.constants as cst
    from greykite.framework.utils.result_summary import patsy_categorical_term

    # Provides all holidays of interest
    holiday_names = ["Christmas Day", "Thanksgiving", "Labor Day"]

    # Adds day-of-holiday interaction with is_weekend
    # (does not add the interaction on days offset from the holiday)
    extra_pred_cols = []
    event_levels = [cst.EVENT_DEFAULT, cst.EVENT_INDICATOR]
    for holiday_name in holiday_names:
        holiday_term = patsy_categorical_term(
            term=f"{cst.EVENT_PREFIX}_{holiday_name}",  # holiday column name
            levels=event_levels)  # levels for holiday categorical variable
        extra_pred_cols += [f"is_weekend:{holiday_term}"]

    # Tells the model to include these parameters
    custom = dict(extra_pred_cols=extra_pred_cols)

.. note::

    Contact us if you're using ``extra_pred_cols``. If your model terms are useful for others, we can add
    them to ``feature_sets_enabled``.


Similarly, you may specify terms to exclude via ``"drop_pred_cols"``:

.. code-block:: none

    drop_pred_cols : `list` [`str`] or None, default None
        Names of predictor columns to be dropped from the final model.
        Ignored if None.

To directly specify all the terms used in the final model, use ``"explicit_pred_cols"``:

.. code-block:: none

    explicit_pred_cols : `list` [`str`] or None, default None
        Names of the explicit predictor columns which will be
        the only variables in the final model. Note that this overwrites
        the generated predictors in the model and may include new
        terms not appearing in the predictors (e.g. interaction terms).
        Ignored if None.

Forecast limits
---------------

You may prevent the forecast from going above or below pre-set values
via ``min_admissible_value`` and ``max_admissible_value``.

This can be useful, for example, if you are forecasting a non-negative
metric.

.. code-block:: none

    min_admissible_value : `float` or `double` or `int` or None, default None
        The lowest admissible value for the forecasts and prediction
        intervals. Any value below this will be mapped back to this value.
        If None, there is no lower bound.
    max_admissible_value : `float` or `double` or `int`, default None
        The highest admissible value for the forecasts and prediction
        intervals. Any value above this will be mapped back to this value.
        If None, there is no upper bound.

Examples:

.. code-block:: python

    # enforce non-negative forecast
    custom = dict(
        min_admissible_value=0
    )

    # specifies an acceptable range
    custom = dict(
        min_admissible_value=1e3
        max_admissible_value=1e6
    )

Normalization
-------------

It can be helpful to normalize features, especially when features have different magnitudes
and regularization is used.

.. code-block:: none

    normalize_method : `str` or None, default None
        The normalization method for the feature matrix.
        Available values are "statistical", "zero_to_one" and "minus_half_to_half".

Examples:

.. code-block:: python

    custom = dict(
        normalize_method="statistical"
    )
    custom = dict(
        normalize_method="zero_to_one"
    )

The ``statistical`` method removes the "mean" and divides by "std" for each column.
The ``zero_to_one`` method removes the "min" and divides by the "max - min"
The ````minus_half_to_half```` method removes the "(min + max)/2" and divides by the "max - min"
for each column. For details, see `~greykite.common.features.normalize.normalize_df`.
