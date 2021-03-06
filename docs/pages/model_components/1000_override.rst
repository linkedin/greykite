Pre-processing, Selective Grid Search
=====================================

Forecasts are generated by a `sklearn.pipeline.Pipeline`. It uses transformers
to pre-process the input data, followed by an estimator to fit the model and make
predictions. Use ``model_components.hyperparameter_override`` to override
pipeline parameters set by the other ``model_components`` attributes.

A few reasons to set this:

  1. Define pre-processing steps before the data is passed to the estimator
     for fitting and prediction. By default, there is no pre-processing except
     to impute null values in the response and numeric regressors.
     The other ``model_components`` attributes set estimator parameters,
     but not pre-processing parameters.

    * Whether/how to classify outliers (default=off)
    * Whether/how to normalize regressors (default=off)
    * The method for missing data interpolation (default=linear)
    * Whether degenerate columns should be removed before fitting (default=off)

  2. Run selective grid search over a list of hyperparameter grids. Without override,
     the template returns a single grid for full grid search over all combinations.
     See details below.

  3. Tune Estimator params that are not accessible via ``model_components``. These parameters
     are omitted by design, and likely do not need to be set. For example:

    * ``estimator__origin_for_time_vars`` (valid for the Silverkite model).
      The time origin is automatically set to the first date of the dataset, but override is possible.
    * ``estimator__silverkite`` (valid for the Silverkite model).
      Allows override of the algorithm used to fit the model.
      It can be used to provide an algorithm instance with customized behavior.

      If using a high-level silverkite template that relies on
      `~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`,
      the value should be an instance of
      `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.SimpleSilverkiteForecast`.
      Otherwise, if using a low-level silverkite template that relies on
      `~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator`,
      the value should be an instance of
      `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast`.

    * ``estimator__silverkite_diagnostics`` (valid for the Silverkite model).
      Allows override of the SilverkiteDiagnostics class that generates plots for the model.
      It can be used to provide an instance with customized behavior for plotting the components.
      Value should be an instance of `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteDiagnostics`.

Pipeline steps
^^^^^^^^^^^^^^

The forecast pipeline contains data preprocessing and estimation steps,
as outlined in the `sklearn.pipeline.Pipeline` below.

The pipeline steps are defined by
`~greykite.framework.pipeline.utils.get_basic_pipeline`,
copied here for reference.

``hyperparameter_override`` lets you set these components. It takes as dictionary whose
keys have the format:
``{named_step}__{parameter_name}`` for the named steps of the `sklearn.pipeline.Pipeline`.

See examples and the default values below.

.. code-block:: python

    pipeline = Pipeline([
        # Transforms the data prior to fitting
        ("input", PandasFeatureUnion([
            # Pass-through of time column.
            ("date", Pipeline([
                ("select_date", ColumnSelector([TIME_COL]))
            ])),
            # Transforms value column (prediction target, as input to `fit`).
            # This column is ignored by `predict`.
            ("response", Pipeline([
                ("select_val", ColumnSelector([VALUE_COL])),
                # For example, set z_cutoff parameter of this ZscoreOutlierTransformer
                # via ``input__response__outlier__z_cutoff``
                ("outlier", ZscoreOutlierTransformer(z_cutoff=None)),
                # Uses linear interpolation to fill in missing response values.
                # To avoid fitting to imputed values, set impute_algorithm=None.
                # Algorithms that require missing values to be imputed must use `impute_algorithm`.
                ("null", NullTransformer(impute_algorithm="interpolate"))
            ])),
            # Transforms numeric regressors.
            ("regressors_numeric", Pipeline([
                ("select_reg", ColumnSelector(regressor_cols)),
                ("select_reg_numeric", DtypeColumnSelector(include="number")),
                ("outlier", ZscoreOutlierTransformer(z_cutoff=None)),
                # For example, set NormalizeTransformer parameters via
                # `input__regressors_numeric__normalize__{parameter_name}`
                ("normalize", NormalizeTransformer(normalize_algorithm=None)),  # no normalization by default
                # Uses linear interpolation to fill in missing regressor values.
                ("null", NullTransformer(impute_algorithm="interpolate"))
            ])),
            # Pass-through of non-numeric regressors.
            ("regressors_other", Pipeline([
                ("select_reg", ColumnSelector(regressor_cols)),
                ("select_reg_non_numeric", DtypeColumnSelector(exclude="number"))
            ]))
        ])),
        # Optionally removes columns with constant value (default=False)
        ("degenerate", DropDegenerateTransformer()),
        # Performs the forecast `fit` and `predict`.
        # "Estimator" is either `SimpleSilverkiteEstimator` or `ProphetEstimator`,
        # as selected by `ForecastConfig.model_template`.
        ("estimator", Estimator(
            score_func=score_func, coverage=coverage, null_model_params=null_model_params
        ))
    ])

Pre-processing
^^^^^^^^^^^^^^

Below are some examples of how to set pre-processing parameters.

.. note::
    Some templates can handle gaps in the timeseries, such as ``SILVERKITE`` and ``PROPHET``.
    They do not require the response value to be imputed.

    For these templates, if you have missing response values in your data, it may be
    beneficial not to impute the values, to avoid fitting the model to imputed values.
    Set ``input__response__null__impute_algorithm`` to None and see if forecast
    quality improves.

    If your timeseries has regular missing values (e.g. observations only
    between 9am-5pm each day, null otherwise), it is especially important to
    set ``input__response__null__impute_algorithm`` to None.

    If a few values are missing at random, you may also try setting it to None,
    or use your expert knowledge to pick a suitable interpolation method
    ("interpolate" or "ts_interpolate" with corresponding ``impute_params``),
    as shown below.

.. code-block:: python

    # The default values (if not specified)
    hyperparameter_override=dict(
        # Transformers for the response (value_col), the fitting target.
        # Allows outlier removal (replace with null), followed by null imputation.
        input__response__outlier__use_fit_baseline=False,
        input__response__outlier__z_cutoff=None,
        input__response__null__impute_algorithm="interpolate",
        input__response__null__impute_all=True,
        input__response__null__impute_params=None,
        input__response__null__max_frac=0.10,
        # Transformers for numeric regressors provided in ``df``,
        # used to predict the response.
        # Allows outlier removal, normalization, and null imputation.
        input__regressors_numeric__outlier__use_fit_baseline=False,
        input__regressors_numeric__outlier__z_cutoff=None,
        input__regressors_numeric__normalize__normalize_algorithm=None,
        input__regressors_numeric__normalize__normalize_params=None,
        input__regressors_numeric__null__impute_algorithm="interpolate",
        input__regressors_numeric__null__impute_all=True,
        input__regressors_numeric__null__impute_params=None,
        input__regressors_numeric__null__max_frac=0.10,
        # Whether to drop degenerate regressors before fitting.
        # May cause fit to fail if a column is dropped and not
        # found by the estimator.
        degenerate__drop_degenerate=False,
    )

    # Example custom pre-processing configuration
    hyperparameter_override=dict(
        # Sets z-cutoff to a high value. Response values above this
        # are replaced with NaN and imputed.
        input__response__outlier__z_cutoff=10.0,
        # Sets response null imputation to use the average value
        # 7 and 14 periods ago (if input data has daily frequency,
        # this is the value 1 week and 2 weeks ago on the same day of week).
        input__response__null__impute_algorithm="ts_interpolate",
        input__response__null__impute_params=dict(
            orders=[7, 14],
            agg_func=np.mean,
            iter_num=5,  # repeats up to 5x if missing values take multiple iterations to fill
        ),
        input__response__null__impute_all=True,  # guarantees all nulls are imputed (default)
        # Sets z-cutoff to a high value. Numeric regressor values above this
        # are replaced with NaN and imputed.
        input__regressors_numeric__outlier__z_cutoff=10.0,
        # Uses `sklearn.preprocessing.RobustScaler` to normalize numeric regressors
        # before fitting. Sets parameters to this scaler via `normalize_params`.
        input__regressors_numeric__normalize__normalize_algorithm="RobustScaler",
        input__regressors_numeric__normalize__normalize_params=dict(
            quantile_range=(10.0, 90.0)
        ),
        # Same null imputation configuration for the numeric regressors
        input__regressors_numeric__null__impute_algorithm="ts_interpolate",
        input__regressors_numeric__null__impute_params=dict(
            orders=[7, 14],
            agg_func=np.mean,
            iter_num=5,
        ),
        input__regressors_numeric__null__impute_all=True,
        # Drops degenerate columns before fitting
        degenerate__drop_degenerate=True,
    )

    # As usual, list can be used for any of the parameters,
    # for grid search over the options.
    hyperparameter_override=dict(
        input__response__outlier__z_cutoff=[4.0, None],
        input__response__null__impute_algorithm=["ts_interpolate", "interpolate", None],
    )

See more details about the transformer parameters here:

  * `~greykite.sklearn.transform.zscore_outlier_transformer.ZscoreOutlierTransformer`
  * `~greykite.sklearn.transform.normalize_transformer.NormalizeTransformer`
  * `~greykite.sklearn.transform.null_transformer.NullTransformer`
  * `~greykite.sklearn.transform.drop_degenerate_transformer.DropDegenerateTransformer`

Do not set the parameters of ``ColumnSelector`` or ``DTypeColumnSelector``.

Selective grid search
^^^^^^^^^^^^^^^^^^^^^

Rather than provide a single grid, `~sklearn.model_selection.RandomizedSearchCV`
allows passing a list of grids.

This can be useful if your search space is large:

  * If you have 6 parameter with 3 values each, the search space is 3^6=729, too large to be practical.
  * You could set ``hyperparameter_budget``, but because this searches at random,
    some parameter values might not be explored.
  * Instead, one strategy is to grid search on each parameter independently, for 3*6=18 options.
  * The optimal values from this initial grid search can rule out "bad" parameter values. Further
    grid search can scan combinations of "good" parameters.

To use this approach, the specified ``model_components`` should be a single "best guess" model
from which to make small changes for exploration.

Modifications to this base model are defined by ``hyperparameter_override``. It can be a dictionary
as shown above, or a list of dictionaries. For selective grid search, provide a list of dictionaries.
Each dictionary in the list updates the hyperparameters specified by ``model_components``
to generate the search space.

To override ``estimator`` parameters, use ``"estimator__{param}"`` in ``hyperparameter_override``,
where ``param`` is a parameter used to initialize.

See more details about the estimator parameters here:

  * `~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator` (SILVERKITE template)
  * `~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator` (PROPHET template)

Examples:

.. code-block:: python

    hyperparameter_override = [
        {
            # Uses the original, unmodified estimator parameters from `model_components`.
            # We can set pre-processing parameters in selective grid search.
            # In this example, we change the null imputation algorithm
            # in all the grids.
            "input__response__null__impute_algorithm": "ts_interpolate",
        },
        {
            # Explores two options for holiday countries
            "estimator__holiday_lookup_countries": [
                [
                    "UnitedStates",
                    "UnitedKingdom",
                    "India",
                    "France",
                    "China",
                ],
                [
                    "UnitedStates",
                    "India",
                ]
            ],
            "input__response__null__impute_algorithm": "ts_interpolate",
        },
        {
            # Explores two options for fit algorithms
            "estimator__fit_algorithm_dict": [
                dict(
                    fit_algorithm="ridge"
                ),
                dict(
                    fit_algorithm="elastic_net"
                ),
            ],
            "input__response__null__impute_algorithm": "ts_interpolate",
        },
        {
            # Tries adding a changepoint
            "estimator__changepoints_dict": [
                dict(
                    method="custom",
                    dates=["2019-08-01-00"],
                    continuous_time_col="ct1",
                    growth_func=lambda x: x
                )
            ],
            "input__response__null__impute_algorithm": "ts_interpolate",
        },
        # etc.
    ]

The illustrates how override updates the parameters set by the
other ``model_components`` attributes.

.. code-block:: python

    # Original grid defined by the other model_components attributes.
    original_grid = {"estimator__param": 5}
    # Override options.
    hyperparameter_override = [
        {},
        {"estimator__param": [10]},
        {"estimator__param1": ["a", "b", "c"]},
        {"estimator__param2": [1.0, 2.0]},
    ]
    # The resulting search space:
    hyperparameter_grid = [
        {"estimator__param": 5},     # original
        {"estimator__param": [10]},  # replaced
        {"estimator__param": 5, "estimator__param1": ["a", "b", "c"]},  # added
        {"estimator__param": 5, "estimator__param2": [1.0, 2.0]},       # added
    ]
    # After auto-list conversion, this grid is passed
    # to `sklearn.model_selection.RandomizedSearchCV`
    # to set the Pipeline parameters.
    hyperparameter_grid = [
        {"estimator__param": [5]},
        {"estimator__param": [10]},
        {"estimator__param": [5], "estimator__param1": ["a", "b", "c"]},
        {"estimator__param": [5], "estimator__param2": [1.0, 2.0]},
    ]

    # The search space has 7 options.
    # To see this more clearly, consider the
    # equivalent (flattened) search space.
    hyperparameter_grid = [
        {"estimator__param": [5]},
        {"estimator__param": [10]},
        {"estimator__param": [5], "estimator__param1": ["a"]},
        {"estimator__param": [5], "estimator__param1": ["b"]},
        {"estimator__param": [5], "estimator__param1": ["c"]},
        {"estimator__param": [5], "estimator__param2": [1.0]},
        {"estimator__param": [5], "estimator__param2": [2.0]},
    ]

    # To cover the above (7) options without using hyperparameter
    # override, we'd need to check 12 combinations (2 x 3 x 2)
    # as shown in the grid below.
    # Thus, the override allows for more precise grid search over the
    # combinations of interest.
    original_grid = [
        {
            "estimator__param": [5, 10],
            "estimator__param1": ["a", "b", "c"],
            "estimator__param2": [1.0, 2.0]
        },
    ]
