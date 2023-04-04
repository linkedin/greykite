Uncertainty Intervals
=====================

Silverkite and Prophet support prediction intervals to quantify uncertainty.

Set :ref:`ForecastConfig.coverage <coverage>` to request a prediction interval
with the desired coverage.

You can use ``model_components.uncertainty`` to configure additional options
about how these intervals are calculated.

Silverkite
----------

Options:

.. code-block:: none

        uncertainty : `dict` [`str`, `dict`] or None, optional
            Along with ``coverage``, specifies the uncertainty interval configuration. Use ``coverage``
            to set interval size. Use ``uncertainty`` to tune the calculation.

            ``"uncertainty_dict"`` : `dict` or "auto" or None or a list of such values for grid search
                 If a dictionary, valid keys are:

                    ``"uncertainty_method"`` : `str`
                        The title of the method.
                        Only ``"simple_conditional_residuals"`` is implemented
                        in `~greykite.algo.common.ml_models.fit_ml_model`
                        which calculates intervals using residuals.

                    ``"params"``: `dict`
                        A dictionary of parameters needed for
                        the requested ``uncertainty_method``. For example, for
                        ``uncertainty_method="simple_conditional_residuals"``, see
                        parameters of `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`:

                            * ``"conditional_cols"``
                            * ``"quantiles"``
                            * ``"quantile_estimation_method"``
                            * ``"sample_size_thresh"``
                            * ``"small_sample_size_method"``
                            * ``"small_sample_size_quantile"``

                        The default value for ``quantiles`` is inferred from coverage.

                If "auto", see
                `~greykite.algo.forecast.silverkite.forecast_silverkite_helper.get_silverkite_uncertainty_dict`
                for the default value. If ``coverage`` is not None and ``uncertainty_dict`` is not provided,
                then the "auto" setting is used.

                If ``coverage`` is None and ``uncertainty_dict`` is None, then no intervals are returned.

Examples:

The only values that should be adjusted are ``conditional_cols``,
``sample_size_thresh``, and ``small_sample_size_quantile``.

.. code-block:: python

    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": None,
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],  # interval size depends on hour of week
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}


.. Note: The `quantiles` parameter is not documented because only the lower/upper
   quantiles are available in `forecast_pipeline` output.


Prophet
-------

Options:

.. code-block:: none

    uncertainty : `dict` [`str`, `any`] or None
        Specifies the uncertainty configuration. A dictionary with the following optional keys:

        mcmc_samples: `int` or None or list of such values for grid search, default 0
            if greater than 0, will do full Bayesian inference with the specified number of MCMC samples.
            If 0, will do MAP estimation.
        uncertainty_samples: `int` or None or list of such values for grid search, default 1000
            Number of simulated draws used to estimate.
            uncertainty intervals. Setting this value to 0 or False will disable
            uncertainty estimation and speed up the calculation.
