.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_gallery_tutorials_0200_templates.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_gallery_tutorials_0200_templates.py:


Templates
=========

This is a basic tutorial for utilizing model templates and adding additional customization.
After reading this tutorial, you will be able to

    - Choose the desired template(s) to use.
    - Customize the parameters in the templates.
    - Perform grid search via multiple templates/parameters.

Model templates include parameters and their default values that are ready to use for forecast models.
These templates can be run with the :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`
function by defining the `~greykite.framework.templates.autogen.forecast_config.ForecastConfig` class.
The intention of using these templates is to easily combine with our pipeline
`~greykite.framework.pipeline.pipeline.forecast_pipeline` and do grid search, cross-validation and evaluation.
In this tutorial, we will go over available templates, their default values and how to customize the parameters.

A General View of the Templates and the ``ForecastConfig`` Class
----------------------------------------------------------------

There are two forecast models: the ``PROPHET`` model and the ``SILVERKITE`` model. The ``PROPHET`` model has an estimator named
`~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator`. The ``SILVERKITE`` model has a high-level
estimator named `~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`
and a low-level estimator named `~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator`.

The estimators only implement fit and predict methods. They are used in the
`~greykite.framework.pipeline.pipeline.forecast_pipeline` function,
for grid search, cross-validation, backtest, and forecast.
To easily configure and run the forecast pipeline, pass
`~greykite.framework.templates.autogen.forecast_config.ForecastConfig`
to the :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config` function.

The `~greykite.framework.templates.autogen.forecast_config.ForecastConfig` class takes the following
parameters

    - ``computation_param``: the `~greykite.framework.templates.autogen.forecast_config.ComputationParam` class,
      defines the computation parameters of the pipeline.
    - ``coverage``: the forecast interval coverage.
    - ``evaluation_metric_param``: the `~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam` class,
      defines the metrics used to evaluate performance and choose the best model.
    - ``evaluation_period_param``: the `~greykite.framework.templates.autogen.forecast_config.EvaluationPeriodParam` class,
      defines the cross-validation and train-test-split rules.
    - ``forecast_horizon``: the forecast horizon.
    - ``metadata_param``: the `~greykite.framework.templates.autogen.forecast_config.MetadataParam` class,
      defines the metadata of the training data.
    - ``model_components_param``: the `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` class,
      defines the model parameters.
    - ``model_template``: the name or dataclass of the base model template, corresponding to one or some pre-defined
      `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` class(es).

Among these parameters, ``model_components_param`` and ``model_template`` define the parameters used in the estimators.

    #. The full estimator parameters can be specified through the
       `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` class,
       as described `here <../../pages/model_components/0100_introduction.html>`_.
    #. We have pre-defined `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` classes
       that serve as default estimator parameters for different use cases. These pre-defined ``ModelComponentsParam`` classes have names.
    #. You can specify in the ``model_template`` parameter a valid model template name.
       The function will automatically map the ``model_template`` input to the corresponding estimator and its default parameters.
    #. To override the default values, you can create a
       `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` class
       with only the parameters you want to override, and pass it to the ``model_components_param`` parameter.

Note that you don't have to specify all values in the
`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`
to override the defaults. If a parameter is not specified, the default value for the parameter
specified by the model template name will be used.
In the later sections we will go over the valid ``model_template`` and ``ModelComponentsParam`` for each of the
three estimators.
For details about how to configure the other parameters and how to use the ``run_forecast_config`` function, see
`Tune your first forecast model <./0100_forecast_tutorial.html>`_.

The three estimators accept different input for ``model_template`` and ``ModelComponentsParam``.
Below are the valid input types for the ``model_template`` parameter.

    - High-level ``Silverkite`` template: for the high-level ``SimpleSilverkiteEstimator``, we have model templates named
      ``"SILVERKITE"``, ``"SILVERKITE_EMPTY"``, ``"SILVERKITE_DAILY_90"``, ``"SILVERKITE_WEEKLY"`` and a set of
      generic naming following some rules. This type of model templates support list input for both
      ``model_template`` and ``model_components_param`` parameters.
      This type of model templates are most recommended for ease of use.
    - Low-level ``Silverkite`` template: for the low-level ``SilverkiteEstimator``, we have a model template
      named ``"SK"``. This template allows you to configure lower-level parameters in the ``Silverkite`` model.
      This template does not support list input.
    - Prophet template: for the ``ProphetEstimator``, we have a model template named ``"PROPHET"``.
      This template does not support list input.

To customize the default parameters in the templates, the
`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` dataclass
takes the following parameters

* ``growth``: defines how the trend of the time series grows.
* ``seasonality``: defines the seasonality components and orders.
* ``changepoints``: defines when trend and/or seasonality should change, including automatic options.
* ``events``: defines short term events and holidays.
* ``autoregression``: defines the lags and aggregations for the past values.
* ``regressors``: defines extra regressors.
* ``uncertainty``: defines the forecast interval parameters.
* ``custom``: defines parameters that do not belong to the other sections.
* ``hyperparameter_override``: used to create overrides for the parameters specified above; useful in grid search.

The model's tuning parameters are set according to the categories above.
However, different estimators take different types of values for these categories.
We will go over each of the three types of templates, their default values, and how to customize the
``ModelComponentsParam`` for them.
For more general details, see :doc:`/pages/model_components/0100_introduction`.


.. code-block:: default
   :lineno-start: 108

    # Imports related libraries.
    import pandas as pd

    from greykite.framework.templates.autogen.forecast_config import ForecastConfig
    from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
    from greykite.framework.templates.model_templates import ModelTemplateEnum
    from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate








The High-level Templates in ``SILVERKITE``
------------------------------------------
The high-level templates in ``SILVERKITE`` provides many good defaults that work under different scenarios.
All templates in this section use `~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`.
The two most basic templates are ``"SILVERKITE"`` and ``"SILVERKITE_EMPTY"``.

``"SILVERKITE"`` is a template with automatic growth, seasonality, holidays, and interactions.
It works best for hourly and daily frequencies.
If you specify ``"SILVERKITE"`` as ``model_template``, the following
`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` class
is used as default template values.


.. code-block:: default
   :lineno-start: 129


    model_components_param_silverkite = ModelComponentsParam(
        growth={
            "growth_term": "linear"
        },
        seasonality={
            "yearly_seasonality": "auto",
            "quarterly_seasonality": "auto",
            "monthly_seasonality": "auto",
            "weekly_seasonality": "auto",
            "daily_seasonality": "auto",
        },
        changepoints={
            "changepoints_dict": None,
            "seasonality_changepoints_dict": None
        },
        events={
            "holidays_to_model_separately": "auto",
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        autoregression={
            "autoreg_dict": None
        },
        regressors={
            "regressor_cols": []
        },
        uncertainty={
            "uncertainty_dict": None
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None,
            },
            "feature_sets_enabled": "auto",  # "auto" based on data freq and size
            "max_daily_seas_interaction_order": 5,
            "max_weekly_seas_interaction_order": 2,
            "extra_pred_cols": [],
            "min_admissible_value": None,
            "max_admissible_value": None,
        }
    )








To customize this template, create a ``ModelComponentsParam`` class like above with the parameters you would like to use
to override the defaults, and feed it to the ``model_components_param`` parameter in ``ForecastConfig``. For example


.. code-block:: default
   :lineno-start: 179


    custom_model_components = ModelComponentsParam(
        seasonality={
            "yearly_seasonality": 15
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            }
        }
    )








These two parameters can be put in the
`~greykite.framework.templates.autogen.forecast_config.ForecastConfig` class.
The parameters used by the model will be those in the ``model_components_param_silverkite``
with ``"yearly_seasonality"`` and ``"fit_algorithm_dict"`` overridden by the custom parameters.


.. code-block:: default
   :lineno-start: 197


    forecast_config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        model_components_param=custom_model_components
    )








Detailed explanations for these parameters are in :doc:`/pages/model_components/0100_introduction`. The following paragraphs
briefly summarized what each parameter does.

The ``growth`` parameter recognizes the key ``"growth_term"``, which describes the growth rate of the time series model.
For ``"SILVERKITE"`` template, the value is ``"linear"`` and indicates linear growth.

The ``seasonality`` parameter recognizes the keys ``"yearly_seasonality"``, ``"quarterly_seasonality"``, ``"monthly_seasonality"``,
``"weekly_seasonality"`` and ``"daily_seasonality"``. Their values are the corresponding Fourier series values.
For ``"SILVERKITE"`` template, the values are ``"auto"`` and all orders will be 5.

The ``changepoints`` parameter recognizes the keys ``"changepoints_dict"`` and ``"seasonality_changepoints_dict"``,
which correspond to trend changepoints and seasonality changepoints.
For more details of configuring these two parameters, see `Changepoints <../quickstart/0200_changepoint_detection.html>`_.
For ``"SILVERKITE"`` template, both parameters are ``None``, indicating that neither trend changepoints nor seasonality changepoints
is included.

The ``events`` parameter recognizes the keys ``"holidays_to_model_separately"``, ``"holiday_lookup_countries"``,
``"holiday_pre_num_days"``, ``"holiday_post_num_days"``, ``"holiday_pre_post_num_dict"`` and ``"daily_event_df_dict"``.
More details can be found at `Holidays and Events <../../pages/model_components/0400_events.html#>`_.
For ``"SILVERKITE"`` template, it automatically looks up holidays in a holiday dictionary and model major holidays
plus minus 2 days with separate indicators.

The ``autoregression`` parameter recognizes the key ``"autoreg_dict"``. You can specify lags and aggregated lags through the
dictionary to trigger autoregressive terms. Specify the value as ``"auto"`` to automatically include recommended
autoregressive terms for the data frequency and forecast horizon.
More details can be found at `Autoregression <../../pages/model_components/0800_autoregression.html#>`_.
For ``"SILVERKITE"`` template, autoregression is not included.

The ``regressors`` parameter recognizes the key ``"regressor_cols"``, which takes a list of regressor column names. These regressor columns
have to be included in the training df for both training and forecast periods. For more details about regressors, see
`Regressors <../../pages/model_components/0700_regressors.html#silverkite>`_.
For ``"SILVERKITE"`` template, no regressors are included.

The ``uncertainty`` parameter recognizes the key ``"uncertainty_dict"``, which takes a dictionary to specify how forecast intervals
are calculated. For more details about uncertainty, see `Uncertainty <../../pages/model_components/0900_uncertainty.html#silverkite>`_.
For ``"SILVERKITE"`` template, the default value is ``None``. If ``coverage`` in ``ForecastConfig`` is not None,
the template uses a default setting based on data frequency. We will see how to set ``coverage`` later.

The ``custom`` parameter recognizes specific keys for ``SILVERKITE`` type of templates that correspond to
`~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`. These keys include

    - ``"fit_algorithm_dict"`` takes a dictionary to specify what regression method is used to fit the time series.
      The default is the ridge regression in `sklearn`. For a detailed list of algorithms, see
      `Algorithms <../../pages/model_components/0600_custom.html#fit-algorithm>`_.
    - ``"feature_sets_enabled"`` defines the interaction terms to be included in the model. A list of pre-defined
      interaction terms can be found at `Feature sets <../../pages/model_components/0600_custom.html#interactions>`_.
      The default is ``None``, which automatically finds the proper interaction terms that fit the data frequency.
    - ``"max_daily_seas_interaction_order"`` is the maximum order of Fourier series components in daily seasonality to
      be used in interactions. The default is 5.
    - ``"max_weekly_seas_interaction_order"`` is the maximum order of Fourier series components in daily seasonality to
      be used in interactions. The default is 2.
    - ``"extra_pred_cols"`` defines extra predictor column names. For details, see
      `Extra predictors <../../pages/model_components/0600_custom.html#extra-predictors>`_.
      The default is no extra predictors.
    - ``"min_admissible_value"`` is the minimum admissible value in forecast. All values below this will be clipped at this value.
      The default is None.
    - ``"max_admissible_value"`` is the maximum admissible value in forecast. All values above this will be clipped at this value.
      The default is None.

All default high-level ``SILVERKITE`` templates are defined through this framework.
The ``"SILVERKITE_EMPTY"`` template is an empty template that does not include any component.
If you provide ``ModelComponentsParam`` via ``model_components_param`` with ``"SILVERKITE_EMPTY"``,
the final model parameter to be used will be exactly what you provided through ``ModelComponentsParam``.
It's not like ``"SILVERKITE"``, where the values you do not provide within ``model_components_param`` will
be filled with the defaults in ``"SILVERKITE"``.
If you choose to use the ``"SILVERKITE_EMPTY"`` template but do not provide any ``ModelComponentsParam``
via ``model_components_param``, the model will only fit the intercept term.

Pre-defined Generic High-level ``SILVERKITE`` Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It can happen that you would like to customize the ``ModelComponentsParam`` but are not sure
which values to set for each parameter.
The high-level ``SILVERKITE`` template pre-defines sets of values for different components,
indexed by human-readable language.
This allows you to try sensible options for the components using a directive language.
For example, "setting seasonality to normal and changepoints to light" is specified by
``sk.SEAS.value.NM`` and ``sk.CP.value.LT``.
This option provides rough tuning knobs before fine tuning the exact parameter values.
This type of template name must be initialized through the
`~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`
dataclass.
You can choose a value for each component and assemble them as a template.


.. code-block:: default
   :lineno-start: 287



    from greykite.framework.templates.simple_silverkite_template_config \
        import SimpleSilverkiteTemplateOptions as st
    from greykite.framework.templates.simple_silverkite_template_config \
        import SILVERKITE_COMPONENT_KEYWORDS as sk
    # The model template specifies
    # hourly frequency, normal seasonality (no quarterly or monthly), linear growth, light trend changepoints,
    # separate holidays with plus/minus 2 days, automatic feature sets, ridge regression, automatic autoregression,
    # automatic max daily seasonality interaction order and automatic max weekly seasonality interaction order.
    model_template = st(
        freq=sk.FREQ.value.HOURLY,
        seas=sk.SEAS.value.NM,
        gr=sk.GR.value.LINEAR,
        cp=sk.CP.value.LT,
        hol=sk.HOL.value.SP2,
        feaset=sk.FEASET.value.AUTO,
        algo=sk.ALGO.value.RIDGE,
        ar=sk.AR.value.AUTO,
        dsi=sk.DSI.value.AUTO,
        wsi=sk.WSI.value.AUTO
    )








This option provides rough tuning knobs to intuitively try out different model component parameters.
You can then fine tune the model using ``ModelComponentsParams`` directly.
A complete list of the key-values are

    - ``FREQ``: the data frequency, can be "HOURLY", "DAILY" or "WEEKLY", default "DAILY".
    - ``SEAS``: the seasonality, can be "LT", "NM", "HV", "NONE", "LTQM", "NMQM" or "HVQM", default "LT".
      The "QM" versions include quarterly and monthly seasonality while the others do not.
    - ``GR``: the growth term, can be "LINEAR" or "NONE", default "LINEAR", corresponding to linear growth or constant growth.
    - ``CP``: the automatically detected trend change points, can be "NONE", "LT", "NM", "HV", default "NONE".
    - ``HOL``: the holidays, can be "NONE", "SP1", "SP2", "SP4" or "TG", default "NONE". The default configuration looks up
      popular holidays in a list of popular countries. The "SP{n}" values models major holidays
      with plus/minus n days around them separately, while "TG" models all holidays along with
      plus/minus 2 days together as one indicator.
    - ``FEASET``: the feature sets that defines the interaction terms, can be "AUTO", "ON" or "OFF", default "OFF".
      "AUTO" choose the pre-defined interaction terms automatically, while "ON" and "OFF" includes
      or excludes all pre-defined interaction terms, respectively.
    - ``ALGO``: the algorithm used to fit the model, can be "LINEAR", "RIDGE", "SGD" or "LASSO", default "LINEAR".
      Ridge and Lasso use cross-validation to identify the tuning parameter, while "SGD"
      (stochastic gradient descent) implements L2 norm regularization with tuning parameter 0.001.
    - ``AR``: the autoregressive terms, can be "AUTO" or "OFF", default "OFF".
    - ``DSI``: the maximum daily seasonality order used for interaction in feature sets, can be "AUTO" or "OFF", default "AUTO".
    - ``WSI``: the maximum weekly seasonality order used for interaction in feature sets, can be "AUTO" or "OFF", default "AUTO".

Note that if you do not specify any parameter, the default value will be used:
``FREQ=DAILY``, ``SEAS=LT``, ``GR=LINEAR``, ``CP=NONE``, ``HOL=NONE``, ``FEASET=OFF``, ``ALGO=LINEAR``,
``AR=OFF``, ``DSI=AUTO``, ``WSI=AUTO``.
To see how these keywords are converted to these model component params, see
`~greykite.framework.templates.simple_silverkite_template_config.COMMON_MODELCOMPONENTPARAM_PARAMETERS`.
However, you can print the ``ModelComponentsParam`` class for a model template with the util function
`~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate.get_model_components_from_model_template`.


.. code-block:: default
   :lineno-start: 340


    sst = SimpleSilverkiteTemplate()
    model_components = sst.get_model_components_from_model_template("SILVERKITE_EMPTY")
    print(model_components[0])  # `model_components` is a list of length 1.





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ModelComponentsParam(autoregression={'autoreg_dict': None}, changepoints={'changepoints_dict': None, 'seasonality_changepoints_dict': None}, custom={'feature_sets_enabled': False, 'fit_algorithm_dict': {'fit_algorithm': 'linear', 'fit_algorithm_params': None}, 'max_daily_seas_interaction_order': 0, 'max_weekly_seas_interaction_order': 0, 'extra_pred_cols': [], 'min_admissible_value': None, 'max_admissible_value': None}, events={'holidays_to_model_separately': [], 'holiday_lookup_countries': [], 'holiday_pre_num_days': 0, 'holiday_post_num_days': 0, 'holiday_pre_post_num_dict': None, 'daily_event_df_dict': None}, growth={'growth_term': None}, hyperparameter_override=None, regressors={'regressor_cols': []}, seasonality={'yearly_seasonality': 0, 'quarterly_seasonality': 0, 'monthly_seasonality': 0, 'weekly_seasonality': 0, 'daily_seasonality': 0}, uncertainty={'uncertainty_dict': None})




You can also pass a dataclass.


.. code-block:: default
   :lineno-start: 347


    model_components = sst.get_model_components_from_model_template(model_template)
    print(model_components[0])  # `model_components` is a list of length 1.





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ModelComponentsParam(autoregression={'autoreg_dict': 'auto'}, changepoints={'changepoints_dict': {'method': 'auto', 'resample_freq': 'D', 'regularization_strength': 0.6, 'potential_changepoint_distance': '7D', 'no_changepoint_distance_from_end': '30D', 'yearly_seasonality_order': 15, 'yearly_seasonality_change_freq': None}, 'seasonality_changepoints_dict': None}, custom={'feature_sets_enabled': 'auto', 'fit_algorithm_dict': {'fit_algorithm': 'ridge', 'fit_algorithm_params': None}, 'max_daily_seas_interaction_order': 5, 'max_weekly_seas_interaction_order': 2, 'extra_pred_cols': [], 'min_admissible_value': None, 'max_admissible_value': None}, events={'holidays_to_model_separately': 'auto', 'holiday_lookup_countries': 'auto', 'holiday_pre_num_days': 2, 'holiday_post_num_days': 2, 'holiday_pre_post_num_dict': None, 'daily_event_df_dict': None}, growth={'growth_term': 'linear'}, hyperparameter_override=None, regressors={'regressor_cols': []}, seasonality={'yearly_seasonality': 15, 'quarterly_seasonality': 0, 'monthly_seasonality': 0, 'weekly_seasonality': 4, 'daily_seasonality': 8}, uncertainty={'uncertainty_dict': None})




Provide a List of Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the high-level ``"SILVERKITE"`` templates through the
`~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator` estimator,
you are allowed to provide a list of ``model_template`` or/and a list of ``model_components_param``.
This option allows you to do grid search and compare over different templates/model component overrides
at the same time.

For ``model_template``, you can provide a list of any templates defined above. For example, you can do


.. code-block:: default
   :lineno-start: 361


    model_templates_list = ["SILVERKITE", "SILVERKITE_EMPTY", model_template]








The `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate.get_model_components_from_model_template`
also takes a list as input.


.. code-block:: default
   :lineno-start: 366

    model_components = sst.get_model_components_from_model_template(model_templates_list)
    print(model_components)  # There are 3 elements.





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ModelComponentsParam(autoregression={'autoreg_dict': None}, changepoints={'changepoints_dict': None, 'seasonality_changepoints_dict': None}, custom={'fit_algorithm_dict': {'fit_algorithm': 'ridge', 'fit_algorithm_params': None}, 'feature_sets_enabled': 'auto', 'max_daily_seas_interaction_order': 5, 'max_weekly_seas_interaction_order': 2, 'extra_pred_cols': [], 'min_admissible_value': None, 'max_admissible_value': None}, events={'holidays_to_model_separately': 'auto', 'holiday_lookup_countries': 'auto', 'holiday_pre_num_days': 2, 'holiday_post_num_days': 2, 'holiday_pre_post_num_dict': None, 'daily_event_df_dict': None}, growth={'growth_term': 'linear'}, hyperparameter_override=None, regressors={'regressor_cols': []}, seasonality={'yearly_seasonality': 'auto', 'quarterly_seasonality': 'auto', 'monthly_seasonality': 'auto', 'weekly_seasonality': 'auto', 'daily_seasonality': 'auto'}, uncertainty={'uncertainty_dict': None}), ModelComponentsParam(autoregression={'autoreg_dict': None}, changepoints={'changepoints_dict': None, 'seasonality_changepoints_dict': None}, custom={'feature_sets_enabled': False, 'fit_algorithm_dict': {'fit_algorithm': 'linear', 'fit_algorithm_params': None}, 'max_daily_seas_interaction_order': 0, 'max_weekly_seas_interaction_order': 0, 'extra_pred_cols': [], 'min_admissible_value': None, 'max_admissible_value': None}, events={'holidays_to_model_separately': [], 'holiday_lookup_countries': [], 'holiday_pre_num_days': 0, 'holiday_post_num_days': 0, 'holiday_pre_post_num_dict': None, 'daily_event_df_dict': None}, growth={'growth_term': None}, hyperparameter_override=None, regressors={'regressor_cols': []}, seasonality={'yearly_seasonality': 0, 'quarterly_seasonality': 0, 'monthly_seasonality': 0, 'weekly_seasonality': 0, 'daily_seasonality': 0}, uncertainty={'uncertainty_dict': None}), ModelComponentsParam(autoregression={'autoreg_dict': 'auto'}, changepoints={'changepoints_dict': {'method': 'auto', 'resample_freq': 'D', 'regularization_strength': 0.6, 'potential_changepoint_distance': '7D', 'no_changepoint_distance_from_end': '30D', 'yearly_seasonality_order': 15, 'yearly_seasonality_change_freq': None}, 'seasonality_changepoints_dict': None}, custom={'feature_sets_enabled': 'auto', 'fit_algorithm_dict': {'fit_algorithm': 'ridge', 'fit_algorithm_params': None}, 'max_daily_seas_interaction_order': 5, 'max_weekly_seas_interaction_order': 2, 'extra_pred_cols': [], 'min_admissible_value': None, 'max_admissible_value': None}, events={'holidays_to_model_separately': 'auto', 'holiday_lookup_countries': 'auto', 'holiday_pre_num_days': 2, 'holiday_post_num_days': 2, 'holiday_pre_post_num_dict': None, 'daily_event_df_dict': None}, growth={'growth_term': 'linear'}, hyperparameter_override=None, regressors={'regressor_cols': []}, seasonality={'yearly_seasonality': 15, 'quarterly_seasonality': 0, 'monthly_seasonality': 0, 'weekly_seasonality': 4, 'daily_seasonality': 8}, uncertainty={'uncertainty_dict': None})]




For ``model_components_param``, you can also create a list of ``ModelComponentsParam`` classes to override
the base templates. Each single ``ModelComponentsParam`` is used to override each single base template.
Therefore, if you provide a list of 4 ``ModelComponentsParam`` via ``model_components_param`` and the list
of 3 base templates above via ``model_template``, a total of 12 different sets of model parameters is expected.
However, only unique sets of parameters will be kept.

There are also pre-defined model templates that are defined through lists.
The ``"SILVERKITE_DAILY_90"`` is a pre-tuned model template on daily data with 90 day's forecast horizon.
It is defined through the data class with 4 sets of parameters.
The ``"SILVERKITE_WEEKLY"`` is a pre-tuned model template on weekly data.
It is defined through the data class with 4 sets of parameters.
The ``"SILVERKITE_HOURLY_1"``, ``"SILVERKITE_HOURLY_24"``, ``"SILVERKITE_HOURLY_168"``, ``"SILVERKITE_HOURLY_336"``
are pre-tuned model templates on hourly data with horizons 1 hour, 1 day, 1 week and 2 weeks, respectively.
They are defined through the data class with 4 sets of parameters each.

You are also allowed to put these names in the ``model_template`` list, for example


.. code-block:: default
   :lineno-start: 387


    model_templates_list2 = ["SILVERKITE_DAILY_90", model_template]








This corresponds to 5 single base templates. Whenever you specify multiple sets of parameters
(list of templates, list of model components, etc.), it's best to have a sufficient number
of cross validation folds so that the model does not pick a biased set of parameters.

The Low-level Templates in ``SILVERKITE``
-----------------------------------------

There is a pre-defined low-level template named ``"SK"`` that takes low-level parameters and uses
`~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator`.

The attributes in ``ModelComponentsParam`` are the same as in ``"SILVERKITE"`` but they take different
types of inputs.


.. code-block:: default
   :lineno-start: 403


    model_components_param_sk = ModelComponentsParam(
        growth={
        },  # growth does not accept any parameters, pass growth term via `extra_pred_cols` instead.
        seasonality={
            "fs_components_df": [pd.DataFrame({
                "name": ["tod", "tow", "tom", "toq", "toy"],
                "period": [24.0, 7.0, 1.0, 1.0, 1.0],
                "order": [3, 3, 1, 1, 5],
                "seas_names": ["daily", "weekly", "monthly", "quarterly", "yearly"]})],
        },
        changepoints={
            "changepoints_dict": [None],
            "seasonality_changepoints_dict": [None]
        },
        events={
            "daily_event_df_dict": [None]
        },
        autoregression={
            "autoreg_dict": [None]
        },
        regressors={
            "regressor_cols": [None]
        },
        uncertainty={
            "uncertainty_dict": [None]
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None,
            },
            "extra_pred_cols": ["ct1"],  # linear growth
            "min_admissible_value": [None],
            "max_admissible_value": [None],
        }
    )








The ``growth`` parameter, the dictionary should be empty. The growth term's name is specified
via ``extra_pred_cols`` in ``custom``. The default growth term is ``"ct1"``, which corresponds to linear growth.

The ``seasonality`` parameter, it recognizes the key ``"fs_components_df"``, which is a pandas dataframe
that specifies the fourier series generation information. For more information, see
`~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator`.
For ``"SK"`` template, the default includes daily, weekly, monthly, quarterly and yearly seasonality
with orders 3, 3, 1, 1, 5, respectively.

The ``changepoints`` parameter recognizes the keys ``"changepoints_dict"`` and ``"seasonality_changepoints_dict"``.
Each of the two keys takes a parameter dictionary that corresponds to trend changepoints and seasonality changepoints.
For more details of configuring these two parameters, see `Changepoints <../quickstart/0200_changepoint_detection.html>`_.
For ``"SK"`` template, both parameters are ``None``, indicating that neither trend changepoints nor seasonality changepoints
is included.

The ``events`` parameter recognizes the key ``"daily_event_df_dict"``.
Specify any events or holidays through the "daily_event_df_dict". The usage is the same as this parameter in ``SILVERKITE``.
For ``"SK"`` template, the default is no daily events (holidays).

The ``autoregression`` parameter recognizes the key ``"autoreg_dict"``. You can specify lags and aggregated lags through the
dictionary to trigger autoregressive terms. Specify the value as ``"auto"`` to automatically include the proper order of lags.
For ``"SK"`` template, autoregression is not included.

The ``regressors`` parameter recognizes the key ``"regressor_cols"``, which takes a list of regressor column names. These regressor columns
have to be included in the training df for both training and forecast periods. For more details about regressors, see
`Regressors <../../pages/model_components/0700_regressors.html#silverkite>`_.
For ``"SK"`` template, no regressors are included.

The ``uncertainty`` parameter recognizes the key ``"uncertainty_dict"``, which takes a dictionary to specify how forecast intervals
are calculated. For more details about uncertainty, see `Uncertainty <../../pages/model_components/0900_uncertainty.html#silverkite>`_.
For ``"SK"`` template, the default value is ``None``. If ``coverage`` in ``ForecastConfig`` is not None, it will automatically finds the
most proper conditional residual to compute forecast intervals. We will see how to set ``coverage`` later.

The ``custom`` parameter recognizes specific keys for ``"SK"`` type of template that correspond to
`~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator`. These keys include

    - ``"fit_algorithm_dict"`` takes a dictionary to specify what regression method is used to fit the time series.
      The default is the linear regression in `sklearn`. For a detailed list of algorithms, see
      `Algorithms <../../pages/model_components/0600_custom.html#fit-algorithm>`_.
    - ``"extra_pred_cols"`` defines extra predictor column names. It accepts any valid patsy model formula term. Every column
      name needs to be either generated by `~greykite.common.features.timeseries_features.build_silverkite_features`
      or included in the data df. For details, see
      `Extra predictors <../../pages/model_components/0600_custom.html#extra-predictors>`_.
      The default is ``["ct1"]``, which is the linear growth term.
    - ``"min_admissible_value"`` is the minimum admissible value in forecast. All values below this will be clipped at this value.
      The default is None.
    - ``"max_admissible_value"`` is the maximum admissible value in forecast. All values above this will be clipped at this value.
      The default is None.

A major difference between the high-level and low-level interfaces is that
the lower-level interface does not have pre-defined holidays or feature sets (interaction terms),
and takes more customizable seasonality information. Note that ``"SK"`` is the only low-level
template in ``SILVERKITE`` estimators, and does not support a list of ``model_template`` or
``model_components_param``.

The ``"PROPHET"`` Template
--------------------------

The ``"PROPHET"`` template uses
`~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator`,
which is a wrapper for the `Prophet model <https://facebook.github.io/prophet/docs/quick_start.html>`_.

The attributes in ``ModelComponentsParam`` are the same as in ``"SILVERKITE"`` but they take different
types of inputs.


.. code-block:: default
   :lineno-start: 507


    model_components_param_prophet = ModelComponentsParam(
        growth={
            "growth_term": ["linear"]
        },
        seasonality={
            "seasonality_mode": ["additive"],
            "seasonality_prior_scale": [10.0],
            "yearly_seasonality": ['auto'],
            "weekly_seasonality": ['auto'],
            "daily_seasonality": ['auto'],
            "add_seasonality_dict": [None]
        },
        changepoints={
            "changepoint_prior_scale": [0.05],
            "changepoints": [None],
            "n_changepoints": [25],
            "changepoint_range": [0.8]
        },
        events={
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": [2],
            "holiday_post_num_days": [2],
            "start_year": 2015,
            "end_year": 2030,
            "holidays_prior_scale": [10.0]
        },
        regressors={
            "add_regressor_dict": [None]
        },
        uncertainty={
            "mcmc_samples": [0],
            "uncertainty_samples": [1000]
        }
    )








The ``growth`` parameter recognizes the key ``"growth_term"``, which describes the growth rate of the time series model.
For ``"PROPHET"`` template, the value indicates linear growth.

The ``seasonality`` parameter recognizes the keys ``"seasonality_mode"``, ``"seasonality_prior_scale"``,
``"yearly_seasonality"``, ``"weekly_seasonality"``, ``"daily_seasonality"`` and ``"add_seasonality_dict"``.
For ``"PROPHET"`` template, the seasonality model is "additive" with prior scale 10 and automatic components.

The ``changepoints`` parameter recognizes the keys ``"changepoint_prior_scale"``, ``"changepoints"``, ``"n_changepoints"``
and ``"changepoint_range"``.
The Prophet model supports trend changepoints only.
For ``"PROPHET"`` template, it puts 25 potential trend changepoints uniformly over the first 80%
data and use regularization with prior scale 0.05.

The ``events`` parameter recognizes the keys ``"holiday_lookup_countries"``,
``"holiday_pre_num_days"``, ``"holiday_post_num_days"``, ``"start_year"``, ``"end_year"`` and ``"holidays_prior_scale"``.
The algorithm automatically looks up holidays in ``"holiday_lookup_countries"``.
For ``"PROPHET"`` template, it automatically looks up holidays between 2015 and 2030 with their
plus/minus 2 days. The holiday prior scale is 10.

The Prophet model does not support autoregression, so the ``autoregression`` value should be empty.

The ``regressors`` parameter recognizes the key ``"add_regressor_dict"``.
For more details about regressors, see
`Regressors <../../pages/model_components/0700_regressors.html#prophet>`_.
For ``"PROPHET"`` template, no regressors are included.

The ``uncertainty`` parameter recognizes the key ``"mcmc_samples"`` and ``"uncertainty_samples"``.
For more details about uncertainty, see `Uncertainty <../../pages/model_components/0900_uncertainty.html#prophet>`_.
For ``"PROPHET"`` template, the default value is to sample 1000 uncertainty samples.

The Prophet model does not have any specific value in the ``custom`` parameter.

Extra Notes
-----------
- All templates take the ``hyperparameter_override`` key in their
  ``ModelComponentsParam`` class, which is used to define extra grid search options.
  For details, see `Grid search <../quickstart/0500_grid_search.html>`_.

- To specify a string as a template name, it is recommended to use the
  `~greykite.framework.templates.model_templates.ModelTemplateEnum`
  to avoid typos. For example,


.. code-block:: default
   :lineno-start: 586


    silverkite_template = ModelTemplateEnum.SILVERKITE.name
    silverkite_templates = [
        ModelTemplateEnum.SILVERKITE_EMPTY.name,
        ModelTemplateEnum.SILVERKITE_DAILY_90.name
    ]
    prophet_template = ModelTemplateEnum.PROPHET.name








.. _sphx_glr_download_gallery_tutorials_0200_templates.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: 0200_templates.py <0200_templates.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: 0200_templates.ipynb <0200_templates.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
