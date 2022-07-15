"""
Grid Search
===========

Forecast models have many hyperparameters that could significantly affect
the accuracy. These hyperparameters control different components
in the model including trend, seasonality, events, etc.
You can learn more about how to configure the components or hyperparameters in
the model tuning tutorial (:doc:`/gallery/tutorials/0100_forecast_tutorial`). Here we
will see a step-by-step example of how to utilize the "grid search" functionality
to choose the best set of hyperparameters.

All model templates support grid search.
Here we continue the model tuning tutorial
example to use the ``SILVERKITE`` model on the Peyton Manning data set.
The mechanism of using grid search in ``PROPHET`` is similar.
"""

import warnings

warnings.filterwarnings("ignore")

from greykite.common.data_loader import DataLoader
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results

# Loads dataset into pandas DataFrame
dl = DataLoader()
df = dl.load_peyton_manning()

# %%
# Grid search hyperparameters
# ---------------------------
#
# In :doc:`/gallery/tutorials/0100_forecast_tutorial`
# we learned how the components affect the prediction and how to choose the potential
# candidate components. We also learned how to interpret the cross-validation results
# for one set of hyperparameters. In this section, we will go over the ``grid_search``
# functionality that allows us to compare different sets of hyperparameters by running
# cross-validation on them automatically.
#
# In the `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` class,
# each attribute contains a dictionary mapping parameter names to parameter values. You may
# specify either a specific parameter value to use, or a list of values to explore via grid search.
# Grid search is done over every possible combination of hyperparameters across the lists.
#
# .. note::
#   You may only provide lists for these attributes' parameter values, not for the parameter values
#   of these attributes' parameter values if they are dictionaries.
#   For example, ``seasonality`` is an attribute in ``ModelComponentsParam``,
#   which has parameter names ``yearly_seasonality``, ``quarterly_seasonality``, etc.
#   We can provide lists for the parameter values of these names.
#   On the other hand, ``changepoints`` is an attribute, too,
#   which has parameter names ``changepoints_dict`` and ``seasonality_changepoints_dict``.
#   Both names take dictionaries as their parameter values.
#   We can provide lists of dictionaries as the values, however, within each dictionary,
#   we are not allowed to further wrap parameters in lists.
#
# Cross-validation will be performed over these sets of hyperparameters, and the best set of hyperparameters
# will be selected based on the metric you pick, specified by ``cv_selection_metric`` in
# `~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam`.
#
# Now consider that we want to compare different yearly seasonalities (10 or 20), trend changepoints (None or "auto")
# and fit algorithms (linear or ridge), while keeping all other model components the same. We could specify:

seasonality = {
    "yearly_seasonality": [10, 20],  # yearly seasonality could be 10 or 20
    "quarterly_seasonality": False,
    "monthly_seasonality": False,
    "weekly_seasonality": False,
    "daily_seasonality": False
}

changepoints = {
    # Changepoints could be None or auto.
    "changepoints_dict": [
        None,
        {"method": "auto"}
    ]
}

# Specifies custom parameters
custom = {
    "fit_algorithm_dict": [
        {"fit_algorithm": "ridge"},
        {"fit_algorithm": "linear", "fit_algorithm_params": dict(missing="drop")}
    ]
}

# Specifies the model components
# Could leave the other components as default,
# or specify them in the normal way.
model_components = ModelComponentsParam(
    seasonality=seasonality,
    changepoints=changepoints,
    custom=custom
)

# Specifies the metrics
evaluation_metric = EvaluationMetricParam(
    # The metrics in ``cv_report_metrics`` will be calculated and reported.
    cv_report_metrics=[EvaluationMetricEnum.MeanAbsolutePercentError.name,
                       EvaluationMetricEnum.MeanSquaredError.name],
    # The ``cv_selection_metric`` will be used to select the best set of hyperparameters.
    # It will be added to ``cv_report_metrics`` if it's not there.
    cv_selection_metric=EvaluationMetricEnum.MeanAbsolutePercentError.name
)

# Specifies the forecast configuration.
# You could also specify ``forecast_horizon``, ``metadata_param``, etc.
config = ForecastConfig(
    model_template=ModelTemplateEnum.SILVERKITE.name,
    model_components_param=model_components,
    evaluation_metric_param=evaluation_metric
)

# %%
# For the configuration above, all other model components parameters are the same but yearly seasonality,
# changepoints and fit algorithm have 2 options each. The model will automatically run
# cross-validation over the 8 cases:
#
#   - yearly seasonality = 10, no changepoints, fit algorithm = "linear".
#   - yearly seasonality = 20, no changepoints, fit algorithm = "linear".
#   - yearly seasonality = 10, automatic changepoints, fit algorithm = "linear".
#   - yearly seasonality = 20, automatic changepoints, fit algorithm = "linear".
#   - yearly seasonality = 10, no changepoints, fit algorithm = "ridge".
#   - yearly seasonality = 20, no changepoints, fit algorithm = "ridge".
#   - yearly seasonality = 10, automatic changepoints, fit algorithm = "ridge".
#   - yearly seasonality = 20, automatic changepoints, fit algorithm = "ridge".
#
# The CV test scores will be reported for all 8 cases using the metrics in ``cv_report_metrics``,
# and the final model will be trained on the best set of hyperparameters according to the
# ``cv_selection_metric``.
#
# Selective grid search
# ---------------------
# Consider the case when you have 6 model components to tune, each with 3 different candidates.
# In this case, there will be 3^6=729 different sets of hyperparameters to grid search from.
# The results might be convincing because of the exhaustive grid search, however, the running
# time is going to pile up.
#
# It's very common that not all of the 729 sets of hyperparameters makes sense to us, so it
# would be good not to run all of them. There are two ways to do selective grid search:
#
#   - Setting ``hyperparameter_budget``.
#   - Utilizing ``hyperparameter_override``.
#
# Setting ``hyperparameter_budget``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The ``hyperparameter_budget`` parameter directly controls how many sets of hyperparameters
# will be used in grid search. If this number is less than the number of all possible sets
# of hyperparameters, the algorithm will randomly pick ``hyperparameter_budget`` number of
# hyperparameter sets. Set ``hyperparameter_budget`` to ``-1`` to search all possible sets.
# You may set the budget in the ``ComputationParam`` class. This is a simple way to search a
# large space of hyperparameters if you are not sure which are likely to succeed. After you
# identify parameter values with better performance, you may run a more precise grid search
# to fine tune around these values.
#
# .. note::
#   If you have a small number of timeseries to forecast, we recommend using the
#   model tuning tutorial (:doc:`/gallery/tutorials/0100_forecast_tutorial`)
#   to help identify good parameters candidates. This is likely more effective than
#   random grid search over a large grid.

# Specifies the hyperparameter_budget.
# Randomly picks 3 sets of hyperparameters.
computation = ComputationParam(
    hyperparameter_budget=3
)
# Specifies the forecast configuration.
# You could also specify ``forecast_horizon``, ``metadata_param``, etc.
config = ForecastConfig(
    model_template=ModelTemplateEnum.SILVERKITE.name,
    model_components_param=model_components,
    evaluation_metric_param=evaluation_metric,
    computation_param=computation
)

# %%
# Utilizing ``hyperparameter_override``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The ``hyperparameter_override`` functionality allows us to customize the sets of hyperparameters
# to search within. The way is to specify the ``hyperparameter_override`` parameter in the
# ``ModelComponentsParam`` class.
# First, model components are translated to the parameters in the corresponding sklearn Estimator
# for the template (`~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`
# and `~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator`). The name is usually the same as the
# key, for example, "estimator__yearly_seasonality" and "estimator__fit_algorithm_dict" (the ``ModelComponentsParam``
# attribute is ignored). This creates a default hyperparameter_grid dictionary. Then for each dict in
# ``hyperparameter_override``, the default grid's values are replaced by the override values, producing a
# list of customized grids to search over. Grid search done across all the grids in the list.
# For more details, see
# `hyperparameter override <../../../pages/model_components/1000_override.html#selective-grid-search>`_.
# Now assume we have the following parameter options, as above:
#
#   - yearly seasonality orders: 10 and 20.
#   - trend changepoints: None and "auto".
#   - fit algorithm: linear and ridge.
#
# We do not want to run all 8 sets of hyperparameters. For example, we think that
# ridge is not needed for the model without changepoints because the model is simple, while linear should
# not be used when there are changepoints because the model is complex. So we want:
#
#   - for no changepoints we use linear regression only.
#   - for automatic changepoints we use ridge regression only.
#
# Then we can specify:

seasonality = {
    "yearly_seasonality": [10, 20],
    "quarterly_seasonality": False,
    "monthly_seasonality": False,
    "weekly_seasonality": False,
    "daily_seasonality": False
}

changepoints = {
    "changepoints_dict": None
}

# Specifies custom parameters
custom = {
    "fit_algorithm_dict": {"fit_algorithm": "linear"}
}

# Hyperparameter override can be a list of dictionaries.
# Each dictionary will be one set of hyperparameters.
override = [
    {},
    {
        "estimator__changepoints_dict": {"method": "auto"},
        "estimator__fit_algorithm_dict": {"fit_algorithm": "ridge"}
    }
]

# Specifies the model components
# Could leave the other components as default,
# or specify them in the normal way.
model_components = ModelComponentsParam(
    seasonality=seasonality,
    changepoints=changepoints,
    custom=custom,
    hyperparameter_override=override
)

# Specifies the evaluation period
evaluation_period = EvaluationPeriodParam(
    test_horizon=365,             # leaves 365 days as testing data
    cv_horizon=365,               # each CV test size is 365 days (same as forecast horizon)
    cv_max_splits=3,              # 3 folds CV
    cv_min_train_periods=365 * 4  # uses at least 4 years for training because we have 8 years data
)

config = ForecastConfig(
    forecast_horizon=365,
    model_template=ModelTemplateEnum.SILVERKITE.name,
    model_components_param=model_components,
    evaluation_metric_param=evaluation_metric,
    evaluation_period_param=evaluation_period
)

# %%
# The forecast configuration above specifies the yearly seasonality orders in
# a list, therefore, both 10 and 20 will be searched. For the hyperparameter override
# list, there are two elements. The first one is an empty dictionary, which corresponds
# to the original changepoint and fit algorithm in the configuration. The second dictionary
# overrides changepoint method with automatic changepoint detection and fit algorithm with ridge.
# In total, the model will run 4 different configurations:
#
#   - yearly seasonality 10, no changepoint, fit algorithm linear.
#   - yearly seasonality 20, no changepoint, fit algorithm linear.
#   - yearly seasonality 10, automatic changepoints, fit algorithm ridge.
#   - yearly seasonality 20, automatic changepoints, fit algorithm ridge.
#
# In this way, we could only search the sets of hyperparameters we need and save a lot of time.
# Also note that the above configuration also configures the CV splits using
# `~greykite.framework.templates.autogen.forecast_config.EvaluationPeriodParam`.
# We can see the configs and evaluations with ``summarize_grid_search_results``.

# Runs the forecast
forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=config
)

# Summarizes the CV results
cv_results = summarize_grid_search_results(
    grid_search=result.grid_search,
    decimals=1,
    # The below saves space in the printed output. Remove to show all available metrics and columns.
    cv_report_metrics=None,
    column_order=["rank", "mean_test", "split_test", "mean_train", "split_train", "mean_fit_time", "mean_score_time", "params"])
cv_results["params"] = cv_results["params"].astype(str)
cv_results.set_index("params", drop=True, inplace=True)
cv_results


# %%
# .. tip::
#   The simple silverkite templates that use
#   `~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`
#   are the easiest templates to do grid search, because they support a list of model templates
#   and a list of ``ModelComponentsParam``. For more information, see
#   :doc:`/gallery/templates/0100_template_overview`.
