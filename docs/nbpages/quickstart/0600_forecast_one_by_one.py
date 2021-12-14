"""
Forecast One By One
===================

A useful feature for short-term forecast in Silverkite model family is autoregression.
Silverkite has an "auto" option for autoregression,
which automatically selects the autoregression lag orders based on the data frequency and forecast horizons.
One important rule of this "auto" option is that the minimum order of autoregression terms
is at least the forecast horizon.
For example, if the forecast horizon is 3 on a daily model,
the minimum order of autoregression is set to 3.
The "auto" option won't have an order of 2 in this case,
because the 3rd day forecast will need the 1st day's observation,
which isn't available at the current time.
Although the model can make predictions with an autoregression lag order less than the forecast horizon
via simulations, it takes longer time to run and is not the preferred behavior in the "auto" option.

However, in many cases, using smaller autoregression lag orders can give more accurate forecast results.
We observe that the only barrier of using an autoregression term of order 2 in the 3-day forecast model
is the 3rd day, while we can use it freely for the first 2 days.
Similarly, we are able to use an autoregression term of order 1 for the 1st day.
In a 3 day forecast, if the accuracy of all 3 days are important, then replacing the first 2 days' models
with shorter autoregression lag orders can improve the accuracy.
The forecast-one-by-one algorithm is designed in this context.

The observations above together bring the idea of the forecast-one-by-one algorithm.
The algorithm allows fitting multiple models with the "auto" option in autoregression,
when one is forecasting with a forecast horizon longer than 1.
For each model, the "auto" option for autoregression selects the smallest
available autoregression lag order and predicts for the corresponding forecast steps,
thus improving the forecast accuracy for the early steps.

In this example, we will cover how to activate the forecast-one-by-one approach
via the ``ForecastConfig`` and the ``Forecaster`` classes.
For a detailed API reference, please see the
`~greykite.framework.templates.autogen.forecast_config.ForecastConfig` and
`~greykite.sklearn.estimator.one_by_one_estimator.OneByOneEstimator` classes.
"""

import warnings

warnings.filterwarnings("ignore")

import plotly
from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results

# Loads dataset into pandas DataFrame
dl = DataLoader()
df = dl.load_peyton_manning()

# %%
# The forecast-one-by-one option
# ------------------------------
#
# The forecast-one-by-one option is specified through the ``forecast_one_by_one`` parameter
# in ``ForecastConfig``.

config = ForecastConfig(
    model_template=ModelTemplateEnum.SILVERKITE.name,
    forecast_horizon=3,
    model_components_param=ModelComponentsParam(
        autoregression=dict(autoreg_dict="auto")
    ),
    forecast_one_by_one=True
)

# %%
# The ``forecast_one_by_one`` parameter can be specified in the following ways
#
#   - **``True``**: every forecast step will be a separate model.
#     The number of models equals the forecast horizon.
#     In this example, 3 models will be fit with the 3 forecast steps.
#   - **``False``**: the forecast-one-by-one method is turned off.
#     This is the default behavior and a single model is used for all forecast steps.
#   - **A list of integers**: each integer corresponds to a model,
#     and it is the number of steps. For example, in a 7 day forecast,
#     specifying ``forecast_one_by_one=[1, 2, 4]`` will result in 3 models.
#     The first model forecasts the 1st day with forecast horizon 1;
#     The second model forecasts the 2nd - 3rd days with forecast horizon 3;
#     The third model forecasts the 4th - 7th days with forecast horizon 7.
#     In this case, the sum of the list entries must equal the forecast horizon.
#   - **an integer ``n``**: every model will account for n steps. The last model
#     will account for the rest <n steps. For example in a 7 day forecast,
#     specifying ``forecast_one_by_one=2`` will result in 4 models,
#     which is equivalent to ``forecast_one_by_one=[2, 2, 2, 1]``.
#
# .. note::
#   ``forecast_one_by_one`` is activated only when there are parameters in
#   the model that depend on the forecast horizon. Currently the only parameter
#   that depends on forecast horizon is ``autoreg_dict="auto"``. If you do not specify
#   ``autoreg_dict="auto"``, the ``forecast_one_by_one`` parameter will be ignored.
#
# .. note::
#   Forecast-one-by-one fits multiple models to increase accuracy,
#   which may cause the training time to increase linearly with the number of models.
#   Please make sure your ``forecast_one_by_one`` parameter and forecast horizon
#   result in a reasonable number of models.
#
# Next, let's run the model and look at the result.

# Runs the forecast
forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df.iloc[-365:].reset_index(drop=True),  # Uses less data to speed up this example.
    config=config
)

# %%
# You may see a few warnings like "The future x length is 0,
# which doesn't match the model forecast horizon 3,
# using only the model with the longest forecast horizon for prediction."
# This is an expected behavior when calculating the training errors.
# Because the models are mapped to the forecast period only,
# but not to the training period. Therefore, only the last model is used to
# get the fitted values on the training period.
# You don't need to worry about it.
#
# Everything on the ``forecast_result`` level is the same as not activating forecast-one-by-one.
# For example, we can view the cross-validation results in the same way.

# Summarizes the CV results
cv_results = summarize_grid_search_results(
    grid_search=result.grid_search,
    decimals=1,
    # The below saves space in the printed output. Remove to show all available metrics and columns.
    cv_report_metrics=None,
    column_order=["rank", "mean_test", "split_test", "mean_train", "split_train", "mean_fit_time", "mean_score_time", "params"])
cv_results["params"] = cv_results["params"].astype(str)
cv_results.set_index("params", drop=True, inplace=True)
cv_results.transpose()

# %%
# When you need to access estimator level attributes, for example, model summary or component plots,
# the returned result will be a list of the original type, because we fit multiple models.
# The model summary list can be accessed in the same way and you can use index to get the model summary
# for a single model.

# Gets the model summary list
one_by_one_estimator = result.model[-1]
summaries = one_by_one_estimator.summary()
# Prints the model summary for 1st model only
print(summaries[0])

# %%
# We can access the component plots in a similar way.

# Gets the fig list
figs = one_by_one_estimator.plot_components()
# Shows the component plot for 1st model only
plotly.io.show(figs[0])
