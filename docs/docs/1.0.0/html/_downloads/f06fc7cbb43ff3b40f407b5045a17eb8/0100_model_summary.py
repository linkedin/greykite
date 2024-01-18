"""
Model Summary
=============

For every forecast model trained with the Silverkite algorithm,
you can print the model summary with only a few lines of code.
The model summary gives you insight into model performance,
parameter significance, etc.

In this example, we will discuss how to utilize the
`~greykite.algo.common.model_summary.ModelSummary`
module to output model summary.

First we'll load a dataset representing ``log(daily page views)``
on the Wikipedia page for Peyton Manning.
It contains values from 2007-12-10 to 2016-01-20. More dataset info
`here <https://facebook.github.io/prophet/docs/quick_start.html>`_.
"""

import warnings

warnings.filterwarnings("ignore")

from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.forecaster import Forecaster

# Loads dataset into pandas DataFrame
dl = DataLoader()
df = dl.load_peyton_manning()

# %%
# Then we create a forecast model with ``SILVERKITE`` template.
# For a simple example of creating a forecast model, see
# :doc:`/gallery/quickstart/0100_simple_forecast`.
# For a detailed tuning tutorial, see
# :doc:`/gallery/tutorials/0100_forecast_tutorial`.

# Specifies dataset information
metadata = MetadataParam(
    time_col="ts",  # name of the time column
    value_col="y",  # name of the value column
    freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
)

# Specifies model parameters
model_components = ModelComponentsParam(
    changepoints={
        "changepoints_dict": {
            "method": "auto",
            "potential_changepoint_n": 25,
            "regularization_strength": 0.5,
            "resample_freq": "7D",
            "no_changepoint_distance_from_end": "365D"}
    },
    uncertainty={
        "uncertainty_dict": "auto",
    },
    custom={
        "fit_algorithm_dict": {
            "fit_algorithm": "linear",
        },
    }
)

# Runs the forecast
forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=365,  # forecasts 365 steps ahead
        coverage=0.95,  # 95% prediction intervals
        metadata_param=metadata,
        model_components_param=model_components
    )
)

# %%
# Creating model summary
# ^^^^^^^^^^^^^^^^^^^^^^
# Now that we have the output from :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`,
# we are able to access the model summary.

# Initializes the model summary class.
# ``max_colwidth`` is the maximum length of predictor names that can be displayed.
summary = result.model[-1].summary(max_colwidth=30)

# %%
# The above command creates a model summary class and derives extra information
# that summarizes the model. Generally the summarized information includes
# the following sections:
#
#   #. **Model parameter section:** includes basic model parameter information such
#      as number of observations, number of features, model name and etc.
#   #. **Model residual section:** includes the five number summary of training residuals.
#   #. **Model coefficients section (for regression model):** the estimated coefficients
#      and their p-values/confidence intervals. For linear regression, these are the
#      conventional results; for ridge regression, these are calculated from bootstrap [1]_;
#      for lasso regression, these are calculated by multi-sample-splitting [2]_.
#   #. **Model coefficients section (for tree model):** the feature significance.
#   #. **Model significance section (for regression model only):** the overall significance
#      of the regression model, including the coefficient of determination, the
#      F-ratio and its p-value, and model AIC/BIC. The results are based on classical
#      statistical inference and may not be reliable for regularized methods (ridge, lasso, etc.).
#   #. **Warning section:** any warnings for the model summary such as high multicollinearity
#      are displayed in this section.
#
# To see the summary, you can either type ``summary`` or ``print(summary)``.

# Prints the summary
print(summary)

# %%
# The model summary provides useful insights:
#
#   #. We can check the ``sig. code`` column to see which features are not significant.
#      For example, the "Independence Day" events are not significant,
#      therefore we could consider removing them from the model.
#   #. We can check the effect of each feature by examing the confidence interval.
#      For example, the Christmas day has a negative effect of -0.57, with a confidence interval
#      of -0.93 to -0.22. The changepoint at 2010-02-15 changes the slope by -2.52, with a
#      confidence interval of -3.60 to -1.44.
#
# For linear regression, the results are the
# same as the regular regression summary in R (the ``lm`` function).
# The usual considerations apply when interpreting the results:
#
#   #. High feature correlation can increase the coefficient variance.
#      This is common in forecasting problems, so we recommend regularized models.
#   #. There is no standard way to calculate confidence intervals and p-values for regularized
#      linear models (ridge, lasso, elastic_net). We follow the approach in [1]_ for ridge
#      inference and [2]_ for lasso inference.
#      The ideas are to use bootstrap and sample-splitting, respectively.
#
#           - For ridge regression, the confidence intervals and p-values are based on biased estimators.
#             This is a remedy for multicollinearity to produce better forecast, but could lower the true
#             effect of the features.
#           - For lasso regression, the confidence intervals and p-values are based on a multi-sample-split
#             procedure. While this approach of generating CIs is optimized for accuracy, they are calculated
#             independently of the coefficient estimates and are not guaranteed to overlap with the estimates.
#             It's worth noting that the probability of a coefficient being nonzero is also reported in the column ``Prob_nonzero``.
#             This probability can be used to interpret the significance of the corresponding feature.
#
# Moreover, if you would like to explore the numbers behind the printed summary,
# they are stored in the ``info_dict`` attribute, which is a python dictionary.

# Prints the keys of the ``info_dict`` dictionary.
print(summary.info_dict.keys())

# %%

# The above coefficient summary can be accessed as a pandas Dataframe.
print(summary.info_dict["coef_summary_df"])

# %%
# Selected features in a category
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# You may have noticed that there are too many features in the forecast model.
# It's not easy to read all of them in the coefficient summary table.
# The model summary class is able to filter the categories of these features.
# This is done by the
# `~greykite.algo.common.model_summary.ModelSummary.get_coef_summary`
# function.
#
# A few filters are available, including:
#
#   - ``is_intercept``: intercept term.
#   - ``is_time_feature``: features defined in `~greykite.common.features.timeseries_features.build_time_features_df`.
#   - ``is_event``: holidays and events.
#   - ``is_trend``: trend features.
#   - ``is_seasonality``: seasonality features.
#   - ``is_lag``: autoregressive features.
#   - ``is_regressor``: extra regressors provided by user.
#   - ``is_interaction``: interaction terms.
#
# All filters set to ``True`` will be joined with the logical operator ``or``,
# while all filters set to ``False`` will be joined with the logical operator ``and``.
# Simply speaking, set what you want to see to ``True`` and what you don't want to see
# to ``False``.
#
# By default, ``is_interaction`` is set to ``True``, this means as long as one feature in
# an interaction term belongs to a category set to ``True``, the interaction term is included
# in the output. However, if one feature in an interaction term belongs to a category set to
# ``False``, the interaction is excluded from the output.
# To hide interaction terms, set ``is_interaction`` to ``False``.

# Displays intercept, trend features but not seasonality features.
summary.get_coef_summary(
    is_intercept=True,
    is_trend=True,
    is_seasonality=False
)

# %%
# There might be too many featuers for the trend (including interaction terms).
# Let's hide the interaction terms.

# Displays intercept, trend features but not seasonality features.
# Hides interaction terms.
summary.get_coef_summary(
    is_intercept=True,
    is_trend=True,
    is_seasonality=False,
    is_interaction=False
)

# %%
# Now we can see the pure trend features, including the continuous growth term and trend changepoints.
# Each changepoint's name starts with "cp" followed by the time point it happens.
# The estimated coefficients are the changes in slope at the corresponding changepoints.
# We can also see the significance of the changepoints by examining their p-values.
#
# We can also retrieve the filtered dataframe by setting ``return_df`` to ``True``.
# This way you could further explore the coefficients.

output = summary.get_coef_summary(
    is_intercept=True,
    is_trend=True,
    is_seasonality=False,
    is_interaction=False,
    return_df=True  # returns the filtered df
)

# %%
# .. [1] Reference: "An Introduction to Bootstrap", Efron 1993.
# .. [2] Reference: "High-Dimensional Inference: Confidence Intervals, p-Values and R-Software hdi", Dezeure, Buhlmann, Meier and Meinshausen.
