"""
Tune your first anomaly detection model
=======================================

This is a basic tutorial for creating and tuning a Greykite AD (Anomaly Detection) model.
It is intended for users who are new to Greykite AD and want to get started quickly.

The Greykite AD is a forecast-based AD method i.e. the forecast is used as the baseline.
A data point is predicted as anomalous if it is outside the forecasted confidence intervals.
The Greykite AD algorithm gives you the flexibility to better model and control the
confidence intervals. A forecast based AD method is inherently dependent on an accurate
forecasting model to achieve satisfactory AD performance.

Throughout this tutorial, we will assume that you are familiar with tuning a
Greykite forecast model. If you are not, please refer to the
:doc:`/gallery/tutorials/0100_forecast_tutorial`.

The anomaly detection config (``ADConfig``) allows the users divide the time series into segments and
learn a different volatility model for each segment. The user can specify the volatility features.
It also allows users to specify objective function, constraints and parameter space to optimize the
confidence intervals.

These features include:

    Volatility Features:
        This allows users to specify the features to segment the time series and learn a
        different volatility model for each segment. For example, if the time series is a daily
        time series, the user can specify the volatility features as ``["dow"]`` to learn a
        different volatility model for each day of the week. The user can also specify multiple
        volatility features. For example, if the time series is a daily time series, the user can
        specify the volatility features as ``[["dow", "is_weekend"]]`` to learn a different
        volatility model for each day of the week and a different volatility model for weekends.

    Coverage Grid:
        This allows users to specify a grid of the confidence intervals. The ``coverage_grid`` is
        specified as a list of floats between 0 and 1. For example, if the ``coverage_grid`` is specified as
        ``[0.5, 0.95]``, the algorithm optimizes over confidence intervals with coverage ``0.5`` and ``0.95``.

    Target Anomaly Percentage:
        This allows users to specify the ``target_anomaly_percent``, which is specified as a float
        between 0 and 1. For example, if ``target_anomaly_percent`` is
        specified as ``0.1``, the anomaly score threshold is optimized such that 10% of the data
        points are predicted as anomalous.

    Target Precision:
        This allows users to specify the ``target_precision``, which is specified as a
        float between 0 and 1. For example, if the ``target_precision`` is specified as ``0.9``, the
        anomaly score threshold is optimized such that at least 90% of the predicted anomalies are true
        anomalies. This is useful when the user has a limited budget to investigate the anomalies.

    Target Recall:
        This allows users to specify the ``target_recall``, which is specified as a float
        between 0 and 1. For example, if the ``target_recall`` is specified as ``0.9``, the anomaly
        score threshold is optimized such that at least 90% of the true anomalies are predicted as
        anomalies. This is useful when the user wants to detect most of the anomalies.

"""

import datetime

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils_anomalies import contaminate_df_with_anomalies
from greykite.common.viz.timeseries_annotate import plot_lines_markers
from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.common.ad_evaluation import precision_score
from greykite.detection.common.ad_evaluation import recall_score
from greykite.detection.detector.ad_utils import partial_return
from greykite.detection.detector.config import ADConfig
from greykite.detection.detector.data import DetectorData
from greykite.detection.detector.greykite import GreykiteDetector
from greykite.detection.detector.reward import Reward
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam

# Evaluation metrics used in the tests.
# F1 score for the True label:
f1 = partial_return(f1_score, True)
# Precision score, for the True label:
precision = partial_return(precision_score, True)
# Recall score for the True label:
recall = partial_return(recall_score, True)

# %%
# Generate a dataset with anomalies
# ----------------------------------
# Let us first generate a dataset with ground truth anomaly labels.
df = generate_df_for_tests(
    freq="D",
    train_start_date=datetime.datetime(2020, 1, 1),
    intercept=50,
    train_frac=0.99,
    periods=200)["df"]

# Specifies anomaly locations.
anomaly_block_list = [
    np.arange(10, 15),
    np.arange(33, 35),
    np.arange(60, 65),
    np.arange(82, 85),
    np.arange(94, 98),
    np.arange(100, 105),
    np.arange(111, 113),
    np.arange(125, 130),
    np.arange(160, 163),
    np.arange(185, 190),
    np.arange(198, 200)]

# Contaminates `df` with anomalies at the specified locations,
# via `anomaly_block_list`.
# If original value is y, the anomalous value is: (1 +/- delta)*y.
df = contaminate_df_with_anomalies(
    df=df,
    anomaly_block_list=anomaly_block_list,
    delta_range_lower=0.25,
    delta_range_upper=0.5,
    value_col=VALUE_COL,
    min_admissible_value=None,
    max_admissible_value=None)

fig = plot_lines_markers(
    df=df,
    x_col=TIME_COL,
    line_cols=["contaminated_y", "y"],
    line_colors=["red", "blue"],
    title="Generation of daily anomalous data")
fig.update_yaxes()
plotly.io.show(fig)

# %%
# The anomalies are generated by adding a random delta to the original value.
# The plot above shows the original data (``y``) in blue and the contaminated data
# (``contaminated_y``) in red. We will drop the original data (``y``) and use the
# contaminated data (``contaminated_y``) as the input to the anomaly detector.

df = df.drop(columns=[VALUE_COL]).rename(
    columns={"contaminated_y": VALUE_COL})
df[ANOMALY_COL] = (df[ANOMALY_COL] == 1)

train_size = int(100)
df_train = df[:train_size].reset_index(drop=True)
df_test = df[train_size:].reset_index(drop=True)


# %%
# Structure of a Greykite AD model
# ---------------------------------
# The Greykite AD takes a ``forecast_config`` and ``ADConfig``
# and builds a detector which uses the forecast as baseline.
# The fit consists of following stages:
#   - Fit a forecast model using the given ``forecast_config``.
#   - Fit a volatility model using the given ``ADConfig``.
#     This builds a `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`
#    model that optimizes over the parameters specified in the ``ADConfig``.

# %%
# Any of the available forecast model
# templates (see :doc:`/pages/stepbystep/0100_choose_model`) work in conjunction
# with the Greykite AD. In this example, we choose the "SILVERKITE_EMPTY" template.

metadata = MetadataParam(
    time_col=TIME_COL,
    value_col=VALUE_COL,
    train_end_date=None,
    anomaly_info=None)

evaluation_period = EvaluationPeriodParam(
    test_horizon=0,
    cv_max_splits=0)

model_components = ModelComponentsParam(
    autoregression={
        "autoreg_dict": {
            "lag_dict": {"orders": [7]},
            "agg_lag_dict": None}},
    events={
        "auto_holiday": False,
        "holiday_lookup_countries": ["US"],
        "holiday_pre_num_days": 2,
        "holiday_post_num_days": 2,
        "daily_event_df_dict": None},
    custom={
        "extra_pred_cols": ["dow"],
        "min_admissible_value": 0,
        "normalize_method": "zero_to_one"})

forecast_config = ForecastConfig(
    model_template="SILVERKITE_EMPTY",
    metadata_param=metadata,
    coverage=None,
    evaluation_period_param=evaluation_period,
    forecast_horizon=1,
    model_components_param=model_components)

# %%
# The Greykite AD algorithm works with or without anomaly labels for training.
# The reward function for the AD algorithm is updated accordingly.
# When no anomaly labels are provided, the AD algorithm uses ``target_anomaly_percent`` to determine
# the anomaly score threshold. If anomaly labels are provided, the AD algorithm uses
# ``precision``, ``recall`` or ``f1`` to determine the anomaly score threshold.

# %%
# Anomaly labels are available
# -----------------------------
# Let us first consider the case where anomaly labels are available for training.
# You can pass the anomaly labels in a few different ways:
#   - As the ``ANOMALY_COL`` column in the training dataframe (``train_data.df``).
#   - As a vector of anomaly labels in the training data (``train_data.y_true``).
#   - As a separate dataframe in the training data (``train_data.anomaly_df``).
#   - As a separate dataframe in the ``metadata_param`` in the ``forecast_config``.
# The detector combines the anomaly labels from all these sources and stores it
# under the ``anomaly_df`` attribute in the ``detector``.

# %%
# In this example, the anomaly labels are passed as ``ANOMALY_COL`` column in the training dataframe.
# When anomalies are available for training, you can use ``precision``, ``recall``,  ``f1`` or a combination
# of these metrics to determine the anomaly score threshold. In this example, we will use ``f1``.

ad_config = ADConfig(
    volatility_features_list=[["dow"], ["is_weekend"]],
    coverage_grid=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
    variance_scaling=True)

def f1_reward(data):
    return f1(
        y_true=data.y_true,
        y_pred=data.y_pred)
reward = Reward(f1_reward)
train_data = DetectorData(df=df_train)

# Initializes the detector.
detector = GreykiteDetector(
    forecast_config=forecast_config,
    ad_config=ad_config,
    reward=reward)
# Fits the model
detector.fit(data=train_data)

# Checks parameter grid.
param_obj_list = detector.fit_info["param_obj_list"]
param_eval_df = pd.DataFrame.from_records(param_obj_list)
param_eval_df["volatility_features"] = param_eval_df["volatility_features"].map(str)
fig = px.line(
    param_eval_df,
    x="coverage",
    y="obj_value",
    color="volatility_features",
    title="'GreykiteDetector' result of parameter search: reward=f1")
plotly.io.show(fig)

# %%
# Plots the training results.
fig = detector.plot(title="'GreykiteDetector' prediction: reward=f1", phase="train")
plotly.io.show(fig)

# %%
# Let us run the model on the test data and plot the results.
# The plot shows the actual data in orange, the forecast in blue, and the
# confidence intervals in grey. The predicted anomalies are marked in red.
test_data = DetectorData(
    df=df_test,
    y_true=df_test[ANOMALY_COL])
test_data = detector.predict(test_data)
fig = detector.plot(title="'GreykiteDetector' prediction: reward=f1")
plotly.io.show(fig)

# %%
# We can see from the plot that our model is able to detect all the anomalies.
# Finally, let's check the evaluation metrics via the ``summary`` method.
# You can see that the model achieved a high precision and recall value.
summary = detector.summary()
print(summary)

# %%
# Examples of other reward functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this section we provide examples of other reward functions that can be used.
# The `~greykite.detection.detector.reward.Reward` class allows users the
# flexibility to specify their own reward functions. This class enables two powerful mechanisms:
#  - taking a simple `reward_func` and construct a penalized version of that
#  - starting from existing objectives building more complex ones by adding /
#  multiplying / dividing them or use same operations with numbers.
#
# These two mechanisms together support robust multi-objective problems.
# Some examples are provided below. All these reward functions can be used as before.

# Builds precision as objective function.
def precision_func(data):
    return precision(
        y_true=data.y_true,
        y_pred=data.y_pred)
precision_obj = Reward(precision_func)

# Builds recall as objective function.
def recall_func(data):
    return recall(
        y_true=data.y_true,
        y_pred=data.y_pred)
recall_obj = Reward(recall_func)

# Builds sum of precision and recall objective function.
additive_obj = precision_obj + recall_obj

# %%
# The class also allows for constrained optimization. For example, in the context
# of anomaly detection if recall is to be optimized
# subject to precision being at least 80 percent, the users can enable this. Let's
# see how this can be done.

# First, let's build a penalized precision objective function that
# penalizes precision values under 0.8 by `penalty == -inf`.
penalized_precision_obj = Reward(
    precision_func,
    min_unpenalized=0.8,
    penalty=-np.inf)

# The constraint can also be passed via the ADConfig.
ad_config = ADConfig(
    target_precision=0.8)

# Builds a combined objective function that optimizes recall
# subject to precision being at least 80 percent.
combined_obj = recall_obj + penalized_precision_obj

# %%
# Users can also combine objectives to achieve more complex objectives from existing ones.
# For example F1 can be easily expressed in terms of precision and recall objectives.
f1_obj = (2 * recall_obj * precision_obj) / (recall_obj + precision_obj)


# %%
# Anomaly labels are *NOT* available
# ---------------------------------
# In this example, we will use an AD config which uses ``target_anomaly_percent`` to
# determine the anomaly score threshold. If not specified, the AD algorithm uses a default
# ``target_anomaly_percent`` of 10%.
ad_config = ADConfig(
    volatility_features_list=[["dow"], ["is_weekend"]],
    coverage_grid=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95, 0.99, 0.999],
    target_anomaly_percent=10.0,
    variance_scaling=True)

detector = GreykiteDetector(
    forecast_config=forecast_config,
    ad_config=ad_config,
    reward=None)
detector.fit(data=train_data)

# Checks parameter grid.
param_obj_list = detector.fit_info["param_obj_list"]
param_eval_df = pd.DataFrame.from_records(param_obj_list)
param_eval_df["volatility_features"] = param_eval_df["volatility_features"].map(str)
fig = px.line(
    param_eval_df,
    x="coverage",
    y="obj_value",
    color="volatility_features",
    title="'GreykiteDetector' result of param search: reward=anomaly_percent")
plotly.io.show(fig)

# %%
# Plots the training results.
fig = detector.plot(title="'GreykiteDetector' prediction: reward=anomaly_percent", phase="train")
plotly.io.show(fig)


# %%
# Let us run the model on the test data and plot the results.
# The plot shows the actual data in orange, the forecast in blue, and the
# confidence intervals in grey. The predicted anomalies are marked in red.

test_data = DetectorData(
    df=df_test,
    y_true=df_test[ANOMALY_COL])
test_data = detector.predict(test_data)
fig = detector.plot(title="'GreykiteDetector' prediction: reward=anomaly_percent")
plotly.io.show(fig)

# %%
# We can see from the plot that our model is able to detect all the anomalies.
# Finally, let's check the evaluation metrics via the ``summary`` method.
# You can see that the model achieved a high precision and recall value.

summary = detector.summary()
print(summary)
