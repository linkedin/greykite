"""
Reconcile Forecasts
===================

This tutorial explains how use the
`~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
class to create forecasts that satisfy inter-forecast additivity constraints.

The inputs are:

1. additive constraints to be satisfied
2. original (base) forecasts (timeseries)
3. actuals (timeseries)

The output is adjusted forecasts that satisfy the constraints.
"""
# %%
# Optimization Approach
# ---------------------
# The adjusted forecasts are computed as a linear transformation of the base forecasts.
# The linear transform is the solution to an optimization problem
# (:doc:`/pages/miscellaneous/reconcile_forecasts`).
#
# In brief, the objective is to minimize the weighted sum of these error terms:
#
# 1. ``Training MSE``: empirical MSE of the adjusted forecasts on the training set
# 2. ``Bias penalty``: estimated squared bias of adjusted forecast errors
# 3. ``Variance penalty``: estimated variance of adjusted forecast errors for an unbiased
#    transformation, assuming base forecasts are unbiased (this underestimates the variance
#    if the transformation is biased).
# 4. ``Adjustment penalty``: regularization term that penalizes large adjustments
#
# Subject to these constraints:
#
# 1. Adjusted forecasts satisfy inter-forecast additivity constraints (required)
# 2. Transform is unbiased (optional)
# 3. Transform matrix entries are between [lower, upper] bound (optional)
#
# `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
# allows you to tune the optimization objective and constraints.
# It also exposes common methods as special cases of this optimization problem.
# The available methods are:
#
# * ``"bottom_up"`` (bottom up)
# * ``"ols"`` (`OLS <https://robjhyndman.com/papers/Hierarchical6.pdf>`_)
# * ``"mint_sample"`` (`MinT <https://robjhyndman.com/papers/mint.pdf>`_ with sample covariance)
# * ``"custom"`` (custom objective and constraints)
#
# .. note::
#
#   ``"bottom_up"`` is applicable when the constraints can be represented as a tree.
#   It produces reconciled forecasts by summing the leaf nodes. This is equivalent to the
#   solution to the optimization that only penalizes adjustment to the leaf nodes' forecasts.
#
#   ``"ols"`` and ``"mint_sample"`` include only the variance penalty and require
#   that the transform be unbiased. The variance penalty depends on forecast error covariances.
#   ``"ols"`` assumes base forecast errors are uncorrelated with equal variance.
#   ``"mint_sample"`` uses sample covariance of the forecast errors.

# %%
# Prepare Input Data
# ------------------
# In this tutorial, we consider a 3-level tree with the parent-child relationships below.
#
# .. code-block:: none
#
#           00        # level 0
#         /   \
#      10       11    # level 1
#     / | \     /\
#   20 21 22   23 24  # level 2
#
# We want the forecasts of parent nodes to equal the sum of the forecasts of their children.

# %%
# First, we need to generate forecasts for each of the nodes.
# One approach is to generate the forecasts independently, using rolling window
# forecasting to get h-step ahead forecasts over time, for some constant ``h``.
# This can be done with the :doc:`benchmark class </gallery/quickstart/03_benchmark/0200_benchmark>`.
# (The variance penalty assumes the residuals have fixed covariance,
# and using constant ``h`` helps with that assumption.)
#
# For this tutorial, we assume that forecasts have already been computed.
# Below, ``forecasts`` and ``actuals`` are pandas DataFrames in long format, where each column
# is a time series, and each row is a time step. The rows are sorted in ascending order.
import logging
import plotly
import warnings

import pandas as pd
import numpy as np

from greykite.algo.reconcile.convex.reconcile_forecasts import ReconcileAdditiveForecasts
from greykite.common.constants import TIME_COL
from greykite.common.data_loader import DataLoader
from greykite.common.viz.timeseries_plotting import plot_multivariate

logger = logging.getLogger()
logger.setLevel(logging.ERROR)  # reduces logging
warnings.simplefilter("ignore", category=UserWarning)  # ignores matplotlib warnings when rendering documentation

dl = DataLoader()
actuals = dl.load_data(data_name="daily_hierarchical_actuals")
forecasts = dl.load_data(data_name="daily_hierarchical_forecasts")
actuals.set_index(TIME_COL, inplace=True)
forecasts.set_index(TIME_COL, inplace=True)
forecasts.head().round(1)

# %%
# .. note::
#
#   To use the reconcile method, dataframe columns should contain
#   only the forecasts or actuals timeseries. Time should
#   not be its own column.
#
#   Above, we set time as the index using ``.set_index()``.
#   Index values are ignored by the reconcile method
#   so you could also choose to drop the column.

# %%
# The rows and columns in forecasts and actuals correspond to each other.
assert forecasts.index.equals(actuals.index)
assert forecasts.columns.equals(actuals.columns)

# %%
# Next, we need to encode the constraints.
# In general, these can be defined by ``constraint_matrix``.
# This is a ``c x m`` array encoding ``c`` constraints in ``m`` variables,
# where ``m`` is the number of timeseries. The columns in this matrix
# correspond to the columns in the forecasts/actuals dataframes below.
# The rows encode additive expressions that must equal 0.
constraint_matrix = np.array([
   # 00  10  11 20 21 22 23 24
    [-1,  1,  1, 0, 0, 0, 0, 0],  # 0 = -1*x_00 + 1*x_10 + 1*x_11
    [ 0, -1,  0, 1, 1, 1, 0, 0],  # 0 = -1*x_10 + 1*x_20 + 1*x_21 + 1*x_22
    [ 0,  0, -1, 0, 0, 0, 1, 1]   # 0 = -1*x_11 + 1*x_23 + 1*x_24
])

# %%
# Alternatively, if the graph is a tree, you can use the ``levels`` parameter. This
# is a more concise way to specify additive tree # constraints, where forecasts of
# parent nodes must equal the sum of the forecasts of their children. It assumes
# the columns in ``forecasts`` and ``actuals`` are in the tree's breadth first
# traversal order: i.e., starting from the root, scan left to right,
# top to bottom, as shown below for our example:
#
# .. code-block:: none
#
#           0
#        /     \
#       1       2
#     / | \    / \
#    3  4  5  6   7
#
# Here is an equivalent specification using the ``levels`` parameter.

# The root has two children.
# Its children have 3 and 2 children, respectively.
levels = [[2], [3, 2]]
# Summarize non-leaf nodes by the number of children
# they have, and iterate in breadth first traversal order.
# Each level in the tree becomes a sublist of `levels`.
#
#          (2)     --> [2]
#         /   \
#      (3)     (2) --> [3, 2]

# %%
# .. note::
#
#   More formally, ``levels`` specifies the number of children of each
#   internal node in the tree. The ith inner list provides the number
#   of children of each node in level i. Thus, the first sublist has one
#   integer, the length of a sublist is the sum of the previous sublist,
#   and all entries in ``levels`` are positive integers.
#   All leaf nodes must have the same depth.

# %%
# For illustration, we plot the inconsistency between forecasts
# of the root node, ``"00"``, and its children.
# Notice that the blue and orange lines do not perfectly overlap.
parent = "00"
children = ["10", "11"]
cols = {
    f"parent-{parent}": forecasts[parent],
    "sum(children)": sum(forecasts[child] for child in children)
}
cols.update({f"child-{child}": forecasts[child] for child in children})
cols[TIME_COL] = forecasts.index
parent_child_df = pd.DataFrame(cols)
fig = plot_multivariate(
    df=parent_child_df,
    x_col=TIME_COL,
    title=f"Forecasts of node '{parent}' and its children violate the constraint",
)
plotly.io.show(fig)

# %%
# Forecast reconciliation
# -----------------------
#
# Training Evaluation
# ^^^^^^^^^^^^^^^^^^^
# To reconcile these forecasts, we use the
# `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts` class.
raf = ReconcileAdditiveForecasts()

# %%
# Fit
# ~~~
# Call ``fit()`` to learn the linear transform.
# Available methods are ``"bottom_up"``, ``"ols"``,
# ``"mint_sample"``, ``"custom"``.
# Let's start with the bottom up method.
_ = raf.fit(
    forecasts=forecasts,
    actuals=actuals,
    levels=levels,
    method="bottom_up",
)

# %%
# Each row in the transform matrix shows how to compute
# the adjusted forecast as a linear combination of the base forecasts.
# For the "bottom up" transform, the matrix simply reflects the tree structure.
raf.transform_matrix

# %%
# We can visualize this matrix to more easily see how forecasts are combined.
# The top row in this plot shows that the adjusted forecast for
# node "00" (tree root) is the sum of all the base forecasts of the leaf nodes.
# "10" and "11" are the sum of their children, and each leaf node keeps its original value.
fig = raf.plot_transform_matrix()
plotly.io.show(fig)

# %%
# Transform
# ~~~~~~~~~
# The ``transform()`` method applies the transform and returns the adjusted (consistent) forecasts.
# If we call it without arguments, it applies the transform to the training set.
adjusted_forecasts = raf.transform()
adjusted_forecasts.head().round(1)

# %%
# The adjusted forecasts on the training set are stored in the ``adjusted_forecasts`` attribute.
assert adjusted_forecasts.equals(raf.adjusted_forecasts)

# %%
# Evaluate
# ~~~~~~~~
# Now that we have the actuals, forecasts, and adjusted forecasts,
# we can check how the adjustment affects forecast quality.
# Here, we do evaluation on the training set.
_ = raf.evaluate(
    is_train=True,         # evaluates on training set
    ipython_display=True,  # displays evaluation table
    plot=True,             # displays plots
    plot_num_cols=2,       # formats plots into two columns
)

# %%
# For better formatting in this documentation, let's display the
# table again. ``evaluation_df`` contains the
# evaluation table for the training set. The errors for
# the leaf nodes are the same, as expected, because their
# forecasts have not changed.
# The error for nodes "00" and "11" have increased.
raf.evaluation_df.round(1)

# %%
# .. note::
#
#   The ``ipython_display`` parameter controls whether to display the evaluation table.
#
#   - The "\*change" columns show the change in error after adjustment.
#   - The "Base\*" columns show evaluation metrics for the original base forecasts.
#   - The "Adjusted\*" columns show evaluation metrics for the adjusted forecasts.
#   - MAPE/MedAPE = mean/median absolute percentage error,
#     RMSE = root mean squared error, pp = percentage point.

# %%
# We can check the diagnostic plots for more information.
# The "Base vs Adjusted" and "Adjustment Size" plots show that
# the forecasts for "00" and "11" are higher after adjustment.
# The "Forecast Error" plot shows that this increased the forecast error.
# (Plots are automatically shown when ``plot=True``.
# To make plots appear inline in this tutorial, we need
# to explicitly show the figures.)
plotly.io.show(raf.figures["base_adj"])

# %%
plotly.io.show(raf.figures["adj_size"])

# %%
plotly.io.show(raf.figures["error"])

# %%
# .. note::
#
#   The ``plot`` parameter controls whether to display
#   diagnostic plots to adjusted to base forecasts.
#
#   - "Base vs Adjusted Forecast" shows base forecast (blue) vs adjusted forecast (orange)
#   - "Adjustment Size (%)" shows the size of the adjustment.
#   - "Forecast Error (%)" shows the % error before (blue) and after (orange) adjustment.
#     Closer to 0 is better.
#   - Note that the y-axes are independent.


# %%
# For completeness, we can verify that the actuals
# and adjusted forecasts satisfy the constraints.
# ``constraint_violation`` shows constraint violation on the training set,
# defined as root mean squared violation
# (averaged across time points and constraints),
# divided by root mean squared actual value.
# It should be close to 0 for "adjusted" and "actual".
# (This is not necessary to check, because
# a warning is printed during fitting if actuals do not satisfy the constraints
# or if there is no solution to the optimization problem.)
raf.constraint_violation

# %%
# Test Set Evaluation
# ^^^^^^^^^^^^^^^^^^^
# Evaluation on the training set is sufficient for the ``"bottom_up"``
# and ``"ols"`` methods, because they do not use the forecasts or actuals
# to learn the transform matrix. The transform depends only on the constraints.
#
# The ``"mint_sample"`` and ``"custom"`` methods use forecasts and actuals
# in addition to the constraints, so we should evaluate accuracy
# on an out-of-sample test set.
#
# .. csv-table:: Information used by each method
#    :header: "", "constraints", "forecasts", "actuals"
#
#    "``bottom_up``", "X", "", ""
#    "``ols``", "X", "", ""
#    "``mint_sample``", "X", "X", "X"
#    "``custom``", "X", "X", "X"
#
# ``"custom"`` always uses the constraints. Whether it uses forecasts
# and actuals depends on the optimization terms:
#
# - ``forecasts``: used for adjustment penalty, train penalty, variance penalty
#   with "sample" covariance, preset weight options ("MedAPE", "InverseMedAPE").
# - ``actuals``: used for bias penalty, train penalty, variance penalty
#   with "sample" covariance, preset weight options ("MedAPE", "InverseMedAPE").

# %%
# Train
# ~~~~~
# We'll fit to the first half of the data and evaluate accuracy
# on the second half.
train_size = forecasts.shape[0]//2
forecasts_train = forecasts.iloc[:train_size,:]
actuals_train = actuals.iloc[:train_size,:]
forecasts_test = forecasts.iloc[train_size:,:]
actuals_test = actuals.iloc[train_size:,:]

# %%
# Let's try the ``"mint_sample"`` method.
# First, fit the transform and apply it on the training set.
# The transform matrix is more complex than before.
raf = ReconcileAdditiveForecasts()
raf.fit_transform(  # fits and transforms the training data
    forecasts=forecasts_train,
    actuals=actuals_train,
    levels=levels,
    method="mint_sample"
)
assert raf.transform_matrix is not None    # train fit result, set by fit
assert raf.adjusted_forecasts is not None  # train transform result, set by transform
fig = raf.plot_transform_matrix()
plotly.io.show(fig)

# %%
# Now, evaluate accuracy on the training set.
# In our example, all the reconciled forecasts have lower error
# than the base forecasts on the training set.
raf.evaluate(is_train=True)
assert raf.evaluation_df is not None         # train evaluation result, set by evaluate
assert raf.figures is not None               # train evaluation figures, set by evaluate
assert raf.constraint_violation is not None  # train constraint violation, set by evaluate
raf.evaluation_df.round(1)

# %%
# Test
# ~~~~
# Next, apply the transform to the test set and evaluate accuracy.
# Not all forecasts have improved on the test set.
# This demonstrates the importance of test set evaluation.
raf.transform_evaluate(  # transform and evaluates on test data
    forecasts_test=forecasts_test,
    actuals_test=actuals_test,
    ipython_display=False,
    plot=False,
)
assert raf.adjusted_forecasts_test is not None    # test transform result, set by transform
assert raf.evaluation_df_test is not None         # test evaluation result, set by evaluate
assert raf.figures_test is not None               # test evaluation figures, set by evaluate
assert raf.constraint_violation_test is not None  # test constraint violation, set by evaluate
raf.evaluation_df_test.round(1)

# %%
# .. note::
#
#   The results for the test set are in the
#   corresponding attributes ending with ``"_test"``.
#
#   As a summary, here are some key attributes containing the results:
#
#   .. code-block:: none
#
#     transform_matrix :          transform learned from train set
#     adjusted_forecasts :        adjusted forecasts on train set
#     adjusted_forecasts_test :   adjusted forecasts on test set
#     evaluation_df :             evaluation result on train set
#     evaluation_df_test :        evaluation result on test set
#     constraint_violation :      normalized constraint violations on train set
#     constraint_violation_test : normalized constraint violations on test set
#     figures :                   evaluation plots on train set
#     figures_test :              evaluation plots on test set
#
#   For full attribute details, see
#   `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.

# %%
# Model Tuning
# ^^^^^^^^^^^^
# Now that you understand the basic usage, we'll introduce some tuning parameters.
# If you have enough holdout data, you can use the out of sample evaluation to tune the model.
#
# First, try the presets for the ``method`` parameter:
# ``"bottom_up"``, ``"ols"``, ``"mint_sample"``, ``"custom"``.
#
# If you'd like to tune further, use the ``"custom"`` method to tune
# the optimization objective and constraints.
# The tuning parameters and their default values are shown below.
# See `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
# for details.
raf = ReconcileAdditiveForecasts()
_ = raf.fit_transform_evaluate(  # fits, transforms, and evaluates on training data
    forecasts=forecasts_train,
    actuals=actuals_train,
    fit_kwargs=dict(  # additional parameters passed to fit()
        levels=levels,
        method="custom",
        # tuning parameters, with their default values for the custom method
        lower_bound=None,     # Lower bound on each entry of ``transform_matrix``.
        upper_bound=None,     # Upper bound on each entry of ``transform_matrix``.
        unbiased=True,        # Whether the resulting transformation must be unbiased.
        lam_adj=1.0,          # Weight for the adjustment penalty (adj forecast - forecast)
        lam_bias=1.0,         # Weight for the bias penalty (adj actual - actual).
        lam_train=1.0,        # Weight for the training MSE penalty (adj forecast - actual)
        lam_var=1.0,          # Weight for the variance penalty (variance of adjusted forecast errors for an unbiased transformation, assuming base forecasts are unbiased)
        covariance="sample",  # Variance-covariance matrix of base forecast errors, used to compute the variance penalty ("sample", "identity" or numpy array)
        weight_adj=None,      # Weight for the adjustment penalty to put a different weight per-timeseries.
        weight_bias=None,     # Weight for the bias penalty to put a different weight per-timeseries.
        weight_train=None,    # Weight for the train MSE penalty to put a different weight per-timeseries.
        weight_var=None,      # Weight for the variance penalty to put a different weight per-timeseries.
    ),
    evaluate_kwargs=dict()  # additional parameters passed to evaluate()
)

# %%
# Using ``"custom"`` with default settings,
# we find good training set performance overall.
raf.evaluation_df.round(1)

# %%
# Test set performance is also good, except for node "24".
raf.transform_evaluate(
    forecasts_test=forecasts_test,
    actuals_test=actuals_test,
    ipython_display=False,
    plot=False
)
raf.evaluation_df_test.round(1)

# %%
# Notice from the tables that node "24" had the most accurate
# base forecast of all nodes. Therefore, we don't want its adjusted
# forecast to change much. It's possible that the above
# transform was overfitting this node.
#
# We can increase the adjustment penalty for node "24"
# so that its adjusted forecast will be closer to the original one.
# This should allow us to get good forecasts overall and
# for node "24" specifically.

# the order of `weights` corresponds to `forecasts.columns`
weight = np.array([1, 1, 1, 1, 1, 1, 1, 5])  # weight is 5x higher for node "24"
raf = ReconcileAdditiveForecasts()
_ = raf.fit_transform_evaluate(
    forecasts=forecasts_train,
    actuals=actuals_train,
    fit_kwargs=dict(
        levels=levels,
        method="custom",
        lower_bound=None,
        upper_bound=None,
        unbiased=True,
        lam_adj=1.0,
        lam_bias=1.0,
        lam_train=1.0,
        lam_var=1.0,
        covariance="sample",
        weight_adj=weight,    # apply the weights to adjustment penalty
        weight_bias=None,
        weight_train=None,
        weight_var=None,
    )
)

# %%
# .. note::
#
#    The default ``weight=None`` puts equal weight on all nodes.
#    Weight can also be ``"MedAPE"`` (proportional to MedAPE
#    of base forecasts), ``"InverseMedAPE"`` (proportional to 1/MedAPE
#    of base forecasts), or a numpy array that specifies the weight
#    for each node.
#
# .. note::
#
#    When the transform is unbiased (``unbiased=True``),
#    the bias penalty is zero, so ``lam_bias`` and
#    ``weight_bias`` have no effect.

# %%
# The training error looks good.
raf.evaluation_df.round(1)

# %%
# Plots of the transform matrix and adjustment size
# show that node "24"'s adjusted forecast is almost the
# same as its base forecast.
fig = raf.plot_transform_matrix()
plotly.io.show(fig)

# %%
plotly.io.show(raf.figures["adj_size"])

# %%
# The test error looks better than before.

# Transform and evaluate on the test set.
raf.transform_evaluate(
    forecasts_test=forecasts_test,
    actuals_test=actuals_test,
    ipython_display=False,
    plot=True,
    plot_num_cols=2,
)
raf.evaluation_df_test.round(1)

# %%
plotly.io.show(raf.figures_test["base_adj"])

# %%
plotly.io.show(raf.figures_test["adj_size"])

# %%
plotly.io.show(raf.figures_test["error"])


# %%
# Tuning Tips
# -----------
#
# If you have enough data, you can use cross validation with multiple test sets
# for a better estimate of test error. You can use test error to select the parameters.
#
# To tune the parameters,
#
# 1. Try all four methods.
# 2. Tune the lambdas and the weights for the custom method.
#
# For example, start with these lambda settings to see
# which penalties are useful:

lambdas = [
    # lam_adj, lam_bias, lam_train, lam_var
    (0, 0, 0, 1),  # the same as "mint_sample" if other params are set to default values.
    (0, 0, 1, 1),
    (1, 0, 0, 1),
    (1, 0, 1, 1),  # the same as "custom" if other params are set to default values.
    (1, 1, 1, 1),  # try this one with unbiased=False
]

# %%
# Tips:
#
# * ``var`` penalty is usually helpful
# * ``train``, ``adj``, ``bias`` penalties are sometimes helpful
# * You can increase the lambda for penalties that are more helpful.
#
# To try a biased transform, set ``(unbiased=False, lam_bias>0)``.
# Avoid ``(unbiased=False, lam_bias=0)``, because that can result in high bias.
#
# Choose weights that fit your needs. For example, you may care about
# the accuracy of some forecasts more than others.
#
# Setting ``weight_adj`` to ``"InverseMedAPE"`` is a convenient way to
# penalize adjustment to base forecasts that are already accurate.
#
# Setting ``weight_bias``, ``weight_train``, or ``weight_var``
# to ``"MedAPE"`` is a convenient way to improve the error
# on base forecasts that start with high error.

# %%
# Debugging
# ---------
# Some tips if you need to debug:

# %%
# 1. Make sure the constraints are properly encoded
# (for the bottom up method, another way is to check
# the transform matrix).
raf.constraint_matrix

# %%
# 2. The constraint violation should be 0 for the actuals.
raf.constraint_violation
raf.constraint_violation_test

# %%
# 3. Check the transform matrix to understand predictions.
#
# .. code-block::
#
#   fig = raf.plot_transform_matrix()
#   plotly.io.show(fig)

# %%
# 4. For all methods besides "bottom_up", check if a solution was found to the optimization problem.
# If False, then the ``transform_matrix`` may be set to a fallback option (bottom up transform, if available).
# A warning is printed when this happens ("Failed to find a solution. Falling back to bottom-up method.").
raf.is_optimization_solution

# %%
# 5. Check ``prob.status`` for details about cvxpy solver status
# and look for printed warnings for any issues. You can pass solver options
# to the ``fit`` method. See
# `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
# for details.
raf.prob.status

# %%
# 6. Inspect the objective function value at the identified
# solution and its breakdown into components. This shows the terms
# in the objective after multiplication by the lambdas/weights.
raf.objective_fn_val

# %%
# 7. Check objective function weights, to make sure
# covariance, etc., match expectations.
raf.objective_weights

# %%
# 8. Check the convex optimization problem.
print(type(raf.prob))
raf.prob
