# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Albert Chen
"""Convex optimization approach to forecast reconciliation."""

import functools
import logging
import sys
import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.exceptions import NotFittedError

from greykite.algo.reconcile.hierarchical_relationship import HierarchicalRelationship
from greykite.common.python_utils import reorder_columns


try:
    from IPython.core import display as ICD
except ImportError:
    pass  # ipython is an optional dependency, by default only enabled for development


def get_weight_matrix(weights, n_forecasts, name, weight_auto=None):
    """Returns a diagonal weight matrix with shape (``n_forecasts``, ``n_forecasts``).

    Parameters
    ----------
    weights : `list` [`float`] or "auto" or None
        What weights to use.

            - If a list, returns a diagonal matrix with the list values on the diagonal.
              These values specify the weight for each timeseries.
              In ``ReconcileAdditiveForecasts``, weights are applied to the matrix
              whose rows are reordered to canonical form (transposed output of ``reorder_columns``)
            - If "auto", determined by ``weight_auto``.
            - If None, the identity matrix (equal weights).

    n_forecasts : `int`
        The number of forecasts (shape of the output).
    name : `str`
        The name of the weight matrix.
    weight_auto : `numpy.array` or None
        Weights to use if ``array="auto"``.
        Should be a square matrix with shape (``n_forecasts``, ``n_forecasts``).
        None corresponds to the identity matrix.

    Returns
    -------
    weight_matrix : `numpy.array`
        Weights to apply to the errors before taking the norm.
        Diagonal matrix with shape (``n_forecasts``, ``n_forecasts``)
    """
    if weights is None:
        weight_matrix = np.eye(n_forecasts)
    elif isinstance(weights, str) and weights == "auto":
        if weight_auto is None:
            logging.info(f"'auto' weight for {name} is the identity matrix")
            weight_matrix = np.eye(n_forecasts)
        else:
            logging.info(f"'auto' weight for {name} is {weight_auto}")
            weight_matrix = weight_auto
    else:
        weight_matrix = np.diag(weights)

    if weight_matrix.shape != (n_forecasts, n_forecasts):
        raise ValueError(
            f"Expected square matrix with size {n_forecasts}, but `{name}` has "
            f"weight matrix with shape {weight_matrix.shape}")
    return weight_matrix


def get_fit_params(method=None):
    """Returns parameters for
    `greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts.fit`
    corresponding to recognized hierarchical reconciliation method.

    Parameters
    ----------
    method : `str` or None
        Which reconciliation method to use
        Valid values are "bottom_up", "ols", "mint_sample".

            - "bottom_up"   : Sums leaf nodes. Unbiased transform that uses only the values of the leaf nodes
                              to propagate up the tree. Each node's value is the sum of its
                              corresponding leaf nodes' values (a leaf node corresponds to a node T if it is
                              a leaf node of the subtree with T as its root, i.e. a descendant of T or T itself).
                              See Dangerfield and Morris 1992 "Top-down or bottom-up: Aggregate versus disaggregate
                              extrapolations" for one discussion of this method.
            - "ols"         : OLS estimate proposed by https://robjhyndman.com/papers/Hierarchical6.pdf
                              (Hyndman et al. 2010, "Optimal combination forecasts for hierarchical time series")
                              Also see https://robjhyndman.com/papers/mint.pdf section 2.4.1.
                              (Wickramasuriya et al. 2019 "Optimal forecast reconciliation for
                              hierarchical and grouped time series through trace minimization".)
                              Unbiased transform that minimizes variance of adjusted residuals,
                              using "identity" estimate of original residual variance.
                              Optimal if original forecast errors are uncorrelated with equal variance (unlikely).
                              Depends only on the structure of the hierarchy, not on the data itself.
            - "mint_sample" : Unbiased transform that minimizes variance of adjusted residuals,
                              using "sample" estimate of original residual variance.
                              See Wickramasuriya et al. 2019 section 2.4.4.
                              Depends only on the structure of the hierarchy and forecast error covariances,
                              not on the data itself.

        If None, returns an empty dictionary.

    Returns
    -------
    estimator_fit_params : `dict` [`str`, Any]
        Parameters to use when calling ``ReconcileAdditiveForecasts.fit()``.
    """
    if method is None:
        estimator_fit_params = {}
    elif method == "bottom_up":
        logging.info("Using bottom_up estimator")
        estimator_fit_params = {}  # already passed to ``fit`` via ``method``
    elif method == "ols":
        logging.info("Using ols estimator")
        estimator_fit_params = dict(
            lower_bound=None,
            upper_bound=None,
            unbiased=True,          # unbiased
            lam_adj=0.0,
            lam_bias=0.0,
            lam_coef=0.0,
            lam_train=0.0,
            lam_var=1.0,            # minimizes variance
            covariance="identity",  # assumes equal uncorrelated variance
            weight_adj=None,
            weight_bias=None,
            weight_coef=None,
            weight_train=None,
            weight_var=None,
        )
    elif method == "mint_sample":
        logging.info("Using mint_sample estimator")
        estimator_fit_params = dict(
            lower_bound=None,
            upper_bound=None,
            unbiased=True,        # unbiased
            lam_adj=0.0,
            lam_bias=0.0,
            lam_coef=0.0,
            lam_train=0.0,
            lam_var=1.0,          # minimizes variance
            covariance="sample",  # in-sample variance estimate
            weight_adj=None,
            weight_bias=None,
            weight_coef=None,
            weight_train=None,
            weight_var=None,
        )
    else:
        raise ValueError(f"`method` '{method}' is not recognized. "
                         f"Must be one of 'bottom_up', 'ols', 'mint_sample' or None")
    return estimator_fit_params


def apply_method_defaults(fit_func):
    """Decorator for `greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts.fit`.
    Fetches parameters based on ``method`` and calls ``fit_func`` with the result.

    Parameters
    ----------
    fit_func : `callable`
        Should be `greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts.fit`.

    Returns
    -------
    apply_defaults_and_fit : callable
        Function that overwrites ``fit_params`` with those specified by ``method``
        and calls ``fit_func`` with those params.
    """
    @functools.wraps(fit_func)
    def apply_defaults_and_fit(
            self,
            forecasts,
            actuals,
            **fit_params):
        """Overwrites ``fit_params`` with those specified by ``fit_params["method"]``.
        Calls ``fit_func`` with those params.

        Parameters
        ----------
        self : `ReconcileAdditiveForecasts`
            Self. Decorator should be applied to ``ReconcileAdditiveForecasts.fit()``.
        forecasts : `pandas.DataFrame`
            Forecasted values to pass to ``fit_func``.
        actuals : `pandas.DataFrame`
            Actual values to pass to ``fit_func``.
        fit_params : `dict`
            Original parameters used to call ``fit_func``.
            These are overridden by the ones corresponding to ``method``.

        Returns
        -------
        output : Any
            Same as the output of ``fit_func``.
        """
        method = fit_params.get("method")
        method_fit_params = get_fit_params(method)
        params = dict(fit_params, **method_fit_params)
        return fit_func(
            self,
            forecasts,
            actuals,
            **params)
    return apply_defaults_and_fit


class ReconcileAdditiveForecasts:
    """Reconciles forecasts to satisfy additive constraints.

    Constraints can be encoded by the tree structure via ``levels``.
    In the tree formulation, a parent's value must be the sum of its children's values.

    Or, constraints can be encoded as a matrix via ``constraint_matrix``,
    specifying additive equations that must sum to 0. The constraints need not have a
    tree representation.

    Provides standard methods such as bottom up, ols, MinT. Also supports a custom method
    that minimizes user-specified types of error. The solution is derived by convex optimization.
    If desired, a constraint is added to require the transformation to be unbiased.

    If not using method="ols" or method="bottom_up", which don't depend on the data,
    forecast reconciliation should be trained once per horizon
    (# periods between forecasted date and train_end_date),
    because the optimal adjustment may differ.

    See http://go/raf for details.

    Attributes
    ----------
    forecasts : `pandas.DataFrame`, shape (n, m)
        Original forecasted values, used to train the method.
        Also known as "base" forecasts.
        Long format where each column is a time series.
        and each row is a time step.
        For proper variance estimates for the variance penalty,
        values should be at a fixed-horizon (e.g. always 7-step ahead).
    actuals : `pandas.DataFrame`, shape (n, m)
        Actual values to train the method, corresponding to ``forecasts``.
        Must have the same shape and column names as ``forecasts``.
    constraint_matrix : `numpy.array`, shape (c, m), or None
        Constraints. ``c x m`` array encoding ``c`` constraints of ``m`` variables.
        We require ``constraint_matrix @ transform_matrix = 0``.
        For example, to encode ``-x1 + x2 + x3 == 0 and -x2 + x4 + x5 == 0``::

            constraint_matrix = np.array([
                [-1, 1, 1, 0, 0],
                [0, -1, 0, 1, 1]])

        Entries are typically in [-1, 0, 1], but this is not required.
        Either ``constraint_matrix`` or ``levels`` must be provided.

    levels : `list` [`list` [`int`]] or None
        A simpler way to encode tree constraints. Overrides ``constraint_matrix`` if provided.
        Specifies the number of children of each parent (internal) node in the tree.
        The number of inner lists is the height of the tree. The ith inner list provides the number
        of children of each node at depth i. For example::

            # root node with 3 children
            levels = [[3]]
            # root node with 3 children, who have 2, 3, 3 children respectively
            levels = [[3], [2, 3, 3]]

        All leaf nodes must have the same depth. Thus, the first sublist must have one
        integer, the length of a sublist must equal the sum of the previous sublist,
        and all integers in ``levels`` must be positive.

        Either ``constraint_matrix`` or ``levels`` must be provided.

    order_dict : `dict` [`str`, `float`] or None
        How to order the columns before fitting.
        The key is the column name, the value is its position.
        When ``levels`` is used, map each column name to the
        order of its corresponding node in a BFS traversal of the tree.
        When ``constraint_matrix`` is used, this shuffles the order
        of the columns before the constraints are applied (thus, columns
        in ``constraint_matrix`` refer to the columns after reordering).

        If None, no reordering is done.

    method : `str` or None
        Which method to use.
        If None, uses the parameters specified to formulate the convex optimization problem.
        If a string, valid values are "bottom_up", "ols", "mint_sample". The other parameters are ignored.

            - "bottom_up"   : Sums leaf nodes. Unbiased transform that uses only the values of the leaf nodes
                              to propagate up the tree. Each node's value is the sum of its
                              corresponding leaf nodes' values (a leaf node corresponds to a node T if it is
                              a leaf node of the subtree with T as its root, i.e. a descendant of T or T itself).
                              See Dangerfield and Morris 1992 "Top-down or bottom-up: Aggregate versus disaggregate
                              extrapolations" for one discussion of this method.
            - "ols"         : OLS estimate proposed by https://robjhyndman.com/papers/Hierarchical6.pdf
                              (Hyndman et al. 2010, "Optimal combination forecasts for hierarchical time series")
                              Also see https://robjhyndman.com/papers/mint.pdf section 2.4.1.
                              (Wickramasuriya et al. 2019 "Optimal forecast reconciliation for
                              hierarchical and grouped time series through trace minimization".)
                              Unbiased transform that minimizes variance of adjusted residuals,
                              using "identity" estimate of original residual variance.
                              Optimal if original forecast errors are uncorrelated with equal variance (unlikely).
                              Depends only on the structure of the hierarchy, not on the data itself.
            - "mint_sample" : Unbiased transform that minimizes variance of adjusted residuals,
                              using "sample" estimate of original residual variance.
                              See Wickramasuriya et al. 2019 section 2.4.4.
                              Depends only on the structure of the hierarchy and forecast error covariances,
                              not on the data itself.

    lower_bound : `float` or None
        Lower bound on each entry of ``transform_matrix``.
        If None, no lower bound is applied.
    upper_bound : `float` or None
        Upper bound on each entry of ``transform_matrix``.
        If None, no upper bound is applied.
    unbiased : `bool`
        Whether the resulting transformation must be unbiased.
    lam_adj : `float`
        Weight for the adjustment penalty.
        The adjustment penalty is the mean squared difference
        between adjusted forecasts and base forecasts.
    lam_bias : `float`
        Weight for the bias penalty.
        The bias penalty is the mean squared difference
        between adjusted actuals and actuals.
        The value is 0 for an unbiased transformation,
        actuals satisfy the constraints.
    lam_coef : `float`
        Weight for the coefficient penalty.
        The coefficient penalty measures the deviation of the
        transformation matrix from the identity matrix.
        This penalty is more ad-hoc, and its units are not on
        the same scale as the other. The unweighted penalty tends
        to be large, so prefer smaller values of ``lam_coef`` relative
        to other penalties, if desired.
    lam_train : `float`
        Weight for the training MSE penalty.
        The train MSE penalty measures the mean squared difference
        between adjusted forecasts and actuals.
    lam_var : `float`
        Weight for the variance penalty.
        The variance penalty measures the variance of
        adjusted forecast errors an unbiased transformation.
        It is reported as the average of the variances across timeseries.
        It is based on the variance of the base forecast error
        variance, ``covariance``. For biased transforms,
        this is an underestimate of the true variance.
    covariance : `numpy.array` of shape (m, m), or "sample" or "identity"
        Variance-covariance matrix of base forecast errors. Used to
        compute the variance penalty.

            - If a `numpy.array`, row/column i corresponds to the
              ith column after reordering by ``order_dict``. Should
              be reported on the original scale of the data.
            - If "sample", the sample covariance of residuals.
            - If "identity", the identity matrix.

    weight_adj : `list` [`float`] of length m, or "auto" or None
        Weight for the adjustment penalty that allows a different
        weight per-timeseries.

            - If a list, values specify the weight for each forecast
              after reordering by ``order_dict``.
            - If "auto", proportional to the MedAPE of the forecast.
            - If None, the identity matrix (equal weights).

    weight_bias : `list` [`float`] of length m, or "auto" or None
        Weight for the bias penalty that allows a different
        weight per-timeseries.

            - If a list, values specify the weight for each forecast
              after reordering by ``order_dict``.
            - If "auto", proportional to the MedAPE of the forecast.
            - If None, the identity matrix (equal weights).

    weight_coef : `numpy.array` of shape (m, m), or "auto" or None
        Weight for the coefficient penalty that allows a different
        weight per-timeseries.

            - If a `numpy.array`, values specify the weight for each entry in
              ``transform_matrix - np.eye(m)``.
              Note that rows/cols in ``transform_matrix``
              correspond to the forecasts after reordering by ``order_dict``.
            - If "auto", a matrix of 1s, with 0s on the diagonal.
            - If None, a matrix of 1s.

    weight_train : `list` [`float`] of length m, or None
        Weight for the train MSE penalty that allows a different
        weight per-timeseries.

            - If a list, values specify the weight for each forecast
              after reordering by ``order_dict``.
            - If None, the identity matrix (equal weights).

    weight_var : `list` [`float`] of length m, or None
        Weight for the variance penalty that allows a different
        weight per-timeseries.

            - If a list, values specify the weight for each forecast
              after reordering by ``order_dict``.
            - If None, the identity matrix (equal weights).

    names : `pandas.Index`
        Names of ``forecast`` columns after reordering by ``order_dict``.
    tree : `greykite.algo.reconcile.hierarchical_relationship.HierarchicalRelationship` or None
        If ``levels`` is provided, represents the tree structure encoded by the levels.
        Else None.
    transform_variable : `cvxpy.Variable`, shape (m, m) or None
        Optimization variable to learn the transform matrix.
        None if a rule-based method is used, e.g. ``method == bottom_up``
    transform_matrix : `numpy.array`, shape (m, m)
        Transformation matrix. Same as ``transform_variable.value``, unless
        the solver failed the find a solution, and a backup value is used.
        Adjusted forecasts are computed by applying the transform from the left
        to reordered and transposed ``forecasts``. See ``transform`` in this class.
    prob : `cvxpy.Problem`
        Convex optimization problem.
    is_optimization_solution : `bool`
        Whether ``transform_matrix`` is a solution found by convex optimization solution.
        If False, then ``transform_matrix`` may be set to a backup value (bottom up transform).
        Check ``prob.status`` for more details about solver status.
    objective_fn : `callable`
        Evaluates the objective function for a given transform matrix and dataset.
        Takes ``transform_matrix, forecast_matrix (optional), actual_matrix (optional)``.
        Return value has same format as ``objective_fn_val``.
        If forecast_matrix/actual_matrix are not provided, uses the fitting datasets.
    objective_fn_val : `dict` [`str`, `float`]
        Dictionary containing the objective value, and its components,
        as evaluated on the training set for the identified optimal solution
        from convex optimization.
        Keys are:

            ``"adj"``    : adjustment size
            ``"bias"``   : bias of estimator
            ``"coef"``   : penalty on off-diagonal coefficients
            ``"train"``  : train set MSE
            ``"var"``    : variance of unbiased estimator
            ``"total"``  : sum of the above

    objective_weights : `dict` [`str`, `np.array` of shape (m, m)]
        Weights used in the objective function, derived from
        ``covariance``, ``weight_*``, ``forecasts``, ``actuals``.
        Keys are:

            - weight_adj
            - weight_bias
            - weight_coef
            - weight_train
            - weight_var
            - covariance

    adjusted_forecasts : `pandas.DataFrame`, shape (n, m)
        Adjusted ``forecasts`` that satisfy the constraints.
    constraint_violation : `dict` [`str`, `float`]
        The normalized constraint violations on training set.
        Keys are "actual", "forecast", and "adjusted".
        Root mean squared constraint violation is divided by root mean squared actual value.
    evaluation_df : `pandas.DataFrame`, shape (m, # metrics)
        DataFrame of evaluation results on training set. Rows are timeseries,
        columns are metrics. See ``evaluate`` in this class.
    forecasts_test : `pandas.DataFrame`, shape (q, m)
        Forecasted values to test the method.
        Long format where each column is a time series
        and each row is a time step.
        Must have the same column names as ``forecasts``.
        Can have a different number of rows (observations).
    actuals_test : `pandas.DataFrame`, shape (q, m)
        Actual values to test the method.
        Must have the same shape and column names as ``forecasts_test``.
    adjusted_forecasts_test : `pandas.DataFrame`, shape (q, m)
        Adjusted ``forecasts_test`` that satisfy the constraints.
    constraint_violation_test : `dict` [`str`, `float`]
        The normalized constraint violations on test set.
        Keys are "actual", "forecast", and "adjusted".
        Root mean squared constraint violation is divided by root mean squared actual value
        on test set.
    evaluation_df_test : `pandas.DataFrame`, shape (m, # metrics)
        DataFrame of evaluation results on test set. Rows are timeseries,
        columns are metrics. See ``evaluate()`` in this class.

    Methods
    -------
    fit : callable
        Fits the ``transform_matrix`` from training data.
    transform : callable
        Adjusts a forecast to satisfy additive constraints using the ``transform_matrix``.
    evaluate : callable
        Evaluates the adjustment quality by its impact to MAPE, MedAPE, and RMSE.
    fit_transform : callable
        Fits and transforms the training data.
    fit_transform_evaluate : callable
        Fits, transforms, and evaluates on training data.
    transform_evaluate : calllable
        Transforms and evaluates on a new test set.
    """
    def __init__(self):
        # training data and input params
        self.forecasts = None
        self.actuals = None
        self.constraint_matrix = None  # updated by `fit` if `levels` is provided
        self.levels = None
        self.order_dict = None
        self.method = None
        self.lower_bound = None
        self.upper_bound = None
        self.unbiased = False
        self.lam_adj = 0.0
        self.lam_bias = 0.0
        self.lam_coef = 0.0
        self.lam_train = 0.0
        self.lam_var = 0.0
        self.covariance = None
        self.weight_adj = None
        self.weight_bias = None
        self.weight_coef = None
        self.weight_train = None
        self.weight_var = None

        # set by `fit`
        self.names = None
        self.tree = None
        self.transform_variable = None
        self.transform_matrix = None
        self.prob = None
        self.is_optimization_solution = None
        self.objective_fn = None
        self.objective_fn_val = None
        self.objective_weights = None

        # training data transformed result
        self.adjusted_forecasts = None
        self.constraint_violation = None
        self.evaluation_df = None
        # test data transformed result
        self.forecasts_test = None
        self.actuals_test = None
        self.adjusted_forecasts_test = None
        self.constraint_violation_test = None
        self.evaluation_df_test = None

    def _form_constraints(
            self,
            transform_variable,
            Ya=None):
        """Forms the constraints for convex optimization.
        The following attributes should be set before calling this method:

            - constraint_matrix or tree (required)
            - unbiased                  (optional)
            - lower_bound               (optional)
            - upper_bound               (optional)

        Parameters
        ----------
        transform_variable : `cvxpy.Variable`, shape (m, m)
            Optimization variable to learn the transform matrix.
        Ya : `numpy.array`, shape (m, n) or None
            Actuals in wide format. Rows are timeseries, columns are timestamps.
            Must have the same shape as ``Yf``.

            Created in ``fit`` by reordering the columns of ``self.actuals``
            according to ``self.order_dict``, taking the transpose,
            and scaling to mean 1.0.

        Returns
        -------
        constraints : `list` [`cvxpy.constraints.constraint.Constraint`]
            Constraints for the convex optimization problem.
            Requires additive consistency as specified by ``self.constraint_matrix``.
            If ``self.unbiased``, requires unbiasedness.
        """
        constraints = []
        if self.lower_bound is not None:
            constraints += [transform_variable >= self.lower_bound]
        if self.upper_bound is not None:
            constraints += [transform_variable <= self.upper_bound]

        if self.tree is not None:
            # Uses tree's constraints if `tree` is provided
            constraint_matrix = self.tree.constraint_matrix
        else:
            constraint_matrix = self.constraint_matrix

        # If this holds, then the constraints are satisfied for any Yf, i.e.
        # constraint_matrix @ Y_{adjusted} = constraint_matrix @ transform_variable @ Yf == 0.
        constraints += [constraint_matrix @ transform_variable == 0]

        if self.unbiased:
            if self.tree is None:
                if Ya is None:
                    raise ValueError("`Ya` must be provided if `unbiased` and `tree` is None.")
                # Since we don't know the structure, we use an
                # empirical check for unbiasedness.
                constraints += [transform_variable @ Ya == Ya]
            else:
                # Let there be `ell` leaf forecasts.
                # S is m x ell, summing matrix against leaf nodes. Y_{actual} = S @ Y_{leaf}
                # See sec. 2.1, Wickramasuriya et al. 2019 "Optimal forecast reconciliation
                # for hierarchical and grouped time series through trace minimization".
                # transform_variable T is defined as S @ P. T is unbiased if T @ S = S @ P @ S = S.
                # (i.e. Forecasts that satisfy hierarchical constraints remain consistent)
                sum_matrix = self.tree.sum_matrix
                constraints += [transform_variable @ sum_matrix == sum_matrix]

        if Ya is not None:
            violation_magnitude = np.sqrt(np.mean(np.square(constraint_matrix @ Ya)))
            observation_magnitude = np.sqrt(np.mean(np.square(Ya)))
            relative_error = violation_magnitude / observation_magnitude
            if relative_error > 1e-3:
                warnings.warn(f"Actuals do not satisfy the constraints! Relative error is {relative_error.round(5)}")
        return constraints

    def _form_objective(
            self,
            Yf,
            Ya,
            transform_variable,
            covariance=None):
        """Forms the objective for convex optimization.

        The objective is a sum of five types of errors::

            obj = adj + bias + coef + train + var

        Where

            ``"adj"`` : MSE of adjusted forecasts vs base forecasts (adjustment size)
            ``"bias"`` : MSE of adjusted actuals vs actuals (squared bias)
            ``"coef"`` : deviation of transformation from identity matrix
            ``"train"`` : MSE of adjusted forecasts vs actuals (training error)
            ``"var"`` : estimated variance of adjusted forecast residuals (variance)

        Relative importance of these errors is specified by ``lam_*``.
        Relative importance of forecasts / coefficients is specified by ``weight_*``.

        The following attributes should be set before calling this method:

            - lam_adj       (optional)
            - lam_bias      (optional)
            - lam_coef      (optional)
            - lam_train     (optional)
            - lam_var       (optional)
            - weight_adj    (optional)
            - weight_bias   (optional)
            - weight_coef   (optional)
            - weight_train  (optional)
            - weight_var    (optional)

        Parameters
        ----------
        Yf : `numpy.array`, shape (m, n)
             Forecasts in wide format. Rows are timeseries, columns are timestamps.
             The goal is to adjust these values to satisfy the constraints.

             Created in ``fit`` by reordering the columns of ``self.forecasts``
             according to ``self.order_dict`` and taking the transpose,
             and scaling down by the mean of the actuals.
        Ya : `numpy.array`, shape (m, n)
             Actuals in wide format. Rows are timeseries, columns are timestamps.
             Must have the same shape as ``Yf``.

             Created in ``fit`` by reordering the columns of ``self.actuals``
             according to ``self.order_dict``, taking the transpose,
             and scaling to mean 1.0.
        transform_variable : `cvxpy.Variable`, shape (m, m)
            Optimization variable. Square matrix with size matching the rows in ``Yf``.

        Returns
        -------
        obj : `cvxpy.problems.objective.Minimize`
            Objective function to minimize.

        Updates ``self.objective_weights`` to contain the
        matrices used in the objective.
        """
        n_forecasts = Yf.shape[0]
        if (isinstance(self.weight_adj, str) and self.weight_adj == "auto") or \
                (isinstance(self.weight_bias, str) and self.weight_bias == "auto"):
            # Default weight is the Med-APE for each timeseries
            weight_auto = np.diag(np.nanmedian(np.abs(Ya - Yf) / np.abs(Ya), axis=1))
            # Rescales the weight so it has the same norm as `m x m` identity matrix
            weight_auto = weight_auto * np.linalg.norm(np.eye(n_forecasts)) / np.linalg.norm(weight_auto)
        else:
            weight_auto = None
        weight_adj = get_weight_matrix(weights=self.weight_adj, n_forecasts=n_forecasts, name="weight_adj", weight_auto=weight_auto)
        weight_bias = get_weight_matrix(weights=self.weight_bias, n_forecasts=n_forecasts, name="weight_bias", weight_auto=weight_auto)
        if self.weight_coef is None:
            weight_coef = np.ones((n_forecasts, n_forecasts))
        elif isinstance(self.weight_coef, str) and self.weight_coef == "auto":
            weight_coef = np.ones((n_forecasts, n_forecasts)) - np.eye(n_forecasts)
        else:
            weight_coef = self.weight_coef
        weight_train = get_weight_matrix(weights=self.weight_train, n_forecasts=n_forecasts, name="weight_train")
        weight_var = get_weight_matrix(weights=self.weight_var, n_forecasts=n_forecasts, name="weight_var")

        if isinstance(covariance, str):
            if covariance == "identity":
                covariance = np.eye(n_forecasts)
            elif covariance == "sample":
                # The sample covariance matrix estimate of h-step ahead forecast errors.
                # Assumes the forecast horizon (h) is fixed.
                # Pass `covariance` to `fit` if a better estimate is available.
                residuals = Ya - Yf
                covariance = np.cov(residuals)
            else:
                raise ValueError("`covariance` not recognized. Provide a valid string in ['identity', 'sample'] or a matrix.")

        self.objective_weights = {
            "weight_adj": weight_adj,
            "weight_bias": weight_bias,
            "weight_coef": weight_coef,
            "weight_train": weight_train,
            "weight_var": weight_var,
            "covariance": covariance
        }

        # Mean squared adjustment size (on observed forecasts distribution, weighted by ``weight_adj``)
        err_adj = cp.sum_squares(cp.norm(weight_adj @ (transform_variable @ Yf - Yf), p="fro")) / np.size(Yf)
        # Mean squared bias (on observed actuals distribution, weighted by ``weight_bias``)
        err_bias = cp.sum_squares(cp.norm(weight_bias @ (transform_variable @ Ya - Ya), p="fro")) / np.size(Ya)
        # Mean squared deviation from identity matrix
        err_coef = cp.sum_squares(cp.norm(cp.multiply(weight_coef, transform_variable - np.eye(n_forecasts)), p="fro")) / np.size(weight_coef)
        # Mean squared train error (proxy for bias^2 + variance) (on observed distribution, weighted by ``weight_train``)
        err_train = cp.sum_squares(cp.norm(weight_train @ (transform_variable @ Yf - Ya), p="fro")) / np.size(Yf)
        # Variance of an unbiased estimator.
        # We can't optimize the trace directly, as specified by Wickramasuriya et al. 2019.
        #   err_var = cp.trace(weight_var @ transform_variable @ covariance @ transform_variable.T) / n_forecasts if covariance is not None else 0.0
        #   "Problem does not follow DCP rules. Specifically:"
        #   "The objective is not DCP."
        # Because `covariance` is symmetric and PSD, we can rewrite the trace as a squared Frobenius norm.
        #   tr(T covariance T') = ||T sqrt(covariance)||_{F]^{2}
        # Note that `weight_var` has a different interpretation than when
        # inside the trace. It matches the interpretation of the other weights.
        # Divides by `n_forecasts` for mean adjusted forecast error variance for a timeseries
        #   (the trace is the sum across timeseries).
        err_var = cp.sum_squares(cp.norm(weight_var @ transform_variable @ sqrtm(covariance), p="fro")) / n_forecasts if covariance is not None else 0.0

        lams = [self.lam_adj, self.lam_bias, self.lam_coef, self.lam_train, self.lam_var]
        errs = [err_adj, err_bias, err_coef, err_train, err_var]
        obj = cp.Minimize(cp.scalar_product(lams, errs))

        def objective_fn(transform_matrix, forecast_matrix=Yf, actual_matrix=Ya):
            """Evaluates the objective function for a given transform matrix and dataset.

            Parameters
            ----------
            transform_matrix : `numpy.array`, shape (m, m)
                Applied to forecast and actual matrices from the left to get the transformed result.
            forecast_matrix : `numpy.array`, shape (m, n), optional
                The forecasts in wide array format.
                Must have the same shape as ``actual_matrix``.
                If not provided, uses the forecasts for fitting.
            actual_matrix : `numpy.array`, shape (m, n), optional
                The actuals in wide array format.
                Must have the same number of rows as ``transform_matrix``.
                If not provided, uses the actuals for fitting.

            Returns
            -------
            objective_value : `dict` [`str`, `float`] or None
                Dictionary containing the objective value, and its components.
                Keys are:

                    ``"adj"``    : adjustment size
                    ``"bias"``   : bias of estimator
                    ``"coef"``   : penalty on off-diagonal coefficients
                    ``"train"``  : train set MSE
                    ``"var"``    : variance of unbiased estimator
                    ``"total"``  : sum of the above

                Returns None if ``transform_matrix`` is None.
            """
            if transform_matrix is None:
                return None
            # NB: Frobenius norm is the default norm
            np_err_adj = np.linalg.norm(weight_adj @ (transform_matrix @ forecast_matrix - forecast_matrix))**2 / np.size(forecast_matrix)
            np_err_bias = np.linalg.norm(weight_bias @ (transform_matrix @ actual_matrix - actual_matrix))**2 / np.size(actual_matrix)
            np_err_coef = np.linalg.norm(np.multiply(weight_coef, transform_matrix - np.eye(n_forecasts)))**2 / np.size(weight_coef)
            np_err_train = np.linalg.norm(weight_train @ (transform_matrix @ forecast_matrix - actual_matrix))**2 / np.size(actual_matrix)
            np_err_var = np.linalg.norm(weight_var @ transform_matrix @ sqrtm(covariance))**2 / n_forecasts if covariance is not None else 0.0
            np_errs = [np_err_adj, np_err_bias, np_err_coef, np_err_train, np_err_var]
            return {
                "adj": self.lam_adj * np_err_adj,        # adjustment size
                "bias": self.lam_bias * np_err_bias,     # bias^2 of estimator
                "coef": self.lam_coef * np_err_coef,     # penalty on off-diagonal coefficients
                "train": self.lam_train * np_err_train,  # train error (bias^2 + variance)
                "var": self.lam_var * np_err_var,        # variance of an *unbiased* estimator
                "total": np.dot(lams, np_errs)           # Sum of the above
            }
        self.objective_fn = objective_fn

        return obj

    @apply_method_defaults
    def fit(self,
            forecasts,
            actuals,
            order_dict=None,
            method=None,
            levels=None,
            constraint_matrix=None,
            lower_bound=None,
            upper_bound=None,
            unbiased=False,
            lam_adj=0.0,
            lam_bias=0.0,
            lam_coef=0.0,
            lam_train=0.0,
            lam_var=0.0,
            covariance="sample",
            weight_adj="auto",
            weight_bias="auto",
            weight_coef="auto",
            weight_train=None,
            weight_var=None,
            **solver_kwargs):
        """Fits the ``transform_matrix`` based on input data, constraint, and objective function.

        Sets the attributes between ``forecasts`` and ``objective_weights`` as noted
        in the class description, including ``transform_matrix``, ``transform_variable``,
        ``prob``, ``objective_fn_val``.

        If method != "bottom_up" and there is no solution, gives a warning and
        ``self.is_optimization_solution`` is set to False. Uses "bottom_up" solution
        as fallback approach if ``levels`` is provided.

        Parameters
        ----------
        forecasts : `pandas.DataFrame`, shape (n, m)
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        actuals : `pandas.DataFrame`, shape (n, m)
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        order_dict : `dict` [`str`, `float`] or None, default None
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        method : `str` or None, default None
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
            If provided, the parameters between ``lower_bound`` and ``weight_var`` below are ignored.
        levels : `list` [`list` [`int`]] or None, default None
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        constraint_matrix : `numpy.array`, shape (c, m) or None, default None
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        lower_bound : `float` or None, default None
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        upper_bound : `float` or None, default None
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        unbiased : `bool`, default True
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        lam_adj : `float`, default 0.0
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        lam_bias : `float`, default 0.0
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        lam_coef : `float`, default 0.0
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        lam_train : `float`, default 0.0
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        lam_var : `float`, default 0.0
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        covariance : `numpy.array` of shape (m, m), or "sample" or "identity", default "sample"
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        weight_adj : `list` [`float`] of length m, or "auto" or None, default "auto"
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        weight_bias : `list` [`float`] of length m, or "auto" or None, default "auto"
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        weight_coef : `numpy.array` of shape (m, m), or "auto" or None, default "auto"
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        weight_train : `list` [`float`] of length m, or None, default None
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        weight_var : `list` [`float`] of length m, or None, default None
            See attributes of `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`.
        solver_kwargs : dict
            Specify the CVXPY solver and parameters. E.g. dict(verbose=True).
            See https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options.

        Returns
        -------
        transform_matrix : `numpy.array`, shape (m, m)
            Transformation matrix. Same as ``transform_variable.value``, unless
            the solver failed the find a solution, and a backup value is used.
            Adjusted forecasts are computed by applying the transform from the left
            to reordered and transposed ``forecasts``. See ``transform()`` in this class.
        """
        self.forecasts = reorder_columns(forecasts, order_dict=order_dict)  # Returns a copy
        self.actuals = reorder_columns(actuals, order_dict=order_dict)
        self.order_dict = order_dict
        self.levels = levels
        self.method = method
        self.constraint_matrix = constraint_matrix
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.unbiased = unbiased
        self.covariance = covariance
        self.lam_adj = lam_adj
        self.lam_bias = lam_bias
        self.lam_coef = lam_coef
        self.lam_train = lam_train
        self.lam_var = lam_var
        self.weight_adj = weight_adj
        self.weight_bias = weight_bias
        self.weight_coef = weight_coef
        self.weight_train = weight_train
        self.weight_var = weight_var

        if constraint_matrix is None and levels is None:
            raise ValueError("Either `constraint_matrix` or `levels` must be provided.")
        if constraint_matrix is not None and levels is not None:
            raise ValueError("Only one of `constraint_matrix` and `levels` can be provided.")
        if (method == "bottom_up") and levels is None:
            raise ValueError("Must provide `levels` if `method='bottom_up'`.")
        if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
            raise ValueError(f"`lower_bound` {lower_bound} should not be greater than `upper_bound` {upper_bound}")
        if not unbiased and isinstance(covariance, str) and lam_var != 0.0:
            warnings.warn(f"Variance of residuals is underestimated if the estimator is biased.")

        if levels is not None:
            tree = HierarchicalRelationship(levels)
            self.tree = tree
            self.constraint_matrix = tree.constraint_matrix  # Overwrites ``constraint_matrix`` if ``levels`` is provided

        self.names = self.forecasts.columns
        if self.forecasts.shape != self.actuals.shape:
            raise ValueError(f"Forecast shape {self.forecasts.shape} does not match actuals shape {self.actuals.shape}")
        if self.constraint_matrix.shape[1] != self.forecasts.shape[1]:
            raise ValueError(f"The number of forecasts {self.forecasts.shape[1]} does not match "
                             f"the number of columns in `constraint_matrix` {self.constraint_matrix.shape[1]}. "
                             f"Make sure `levels` or `constraint_matrix` matches the data.")

        if method == "bottom_up":
            # Uses bottom-up method
            self.transform_matrix = self.tree.bottom_up_transform
        else:
            # Solves convex optimization problem to find transformation
            # Selects and reorders columns before matrix operations in constraint and objective
            Yf = np.array(self.forecasts).T  # rows are forecasts (wide-format)
            Ya = np.array(self.actuals).T

            # For numerical stability, scale units so that actuals have mean 1 before fitting.
            # The solution should be scale invariant.
            scale = Ya.mean()
            Yf /= scale
            Ya /= scale
            if covariance is not None and not isinstance(covariance, str):
                covariance = covariance / scale**2

            # Creates a matrix optimization variable.
            m = Yf.shape[0]
            transform_variable = cp.Variable((m, m))  # Linear transform matrix. Y_{adjusted} = transform_variable @ Yf

            # Creates constraints.
            constraints = self._form_constraints(
                transform_variable=transform_variable,
                Ya=Ya)

            # Forms objective.
            obj = self._form_objective(
                Yf=Yf,
                Ya=Ya,
                transform_variable=transform_variable,
                covariance=covariance)

            # Forms and solves problem.
            prob = cp.Problem(obj, constraints)
            try:
                prob.solve(**solver_kwargs)
            except cp.SolverError as e:
                warnings.warn(str(e))

            self.transform_variable = transform_variable
            self.transform_matrix = transform_variable.value
            self.prob = prob
            logging.info(f"Solver status: {prob.status}")
            logging.info(f"Solver optimal value: {prob.value}")
            logging.info(f"Adjustment matrix:\n {self.transform_matrix}")

            if self.transform_matrix is None:
                # No optimization solution is found.
                self.is_optimization_solution = False
                if self.tree is not None:
                    warnings.warn("Failed to find a solution. Falling back to bottom-up method.")
                    self.transform_matrix = self.tree.bottom_up_transform
                else:
                    warnings.warn("Failed to find a solution. Try setting CVXPY solver parameters, changing the "
                                  "constraints, or changing the objective weights")
            else:
                self.is_optimization_solution = True
            self.objective_fn_val = self.objective_fn(self.transform_matrix)
        return self.transform_matrix

    def transform(self, forecasts_test=None):
        """Transforms the provided forecasts using the
        fitted ``self.transform_matrix``.

        Parameters
        ----------
        forecasts_test : `pandas.DataFrame`, shape (r, m) or None
            Forecasted values to transform.
            Must have the same columns as ``self.forecasts``.
            If None, uses ``self.forecasts``.

        Returns
        -------
        adjusted_forecasts : `pandas.DataFrame`, shape (r, m)
            Adjusted forecasts that satisfy additive constraints.
            Columns are reordered according to ``self.order_dict``.

        If ``forecasts`` is None, results are stored to ``self.adjusted_forecasts``.
        Else, results are stored to ``self.adjusted_forecasts_test``, and the
        provided ``forecasts_test`` to ``self.forecasts_test``.
        """
        if self.transform_matrix is None:
            raise NotFittedError("Must call `fit` first.")

        is_train = forecasts_test is None
        if is_train:
            forecasts = self.forecasts
        else:
            forecasts = forecasts_test

        forecasts = reorder_columns(forecasts, order_dict=self.order_dict)  # Returns a copy
        Yf = np.array(forecasts).T  # Converts to wide format for transform
        adjusted_forecasts = pd.DataFrame(
            np.transpose(self.transform_matrix @ Yf),  # Converts back to long format
            columns=forecasts.columns,
            index=forecasts.index)

        if is_train:
            self.adjusted_forecasts = adjusted_forecasts
        else:
            self.forecasts_test = forecasts
            self.adjusted_forecasts_test = adjusted_forecasts

        return adjusted_forecasts

    def evaluate(self, is_train, actuals_test=None, plot=False, ipython_display=False):
        """Evaluates the adjustment quality. Computes the following metrics for each
        of the `m` timeseries:

             "Base MAPE"        : MAPE of base forecasts
             "Base MedAPE"      : MedAPE of base forecasts
             "Base RMSE"        : RMSE of base forecasts
             "Adjusted MAPE"    : MAPE of adjusted forecasts
             "Adjusted MedAPE"  : MedAPE of adjusted forecasts
             "Adjusted RMSE"    : RMSE of adjusted forecasts
             "RMSE % change"    : (Adjusted RMSE) / (Base RMSE) - 1
             "MAPE pp change"   : (Adjusted MAPE) - (Base MAPE)
             "MedAPE pp change" : (Adjusted MedAPE) - (Base MedAPE)

        Must call ``fit`` and ``transform`` before calling this method.

        Parameters
        ----------
        is_train : `bool`
            Whether to evaluate on training set or test set.
            If True, evaluates training adjustment quality.
            Else, evaluates test adjustment quality. In this
            case, ``actuals_test`` must be provided.
        actuals_test :  `pandas.DataFrame`
            Actual values on test set, required if ``is_train==False``.
            Must have the same shape as the forecasts passed to
            ``transform()``, i.e. ``self.forecasts_test.shape``.
        plot : `bool`, default False
            Whether to plot the results. Function is basic.
        ipython_display : `bool`, default False
            Whether to display the evaluation statistics.

        Returns
        -------
        evaluation_result : `dict` [`str`, `dict`, or `pandas.DataFrame`]

            - ``"constraint_violation"`` : `dict` [`str`, `float`]
                The normalized constraint violations.
                Keys are "actual", "forecast", and "adjusted".
                The value is root mean squared constraint violation divided
                by root mean squared actual value. Constraint violation of
                actuals should be close to 0.
            - ``"evaluation_df"`` : `pandas.DataFrame`, shape (m, # metrics)
                Evaluation results. DataFrame with one row for each timeseries,
                and a column for each metric listed above.

        If ``is_train``, results are stored to ``self.constraint_violation``,  ``self.evaluation_df``.
        Otherwise, they are stored to ``self.constraint_violation_test``, ``self.evaluation_df_test``.
        """
        if is_train:
            forecasts = self.forecasts
            actuals = self.actuals
            adjusted_forecasts = self.adjusted_forecasts
        else:
            if actuals_test is None:
                raise ValueError("`actuals_test` must be provided to evaluate on test set")
            self.actuals_test = actuals_test
            forecasts = self.forecasts_test
            actuals = self.actuals_test
            adjusted_forecasts = self.adjusted_forecasts_test

        forecasts = reorder_columns(forecasts, order_dict=self.order_dict)
        actuals = reorder_columns(actuals, order_dict=self.order_dict)
        forecasts_arr = np.array(forecasts)  # rows are dates (long-format), better for plotting
        actuals_arr = np.array(actuals)
        adjusted_forecasts_arr = np.array(adjusted_forecasts)

        # Constraint violation
        observation_magnitude = np.sqrt(np.mean(np.square(actuals_arr)))
        constraint_violation = {
            "adjusted": np.sqrt(np.mean(np.square(self.constraint_matrix @ adjusted_forecasts_arr.T))) / observation_magnitude,
            "forecast": np.sqrt(np.mean(np.square(self.constraint_matrix @ forecasts_arr.T))) / observation_magnitude,
            "actual": np.sqrt(np.mean(np.square(self.constraint_matrix @ actuals_arr.T))) / observation_magnitude,
        }

        # Adjusted forecast and original forecast error
        evaluation_df = pd.DataFrame({
            "Adjusted MAPE": 100 * np.nanmean(np.abs(adjusted_forecasts_arr - actuals_arr) / np.abs(actuals_arr), axis=0),
            "Base MAPE": 100 * np.nanmean(np.abs(forecasts_arr - actuals_arr) / np.abs(actuals_arr), axis=0),
            "Adjusted MedAPE": 100 * np.nanmedian(np.abs(adjusted_forecasts_arr - actuals_arr) / np.abs(actuals_arr), axis=0),
            "Base MedAPE": 100 * np.nanmedian(np.abs(forecasts_arr - actuals_arr) / np.abs(actuals_arr), axis=0),
            "Adjusted RMSE": np.sqrt(np.mean(np.square(actuals_arr - adjusted_forecasts_arr), axis=0)),
            "Base RMSE": np.sqrt(np.mean(np.square(actuals_arr - forecasts_arr), axis=0)),
        }, index=actuals.columns)

        evaluation_df["RMSE % change"] = 100 * (evaluation_df["Adjusted RMSE"] / evaluation_df["Base RMSE"] - 1.0)
        evaluation_df["MAPE pp change"] = evaluation_df["Adjusted MAPE"] - evaluation_df["Base MAPE"]
        evaluation_df["MedAPE pp change"] = evaluation_df["Adjusted MedAPE"] - evaluation_df["Base MedAPE"]
        column_order = [
            "RMSE % change",
            "MAPE pp change",
            "MedAPE pp change",
            "Base MAPE",
            "Base MedAPE",
            "Base RMSE",
            "Adjusted MAPE",
            "Adjusted MedAPE",
            "Adjusted RMSE",
        ]
        evaluation_df = evaluation_df[column_order]

        if ipython_display:
            if "IPython.core" in sys.modules:
                ICD.display(evaluation_df.round(1))
            else:
                print(evaluation_df.round(1))

        if plot:
            # Basic plots for convenience.
            num_cols = forecasts.shape[1]
            height = 6
            # See the adjustments on original scale
            plt.figure(figsize=(height * num_cols, height))
            for i, dim in enumerate(forecasts.columns):
                plt.subplot(int(f"1{num_cols}{i+1}"))
                y1 = np.array(forecasts[dim])
                y2 = np.array(adjusted_forecasts[dim])
                plt.plot(y1, label="Base forecast")
                plt.plot(y2, label="Adjusted forecast")
                plt.xlabel("time index")
                plt.ylabel("value")
                plt.title(dim)
            plt.suptitle("Base vs Adjusted Forecast")
            plt.legend()
            plt.show()

            # Relative difference
            plt.figure(figsize=(height * num_cols, height))
            for i, dim in enumerate(forecasts.columns):
                plt.subplot(int(f"1{num_cols}{i+1}"))
                y1 = np.array(forecasts[dim])
                y2 = np.array(adjusted_forecasts[dim])
                plt.plot(100 * (y2/y1 - 1))
                plt.xlabel("time index")
                plt.ylabel("% difference")
                plt.title(dim)
            plt.suptitle("% difference: Base vs Adjusted Forecast")
            plt.show()

            # Change in % error
            plt.figure(figsize=(height * num_cols, height))
            for i, dim in enumerate(forecasts.columns):
                plt.subplot(int(f"1{num_cols}{i+1}"))
                y1 = np.array(actuals[dim])
                y2 = np.array(forecasts[dim])
                y3 = np.array(adjusted_forecasts[dim])
                plt.plot(y2/y1 - 1, label='Base Forecast')
                plt.plot(y3/y1 - 1, label='Adjusted Forecast')
                plt.xlabel("time index")
                plt.ylabel("% error")
                plt.title(dim)
            plt.suptitle('Forecast Percent Error (vs actual)')
            plt.legend()
            plt.show()

        if is_train:
            self.constraint_violation = constraint_violation
            self.evaluation_df = evaluation_df
        else:
            self.constraint_violation_test = constraint_violation
            self.evaluation_df_test = evaluation_df

        return {
            "constraint_violation": constraint_violation,
            "evaluation_df": evaluation_df
        }

    def fit_transform(self, forecasts, actuals, **fit_kwargs):
        """Fits and transforms training data.

        Parameters
        ----------
        forecasts : `pandas.DataFrame`
            Forecasts to fit the adjustment. See ``fit``.
        actuals : `pandas.DataFrame`
            Actuals to fit the adjustment. See ``fit``.
        fit_kwargs : `dict`, optional
            Additional parameters to pass to ``fit``.

        Returns
        -------
        adjusted_forecasts : `pandas.DataFrame`
            Adjusted forecasts.
        """
        self.fit(forecasts, actuals, **fit_kwargs)
        self.transform()
        return self.adjusted_forecasts

    def fit_transform_evaluate(self, forecasts, actuals, fit_kwargs=None, evaluate_kwargs=None):
        """Fits, transforms, and evaluates on training data.

        Parameters
        ----------
        forecasts : `pandas.DataFrame`
            Forecasts to fit the adjustment. See ``fit``.
        actuals : `pandas.DataFrame`
            Actuals to fit the adjustment. See ``fit``.
        fit_kwargs : `dict`, optional, default None
            Additional parameters to pass to ``fit``.
        evaluate_kwargs : `dict`, optional, default None
            Additional parameters to pass to ``evaluate``.

        Returns
        -------
        evaluation_df : `pandas.DataFrame`
            Evaluation results on provided ``forecasts``.
        """
        if fit_kwargs is None:
            fit_kwargs = {}
        if evaluate_kwargs is None:
            evaluate_kwargs = {}
        self.fit(forecasts, actuals, **fit_kwargs)
        self.transform()
        self.evaluate(is_train=True, **evaluate_kwargs)
        return self.evaluation_df

    def transform_evaluate(self, forecasts_test, actuals_test, **evaluate_kwargs):
        """Transforms and evaluates on test data.

        Must call ``fit`` before calling this method.

        forecasts_test : `pandas.DataFrame`
            Forecasts to make consistent. Should be different from the training data.
        actuals_test : `pandas.DataFrame`
            Actuals to check quality of the adjustment.
        evaluate_kwargs : `dict`, optional, default None
            Additional parameters to pass to ``evaluate``.

        Returns
        -------
        evaluation_df : `pandas.DataFrame`
            Evaluation results on provided ``forecasts_test``.
        """
        self.transform(forecasts_test)
        self.evaluate(is_train=False, actuals_test=actuals_test, **evaluate_kwargs)
        return self.evaluation_df
