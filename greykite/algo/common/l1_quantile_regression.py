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
# original author: Kaixu Yang
"""Implements Quantile Regression, Weighted Quantile Regression,
L1-norm Regularized Quantile Regression,
Adaptive L1-norm Regularized Quantile Regression.
"""

from __future__ import annotations

from typing import Dict
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from greykite.algo.common.partial_regularize_regression import constant_col_finder
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message


def ordinary_quantile_regression(
        x: np.array,
        y: np.array,
        q: float,
        sample_weight: Optional[np.array] = None,
        max_iter: int = 200,
        tol: float = 1e-2) -> np.array:
    """Implements the quantile regression without penalty terms.
    Supports sample weight.

    Applies the iterative re-weighted least square (IRLS) algorithm
    to solve the quantile regression without regularization. Minimizes

        beta = argmin 1/n * sum {w_i * [(1 - q) * (y_i - x_i^T beta)_- + q * (y_i - x_i^T beta)_+]}

    This works for singular matrix and supports sample weights, while statsmodels does not.
    Please make sure the design matrix has an intercept as intercept is very important to quantile regression.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    q : `float` between 0.0 and 1.0
        The quantile of the response to model.
    sample_weight : `numpy.array` or None, default None
        The sample weight in the loss function.
    max_iter : int, default 200
        The maximum number of iterations.
    tol : float, default 1e-2
        The tolerance to stop the algorithm.

    Returns
    -------
    beta : numpy.array
        The estimated coefficients.
    """
    # Uses IRLS to solve the problem.
    n, p = x.shape
    if sample_weight is None:
        sample_weight = np.ones(n)
    # Minimum constant for denominator to prevent from dividing by zero.
    delta = 1e-6
    beta = np.zeros(p)
    for i in range(max_iter):
        eta = x @ beta
        eps = np.abs(y - eta)
        # Re-weights.
        left = np.where(y < eta, q, 1 - q) / sample_weight
        right = np.where(eps > delta, eps, delta)
        s = (left * right).reshape(-1, 1)
        # Solves with formula.
        # Multiplies weights to x first to speed up.
        x_l = x / s
        beta_new = np.linalg.pinv(x_l.T @ x) @ (x_l.T @ y)
        err = (np.abs(beta_new - beta) / np.abs(beta)).max()
        beta = beta_new
        if err < tol:
            break
        if i == max_iter - 1:
            log_message(
                message=f"Max number of iterations reached. "
                        f"Deviation is {err}. "
                        f"Consider increasing max_iter.",
                level=LoggingLevelEnum.WARNING
            )

    return beta


def l1_quantile_regression(
        x: np.array,
        y: np.array,
        q: float,
        alpha: float,
        sample_weight: Optional[np.array] = None,
        feature_weight: Optional[np.array] = None,
        include_intercept: bool = True) -> Dict[str, any]:
    """Implements the quantile regression with penalty terms.
    Supports sample weight and feature weights in penalty.

    This is solved with linear programming formulation:

        min c^Tx s.t. ax<=b

    where x is [beta+, beta-, alpha+, alpha-, (y-alpha-X beta)+, (y-alpha-X beta)-].
    Please make sure the design matrix is normalized.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    q : `float` between 0.0 and 1.0
        The quantile of the response to model.
    alpha : `float`
        The regularization parameter.
    sample_weight : `numpy.array` or None, default None
        The sample weight in the loss function.
    feature_weight : `numpy.array` or None, default None
        The feature weight in the penalty term.
        This parameter enables adaptive L1 norm regularization.
    include_intercept : `bool`, default True
        Whether to include intercept.
        If True, will fit an intercept separately and drop any degenerate columns.
        False is not recommended because intercept is very useful in distinguishing the quantiles
        (it shifts the predictions up and down).

    Returns
    -------
    coefs : `dict` [`str`, any]
        The coefficients dictionary with the following elements:
            intercept : `float`
                The intercept.
            beta : `numpy.array`
                The estimated coefficients.

    """
    n, p = x.shape
    if sample_weight is None:
        sample_weight = np.ones(n)
    if feature_weight is None:
        feature_weight = np.ones(p)
    # Linear coefficients.
    c = [alpha * feature_weight, alpha * feature_weight]
    if include_intercept:
        c += [np.zeros(2)]
    c += [q * sample_weight, (1 - q) * sample_weight]
    c = np.concatenate(c, axis=0)
    # Variable.
    var_size = 2*p + 2*n + 2*include_intercept
    var = cp.Variable(var_size)
    # Equality constraint coefficients.
    x_vec = [x, -x]
    if include_intercept:
        x_vec += [np.ones([n, 1]), -np.ones([n, 1])]
    x_vec += [np.eye(n), -np.eye(n)]
    x_vec = np.concatenate(x_vec, axis=1)
    # The problem.
    prob = cp.Problem(cp.Minimize(c.T @ var),
                      [var >= 0, x_vec @ var == y])
    prob.solve()

    # Gets the results.
    if include_intercept:
        intercept = var.value[2 * p] - var.value[2 * p + 1]
    else:
        intercept = 0
    coef = var.value[:p] - var.value[p: 2 * p]
    return {
        "intercept": intercept,
        "coef": coef
    }


class QuantileRegression(RegressorMixin, BaseEstimator):
    """Implements the quantile regression model.

    Supports weighted sample, l1 regularization
    and weighted l1 regularization.
    These options can be configured to support different use cases.
    For example, specifying quantile to be 0.5 and sample weight to be the
    inverse absolute value of response minimizes the MAPE.
    """

    def __init__(
            self,
            quantile: float = 0.9,
            alpha: float = 0.001,
            sample_weight: Optional[np.typing.ArrayLike] = None,
            feature_weight: Optional[np.typing.ArrayLike] = None,
            max_iter: int = 100,
            tol: float = 1e-2,
            fit_intercept: bool = True):
        self.quantile = quantile
        self.alpha = alpha
        self.sample_weight = sample_weight
        self.feature_weight = feature_weight
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

        # Parameters, set by ``fit`` method.
        self.n = None
        self.p = None
        self.constant_cols = None  # Detected degenerate columns.
        self.nonconstant_cols = None  # Detected non-degenerate columns.
        self.intercept_ = None
        self.coef_ = None

    def _process_input(
            self,
            x: np.typing.ArrayLike,
            y: np.typing.ArrayLike) -> [np.array, np.array]:
        """Checks validity of input.

        Parameters
        ----------
        x : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The design matrix.
        y : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The response vector.

        Returns
        -------
        x : `numpy.array`
            The processed x.
        y : `numpy.array`
            The processed y.
        """
        # Type conversion and value checks.
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if len(y.shape) > 1:
            log_message(
                message=f"A column-vector y was passed when a 1d array was expected. "
                        f"Please change the shape of y to (n_samples,), "
                        f"for example using ravel().",
                level=LoggingLevelEnum.WARNING
            )
            y = y.ravel()
        self.n, self.p = x.shape
        if y.shape[0] != self.n:
            raise ValueError(f"The shapes of x and y do not match. "
                             f"x has {self.n} observations while y has "
                             f"{y.shape[0]} observations.")
        if self.quantile < 0 or self.quantile > 1:
            raise ValueError(f"Quantile q must be between 0.0 and 1.0, found {self.quantile}.")
        if self.sample_weight is None:
            self.sample_weight = np.ones(self.n)
        else:
            self.sample_weight = np.array(self.sample_weight).ravel()
            if self.sample_weight.shape[0] != self.n:
                raise ValueError(f"The length of sample weight must match the number of observations"
                                 f" {self.n}, but found {self.sample_weight.shape[0]}.")
        if self.feature_weight is None:
            self.feature_weight = np.ones(self.p)
        else:
            self.feature_weight = np.array(self.feature_weight).ravel()
            if self.feature_weight.shape[0] != self.p:
                raise ValueError(f"The length of feature weight must match the number of features"
                                 f" {self.p}, but found {self.feature_weight.shape[0]}.")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be a positive integer, found {self.max_iter}.")
        if self.tol <= 0:
            raise ValueError(f"tol must be a positive number, found {self.tol}.")

        # Handles intercept and degenerate columns.
        # If ``self.fit_intercept`` is True,
        # the degenerate columns are dropped.
        # It will append an intercept column for the alpha = 0 case.
        # For the alpha > 0 case, the intercept is fitted separately.
        if self.fit_intercept:
            self.constant_cols = constant_col_finder(x)
            self.nonconstant_cols = [i for i in range(self.p) if i not in self.constant_cols]
            if self.alpha == 0:
                x = np.concatenate([np.ones([self.n, 1]), x[:, self.nonconstant_cols]], axis=1)
        return x, y

    def fit(
            self,
            X: np.typing.ArrayLike,
            y: np.typing.ArrayLike) -> QuantileRegression:
        """Fits the quantile regression model.

        Parameters
        ----------
        X : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The design matrix.
        y : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The response vector.

        Returns
        -------
        self
        """
        x, y = self._process_input(X, y)

        # Unregularized version. Solved with IRLS.
        if self.alpha == 0:
            beta = ordinary_quantile_regression(
                x=x,
                y=y,
                q=self.quantile,
                sample_weight=self.sample_weight,
                max_iter=self.max_iter,
                tol=self.tol
            )
            if self.fit_intercept:
                self.intercept_ = beta[0]
                self.coef_ = np.zeros(self.p)
                self.coef_[self.nonconstant_cols] = beta[1:]
            else:
                self.intercept_ = 0
                self.coef_ = beta
        else:
            coefs = l1_quantile_regression(
                x=x,
                y=y,
                q=self.quantile,
                alpha=self.alpha,
                sample_weight=self.sample_weight,
                feature_weight=self.feature_weight,
                include_intercept=self.fit_intercept
            )
            self.intercept_ = coefs["intercept"]
            self.coef_ = coefs["coef"]
        return self

    def predict(
            self,
            X: np.typing.ArrayLike) -> np.array:
        """Makes prediction for a given x.

        Parameters
        ----------
        X : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The design matrix used for prediction.
        """
        if X.shape[1] != self.p:
            raise ValueError(f"The predict X should have {self.p} columns, but found {X.shape[1]}.")
        return self.intercept_ + X @ self.coef_
