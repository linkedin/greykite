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
"""Implements partially regularized regression."""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.model_selection import check_cv

from greykite.common.python_utils import ignore_warnings


def constant_col_finder(x, exclude_cols=None):
    """Finds constant columns in x.

    A column is considered as a constant column if all rows have the same value
    and the value is not zero.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    exclude_cols : `list`[`int`], default None
        Columns in ``exlucde_cols`` are not considered as constant columns.

    Returns
    -------
    constant_cols : `list`[`int`]
        A list of indices that satisfy the constant column conditions.
    """
    if exclude_cols is None:
        exclude_cols = []
    var = x.var(axis=0)
    constant_cols = [i for i in range(len(var)) if var[i] == 0 and x[0, i] != 0 and i not in exclude_cols]
    return constant_cols


class PartialRegularizeRegression:
    """Class to implement a partially regularized regression.

    A partially regularized regression is defined as

        beta_0, beta_1, beta_2 = argmin ||y - X_0beta_0 - X_1beta_1 - X_2beta_2||_2^2 + a_1 * ||beta_1||_1 + a_2 * ||beta_2||_2^2

    This regression formulation allows user to specify which regressor to have what type of regularization.
    For example, in automatic changepoint detection, "ct1" should not be penalized, yearly seasonality can have light L2 norm regularization,
    while the changepoint regressors need L1 norm regularization to be sparse.

    Attributes
    ----------
    l1_index : `list`[`int`, `str`], default None
        The index of columns that are to be penalized with L1 norm.
    l1_alpha : `float`, default 0.001
        The L1 norm regularization parameter.
    l2_index : `list`[`int`, `str`], default None
        The index of columns that are to be penalized with L2 norm.
    l2_alpha : `float`, default 0.001
        The L2 norm regularization parameter.
    has_constant : `list`[`int`]
        A list of detected constant column indices by `intercept_checker`.
    constant_index : `list`[`int`]
        A list of constant column indices after preprocessing.
    non_penalize_index : `list`[`int`]
        A list of column indices whose corresponding regressors are not to be penalized.
    intercept_ : `float`
        The fitted intercept.
    coef_ : `numpy.array`
        The fitted coefficients.
    """
    def __init__(
            self,
            l1_index=None,
            l1_alpha=0.001,
            l2_index=None,
            l2_alpha=0.001):
        """Initializes instance."""
        if l1_index is not None and l2_index is not None and len([x for x in l1_index if x in l2_index]) > 0:
            raise ValueError("l1_index and l2_index should not overlap.")
        self.l1_index = l1_index
        self.l1_alpha = l1_alpha
        self.l2_index = l2_index
        self.l2_alpha = l2_alpha
        self.has_constant = None
        self.constant_index = None
        self.non_penalize_index = None
        self.intercept_ = None
        self.intercept_ = None
        self.coef_ = None

    @ignore_warnings(category=FutureWarning)
    def fit(self, x, y):
        """Fits the partial regularized regression.

        Parameters
        ----------
        x : `numpy.array` or `pandas.DataFrame`
            The design matrix.
        y : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The response vector.

        Returns
        -------
        self : `~greykite.algo.common.partial_regularize_regression.PartialRegularizeRegression`
            The current class instance.
        """
        # Checks if the config matches a special case.
        # Because we already checked there's no overlapping in `l1_index` and `l2_index`,
        # so we have the following 3 cases:
        #   - `l1_index` includes all columns of x: this is the same as the Lasso.
        #   - `l2_index` includes all columns of x: this is the same as the Ridge.
        #   - Both `l1_index` and `l2_index` are None: this is the same as the LinearRegression.
        if self.l1_index is not None and len(self.l1_index) == x.shape[1]:
            model = Lasso(alpha=self.l1_alpha).fit(x, y)
            self.intercept_ = model.intercept_
            self.coef_ = model.coef_
            return self
        if self.l2_index is not None and len(self.l2_index) == x.shape[1]:
            model = Ridge(alpha=self.l2_alpha).fit(x, y)
            self.intercept_ = model.intercept_
            self.coef_ = model.coef_
            return self
        if self.l1_index is None and self.l2_index is None:
            model = LinearRegression().fit(x, y)
            self.intercept_ = model.intercept_
            self.coef_ = model.coef_
            return self
        # In the above three cases, we have one type of regularization on all columns of x.
        # The cases below will be a mix of at least two regularizations.
        # We'll handle this with the core algorithm in `self._fit_analytic`.
        x, y = self._check_input(x, y)
        lasso_input = self._get_lasso_input(x, y, self.l2_alpha)
        # Fits the lasso part.
        if lasso_input["x_lasso"].shape[1] > 0:
            lasso = Lasso(alpha=self.l1_alpha, fit_intercept=False).fit(
                X=lasso_input["x_lasso"],
                y=lasso_input["y_lasso"]
            )
            beta1 = lasso.coef_
        else:
            beta1 = np.array([])
        # Computes the L2 and non-penalize part.
        beta02 = lasso_input["hl02_r"] @ (y - lasso_input["x1"] @ beta1)
        coef = np.zeros(x.shape[1])
        coef[self.non_penalize_index] = beta02[:lasso_input["x0"].shape[1]]
        coef[self.l1_index] = beta1
        coef[self.l2_index] = beta02[lasso_input["x0"].shape[1]:]
        # Extracts the intercept and coefficients.
        if not self.has_constant:
            self.intercept_ = coef[-1]
            self.coef_ = coef[:-1]
        else:
            self.intercept_ = np.sum(coef[self.constant_index])
            self.coef_ = coef
            self.coef_[self.constant_index] = 0
        return self

    def predict(self, x):
        """Predicts for new data.

        Parameters
        ----------
        x : `numpy.array` or `pandas.DataFrame`
            The new data matrix.

        Returns
        -------
        y_pred : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The predicted values.
        """
        pred = x @ self.coef_ + self.intercept_
        return pred

    def _check_input(self, x, y):
        """Preprocesses the input design matrix, the response vector, l1_index, l2_index and constant columns.

        Parameters
        ----------
        x : `numpy.array` or `pandas.DataFrame`
            The design matrix.
        y : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The response vector.

        Returns
        -------
        x : `numpy.array`
            The processed design matrix.
        y : `numpy.array`
            The processed response vector.
        """
        if self.l1_index is None:
            self.l1_index = []
        if self.l2_index is None:
            self.l2_index = []
        if isinstance(x, pd.DataFrame):
            columns = x.columns
        else:
            columns = []
        if not all([isinstance(x, int) for x in self.l1_index]):
            l1_index = [i for i, col in enumerate(columns) if col in self.l1_index]
            if len(l1_index) != len(self.l1_index):
                raise ValueError("l1_index takes either a list of integer indices or a list of "
                                 "a subset of the columns names in x (if x is pandas.DataFrame).")
            self.l1_index = l1_index
        if not all([isinstance(x, int) for x in self.l2_index]):
            l2_index = [i for i, col in enumerate(columns) if col in self.l2_index]
            if len(l2_index) != len(self.l2_index):
                raise ValueError("l2_index takes either a list of integer indices or a list of "
                                 "a subset of the columns names in x (if x is pandas.DataFrame).")
            self.l2_index = l2_index
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, Union[pd.DataFrame, pd.Series].__args__):
            y = y.values
        self.has_constant = constant_col_finder(x)
        if not self.has_constant:
            x = np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)
            self.constant_index = x.shape[1] - 1
        else:
            self.constant_index = self.has_constant
        self.non_penalize_index = [i for i in range(x.shape[1]) if i not in self.l1_index and i not in self.l2_index]
        return x, y

    def _get_lasso_input(self, x, y, l2_alpha):
        """Computes the transformed lasso input quantities.

        Parameters
        ----------
        x : `numpy.array`
            The design matrix.
        y : `numpy.array`
            The response vector.
        l2_alpha : `float`
            The L2 norm regularization parameter.

        Returns
        -------
        lasso_input : `dict`
            The quantites need in the lasso computation.
        """
        x0 = x[:, self.non_penalize_index]
        x1 = x[:, self.l1_index]
        x2 = x[:, self.l2_index]
        x02 = np.concatenate([x0, x2], axis=1)
        d = np.eye(x02.shape[1])
        d[:x0.shape[1], :x0.shape[1]] = np.zeros([x0.shape[1], x0.shape[1]])  # Non-penalized part is all zero.
        # This is the matrix to be multiplied to the response vector to produce the estimated beta
        # for the non-penalized and L2 norm penalized part.
        hl02_r = np.linalg.pinv(x02.T @ x02 + l2_alpha * d) @ x02.T
        hl02 = x02 @ hl02_r
        i02 = np.eye(x.shape[0])
        return {
            "x_lasso": (i02 - hl02) @ x1,
            "y_lasso": (i02 - hl02) @ y,
            "hl02": hl02,
            "hl02_r": hl02_r,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x02": x02,
            "i02": i02
        }


class PartialRegularizeRegressionCV:
    """Implements the cross-validation version of
    `~greykite.algo.common.partial_regularize_regression.PartialRegularizeRegression`.

    Attributes
    ----------
    l1_index : `list`[`int`, `str`], default None
        The index of columns that are to be penalized with L1 norm.
    l1_alphas : `list`[`float`], default None
        The L1 norm regularization parameters.
    n_l1_alphas : `int`, default 50.
        The number of L1 norm regularization parameters. Will not be used if `l1_alphas` is provided.
    l2_index : `list`[`int`, `str`], default None
        The index of columns that are to be penalized with L2 norm.
    l2_alphas : `list`[`float`], default (0.001, 0.1, 10)
        The L2 norm regularization parameter.
    cv : `int`, cross-validation generator, or iterable, default 5
        Determines the cross-validation splitting strategy. This is the same as the ``cv`` in `sklearn` modules.
        Possible inputs for cv are:
            - None, to use the default 5-fold cross validation,
            - integer, to specify the number of folds.
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.
    has_constant : `list`[`int`]
        A list of detected constant column indices by `intercept_checker`.
    constant_index : `list`[`int`]
        A list of constant column indices after preprocessing.
    non_penalize_index : `list`[`int`]
        A list of column indices whose corresponding regressors are not to be penalized.
    best_model : `~greykite.algo.common.partial_regularize_regression.PartialRegularizeRegression`
        The best model selected by cross validation.
    best_mse : `float`
        The best cross-validated test mse.
    l1_alpha_ : `float`
        The best L1 norm regularization parameter.
    l2_alpha_ : `float`
        The best L2 norm regularization parameter.
    intercept_ : `float`
        The fitted intercept.
    coef_ : `numpy.array`
        The fitted coefficients.
    """
    def __init__(
            self,
            l1_index=None,
            l1_alphas=None,
            n_l1_alphas=50,
            l2_index=None,
            l2_alphas=(0.001, 0.1, 10),
            cv=5):
        """Initializes instance."""
        self.l1_index = l1_index
        self.l1_alphas = l1_alphas
        self.n_l1_alphas = n_l1_alphas
        self.l2_index = l2_index
        self.l2_alphas = l2_alphas
        self.cv = cv
        self.has_constant = None
        self.constant_index = None
        self.non_penalize_index = None
        self.best_model = None
        self.best_mse = None
        self.l1_alpha_ = None
        self.l2_alpha_ = None
        self.intercept_ = None
        self.coef_ = None

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, x, y):
        """Fits the partial regularized regression.

        Parameters
        ----------
        x : `numpy.array` or `pandas.DataFrame`
            The design matrix.
        y : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The response vector.

        Returns
        -------
        self : `~greykite.algo.common.partial_regularize_regression.PartialRegularizeRegression`
            The current class instance.
        """
        if self.l1_index is not None and len(self.l1_index) == x.shape[1]:
            model = LassoCV(alphas=self.l1_alphas, n_alphas=self.n_l1_alphas).fit(x, y)
            self.intercept_ = model.intercept_
            self.coef_ = model.coef_
            self.best_model = model
            return self
        if self.l2_index is not None and len(self.l2_index) == x.shape[1]:
            model = RidgeCV(alphas=self.l2_alphas).fit(x, y)
            self.intercept_ = model.intercept_
            self.coef_ = model.coef_
            self.best_model = model
            return self
        if self.l1_index is None and self.l2_index is None:
            model = LinearRegression().fit(x, y)
            self.intercept_ = model.intercept_
            self.coef_ = model.coef_
            self.best_model = model
            return self
        x, y = self._check_input(x, y)
        cv = check_cv(self.cv)
        best_model = None
        best_mse = np.inf
        # Iterates over `l2_alphas`, because we can use `lasso_path` to iterate over `l1_alphas`.
        for l2_alpha in self.l2_alphas:
            # Finds the best `l1_alpha` and its MSE under the current `l2_alpha`.
            l1_alpha, mse = self._fit_analytic(
                x=x,
                y=y,
                l2_alpha=l2_alpha,
                cv=cv
            )
            # Updates model if improved.
            if mse < best_mse:
                best_mse = mse
                best_model = PartialRegularizeRegression(
                    l1_index=self.l1_index,
                    l1_alpha=l1_alpha,
                    l2_index=self.l2_index,
                    l2_alpha=l2_alpha,
                )
        # Fits on the entire data set with the best `l1_alpha` and `l2_alpha`.
        if not self.has_constant:
            best_model.fit(x[:, :-1], y)
        else:
            best_model.fit(x, y)
        self.best_model = best_model
        self.best_mse = best_mse
        self.l1_alpha_ = best_model.l1_alpha
        self.l2_alpha_ = best_model.l2_alpha
        self.intercept_ = best_model.intercept_
        self.coef_ = best_model.coef_
        return self

    def predict(self, x):
        """Predicts for new data.

        Parameters
        ----------
        x : `numpy.array` or `pandas.DataFrame`
            The new data matrix.

        Returns
        -------
        y_pred : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The predicted values.
        """
        pred = self.best_model.predict(x)
        return pred

    def _check_input(self, x, y):
        """Preprocesses the input design matrix, the response vector, l1_index, l2_index and constant columns.

        Parameters
        ----------
        x : `numpy.array` or `pandas.DataFrame`
            The design matrix.
        y : `numpy.array`, `pandas.DataFrame` or `pandas.Series`
            The response vector.

        Returns
        -------
        x : `numpy.array`
            The processed design matrix.
        y : `numpy.array`
            The processed response vector.
        """
        if self.l1_index is None:
            self.l1_index = []
        if self.l2_index is None:
            self.l2_index = []
        if isinstance(x, pd.DataFrame):
            columns = x.columns
        else:
            columns = []
        if not all([isinstance(x, int) for x in self.l1_index]):
            l1_index = [i for i, col in enumerate(columns) if col in self.l1_index]
            if len(l1_index) != len(self.l1_index):
                raise ValueError("l1_index takes either a list of integer indices or a sub-list of "
                                 "columns in x (if x is pandas.DataFrame).")
            self.l1_index = l1_index
        if not all([isinstance(x, int) for x in self.l2_index]):
            l2_index = [i for i, col in enumerate(columns) if col in self.l2_index]
            if len(l2_index) != len(self.l2_index):
                raise ValueError("l2_index takes either a list of integer indices or a sub-list of "
                                 "columns in x (if x is pandas.DataFrame).")
            self.l2_index = l2_index
        if len([x for x in range(x.shape[1]) if x in self.l1_index and x in self.l2_index]) > 0:
            raise ValueError("l1_index and l2_index can not overlap.")
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, Union[pd.DataFrame, pd.Series].__args__):
            y = y.values
        self.has_constant = constant_col_finder(x)
        if not self.has_constant:
            x = np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)
            self.constant_index = x.shape[1] - 1
        else:
            self.constant_index = self.has_constant
        self.non_penalize_index = [i for i in range(x.shape[1]) if i not in self.l1_index and i not in self.l2_index]
        return x, y

    def _fit_analytic(self, x, y, l2_alpha, cv):
        """Fits the regression problem with a single L2 norm regularization parameter and
        cross validation over many L1 norm regularization parameters.

        The algorithm fits the problem with `sklearn.linear_model.LassoCV` on transformed x and y,
        then estimates the non-penalized and L2 norm penalized coefficients.

        Parameters
        ----------
        x : `numpy.array`
            The processed design matrix.
        y : `numpy.array`
            The processed response vector.
        l2_alpha : `float`
            The L2 norm regularization parameter.
        cv : `int`, cross-validation generator, or iterable, default 5
            Determines the cross-validation splitting strategy. This is the same as the ``cv`` in `sklearn` modules.
            Possible inputs for cv are:
                - None, to use the default 5-fold cross validation,
                - integer, to specify the number of folds.
                - :term:`CV splitter`,
                - An iterable yielding (train, test) splits as arrays of indices.

        Returns
        -------
        best_l1_alpha : `float`
            The best L1 regularization parameter model.
        best_mse : `float`
            The best mse.
        """
        mses = np.zeros(self.n_l1_alphas) if self.l1_index else np.zeros(1)
        if self.l1_alphas is not None:
            l1_alphas = self.l1_alphas
        else:
            l1_alphas = self._get_alphas(x, y, l2_alpha)
        for train, test in cv.split(x):
            x_train = x[train]
            x_test = x[test]
            y_train = y[train]
            y_test = y[test]
            lasso_input = self._get_lasso_input(x_train, y_train, l2_alpha)
            if l1_alphas is not None:
                # Fits all `l1_alphas`.
                path = LassoCV().path(
                    alphas=l1_alphas,
                    fit_intercept=False,
                    X=lasso_input["x_lasso"],
                    y=lasso_input["y_lasso"]
                )
                # Makes predictions and gets the best MSE.
                beta1 = path[1]
            else:
                beta1 = np.array([])
            beta02 = lasso_input["hl02_r"] @ (y_train.reshape(-1, 1) - (lasso_input["x1"] @ beta1).reshape(y_train.shape[0], -1))
            coef = np.zeros([x.shape[1], self.n_l1_alphas]) if l1_alphas is not None else np.zeros([x.shape[1], 1])
            coef[self.non_penalize_index, :] = beta02[:lasso_input["x0"].shape[1], :]
            coef[self.l1_index, :] = beta1 if l1_alphas is not None else 0
            coef[self.l2_index] = beta02[lasso_input["x0"].shape[1]:, :]
            y_pred = x_test @ coef
            mses += ((y_test.reshape(-1, 1) - y_pred) ** 2).mean(axis=0)
        best_l1_alpha = l1_alphas[mses.argmin()] if l1_alphas is not None else None
        return best_l1_alpha, np.min(mses)

    def _get_lasso_input(self, x, y, l2_alpha):
        """Computes the transformed Lasso input quantities.

        Parameters
        ----------
        x : `numpy.array`
            The design matrix.
        y : `numpy.array`
            The response vector.
        l2_alpha : `float`
            The L2 norm regularization parameter.

        Returns
        -------
        lasso_input : `dict`
            The quantites need in the Lasso computation.
        """
        x0 = x[:, self.non_penalize_index]
        x1 = x[:, self.l1_index]
        x2 = x[:, self.l2_index]
        x02 = np.concatenate([x0, x2], axis=1)
        d = np.eye(x02.shape[1])
        d[:x0.shape[1], :x0.shape[1]] = np.zeros([x0.shape[1], x0.shape[1]])
        hl02_r = np.linalg.pinv(x02.T @ x02 + l2_alpha * d) @ x02.T
        hl02 = x02 @ hl02_r
        i02 = np.eye(x.shape[0])
        return {
            "x_lasso": (i02 - hl02) @ x1,
            "y_lasso": (i02 - hl02) @ y,
            "hl02": hl02,
            "hl02_r": hl02_r,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x02": x02,
            "i02": i02}

    def _get_alphas(self, x, y, l2_alpha):
        """Gets the Lasso alpha grid.

        Parameters
        ----------
        x : `numpy.array`
            The design matrix.
        y : `numpy.array`
            The response vector.
        l2_alpha : `float`
            The L2 norm regularization parameter.

        Returns
        -------
        l1_alpha : `list`[`float`] or None
            The alphas for L1 norm regularization to search from.
        """
        lasso_input = self._get_lasso_input(x, y, l2_alpha)
        if lasso_input["x_lasso"].shape[1] > 0:
            l1_alpha = _alpha_grid(
                lasso_input["x_lasso"],
                lasso_input["y_lasso"],
                n_alphas=self.n_l1_alphas
            )
        else:
            l1_alpha = None
        return l1_alpha
