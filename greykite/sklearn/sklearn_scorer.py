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
"""Modification of `sklearn.metrics.make_scorer` and
`sklearn.metrics.scorer._PredictScorer` for
sklearn estimators whose predict function return a
`pandas.DataFrame` with the predicted value.
"""

import warnings
from functools import partial

import numpy as np

from greykite.common.constants import PREDICTED_COL


def _cached_call(cache, estimator, method, *args, **kwargs):
    """Call estimator with method and args and kwargs.
    This code is private in scikit-learn 0.24, so it is copied here.
    """
    if cache is None:
        return getattr(estimator, method)(*args, **kwargs)

    try:
        return cache[method]
    except KeyError:
        result = getattr(estimator, method)(*args, **kwargs)
        cache[method] = result
        return result


class _BaseScorer:
    """This code is private in scikit-learn 0.24, so it is copied here."""
    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign

    @staticmethod
    def _check_pos_label(pos_label, classes):
        if pos_label not in list(classes):
            raise ValueError(
                f"pos_label={pos_label} is not a valid label: {classes}"
            )

    def _select_proba_binary(self, y_pred, classes):
        """Select the column of the positive label in `y_pred` when
        probabilities are provided.
        Parameters
        ----------
        y_pred : ndarray of shape (n_samples, n_classes)
            The prediction given by `predict_proba`.
        classes : ndarray of shape (n_classes,)
            The class labels for the estimator.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Probability predictions of the positive class.
        """
        if y_pred.shape[1] == 2:
            pos_label = self._kwargs.get("pos_label", classes[1])
            self._check_pos_label(pos_label, classes)
            col_idx = np.flatnonzero(classes == pos_label)[0]
            return y_pred[:, col_idx]

        err_msg = (
            f"Got predict_proba of shape {y_pred.shape}, but need "
            f"classifier with two classes for {self._score_func.__name__} "
            f"scoring"
        )
        raise ValueError(err_msg)

    def __repr__(self):
        kwargs_string = "".join([", %s=%s" % (str(k), str(v))
                                 for k, v in self._kwargs.items()])
        return ("make_scorer(%s%s%s%s)"
                % (self._score_func.__name__,
                   "" if self._sign > 0 else ", greater_is_better=False",
                   self._factory_args(), kwargs_string))

    def __call__(self, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        return self._score(partial(_cached_call, None), estimator, X, y_true,
                           sample_weight=sample_weight)

    def _factory_args(self):
        """Return non-default make_scorer arguments for repr."""
        return ""


class _PredictScorerDF(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        """Evaluates predicted target values for ``X`` relative to ``y_true``.

        Modified from `sklearn.metrics.scorer._PredictScorer` to work with
        estimators that return a `pandas.DataFrame` with multiple
        columns, one of which contains the predicted values (``PREDICTED_COL``).

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        estimator : `BaseForecastEstimator`
            Trained estimator to use for scoring. Must have a `predict`
            method that returns a pandas.DataFrame with a column
            :const:`~greykite.common.constants.PREDICTED_COL`.
            This column is the predicted value used to compute the score.
        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.
        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_pred = method_caller(estimator, "predict", X)[PREDICTED_COL]
        if sample_weight is not None:
            score = self._score_func(y_true, y_pred,
                                     sample_weight=sample_weight,
                                     **self._kwargs)
        else:
            score = self._score_func(y_true, y_pred,
                                     **self._kwargs)

        if score is None:
            # Set to np.nan if undefined
            # (e.g. if MAPE, correlation, R2, or R2_null_model_score has division by 0).
            # This function cannot not return None to avoid the following error in
            # sklearn grid search (sklearn/model_selection/_validation.py:618):
            #   `scoring must return a number, got None (<class 'NoneType'>) instead. (scorer=score)`
            # `np.nan` is also the default value for `error_score` in
            # `~sklearn.model_selection.RandomizedSearchCV`. This split score is set
            # to this value if fit fails.
            warnings.warn("Score is undefined for this split, setting to `np.nan`.")
            score = np.nan

        return self._sign * score


def make_scorer_df(score_func, greater_is_better=True, **kwargs):
    """Make a scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in GridSearchCV
    and cross_val_score. It takes a score function, such as ``accuracy_score``,
    ``mean_squared_error``, ``adjusted_rand_index`` or ``average_precision``
    and returns a callable that scores an estimator's output.

    Modified from `sklearn.metrics.make_scorer` to work with
    estimators whose predict function returns a `pandas.DataFrame`
    with multiple columns, one of which contains the predicted values.

    Enabled any standard `score_func` to be used in grid search on a
    `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`.

    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1
    cls = _PredictScorerDF
    return cls(score_func, sign, kwargs)
