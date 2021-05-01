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
"""sklearn estimator for a dummy model that returns a constant forecast."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message


class DummyEstimator(BaseEstimator, RegressorMixin):
    """Pandas Wrapper for DummyRegressor. Unlike DummyRegressor, it uses X for fitting instead of y
    Otherwise, the interface is identical.

    This allows the DummyRegressor to be part of a Pipeline which transforms X[value_col] before fitting.

    Does not implement BaseForecastEstimator, because it is imported by that class.

    Parameters
    ----------
    strategy : str
        Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.

    constant : int or float or array of shape = [n_outputs]
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0]
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    score_func : callable, optional, default=mean_squared_error
        Function to calculate model R2_null_model_score score,
        with signature (actual, predicted).
        `actual`, `predicted` are array-like with the same shape.
        Smaller values are better.

    Attributes
    ----------
    coverage : None
        Model coverage.
        Always None, provided for compatibility with
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
    null_model: None
        Null model.
        Always None, provided for compatibility with
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
    model : `~sklearn.dummy.DummyRegressor`
        Trained model
    time_col_ : `str`
        Name of timestamp column
    value_col_ : `str`
        Name of value column
    """
    def __init__(self, strategy="mean", constant=None, quantile=None, score_func=mean_squared_error):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile
        self.score_func = score_func

        # Always None, provided for compatibility with `forecast_pipeline`
        self.coverage = None
        self.null_model = None

        # Initializes attributes defined in fit
        self.model = None
        self.time_col_ = None
        self.value_col_ = None

    def fit(self, X, y=None, time_col=TIME_COL, value_col=VALUE_COL, sample_weight=None):
        """Fits the regressor using X[value_col]

        Parameters
        ----------
        X : pd.DataFrame with columns time_col and value_col
            Training data, requires length = n_samples

        y : ignored

        time_col: string, optional
            name of timestamp column in X

        value_col: string, optional
            name of value column in X

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        self : object
        """
        self.time_col_ = time_col  # to be used in `predict` to select proper column
        self.value_col_ = value_col
        # Model cannot be initialized in __init__, otherwise scikit-learn
        # grid search will not be able to set the parameters.
        # See https://scikit-learn.org/stable/developers/develop.html#instantiation.
        self.model = DummyRegressor(
            strategy=self.strategy,
            constant=self.constant,
            quantile=self.quantile)

        # Drops records with missing values
        X = X.dropna(subset=[value_col])

        # Converts pd.DataFrame to np.array. Uses X[value_col] as y for fitting
        x_array = X[[value_col]].to_numpy()
        y_array = X[value_col].to_numpy()
        self.model.fit(X=x_array, y=y_array, sample_weight=sample_weight)
        return self

    def predict(self, X, y=None):
        """Creates forecast for dates specified in X

        Parameters
        ----------
        X : pd.DataFrame with column (self.time_col_), the dates for future prediction. Other columns are ignored.
            timestamps to make forecast

        y : ignored

        Returns
        -------
        df : pd.DataFrame with shape = (X.shape[0], 2)
            Forecasted values for dates in X
        """
        x_array = X[[self.time_col_]].to_numpy()
        predictions = self.model.predict(X=x_array)

        df = pd.DataFrame({
            TIME_COL: X[self.time_col_],
            PREDICTED_COL: predictions
        })

        return df

    def summary(self):
        """Prints input parameters and DummyRegressor model parameters
        """
        log_message(self, LoggingLevelEnum.DEBUG)
        log_message(self.model, LoggingLevelEnum.DEBUG)

    def score(self, X, y, sample_weight=None):
        """Default scorer for the estimator. Returns error based on ``score_func``.
        Because this is often used as null model, it is more informative to return ``score_func`` of the null model
        rather than R2_null_model_score of null model against something else"""
        y_pred = self.predict(X)[PREDICTED_COL]
        return self.score_func(y, y_pred)
