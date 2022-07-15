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
"""All forecast estimators must inherit from this class."""

from abc import ABC
from abc import abstractmethod
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error

from greykite.common import constants as cst
from greykite.common.evaluation import r2_null_model_score
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.sklearn.estimator.null_model import DummyEstimator
from greykite.sklearn.uncertainty.exceptions import UncertaintyError
from greykite.sklearn.uncertainty.uncertainty_methods import UncertaintyMethodEnum


class BaseForecastEstimator(BaseEstimator, RegressorMixin, ABC):
    """A base class for forecast models. Fits a timeseries and predicts future values

    Parameters
    ----------
    score_func : callable, optional, default=mean_squared_error
        Function to calculate model R2_null_model_score score,
        with signature (actual, predicted).
        `actual`, `predicted` are array-like with the same shape.
        Smaller values are better.

    coverage : float, optional, default=0.95
        intended coverage of the prediction bands (0.0 to 1.0)
        If None, the upper/lower predictions are not returned by `predict`

        Every subclass must use `coverage` to set prediction band width. This ensures a common
        BaseForecastEstimator interface for parameters used during fitting and forecast evaluation

    null_model_params : dict with arguments passed to DummyRegressor, optional, default=None
        Dictionary keys must be in ("strategy", "constant", "quantile")
        Defines null model. model score is R2_null_model_score of model error relative to null model, evaluated by score_func
        If None, model score is score_func of the model itself

    Attributes
    ----------
    null_model : DummyEstimator
        null model used to measure model score
    time_col_ : str
        Name of input data time column
    value_col_ : str
        Name of input data value column
    last_predicted_X_ : `pandas.DataFrame` or None
        The ``X`` last passed to ``self.predict()``.
        Used to speed up predictions if the same ``X`` is passed repeatedly.
        Resets to None when ``self.fit()`` is called.
    cached_predictions_ : `pandas.DataFrame` or None
        The return value of the last call to ``self.predict()``.
        Used to speed up predictions if the same ``X`` is passed repeatedly.
        Resets to None when ``self.fit()`` is called.
    """
    @abstractmethod
    def __init__(self, score_func=mean_squared_error, coverage=0.95, null_model_params=None):
        """Initializes attributes common to every BaseForecastEstimator

        Subclasses must also have these parameters. Every subclass must call:

            super().__init__(score_func=score_func, coverage=coverage, null_model_params=null_model_params)

        """
        self.score_func = score_func
        self.coverage = coverage
        self.null_model_params = null_model_params

        # initializes attributes defined in fit
        self.null_model = None
        self.time_col_ = None
        self.value_col_ = None

        # initializes attributes defined in predict
        self.last_predicted_X_ = None    # the most recent X passed to self.predict()
        self.cached_predictions_ = None  # the most recent return value of self.predict()

        # initializes attributes defined in ``fit_uncertainty``
        self.uncertainty_model = None  # the uncertainty model

    @abstractmethod
    def fit(self, X, y=None, time_col=cst.TIME_COL, value_col=cst.VALUE_COL, **fit_params):
        """Fits a model to training data
        Also fits the null model, if specified, for use in evaluating the `score` function

        Every subclass must call this::
            super().fit(X, y=y, time_col=time_col, value_col=value_col, **fit_params)

        Parameters
        ----------
        X : `pandas.DataFrame`
            Input timeseries, with timestamp column,
            value column, and any additional regressors.
            The value column is the response, included in
            ``X`` to allow transformation by `sklearn.pipeline`.
        y : ignored
            The original timeseries values, ignored.
            (The y for fitting is included in X.)
        time_col : `str`
            Time column name in X.
        value_col : `str`
            Value column name in X.
        fit_params : `dict`
            Additional parameters supported by subclass `fit` or null model.
        """
        self.time_col_ = time_col  # to be used in `predict` to select proper column
        self.value_col_ = value_col
        # Null model must be initialized here, otherwise scikit-learn
        # grid search will not be able to set the parameters.
        # See https://scikit-learn.org/stable/developers/develop.html#instantiation.
        if self.null_model_params is not None:
            # Adds score function to null model parameters, and initializes null model
            self.null_model_params["score_func"] = self.score_func
            self.null_model = DummyEstimator(**self.null_model_params)
            # Passes `sample_weight` rather than `**fit_params` to avoid unexpected keyword argument from the main
            #   estimator's parameters
            sample_weight = fit_params.get("sample_weight")
            self.null_model.fit(X, y=y, time_col=time_col, value_col=value_col, sample_weight=sample_weight)

        # Clears the cached result, because it is no longer valid for the updated model
        self.last_predicted_X_ = None
        self.cached_predictions_ = None

    @abstractmethod
    def predict(self, X, y=None):
        """Creates forecast for dates specified in X

        To enable caching, every subclass must call this at the beginning
        of its ``.predict()``. Before returning the result, the subclass
        ``.predict()`` must set ``self.cached_predictions_`` to the return value.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Input timeseries with timestamp column and any additional regressors.
            Timestamps are the dates for prediction.
            Value column, if provided in X, is ignored.
        y : ignored

        Returns
        -------
        predictions : `pandas.DataFrame`
            Forecasted values for the dates in X. Columns:

                - TIME_COL dates
                - PREDICTED_COL predictions
                - PREDICTED_LOWER_COL lower bound of predictions, optional
                - PREDICTED_UPPER_COL upper bound of predictions, optional
                - [other columns], optional

            ``PREDICTED_LOWER_COL`` and ``PREDICTED_UPPER_COL`` are present
            if ``self.coverage`` is not None.
        """
        if self.cached_predictions_ is not None and X.equals(self.last_predicted_X_):
            log_message("Returning cached predictions.", LoggingLevelEnum.DEBUG)
            return self.cached_predictions_
        else:
            # Updates `last_predicted_X` to the new value.
            # To enable caching, the subclass must set
            # `self.cached_predictions` to the returned result.
            self.last_predicted_X_ = X
            return None

    def summary(self):
        """Creates human readable string of how the model works, including relevant diagnostics
        These details cannot be extracted from the forecast alone
        Prints model configuration. Extend this in child class to print the trained model parameters.

        Log message is printed to the cst.LOGGER_NAME logger.
        """
        log_message(self, LoggingLevelEnum.DEBUG)  # print model input parameters

    def score(self, X, y, sample_weight=None):
        """Default scorer for the estimator (Used in GridSearchCV/RandomizedSearchCV if scoring=None)

        Notes
        -----
        If null_model_params is not None, returns R2_null_model_score of model error
        relative to null model, evaluated by score_func.

        If null_model_params is None, returns score_func of the model itself.

        By default, grid search (with no `scoring` parameter) optimizes improvement of ``score_func``
        against null model.

        To optimize a different score function, pass `scoring` to GridSearchCV/RandomizedSearchCV.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Input timeseries with timestamp column and any additional regressors.
            Value column, if provided in X, is ignored
        y : `pandas.Series` or  `numpy.array`
            Actual value, used to compute error
        sample_weight : `pandas.Series` or  `numpy.array`
            ignored

        Returns
        -------
        score : `float` or None
            Comparison of predictions against null predictions, according to specified score function
        """
        y_pred = self.predict(X)[cst.PREDICTED_COL]
        if self.null_model is not None:
            y_pred_null = self.null_model.predict(X)[cst.PREDICTED_COL]
            score = r2_null_model_score(
                y,
                y_pred,
                y_pred_null=y_pred_null,
                loss_func=self.score_func)
        else:
            score = self.score_func(y, y_pred)
        return score

    def fit_uncertainty(
            self,
            df: pd.DataFrame,
            uncertainty_dict: dict,
            fit_params: Optional[dict] = None,
            **kwargs):
        """Fits the uncertainty model with a given ``df`` and ``uncertainty_dict``.

        Parameters
        ----------
        df : `pandas.DataFrame`
            A dataframe representing the data to fit the uncertainty model.
        uncertainty_dict : `dict` [`str`, any]
            The uncertainty model specification. It should have the following keys:

                "uncertainty_method": a string that is in
                    `~greykite.sklearn.uncertainty.uncertainty_methods.UncertaintyMethodEnum`.
                "params": a dictionary that includes any additional parameters needed by the uncertainty method.

        fit_params : `dict` [`str`, any] or None, default None
            Parameters to be passed to the ``fit`` function.
        kwargs : additional parameters to be fed into the uncertainty method.
            These parameters are from the estimator attributes, not given by user.

        Returns
        -------
        The function sets ``self.uncertainty_model`` and does not return anything.
        """
        # When ``uncertainty_dict`` is not provided but ``self.coverage`` is provided,
        # it falls back to the "simple conditional residuals" uncertainty method.
        if (uncertainty_dict is None or uncertainty_dict == {}) and self.coverage is not None:
            uncertainty_dict = dict(
                uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
                params=dict(
                    value_col=self.value_col_,
                    is_residual_based=True,
                    offset_col=cst.PREDICTED_COL
                )
            )
        # Gets the uncertainty method.
        if uncertainty_dict is not None:
            uncertainty_method = uncertainty_dict.get("uncertainty_method", None)
        else:
            uncertainty_method = None
        # Tries to find the uncertainty method from Enum.
        if uncertainty_method is not None:
            try:
                uncertainty_model = UncertaintyMethodEnum[uncertainty_method].value.model_class
            except (KeyError, AttributeError):
                # If not found in Enum, do not stop, return.
                # The ``self.uncertainty_model`` is not set and will not be used in prediction.
                log_message(
                    message=f"Uncertainty method {uncertainty_method} is not found in `UncertaintyMethodEnum`, "
                            f"uncertainty fitting is skipped. Valid methods are "
                            f"{', '.join(UncertaintyMethodEnum.__dict__['_member_names_'])}.",
                    level=LoggingLevelEnum.WARNING
                )
                return
            uncertainty_dict = self._populate_uncertainty_params(uncertainty_dict)
            # Tries to fit the model. If ``UncertaintyError`` happened,
            # catch it and do not raise an error.
            try:
                self.uncertainty_model = uncertainty_model(
                    uncertainty_dict=uncertainty_dict,
                    time_col=self.time_col_,
                    coverage=self.coverage,
                    **kwargs
                )
                if fit_params is None:
                    fit_params = {}
                self.uncertainty_model.fit(
                    df,
                    **fit_params
                )
            except UncertaintyError as e:
                self.uncertainty_model = None
                log_message(
                    message=f"The following errors occurred during fitting the uncertainty model, "
                            f"the uncertainty model is skipped. {e}",
                    level=LoggingLevelEnum.WARNING
                )

    def predict_uncertainty(
            self,
            df: pd.DataFrame,
            predict_params: Optional[dict] = None):
        """Makes predictions of prediction intervals for ``df`` based on the predictions
        and ``self.uncertainty_model``.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The dataframe to calculate prediction intervals upon.
            It should have either ``self.value_col_`` or
            `~greykite.common.constants.PREDICT_COL` which the
            prediction interval is based on.
        predict_params : `dict` [`str`, any] or None, default None
            Parameters to be passed to the ``predict`` function.

        Returns
        -------
        result_df : `pandas.DataFrame`
            The ``df`` with prediction interval columns.
        """
        # Skip prediction when no uncertainty model is trained.
        if self.uncertainty_model is None:
            log_message(
                message=f"The uncertainty model is not trained.",
                level=LoggingLevelEnum.WARNING
            )
            return

        result_df = None
        # Tries to predict prediction intervals,
        # and skip if ``UncertaintyError`` occurred.
        try:
            if predict_params is None:
                predict_params = {}
            result_df = self.uncertainty_model.predict(df, **predict_params)
        except UncertaintyError as e:
            log_message(
                message=f"The following errors occurred during predicting the uncertainty model, "
                        f"the uncertainty model is skipped. {e}",
                level=LoggingLevelEnum.WARNING
            )
        return result_df

    def _populate_uncertainty_params(
            self,
            uncertainty_dict: dict):
        """If any parameters of the ``uncertainty_dict`` for specific uncertainty methods
        need to be populated from the estimator, they should be added here.

        Parameters
        ----------
        uncertainty_dict : `dict` [`str`, any]
            The original uncertainty dictionary.

        Returns
        -------
        uncertainty_dict : `dict` [`str`, any]
            The uncertainty dictionary with additional parameters populated from the estimator.
        """
        if uncertainty_dict.get("uncertainty_method") == UncertaintyMethodEnum.simple_conditional_residuals.name:
            # Populates ``value_col`` and ``residual_col`` for "simple_conditional_residuals" method.
            # The method always uses residual based prediction intervals.
            # If ``value_col`` is not in ``params``, ``self.value_col_`` will be used.
            # If ``residual_col`` is not in ``params``,
            # a residual column will be calculated based on the ``cst.PREDICT_COL``.
            # The algorithm assumes there's always ``cst.PREDICT_COL`` in the input df.
            params = uncertainty_dict.get("params", {})
            if "value_col" not in params:
                params["value_col"] = self.value_col_
            if "residual_col" not in params:
                params["residual_col"] = "residual_col"
            uncertainty_dict["params"] = params
            return uncertainty_dict
        if uncertainty_dict.get("uncertainty_method") == UncertaintyMethodEnum.quantile_regression.name:
            params = uncertainty_dict.get("params", {})
            if "value_col" not in params:
                params["value_col"] = self.value_col_
            uncertainty_dict["params"] = params
            return uncertainty_dict
