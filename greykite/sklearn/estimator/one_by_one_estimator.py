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
# original author: Kaixu Yang, Reza Hosseini
"""Forecast one by one estimator.

Applies different models to forecast different forecast horizons.
Some parameters depends on forecast horizon, for example, when ``autoreg_dict="auto"``.
A model trained with autoreg_dict from forecast horizon 7 may not predict the first
time point as well as a model trained with autoreg_dict from forecast horizon 1.

This estimator takes an estimator, checks if any of the parameters depends on forecast
horizon, and initializes one or multiple instances with different horizons.
The forecast is a splice of the estimators' forecasts.
"""

from copy import deepcopy
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from sklearn.metrics import mean_squared_error

from greykite.common import constants as cst
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator
from greykite.sklearn.estimator.prophet_estimator import ProphetEstimator
from greykite.sklearn.estimator.silverkite_estimator import SilverkiteEstimator
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator


ONEBYONE_ESTIMATORS = {
    "SilverkiteEstimator": {
        "class": SilverkiteEstimator,
        "params_depending_on_horizon": {
            "autoreg_dict": ["auto"]
        },
        "forecast_horizon_param": "forecast_horizon"
    },
    "SimpleSilverkiteEstimator": {
        "class": SimpleSilverkiteEstimator,
        "params_depending_on_horizon": {
            "autoreg_dict": ["auto"]
        },
        "forecast_horizon_param": "forecast_horizon"
    },
    "ProphetEstimator": {
        "class": ProphetEstimator,
        "params_depending_on_horizon": None,
        "forecast_horizon_param": None
    }
}
"""Estimators supported by forecast one-by-one algorithm.
Each estimator has key as EstimatorClass.__name__.
We use the classes' string names as index because in sklearn
cloning nested classes may cause error.
Each estimator has the following keys

    "class": class
        The estimator class.
    "params_depending_on_horizon": `dict` [`str`, `list`]
        The estimator init parameters and their values
        that depends on forecast horizon. For example, in SimpleSilverkiteEstimator,
        the ``autoreg_dict`` param will be calculated based on forecast horizon if
        the input value is "auto". Forecast one by one is activated only when there
        are model parameters that meet these key-values.
    "forecast_horizon_param": `str`
        The name of parameter that passes forecast horizon to the estimator.

Add any estimator here to allow using forecast one-by-one here.
"""


class OneByOneEstimator(BaseForecastEstimator):
    """Forecast one-by-one estimator.

    Parameters
    ----------
    score_func : `callable`, default mean_squared_error
        Function to calculate model R2_null_model_score score,
        with signature (actual, predicted).
        ``actual``, ``predicted`` are array-like with the same shape.
        Smaller values are better.

    coverage : `float`, default 0.95
        Intended coverage of the prediction bands (0.0 to 1.0).
        If None, the upper/lower predictions are not returned by ``predict``.

    null_model_params : `dict`, default None
        Dictionary keys must be in ("strategy", "constant", "quantile").
        Defines null model. model score is R2_null_model_score of model error relative to null model,
        evaluated by score_func.
        If None, model score is score_func of the model itself.

    estimator : `str`
        The estimator used in forecast one-by-one.
        Must be one of the keys in ``ONEBYONE_ESTIMATORS``.

    forecast_horizon : `int`
        The forecast horizon.

    estimator_map : `bool`, `int` or `list` [`int`], default None
        The map between estimators and forecast horizons.

            If True, each time point in the forecasting period has a different model.
            If False, forecast one-by-one is turned off.
            If `int`, every ``estimator_map`` time points have a different model.
            If `list`, the sum must equal to the forecast horizon.
                The values are the number of timepoints in each estimator.

        For example, if forecast horizon is 7

            `estimator_map = True` corresponds to [1, 1, 1, 1, 1, 1, 1], 7 models.
            `estimator_map = 1` corresponds to [1, 1, 1, 1, 1, 1, 1], 7 models.
            `estimator_map = 2` corresponds to [2, 2, 2, 1], 4 models.
            `estimator_map = [1, 2, 4]` corresponds to 3 models.
            `estimator_map = False` corresponds to [7], 1 model.
            `estimator_map = None` corresponds to [7], 1 model.


    Attributes
    ----------
    estimator : `str`
        The estimator name.
    estimator_params : `dict`
        The parameters used in the estimator.
    forecast_horizon : `int`
        The forecast horizon.
    estimator_map : `bool`, `int`, `list` [`int]`
        The map between estimators and time points.
    estimator_map_list : `list` [`int`]
        The processed estimator_map as a list.
        We don't want to overwrite the original self.estimator_map.
    estimator_param_names : `list` [`str`]
        All available parameter names in estimator.
    estimator_class : `class`
        The estimator class.
    estimators : `list` [`class`]
        The list of estimator class instances.
    model_results : `list`
        The list of estimator fit method outputs.
    pred_indices : `list` [`int`]
        The list of indices used to splice forecasts.
    forecast : `pandas.DataFrame`
        The cached forecast results from the last call of ``self.predict``.
    """
    def __init__(
            self,
            estimator: str,
            forecast_horizon: int,
            estimator_map: Optional[Union[int, List[int]]] = None,
            score_func: Callable = mean_squared_error,
            coverage: float = 0.95,
            null_model_params: Optional[dict] = None,
            estimator_params: Optional[dict] = None):
        """init function."""
        # Superclass init.
        super().__init__(
            score_func=score_func,
            coverage=coverage,
            null_model_params=null_model_params
        )

        self.estimator = estimator
        self.estimator_params = estimator_params
        self.forecast_horizon = forecast_horizon
        self.estimator_map = estimator_map

        # Set by `fit` method.
        self.estimator_map_list = None  # List of forecast horizons for each estimator
        self.estimator_param_names = None  # Estimator parameter names
        self.estimator_class = None  # Estimator class
        self.estimators = None  # Estimator instances
        self.model_results = None  # Estimator fit outputs
        self.pred_indices = None  # Indices to splice forecasts
        self.train_end_date = None  # Last training timestamp

        # Set by `predict` method.
        self.forecast = None

    def set_params(self, **params):
        """Overrides the set_params method in `sklearn.base.BaseEstimator`
        to recognize estimator parameters.

        The set_params in sklearn requires each parameter to appear explicitly
        in the __init__ function. However, we want the estimator to be able to
        set estimator parameters as well. The estimators are used to update
        ``self.estimator_params``.

        Parameters
        ----------
        **params : keyword arguments
            The parameters to be set to estimator class attributes.
        """
        if not params:
            return self
        self.estimator_param_names = list(
            ONEBYONE_ESTIMATORS[self.estimator]["class"]().get_params().keys())  # Estimator parameters
        # If ``estimator_params`` is provided, it will be used to update
        # the parameters first.
        if self.estimator_params is None:
            self.estimator_params = {}
        if "estimator_params" in params:
            self.estimator_params.update(params["estimator_params"])
            del params["estimator_params"]
        for key, value in params.items():
            if key in ["forecast_horizon", "estimator", "estimator_map", "score_func", "coverage", "null_model_params"]:
                # Class level parameters.
                setattr(self, key, value)
            elif key in self.estimator_param_names:
                # Estimator level parameters.
                self.estimator_params[key] = value
            else:
                # KeyError, copied from sklearn error.
                raise ValueError(f"Invalid parameter {key} for estimator OneByOneEstimator. "
                                 f"Check the list of available parameters "
                                 f"with `estimator.get_params().keys()`.")
        return self

    def _get_estimators(self):
        """Gets the estimators for forecast one-by-one.

        If the given parameters indicate that multiple estimators are need
        for the forecast one-by-one algorithm, these estimators with proper
        parameters are initialized.

        Sets ``self.estimator_class``, ``self.estimators``, ``self.pred_indices``
        and ``self.estimator_map_list``.
        """
        # Only estimators in ``ONEBYONE_ESTIMATORS`` supports forecast one-by-one.
        if self.estimator not in ONEBYONE_ESTIMATORS:
            raise ValueError(f"Estimator {self.estimator} does not support forecast"
                             f" one-by-one.")

        self.estimator_class = ONEBYONE_ESTIMATORS[self.estimator]["class"]
        if self.estimator_params is None:
            self.estimator_params = {}

        # Sets estimator base parameters, so the prediction confidence intervals can be pulled.
        if "score_func" not in self.estimator_params:
            self.estimator_params["score_func"] = self.score_func
        if "coverage" not in self.estimator_params:
            self.estimator_params["coverage"] = self.coverage
        if "null_model_params" not in self.estimator_params:
            self.estimator_params["null_model_params"] = self.null_model_params

        # Checks if any provided parameters depend on forecast horizon.
        params_depending_on_horizon = ONEBYONE_ESTIMATORS[self.estimator]["params_depending_on_horizon"]
        depending_on_horizon = False
        if params_depending_on_horizon is not None:
            for param, values in params_depending_on_horizon.items():
                if param in self.estimator_params:
                    input_value = self.estimator_params.get(param)
                    if input_value in values:
                        depending_on_horizon = True
        if not depending_on_horizon:
            log_message(
                message="No parameters depending on forecast horizon found. "
                        "Forecast one-by-one is not activated.",
                level=LoggingLevelEnum.INFO)

        # Checks if forecast horizon is a parameter in ``estimator_params``.
        # Forecast horizon should be different for different models.
        # If forecast horizon is a parameter, it need to be removed.
        # It will be added back differently for each estimator.
        forecast_horizon_param = ONEBYONE_ESTIMATORS[self.estimator]["forecast_horizon_param"]
        if depending_on_horizon:
            if forecast_horizon_param in self.estimator_params:
                del self.estimator_params[forecast_horizon_param]

        # Initializes estimator instances.
        if depending_on_horizon and self.estimator_map is not False:
            if self.estimator_map is None or self.estimator_map is True:
                self.estimator_map_list = [1] * self.forecast_horizon
            elif isinstance(self.estimator_map, int):
                estimator_map = [self.estimator_map for _ in range(self.forecast_horizon // self.estimator_map)]
                if self.forecast_horizon % self.estimator_map:
                    estimator_map.append(self.forecast_horizon % self.estimator_map)
                self.estimator_map_list = estimator_map
            else:
                if sum(self.estimator_map) != self.forecast_horizon:
                    raise ValueError(
                        "Sum of forecast one by one estimator map must equal to forecast horizon.")
                self.estimator_map_list = deepcopy(self.estimator_map)
            self.estimators = []
            self.pred_indices = [0]
            current_horizon = 0
            for i in self.estimator_map_list:
                current_horizon += i
                self.estimator_params[forecast_horizon_param] = current_horizon
                self.estimators.append(deepcopy(self.estimator_class(**self.estimator_params)))
                self.pred_indices.append(current_horizon)
        else:
            self.estimator_map_list = [self.forecast_horizon]
            if forecast_horizon_param is not None:
                self.estimator_params[forecast_horizon_param] = self.forecast_horizon
            self.estimators = [deepcopy(self.estimator_class(**self.estimator_params))]

    def fit(
            self,
            X,
            y=None,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            **fit_params):
        """Fits a model to training data.

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
            Additional parameters supported by subclass ``fit`` or null model.
        """
        super().fit(
            X=X,
            y=y,
            time_col=time_col,
            value_col=value_col,
            **fit_params)
        self._get_estimators()
        self.train_end_date = pd.to_datetime(X[time_col]).max()
        self.model_results = [
            estimator.fit(
                X=X,
                y=y,
                time_col=time_col,
                value_col=value_col)
            for estimator in self.estimators
        ]
        return self

    def predict(self, X, y=None):
        """Creates forecast for dates specified in X

        The forecast one-by-one is supposed to forecast for the specified forecast horizon
        with the specified estimator-horizon mapping. If the size of ``X`` is different from
        forecast horizon, only the last model (trained with the longest forecast horizon)
        will be used.

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
        # Returns the cached result if applicable
        cached_predictions = super().predict(X=X)
        if cached_predictions is not None:
            return cached_predictions

        # Only one model.
        if len(self.estimators) == 1:
            return self.estimators[0].predict(X=X)

        # Forecast one-by-one is forecast-horizon-sensitive.
        # Checks future forecast horizon length to decide
        # how to make forecasts.
        is_future = pd.to_datetime(X[self.time_col_]) > self.train_end_date
        x_future = X.loc[is_future]

        # If the future prediction length is different from the forecast horizon,
        # use the last model only.
        if len(x_future) != self.forecast_horizon:
            log_message(
                message=f"The future x length is {len(x_future)}, "
                        f"which doesn't match the model forecast horizon {self.forecast_horizon}, "
                        f"using only the model with the longest forecast horizon for prediction.",
                level=LoggingLevelEnum.WARNING)
            return self.estimators[-1].predict(X)

        # The past df is always forecasted with the last estimator.
        past_prediction = None
        if not is_future.all():
            past_prediction = self.estimators[-1].predict(X.loc[~is_future])

        # From now on assume ``x_future`` is exactly the forecast horizon.
        # Makes predictions according to the estimator map.
        predictions = [
            estimator.predict(x_future.iloc[self.pred_indices[i]: self.pred_indices[i+1]])
            for i, estimator in enumerate(self.estimators)
        ]
        predictions = pd.concat([past_prediction] + predictions, axis=0).reset_index(drop=True)
        self.forecast = predictions

        return predictions

    def summary(self):
        """Returns the model summary.

        If only one estimator is used, its model summary is returned.
        If multiple estimators are used, a list of their summaries are returned.

        Returns
        -------
        summaries : `list` [`class`] or `class`
            The model summary(ies).
        """
        try:
            summaries = [estimator.summary() for estimator in self.estimators]
        except AttributeError:
            summaries = self.estimator
        return summaries

    def plot_components(self):
        """Makes the component plots.

        If only one estimator is used, its component plot is returned.
        If multiple estimators are used, a list of their component plots are returned.

        Returns
        -------
        component_plots : `list` [`dict`]
            The component plot(s).
        """
        try:
            fig = [estimator.plot_components() for estimator in self.estimators]
        except AttributeError:
            raise AttributeError(f"Estimator {self.estimator} does not support component plots")
        return fig
