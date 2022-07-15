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
# original author: Sayan Patra


from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from pmdarima.arima import AutoARIMA
from sklearn.metrics import mean_squared_error

from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator


class AutoArimaEstimator(BaseForecastEstimator):
    """Wrapper for ``pmdarima.arima.AutoARIMA``.
    It currently does not handle the regressor issue when there is
    gap between train and predict periods.

    Parameters
    ----------
    score_func : callable
        see ``BaseForecastEstimator``.
    coverage : float between [0.0, 1.0]
        see ``BaseForecastEstimator``.
    null_model_params : dict with arguments to define DummyRegressor null model, optional, default=None
        see ``BaseForecastEstimator``.
    regressor_cols: `list` [`str`], optional, default None
        A list of regressor columns used during training and prediction.
        If None, no regressor columns are used.

    See ``AutoArima`` documentation for rest of the parameter descriptions:

            * https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html#pmdarima.arima.AutoARIMA

    Attributes
    ----------
    model : ``AutoArima`` object
        Auto arima model object
    fit_df : `pandas.DataFrame` or None
        The training data used to fit the model.
    forecast : `pandas.DataFrame`
        Output of the predict method of ``AutoArima``.
    """
    def __init__(
            self,
            # Null model parameters
            score_func: callable = mean_squared_error,
            coverage: float = 0.90,
            null_model_params: Optional[Dict] = None,
            # Additional parameters
            regressor_cols: Optional[List[str]] = None,
            freq: Optional[float] = None,
            # pmdarima fit parameters
            start_p: Optional[int] = 2,
            d: Optional[int] = None,
            start_q: Optional[int] = 2,
            max_p: Optional[int] = 5,
            max_d: Optional[int] = 2,
            max_q: Optional[int] = 5,
            start_P: Optional[int] = 1,
            D: Optional[int] = None,
            start_Q: Optional[int] = 1,
            max_P: Optional[int] = 2,
            max_D: Optional[int] = 1,
            max_Q: Optional[int] = 2,
            max_order: Optional[int] = 5,
            m: Optional[int] = 1,
            seasonal: Optional[bool] = True,
            stationary: Optional[bool] = False,
            information_criterion: Optional[str] = 'aic',
            alpha: Optional[int] = 0.05,
            test: Optional[str] = 'kpss',
            seasonal_test: Optional[str] = 'ocsb',
            stepwise: Optional[bool] = True,
            n_jobs: Optional[int] = 1,
            start_params: Optional[Dict] = None,
            trend: Optional[str] = None,
            method: Optional[str] = 'lbfgs',
            maxiter: Optional[int] = 50,
            offset_test_args: Optional[Dict] = None,
            seasonal_test_args: Optional[Dict] = None,
            suppress_warnings: Optional[bool] = True,
            error_action: Optional[str] = 'trace',
            trace: Optional[Union[int, bool]] = False,
            random: Optional[bool] = False,
            random_state: Optional[Union[int, callable]] = None,
            n_fits: Optional[int] = 10,
            out_of_sample_size: Optional[int] = 0,
            scoring: Optional[str] = 'mse',
            scoring_args: Optional[Dict] = None,
            with_intercept: Optional[Union[bool, str]] = "auto",
            # pmdarima predict parameters
            return_conf_int: Optional[bool] = True,
            dynamic: Optional[bool] = False):
        # Every subclass of BaseForecastEstimator must call super().__init__
        super().__init__(
            score_func=score_func,
            coverage=coverage,
            null_model_params=null_model_params)
        self.regressor_cols = regressor_cols
        self.freq = freq
        self.start_p = start_p
        self.d = d
        self.start_q = start_q
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.start_P = start_P
        self.D = D
        self.start_Q = start_Q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.max_order = max_order
        self.m = m
        self.seasonal = seasonal
        self.stationary = stationary
        self.information_criterion = information_criterion
        self.alpha = alpha
        self.test = test
        self.seasonal_test = seasonal_test
        self.stepwise = stepwise
        self.n_jobs = n_jobs
        self.start_params = start_params
        self.trend = trend
        self.method = method
        self.maxiter = maxiter
        self.offset_test_args = offset_test_args
        self.seasonal_test_args = seasonal_test_args
        self.suppress_warnings = suppress_warnings
        self.error_action = error_action
        self.trace = trace
        self.random = random
        self.random_state = random_state
        self.n_fits = n_fits
        self.out_of_sample_size = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = scoring_args
        self.with_intercept = with_intercept
        self.return_conf_int = return_conf_int
        self.coverage = coverage
        self.dynamic = dynamic

        # set by the fit method
        self.model = None
        self.fit_df = None
        # set by the predict method
        self.forecast = None

    def fit(self, X, y=None, time_col=TIME_COL, value_col=VALUE_COL, **fit_params):
        """Fits ``ARIMA`` forecast model.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Input timeseries, with timestamp column,
            value column, and any additional regressors.
            The value column is the response, included in
            X to allow transformation by `sklearn.pipeline.Pipeline`
        y : ignored
            The original timeseries values, ignored.
            (The y for fitting is included in ``X``.)
        time_col : `str`
            Time column name in ``X``
        value_col : `str`
            Value column name in ``X``
        fit_params : `dict`
            additional parameters for null model
        Returns
        -------
        self : self
            Fitted model is stored in ``self.model``.
        """
        X = X.sort_values(by=time_col)
        # fits null model
        super().fit(X, y=y, time_col=time_col, value_col=value_col, **fit_params)

        self.fit_df = X
        # fits AutoArima model
        self.model = AutoARIMA(
            start_p=self.start_p,
            d=self.d,
            start_q=self.start_q,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            start_P=self.start_P,
            D=self.D,
            start_Q=self.start_Q,
            max_P=self.max_P,
            max_D=self.max_D,
            max_Q=self.max_Q,
            max_order=self.max_order,
            m=self.m,
            seasonal=self.seasonal,
            stationary=self.stationary,
            information_criterion=self.information_criterion,
            alpha=self.alpha,
            test=self.test,
            seasonal_test=self.seasonal_test,
            stepwise=self.stepwise,
            n_jobs=self.n_jobs,
            start_params=self.start_params,
            trend=self.trend,
            method=self.method,
            maxiter=self.maxiter,
            offset_test_args=self.offset_test_args,
            seasonal_test_args=self.seasonal_test_args,
            suppress_warnings=self.suppress_warnings,
            error_action=self.error_action,
            trace=self.trace,
            random=self.random,
            random_state=self.random_state,
            n_fits=self.n_fits,
            out_of_sample_size=self.out_of_sample_size,
            scoring=self.scoring,
            scoring_args=self.scoring_args,
            with_intercept=self.with_intercept,
            return_conf_int=self.return_conf_int,
            dynamic=self.dynamic,
            regressor_cols=self.regressor_cols
        )

        # fits auto-arima
        if self.regressor_cols is None:
            reg_df = None
        else:
            reg_df = X[self.regressor_cols]
        self.model.fit(y=X[[value_col]], X=reg_df)

        return self

    def predict(self, X, y=None):
        """Creates forecast for the dates specified in ``X``.
        Currently does not support the regressor case where there is gap between
        train and predict periods.

        Parameters
        ----------
        X: `pandas.DataFrame`
            Input timeseries with timestamp column and any additional regressors.
            Timestamps are the dates for prediction.
            Value column, if provided in ``X``, is ignored.
        y: ignored.

        Returns
        -------
        predictions: `pandas.DataFrame`
            Forecasted values for the dates in ``X``. Columns:

                - ``TIME_COL``: dates
                - ``PREDICTED_COL``: predictions
                - ``PREDICTED_LOWER_COL``: lower bound of predictions
                - ``PREDICTED_UPPER_COL``: upper bound of predictions
        """
        X = X.sort_values(by=self.time_col_)
        # Returns the cached result if applicable
        cached_predictions = super().predict(X=X)
        if cached_predictions is not None:
            return cached_predictions

        # Currently does not support the regressor case where
        # there is gap between train and predict periods
        if self.regressor_cols is None:
            fut_reg_df = None
        else:
            fut_df = X[X[self.time_col_] > self.fit_df[self.time_col_].iloc[-1]]
            fut_reg_df = fut_df[self.regressor_cols]  # Auto-arima only accepts regressor values beyond `fit_df`

        if self.freq is None:
            self.freq = pd.infer_freq(self.fit_df[self.time_col_])
        if self.freq == "MS":
            timedelta_freq = "M"  # `to_period` does not recognize non-traditional frequencies
        else:
            timedelta_freq = self.freq
        chosen_d = self.model.model_.order[1]  # This is the value of the d chosen by auto-arima
        forecast_start = (X[self.time_col_].iloc[0].to_period(timedelta_freq) - self.fit_df[self.time_col_].iloc[0].to_period(timedelta_freq)).n
        if forecast_start < chosen_d:
            append_length = chosen_d - forecast_start  # Number of NaNs to append to `pred_df`
            forecast_start = chosen_d  # Auto-arima can not predict below the chosen d
        else:
            append_length = 0
        forecast_end = (X[self.time_col_].iloc[-1].to_period(timedelta_freq) - self.fit_df[self.time_col_].iloc[0].to_period(timedelta_freq)).n

        predictions = self.model.predict_in_sample(
            X=fut_reg_df,
            start=forecast_start,
            end=forecast_end,
            dynamic=self.dynamic,
            return_conf_int=self.return_conf_int,
            alpha=(1-self.coverage)
        )

        if append_length > 0:
            pred_df = pd.DataFrame({
                TIME_COL: X[self.time_col_],
                PREDICTED_COL: np.append(np.repeat(np.nan, append_length), predictions[0]),
                PREDICTED_LOWER_COL: np.append(np.repeat(np.nan, append_length), predictions[1][:, 0]),
                PREDICTED_UPPER_COL: np.append(np.repeat(np.nan, append_length), predictions[1][:, 1])
            })
        else:
            pred_df = pd.DataFrame({
                TIME_COL: X[self.time_col_],
                PREDICTED_COL: predictions[0],
                PREDICTED_LOWER_COL: predictions[1][:, 0],
                PREDICTED_UPPER_COL: predictions[1][:, 1]
            })
        self.forecast = pred_df

        # Caches the predictions
        self.cached_predictions_ = pred_df

        return pred_df

    def summary(self):
        BaseForecastEstimator.summary(self)
        # AutoArima summary
        return self.model.summary()
