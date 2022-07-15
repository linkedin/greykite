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
"""Lag based estimator.
Uses past observations with aggregation function as predictions.
"""

from enum import Enum
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error

from greykite.common import constants as cst
from greykite.common.aggregation_function_enum import AggregationFunctionEnum
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.time_properties import fill_missing_dates
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator


class LagUnitEnum(Enum):
    """Defines the lag units available in
    `~greykite.sklearn.estimator.lag_based_estimator.LagBasedEstimator`.
    The keys are available string names and the values are the corresponding
    `dateutil.relativedelta.relativedelta` objects.
    """
    minute = relativedelta(minutes=1)
    hour = relativedelta(hours=1)
    day = relativedelta(days=1)
    week = relativedelta(weeks=1)
    month = relativedelta(months=1)
    year = relativedelta(years=1)


class LagBasedEstimator(BaseForecastEstimator):
    """The lag based estimator, using lagged observations with aggregation functions
    to forecast the future. This estimator includes the common week-over-week estimation method.

    The algorithm support specifying the following:

        lag_unit : the unit to calculate lagged values. One of the values in
            `~greykite.sklearn.estimator.lag_based_estimator.LagUnitEnum`.
        lags : a list of lags indicating which lagged ``lag_unit`` data are used in prediction.
            For example, [1, 2] indicating using the past two ``lag_unit`` same time data.
        agg_func : the aggregation function used over the lagged observations.
        agg_func_params : extra parameters used for ``agg_func``.


    When certain lags are not available, extra data will be extrapolated.
    When predicting into the future and future data is not available,
    predicted values will be used.

    Parameters
    ----------
    freq : `str` or None, default None
        The data frequency, used to validate lags.
    lag_unit : `str`, default "week"
        The unit to calculate lagged observations.
        Available options are in `~greykite.sklearn.estimator.lag_based_estimator.LagUnitEnum`.
    lags : `list` [`int`] or None, default None
        The lags in ``lag_unit``'s. [1, 2] indicates using the past two ``lag_unit`` same time values.
        If not provided, the default is to use lag 1 observation only.
    agg_func : `str` or callable, default "mean"
        The aggregation functions used over lagged observations.
    agg_func_params : `dict` or None, default None
        Extra parameters used for ``agg_func``.
    uncertainty_dict: `dict` or None, default None
        How to fit the uncertainty model.
        See `~greykite.sklearn.uncertainty.uncertainty_methods.UncertaintyMethodEnum`.
        If not provided but ``coverage`` is given, this falls back to
        `~greykite.sklearn.uncertainty.simple_conditional_residuals_model.SimpleConditionalResidualsModel`.
    past_df : `pandas.DataFrame` or None, default None
        The past data used to append to the training data.
        If not provided the past data needed will be interpolated.
    series_na_fill_func : `callable` or None, default `lambda s: s.bfill().ffill()`
        The function to fill NAs when they exist.

    Attributes
    ----------
    df : `pandas.DataFrame` or None
        The fitted and interpolated training data.
    uncertainty_model : any or None
        The trained uncertainty model.
    max_lag_order : `int` or None
        The maximum lag order.
    min_lag_order : `int` or None
        The minimum lag order.
    train_start : `pandas.Timestamp` or None
        The training start timestamp.
    train_end : `pandas.Timestamp` or None
        The training end timestamp.
    """

    def __init__(
            self,
            score_func=mean_squared_error,
            coverage=None,
            null_model_params=None,
            freq: Optional[str] = None,
            lag_unit: str = "week",
            lags: Optional[Union[int, List[int]]] = None,
            agg_func: Union[str, callable] = "mean",
            agg_func_params: Optional[dict] = None,
            uncertainty_dict: Optional[dict] = None,
            past_df: Optional[pd.DataFrame] = None,
            series_na_fill_func: Optional[callable] = None):
        super().__init__(
            score_func=score_func,
            coverage=coverage,
            null_model_params=null_model_params
        )
        self.freq = freq
        self.lag_unit = lag_unit
        self.lags = lags
        self.agg_func = agg_func
        self.agg_func_params = agg_func_params
        self.uncertainty_dict = uncertainty_dict
        self.past_df = past_df
        self.series_na_fill_func = series_na_fill_func

        # set by ``fit`` method
        self.lag_unit_delta = None
        self.agg_func_wrapper = None
        self.df = None
        self.uncertainty_model = None
        self.max_lag_order = None
        self.min_lag_order = None
        self.train_start = None
        self.train_end = None

    def fit(
            self,
            X,
            y=None,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            **fit_params):
        """Fits the lag based forecast model.

        Parameters
        ----------
        X: `pandas.DataFrame`
            Input timeseries, with timestamp column and value column.
            The value column is the response, included in
            ``X`` to allow transformation by `sklearn.pipeline`.
        y: ignored
            The original timeseries values, ignored.
            (The ``y`` for fitting is included in ``X``).
        time_col: `str`
            Time column name in ``X``.
        value_col: `str`
            Value column name in ``X``.
        fit_params: `dict`
            additional parameters for null model.

        Returns
        -------
        self : self
            Fitted class instance.
        """
        super().fit(
            X=X,
            y=y,
            time_col=time_col,
            value_col=value_col,
            **fit_params
        )
        if X is None or len(X) == 0:
            raise ValueError("The input df is empty!")
        # Validates model parameters.
        self._validate_params()
        self.df = X[[time_col, value_col]].copy()
        # Prepares input data for prediction.
        self._prepare_df()
        # Fits the uncertainty model.
        if self.coverage is not None or self.uncertainty_dict is not None:
            fit_df = self._get_predictions(fut_df=X)
            self.fit_uncertainty(
                df=fit_df,
                uncertainty_dict=self.uncertainty_dict
            )
        return self

    def predict(self, X, y=None):
        """Creates forecast for the dates specified in ``X``.

        Parameters
        ----------
        X: `pandas.DataFrame`
            Input timeseries with timestamp column.
            Timestamps are the dates for prediction.
            Value column, if provided in ``X``, is ignored.
        y: ignored.

        Returns
        -------
        predictions: `pandas.DataFrame`
            Forecasted values for the dates in ``X``. Columns:

                - ``TIME_COL``: dates
                - ``PREDICTED_COL``: predictions
                - ``PREDICTED_LOWER_COL``: lower bound of predictions, optional
                - ``PREDICTED_UPPER_COL``: upper bound of predictions, optional

            ``PREDICTED_LOWER_COL`` and ``PREDICTED_UPPER_COL`` are present
            if ``self.coverage`` is not None.
        """
        if self.df is None:
            raise NotImplementedError("Please run ``fit`` first.")
        # Uses cached predictions if available.
        cached_predictions = super().predict(X=X)
        if cached_predictions is not None:
            return cached_predictions

        # Gets the predictions.
        pred_df = self._get_predictions(fut_df=X.copy())
        # Fits the uncertainty model.
        if self.uncertainty_model is not None:
            pred_with_uncertainty = self.predict_uncertainty(
                df=pred_df
            )
            if pred_with_uncertainty is not None:
                pred_df = pred_with_uncertainty
        self.cached_predictions_ = pred_df
        return pred_df

    def _validate_params(self):
        """Validates model parameters.

        This function checks the following class attributes:

            - self.lag_unit
            - self.lags
            - self.agg_func
            - self.agg_func_params

        """
        # Validates lags.
        try:
            self.lag_unit_delta = LagUnitEnum[self.lag_unit].value
        except KeyError:
            raise ValueError(f"The lag unit '{self.lag_unit}' is not recognized.")

        if self.lags is None:
            # If no customized lags is provided,
            # the default is 1.
            log_message(
                message="Lags not provided, setting lags = [1].",
                level=LoggingLevelEnum.DEBUG
            )
            self.lags = [1]
        if not isinstance(self.lags, list):
            raise ValueError(f"The lags must be a list of integers, found '{self.lags}'.")
        # All lags must be able to be converted to integers.
        try:
            self.lags = [int(x) for x in self.lags]
        except ValueError:
            raise ValueError(f"Not all lags in '{self.lags}' can be converted to integers.")
        self.max_lag_order = max(self.lags)
        self.min_lag_order = min(self.lags)
        if self.min_lag_order <= 0:
            raise ValueError("All lags must be positive integers.")

        # Validates aggregation function.
        if isinstance(self.agg_func, str):
            try:
                self.agg_func_wrapper = AggregationFunctionEnum[self.agg_func].value
            except KeyError:
                raise ValueError(f"The aggregation function '{self.agg_func}' is not recognized as a string. "
                                 f"Please either pass a known string or a function.")
        elif callable(self.agg_func):
            self.agg_func_wrapper = self.agg_func
        else:
            raise ValueError(f"The aggregation function must be a valid string or a callable.")
        if self.agg_func_params is None:
            self.agg_func_params = {}

    def _prepare_df(self):
        """Processes the training data. Including appending ``past_df``, interpolation, etc.

        Returns
        -------
        This function modifies class instance attributes and does not return anything.
        """
        self.df[self.time_col_] = pd.to_datetime(self.df[self.time_col_])
        self.df = self.df.sort_values(by=self.time_col_).reset_index(drop=True)
        # Infers data frequency.
        freq = pd.infer_freq(self.df[self.time_col_])
        if self.freq is None:
            self.freq = freq
        if freq is not None and self.freq != freq:
            log_message(
                message=f"The inferred frequency '{freq}' is different from the provided '{self.freq}'. "
                        f"Using the provided frequency.",
                level=LoggingLevelEnum.INFO
            )
        if self.freq is None:
            raise ValueError("Frequency can not be inferred. Please provide frequency.")
        # Currently "M" frequency is not handled well with `dateutil.relativedelta`.
        # For example, 2020-02-29 minus 1 month is 2020-01-29 instead of 2020-01-31,
        # which does not match the last data point.
        # Since this is not a typical use case, we will fix it later and raise a warning for now.
        # "MS" is preferred.
        if self.freq == "M":
            log_message(
                message="The data frequency is 'M' which may lead to unexpected behaviors. "
                        "Please convert to 'MS' if applicable.",
                level=LoggingLevelEnum.WARNING
            )
        # Checks ``self.lag_unit`` is at least ``self.freq``.
        example_ts = pd.date_range(
            start=self.df[self.time_col_].min(), freq=self.freq, periods=2)  # makes a ts with data frequency
        if example_ts[0] + self.lag_unit_delta < example_ts[1]:
            raise ValueError(
                f"The lag unit '{self.lag_unit}' must be at least equal to the data frequency '{self.freq}'.")
        # Gets the earliest timestamp needed.
        self.train_start = self.df[self.time_col_].iloc[0]
        self.train_end = self.df[self.time_col_].iloc[-1]
        earliest_timestamp = self.train_start - self.lag_unit_delta * self.max_lag_order
        df_earliest_timestamp = pd.DataFrame({
            self.time_col_: [earliest_timestamp],
            self.value_col_: [np.nan]
        })
        # Gets all timestamps needed.
        self.df = fill_missing_dates(
            df=pd.concat([df_earliest_timestamp, self.df], axis=0).reset_index(drop=True),
            time_col=self.time_col_,
            freq=self.freq
        )[0]
        # Takes values from ``past_df`` if exist.
        if self.past_df is not None and len(self.past_df) > 0:
            self.past_df[self.time_col_] = pd.to_datetime(self.past_df[self.time_col_])
            self.past_df = self.past_df.sort_values(by=self.time_col_).reset_index(drop=True)
            merge_df = self.df.merge(
                self.past_df[[self.time_col_, self.value_col_]].rename(
                    columns={self.value_col_: f"{self.value_col_}_p"}),
                on=self.time_col_,
                how="left"
            )
            self.df[self.value_col_] = merge_df[self.value_col_].combine_first(merge_df[f"{self.value_col_}_p"])
        # Fills missing values.
        if self.series_na_fill_func is not None:
            self.df[self.value_col_] = self.series_na_fill_func(self.df[self.value_col_])

    def _get_predictions(
            self,
            fut_df: pd.DataFrame):
        """Gets the lag based predictions.

        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The input data that contains ``self.time_col_`` as the prediction periods.

        Returns
        -------
        predictions: `pandas.DataFrame`
            Forecasted values for the dates in ``fut_df``. Columns:

                - ``TIME_COL``: dates
                - ``PREDICTED_COL``: predictions
                - ``PREDICTED_LOWER_COL``: lower bound of predictions, optional
                - ``PREDICTED_UPPER_COL``: upper bound of predictions, optional

            ``PREDICTED_LOWER_COL`` and ``PREDICTED_UPPER_COL`` are present
            if ``self.coverage`` is not None.
        """
        # Gets the full df from the earliest timestamp during training to the
        # last timestamp in prediction.
        pred_df = fut_df[[self.time_col_]].copy()
        pred_df[self.time_col_] = pd.to_datetime(pred_df[self.time_col_])
        pred_df = pred_df.sort_values(by=self.time_col_).reset_index(drop=True)
        if pred_df[self.time_col_].iloc[0] < self.train_start:
            # Does not predict before the training period.
            raise ValueError("The lag based estimator does not support hindcasting.")
        if pred_df[self.time_col_].iloc[-1] > self.train_end:
            df_end_timestamp = pd.DataFrame({
                self.time_col_: [pred_df[self.time_col_].iloc[-1]],
                self.value_col_: [np.nan]
            })
            # Fills the timestamps using frequency, in case the prediction df has wrong frequencies.
            df = pd.concat([self.df, df_end_timestamp], axis=0).reset_index(drop=True)
            df = fill_missing_dates(
                df=df,
                time_col=self.time_col_,
                freq=self.freq
            )[0]
        else:
            df = self.df.copy()

        # Gets the predicted values.
        df[cst.PREDICTED_COL] = np.nan
        # Gets the number of prediction rounds in case lag is not enough for forecast horizon.
        n_rounds = self._get_n_rounds(pred_df=pred_df)
        for _ in range(n_rounds):
            # The past timestamps needed in lags.
            # Uses time-based calculations to avoid messing up daylight saving times.
            past_timestamps = [df[self.time_col_].apply(lambda x: x - self.lag_unit_delta * lag) for lag in self.lags]
            # Gets the available past values.
            available_past = [
                df[df[self.time_col_].isin(timestamps)][self.value_col_].combine_first(
                    df[df[self.time_col_].isin(timestamps)][cst.PREDICTED_COL])
                for timestamps in past_timestamps
            ]
            # Fills NAs for the earliest unavailable timestamps to match the lengths.
            available_past = [
                pd.concat([pd.Series([np.nan] * (len(df) - len(past))), past], axis=0).reset_index(drop=True)
                for past in available_past
            ]
            # Makes a dataframe.
            available_past = pd.concat(available_past, axis=1)
            # Applies aggregation function for prediction.
            pred = self.agg_func_wrapper(
                available_past,
                axis=1,
                **self.agg_func_params
            )
            # Writes to ``df``.
            df[cst.PREDICTED_COL] = pred
        # Gets the predictions during the input periods.
        pred_df = df[df[self.time_col_].isin(pred_df[self.time_col_])].reset_index(drop=True)
        if len(pred_df) < len(fut_df):
            log_message(
                message=f"Some timestamps in the provided time periods for prediction do not match the "
                        f"training frequency. Returning the matched timestamps.",
                level=LoggingLevelEnum.WARNING
            )
        return pred_df

    def summary(self):
        """The summary of model."""
        super().summary()
        text = (f"This is a lag based forecast model that uses lags '{self.lags}', "
                f"with unit '{self.lag_unit}' and aggregation function '{self.agg_func}'.")
        log_message(
            message=text,
            level=LoggingLevelEnum.INFO
        )
        return text

    def _get_n_rounds(
            self,
            pred_df: pd.DataFrame):
        """Gets the number of rounds of predictions needed.

        For example, when the prediction is 3 weeks while the lag is 1 week,
        it will need to make prediction for the first week in the first round.
        Then it uses the predictions of the first week to predict the second week, etc.
        The number of rounds here is 3.

        If the prediction period is within training or at most ``self.lag_unit_delta`` * ``self.min_lag_order``
        more than the training period, then just one round is needed.

        Parameters
        ----------
        pred_df : `pandas.DataFrame`
            The prediction df with ``self.time_col_``.

        Returns
        -------
        n_rounds : `int`
            The number of rounds of predictions needed.
        """
        train_end = self.train_end
        step_size = self.lag_unit_delta * self.min_lag_order
        pred_end = pred_df[self.time_col_].max()
        n_rounds = 1
        while train_end + step_size < pred_end:
            train_end += step_size
            n_rounds += 1
        return n_rounds
