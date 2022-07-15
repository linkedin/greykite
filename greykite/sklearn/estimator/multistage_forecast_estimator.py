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
"""Multistage Forecast estimator."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.frequencies import to_offset
from sklearn.metrics import mean_squared_error

from greykite.common import constants as cst
from greykite.common.aggregation_function_enum import AggregationFunctionEnum
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator


@dataclass
class MultistageForecastModelConfig:
    """The dataclass to store Multistage Forecast model config for a single model.

    Attributes
    ----------
    train_length : `str`, default "392D"
        The length of data used for training. For example, "56D".
    fit_length : `str` or None, default None
        The length of data where fitted values to be calculated.
        Specify if ``fit_length`` is to be longer than ``train_length``.
    agg_func : str or Callable, default "nanmean"
        The aggregation function.
    agg_freq : `str` or None, default None
        The aggregation period. If None, no aggregation will be used.
    estimator : `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`,
        default `~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`
        The estimator to fit the time series.
    estimator_params : `dict` or None, default None
        The init parameters for ``estimator``.
        When the estimator is in the Silverkite family, the parameters shouldn't include
        ``forecast_horizon`` or ``past_df``, as they will be passed automatically.
    """
    train_length: str = "392D"  # 56 weeks
    fit_length: Optional[str] = None
    agg_func: Union[str, Callable] = "nanmean"
    agg_freq: Optional[str] = None
    estimator: Type[BaseForecastEstimator] = SimpleSilverkiteEstimator
    estimator_params: Optional[dict] = None


class MultistageForecastEstimator(BaseForecastEstimator):
    """The Multistage Forecast Estimator class.
    Implements the Multistage forecast method.

    The Multistage forecast method allows users to fit multiple stages of
    models with each stage in the following fashions:

        (1) subseting: take a subset of data from the end of training data;
        (2) aggregation: aggregate the subset of data into desired frequency;
        (3) training: train a model with the desired estimator and parameters.

    Users can just use one stage model to train on a subset/aggregation of the original data,
    or can specify multiple stages, where the later stages will be trained on the fitted
    residuals of the previous stages.

    This can significantly speed up the training process if the original data is long
    and in fine granularity.

    Notes
    -----
    The following assumptions or special implementations are made in this class:

        - The actual ``fit_length``, the length of data where the fitted values are calculated,
          is the longer of ``train_length`` and ``fit_length``. The reason is that there is no
          benefit of calculating a shorter period of fitted values. The fitted values are already
          available during training (in Silverkite) so there is no loss to calculate fitted values
          on a super set of the training data.
        - The estimator sorts the ``model_configs`` according to the ``train_length`` in descending order.
          The corresponding aggregation frequency, aggregation function, fit length,
          estimator and parameters will be sorted accordingly.
          This is to ensure that we have enough data to use from the previous model
          when we fit the next model.
        - When calculating the length of training data, the length of past df, etc,
          the actual length used may include 1 more period to avoid missing timestamps.
          For example, for an AR order of 5, you may see the length of ``past_df`` to be 6;
          or for a train length of "365D", you may see the actual length to be 366.
          This is expected, just to avoid potential missing timestamps after dropping
          incomplete aggregation periods.
        - Since the models in each stage may not fit on the entire training data,
          there could be periods where fitted values are not calculated.
          Leading fitted values in the training period may be NA.
          These values are ignored when computing evaluation metrics.

    Attributes
    ----------
    model_configs : `list` [`MultistageForecastModelConfig`]
        A list of model configs for Multistage Forecast estimator,
        representing the stages in the model.
    forecast_horizon : `int`
        The forecast horizon on the original data frequency.
    freq : `str` or None
        The frequency of the original data.
    train_lengths : `list` [`str`] or None
        A list of training data lengths for the models.
    fit_lengths : `list` [`str`] or None
        A list of fitting data lengths for the models.
    agg_funcs : `list` [`str` or Callable] or None
        A list of aggregation functions for the models.
    agg_freqs : `list` [`str`] or None
        A list of aggregation frequencies for the models.
    estimators : `list` [`BaseForecastEstimator`] or None
        A list of estimators used in the models.
    estimator_params : `list` [`dict` or None] or None
        A list of estimator parameters for the estimators.
    train_lengths_in_seconds : `list` [`int`] or None
        The list of training lengths in seconds.
    fit_lengths_in_seconds: : `list` [`int`] or None
        The list of fitting lengths in seconds.
        If the original ``fit_length`` is None or is shorter than the corresponding
        ``train_length``, it will be replaced by the corresponding ``train_length``.
    max_ar_orders : `list` [`int`] or None
        A list of maximum AR orders in the models.
    data_freq_in_seconds : `int` or None
        The data frequency in seconds.
    num_points_per_agg_freqs : `list` [`int`] or None
        Number of data points in each aggregation frequency.
    models : `list` [`BaseForecastEstimator`]
        The list of model instances.
    fit_df : `pandas.DataFrame` or None
        The prediction df.
    train_end : `pandas.Timestamp` or None
        The train end timestamp.
    forecast_horizons : `list` [`int`]
        The list of forecast horizons for all models in terms of the aggregated frequencies.
    """

    def __init__(
            self,
            model_configs: List[MultistageForecastModelConfig],
            forecast_horizon: int,
            freq: Optional[str] = None,
            uncertainty_dict: Optional[dict] = None,
            score_func: Callable = mean_squared_error,
            coverage: Optional[float] = None,
            null_model_params: Optional[dict] = None):
        """Instantiates the class.

        Parameters
        ----------
        model_configs : `list` [MultistageForecastModelConfig]
            A list of
            `~greykite.sklearn.estimator.multistage_forecast_estimator.MultistageModelConfig`
            objects. Defines the stages in the Multistage Forecast model.
        forecast_horizon : `int`
            The forecast horizon in the original data frequency.
        freq : `str` or None, default None
            The training data frequency.
            This parameter is important in Multistage Forecast model,
            since calculation of aggregation and dropping incomplete aggregated periods depends on this.
            If None, the model will try to infer it from data.
            If inferring from data failed, the model fit will raise an error.
        uncertainty_dict: `dict` or None, default None
            How to fit the uncertainty model.
            See `~greykite.sklearn.uncertainty.uncertainty_methods.UncertaintyMethodEnum`.
            If not provided but ``coverage`` is given, this falls back to
            `~greykite.sklearn.uncertainty.simple_conditional_residuals_model.SimpleConditionalResidualsModel`.
        """
        # every subclass of BaseForecastEstimator must call super().__init__
        super().__init__(
            score_func=score_func,
            coverage=coverage,
            null_model_params=null_model_params)

        self.model_configs: List[MultistageForecastModelConfig] = model_configs
        self.forecast_horizon: int = forecast_horizon
        self.freq: Optional[str] = freq
        self.uncertainty_dict = uncertainty_dict

        # Derived from ``self.model_configs``.
        self.train_lengths: Optional[List[str]] = None
        self.fit_lengths: Optional[List[Optional[str]]] = None
        self.agg_funcs: Optional[List[Union[str, Callable]]] = None
        self.agg_freqs: Optional[List[str]] = None
        self.estimators: Optional[List[Type[BaseForecastEstimator]]] = None
        self.estimator_params: Optional[List[Optional[dict]]] = None
        self.train_lengths_in_seconds: Optional[List[int]] = None
        self.fit_lengths_in_seconds: Optional[List[int]] = None

        # Set by ``fit`` method.
        self.max_ar_orders: Optional[List[int]] = None
        self.data_freq_in_seconds: Optional[int] = None
        self.num_points_per_agg_freqs: Optional[List[int]] = None
        self.models: Optional[List[BaseForecastEstimator]] = None
        self.fit_df: Optional[pd.DataFrame] = None
        self.train_end: Optional[pd.Timestamp] = None
        self.forecast_horizons: Optional[List[int]] = None

    def fit(
            self,
            X,
            y=None,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            **fit_params):
        """Fits ``MultistageForecast`` forecast model.

        Parameters
        ----------
        X: `pandas.DataFrame`
            Input timeseries, with timestamp column,
            value column, and any additional regressors.
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
            Fitted model is stored in ``self.model_dict``.
        """
        # Fits null model
        super().fit(
            X=X,
            y=y,
            time_col=time_col,
            value_col=value_col,
            **fit_params)
        if self.freq is None:
            self.freq = pd.infer_freq(X[time_col])
        if self.freq is None:
            raise ValueError("Failed to infer frequency from data, please provide during "
                             "instantiation. Data frequency is required for aggregation.")

        self._initialize()

        # Gets the forecast horizons for all models.
        # For each model, the forecast horizon is the length of the aggregated test df.
        self.forecast_horizons = []
        for agg_freq in self.agg_freqs:
            # Constructs a sample prediction df with the current freq and forecast horizon.
            sample_df = pd.DataFrame({
                time_col: pd.date_range(X[time_col].max(), freq=self.freq, periods=self.forecast_horizon + 1)[1:],
                value_col: 0
            })
            sample_df_agg = sample_df.resample(agg_freq, on=time_col).mean()  # The aggregation function is not needed.
            # The forecast horizon for the current model is the length of the aggregated df.
            # The forecast horizon differ when the aggregation covers various periods of the aggregation frequency.
            # For example, if the prediction period is 2020-01-01 23:00, 2020-01-02 00:00, 2020-01-02 01:00,
            # and the aggregation frequency is "D", although the length of prediction is less than a day,
            # but after aggregation, it will become 2020-01-01 and 2020-01-02.
            # On the other hand, if the prediction period is 2020-01-01 21:00, 2020-01-01 22:00, 2020-01-01 23:00,
            # and the aggregation frequency is "D", then after aggregation, it will be 2020-01-01 only.
            # In each stage of model, the model will get the appropriate forecast horizon.
            self.forecast_horizons.append(sample_df_agg.shape[0])

        min_agg_freq = min([to_offset(freq) for freq in self.agg_freqs])
        if min_agg_freq < to_offset(self.freq):
            raise ValueError(f"The minimum aggregation frequency {min_agg_freq} "
                             f"is less than the data frequency {self.freq}. Please make sure "
                             f"the aggregation frequencies are at least the data frequency.")

        self.train_end = X[time_col].max()

        # Trains the model.
        fit_df = self._train(df=X)
        self.fit_df = fit_df

        # Fits the uncertainty model
        self._fit_uncertainty()

        return self

    def predict(self, X, y=None):
        """Creates forecast for the dates specified in ``X``.

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
                - ``PREDICTED_LOWER_COL``: lower bound of predictions, optional
                - ``PREDICTED_UPPER_COL``: upper bound of predictions, optional

            ``PREDICTED_LOWER_COL`` and ``PREDICTED_UPPER_COL`` are present
            if ``self.coverage`` is not None.
        """
        pred = self._predict(X)
        if self.uncertainty_model is not None:
            pred_with_uncertainty = self.predict_uncertainty(
                df=pred
            )
            if pred_with_uncertainty is not None:
                pred = pred_with_uncertainty
        return pred

    def _initialize(self):
        """Sets the derived attributes from model init parameters."""
        self.train_lengths: List[str] = [config.train_length for config in self.model_configs]
        self.fit_lengths: List[Optional[str]] = [config.fit_length for config in self.model_configs]
        self.agg_funcs: List[Union[str, Callable]] = [
            self._get_agg_func(config.agg_func) for config in self.model_configs]
        self.agg_freqs: List[str] = [
            config.agg_freq if config.agg_freq is not None else self.freq for config in self.model_configs]
        self.estimators: List[Type[BaseForecastEstimator]] = [config.estimator for config in self.model_configs]
        self.estimator_params: List[Optional[dict]] = [config.estimator_params for config in self.model_configs]
        # Assumes train length is integer multiples of 1 second, which is most of the cases.
        self.train_lengths_in_seconds: List[int] = [
            to_offset(length).delta // timedelta(seconds=1) for length in self.train_lengths]
        self.fit_lengths_in_seconds: List[int] = [
            to_offset(length).delta // timedelta(seconds=1)
            if length is not None else None for length in self.fit_lengths]
        # If ``fit_length`` is None or is shorter than ``train_length``, it will be replaced by ``train_length``.
        fit_lengths_in_seconds = [
            fit_length if fit_length is not None and fit_length >= train_length
            else train_length
            for fit_length, train_length in zip(self.fit_lengths_in_seconds, self.train_lengths_in_seconds)
        ]
        if fit_lengths_in_seconds != self.fit_lengths_in_seconds:
            self.fit_lengths_in_seconds = fit_lengths_in_seconds
            log_message(
                message="Some `fit_length` is None or is shorter than `train_length`. "
                        "These `fit_length` have been replaced with `train_length`.",
                level=LoggingLevelEnum.INFO
            )
        self.models: List[BaseForecastEstimator] = [
            config.estimator(**config.estimator_params) for config in self.model_configs]
        self.data_freq_in_seconds = to_offset(self.freq).delta // timedelta(seconds=1)

    @staticmethod
    def _get_agg_func(agg_func: Optional[Union[str, Callable]]):
        """Gets the aggregation function.

        Returns the input if it's None or a callable.
        Finds the corresponding callable from
        `~greykite.common.aggregation_function_enum.AggregationFunctionEnum`
        and raises an error if no corresponding aggregation function is found.

        Parameters
        ----------
        agg_func : `str`, Callable or None
            The input of aggregation function.

        Returns
        -------
        agg_func : Callable
            The corresponding aggregation function if input is a string otherwise the input itself.
        """
        if not isinstance(agg_func, str):
            return agg_func
        try:
            agg_func = AggregationFunctionEnum[agg_func].value
            return agg_func
        except KeyError:
            raise ValueError(f"The aggregation function {agg_func} is not recognized as a string. "
                             f"Please either pass a known string or a function.")

    @staticmethod
    def _get_num_points_per_agg_freq(
            data_freq: str,
            agg_freqs: List[str]):
        """Gets the number of data points in a aggregation period.

        Parameters
        ----------
        data_freq : `str`
            The data frequency. For example, "5T".
        agg_freqs : `list` [`str`]
            A list of aggregation frequencies.

        Returns
        -------
        num_points : `list` [`int`]
            The number of points in each aggregation period.
        """
        return [to_offset(freq).delta // to_offset(data_freq).delta for freq in agg_freqs]

    def _get_freq_col(self, index: int, freq: str):
        """Gets the column name for a specific stage/frequency.
        The name will be f"{self.time_col_}__{index}__{freq}".

        Parameters
        ----------
        index : `int`
            The index for the currently stage of model.
            This is used to distinguish models with the same frequency.
        freq : `str`
            The aggregation frequency.

        Returns
        -------
        freq_col_name : `str`
            The time column name for the frequency, f"{self.time_col_}__{index}__{freq}".
        """
        return f"{self.time_col_}__{index}__{freq}"

    def _get_non_time_cols(self, columns: List[str]):
        """Gets the non time columns in a df.
        Non time columns do not have f"{self.time_col}__" in it or do not equal to self.time_col.

        Parameters
        ----------
        columns : `list` [`str`]
            The columns in a df.

        Returns
        -------
        non_time_columns : `list` [`str`]
            The non time columns.
            Non time columns do not have f"{self.time_col}__" in it or do not equal to self.time_col.
        """
        return [col for col in columns if f"{self.time_col_}__" not in col and col != self.time_col_]

    def _add_agg_freq_cols(
            self,
            df: pd.DataFrame):
        """Appends the resample time columns to ``df``.

        For example, the original df has hourly data with columns "ts" and "y".
        The original time column looks like

            "2020-01-01 00:00:00, 2020-01-01 01:00:00, 2020-01-01 02:00:00,
             2020-01-01 03:00:00, 2020-01-01 04:00:00, 2020-01-01 05:00:00,
             2020-01-01 06:00:00, 2020-01-01 07:00:00, 2020-01-01 08:00:00..."

        The resample frequencies are ["3H", "D"].
        The function adds two extra columns to ``df`` with names "ts__0__3H" and "ts__1__D".
        The "ts__0__3H" will have the same value for every 3 hours, such as

            "2020-01-01 00:00:00, 2020-01-01 00:00:00, 2020-01-01 00:00:00,
             2020-01-01 03:00:00, 2020-01-01 03:00:00, 2020-01-01 03:00:00,
             2020-01-01 06:00:00, 2020-01-01 06:00:00, 2020-01-01 06:00:00..."

         and "ts__1__D" will have the same value for every day, such as

            "2020-01-01 00:00:00, 2020-01-01 00:00:00, ...
             ...
             2020-01-01 00:00:00, 2020-01-01 00:00:00, 2020-01-01 00:00:00,
             2020-01-02 00:00:00, 2020-01-02 00:00:00, 2020-01-02 00:00:00..."

        Parameters
        ----------
        df : `pandas.DataFrame`
            The original data frame.

        Returns
        -------
        df : `pandas.DataFrame`
            The augmented df with resampled time columns.
        """
        # Original df has ``self.time_col_`` as the original time column.
        df = df.copy()
        df[self.time_col_] = pd.to_datetime(df[self.time_col_])

        for i, freq in enumerate(self.agg_freqs):
            col = self._get_freq_col(i, freq)  # New column name for resampled time column.
            # Borrows the value column for aggregation.
            df_time = df[[self.time_col_, self.value_col_]].set_index(self.time_col_, drop=False)

            if len(df_time) == 0:
                raise ValueError(f"The df size is zero. Does your input have NANs that are dropped?")

            # Gets the resampled frequency column.
            # This solution is fast and no need to further improve.
            df_time = (df_time
                       .resample(freq, on=self.time_col_)  # resample, the index is resampled time column
                       .mean()  # this function doesn't matter
                       .reset_index(drop=False)  # adds the resampled time column to columns
                       .set_index(self.time_col_, drop=False)  # copies the resampled time column to index
                       .reindex(df_time.index, method='ffill')  # sets the original freq as index, fill resampled column
                       .rename(columns={self.time_col_: col})  # renames the filled resampled column
                       .reset_index(drop=False))  # copies the original freq time column to columns

            # Merges new resampled frequency column into original df.
            df = df.merge(df_time[[self.time_col_, col]], on=self.time_col_)

        return df

    def _drop_incomplete_agg(
            self,
            df: pd.DataFrame,
            agg_freq: str,
            location: int,
            num_points_per_agg_freq: int,
            index: int):
        """Drops aggregations with incomplete periods.

        For example, a daily aggregation of hourly data will have a biased aggregation if the data starts from
        07:00:00, because the first day will be the aggregation of 07:00:00 to 23:00:00. This is not
        representative and should be dropped.

        The returned df's indices are reset.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The input dataframe with augmented aggregation frequency time columns.
        agg_freq : `str`
            The aggregation frequency.
        location : `int`
            Where to drop the incomplete aggregation periods.
            Usually the incomplete aggregation periods happen at the begin and end of the df.
            Specify location = 0 indicates the start, and specify location = -1 indicates the end.
        num_points_per_agg_freq : `int`
            The number of rows expected in a full period.
        index : `int`
            The index for the currently stage of model.
            This is used to distinguish models with the same frequency.

        Returns
        -------
        df : `pandas.DataFrame`
            The dataframe after dropping incomplete aggregation periods.
            The df's indices are reset.
        """
        if df.shape[0] == 0:
            return df
        freq_col = self._get_freq_col(index, agg_freq)
        if (len(df[df[freq_col] == df[freq_col].iloc[location]])
                < num_points_per_agg_freq):
            df = df[df[freq_col] != df[freq_col].iloc[location]]
        return df.reset_index(drop=True)

    def _aggregate_values(
            self,
            df: pd.DataFrame,
            agg_freq: str,
            agg_func: Optional[callable],
            index: int):
        """Aggregates the ``df`` with the given ``agg_freq`` and applies the ``agg_func``.

        All columns whose names do not start with f"{time_col}__" will be kept and aggregated.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The input dataframe.
        agg_freq : `str`
            The aggregation frequency.
        agg_func : `str`, `callable` or None
            The function used for aggregation. If None, no aggregation will be performed.
        index : `int`
            The index for the currently stage of model.
            This is used to distinguish models with the same frequency.

        Returns
        -------
        df : `pandas.DataFrame`
            The aggregated dataframe with f"{time_col}" being the timestamps and all aggregated columns.
        """
        columns = [col for col in df.columns if f"{self.time_col_}__" not in col]
        freq_col = self._get_freq_col(index, agg_freq)
        if agg_func is not None:
            df = (df
                  .groupby(freq_col)
                  .agg({col: agg_func for col in columns})
                  .reset_index(drop=False)
                  .rename(columns={freq_col: self.time_col_})
                  )
        else:
            df = df.rename(columns={freq_col: self.time_col_})
        return df

    def _drop_incomplete_agg_and_aggregate_values(
            self,
            df: pd.DataFrame,
            agg_freq: str,
            agg_func: Optional[callable],
            num_points_per_agg_freq: int,
            drop_incomplete: bool,
            index: int):
        """Drops incomplete periods from the begin and end, and gets aggregated values.

        Calls ``self._drop_incomplete_agg`` with locations 0 and -1, then calls
        ``self._aggregate_values``.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The input dataframe with augmented aggregation frequency time columns.
        agg_freq : `str`
            The aggregation frequency.
        agg_func : `str`, `callable` or None
            The function used for aggregation. If None, no aggregation will be performed.
        num_points_per_agg_freq : `int`
            The number of rows expected in a full period.
        drop_incomplete : `bool`
            Whether to drop incomplete periods from the begin and end.
            This shouldn't be done when calculating fitted or prediction values,
            because dropping may result in missing time points to predict.
        index : `int`
            The index for the currently stage of model.
            This is used to distinguish models with the same frequency.

        Returns
        -------
        agg_df : `pandas.DataFrame`
            The aggregated dataframe with f"{time_col}" being the timestamps and all aggregated columns.
        """
        df = df.copy()
        if df.shape[0] == 0:
            return df
        # Drops incomplete periods.
        if drop_incomplete:
            df = self._drop_incomplete_agg(
                df=df,
                agg_freq=agg_freq,
                location=0,
                num_points_per_agg_freq=num_points_per_agg_freq,
                index=index
            )
            df = self._drop_incomplete_agg(
                df=df,
                agg_freq=agg_freq,
                location=-1,
                num_points_per_agg_freq=num_points_per_agg_freq,
                index=index
            )
            # Checks if there are any missing timestamps in ``df``.
            # This may result in incomplete periods.
            df_check_incomplete_period = df[[self.time_col_, self.value_col_]].resample(
                agg_freq, on=self.time_col_).count()
            df_with_incomplete_periods = df_check_incomplete_period[
                df_check_incomplete_period[self.value_col_] < num_points_per_agg_freq]
            if df_with_incomplete_periods.shape[0] > 0:
                log_message(
                    message=f"There are missing timestamps in `df` when performing aggregation with "
                            f"frequency {agg_freq}. These points are {df_with_incomplete_periods}. "
                            f"This may cause the aggregated values to be biased.",
                    level=LoggingLevelEnum.WARNING
                )

        # Aggregates values
        df = self._aggregate_values(
            df=df[[self._get_freq_col(index, agg_freq)] + self._get_non_time_cols(list(df.columns))],
            agg_freq=agg_freq,
            agg_func=agg_func,
            index=index
        )
        return df

    def _get_agg_dfs(
            self,
            df: pd.DataFrame,
            agg_freq: str,
            agg_func: Optional[callable],
            train_length_in_seconds: int,
            fit_length_in_seconds: Optional[int],
            num_points_per_agg_freq: int,
            max_ar_order: int,
            index: int):
        """Given a dataframe, training/fitting configuration, gets the training data and fit data.

        If training data include incomplete periods during aggregation, the periods will be dropped.
        If fit length is shorter than train length, fit length will be replaced by train length.
        Training data is the data that the model is to be trained on.
        Fit data is the data that the fitted values are to be calculated on.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The input dataframe.
        agg_freq : `str`
            The aggregation frequency. For example, "D".
        agg_func : Callable
            The aggregation function. For example, `numpy.nanmean`.
        train_length_in_seconds : `int`
            The training data length in seconds.
        fit_length_in_seconds : `int` or None
            The fit data length in seconds.
            If None, will use ``train_length_in_seconds``.
        num_points_per_agg_freq : `int`
            For ``agg_freq``, how many data points in data frequency should be in an entire period.
        max_ar_order : `int`
            The maximum order of AR. Used to generate ``past_df`` to be fed into the Silverkite models
            to generate AR terms.
        index : `int`
            The index for the currently stage of model.
            This is used to distinguish models with the same frequency.

        Returns
        -------
        result : `dict`
            A dictionary with the following keys:

                train_df : `pandas.DataFrame`
                    The training df with aggregated frequency.
                fit_df : `pandas.DataFrame`
                    The fit df with aggregated frequency.
                df_past : `pandas.DataFrame`
                    The past df used to generate AR terms.
                fit_df_has_incomplete_period : `bool`
                    Whether ``fit_df`` has incomplete period at the end.
        """
        # Selects the columns in ``df`` excluding irrelevant aggregated time columns.
        freq_col = self._get_freq_col(index, agg_freq)
        non_time_cols = self._get_non_time_cols(list(df.columns))
        df = df[[self.time_col_, freq_col] + non_time_cols]

        train_end = df[self.time_col_].max()
        # Subtracts 1 extra full aggregation period for completion.
        # Because we drop incomplete periods in a later step before aggregation.
        # If there are incomplete periods and we don't take the extra period,
        # the actual length will be the desired length minus 1.
        # The only case that this will add an extra period is when there is no incomplete period,
        # and having an extra period does not lose anything from there.
        train_start = train_end - relativedelta(seconds=train_length_in_seconds) - to_offset(agg_freq)
        fit_start = train_end - relativedelta(seconds=fit_length_in_seconds) - to_offset(agg_freq)

        train_df = df[(df[self.time_col_] >= train_start) & (df[self.time_col_] <= train_end)]
        fit_df = df[(df[self.time_col_] >= fit_start) & (df[self.time_col_] <= train_end)]

        # Checks if there are incomplete periods in ``fit_df`` at the end.
        # They won't be dropped for not missing any prediction periods,
        # but if there are regressors, we record this.
        # Because the aggregated regressor value could be biased.
        # A warning will be raised if such regressor exists in the model.
        fit_df_has_incomplete_period = False
        if fit_df[fit_df[freq_col] == fit_df[freq_col].iloc[-1]].shape[0] < num_points_per_agg_freq:
            fit_df_has_incomplete_period = True

        # Removes incomplete periods aggregations since the aggregated values may be biased.
        # We only drop incomplete aggregations for the training periods in case they affect
        # training by including incorrect aggregated values.
        # For fit/predict we don't drop incomplete aggregations because we want to make prediction
        # for all data points in the original frequency.
        train_df = self._drop_incomplete_agg_and_aggregate_values(
            df=train_df,
            agg_freq=agg_freq,
            agg_func=agg_func,
            num_points_per_agg_freq=num_points_per_agg_freq,
            drop_incomplete=True,
            index=index
        )
        fit_df = self._drop_incomplete_agg_and_aggregate_values(
            df=fit_df,
            agg_freq=agg_freq,
            agg_func=agg_func,
            num_points_per_agg_freq=num_points_per_agg_freq,
            drop_incomplete=False,
            index=index
        )

        # Generates past dataframe for AR terms, if needed.
        past_df = None
        if max_ar_order > 0:
            # Adds 2 complete periods to ensure we don't miss any data in ``past_df``.
            past_df_end = train_start + 2 * to_offset(agg_freq)
            # By +1, we ensure that the ``past_df`` still has enough length after dropping incomplete periods.
            past_df_start = (fit_start
                             - relativedelta(seconds=to_offset(agg_freq).delta.total_seconds() * (max_ar_order + 1)))
            past_df = df[(df[self.time_col_] >= past_df_start) & (df[self.time_col_] <= past_df_end)]
            past_df = self._drop_incomplete_agg_and_aggregate_values(
                df=past_df,
                agg_freq=agg_freq,
                agg_func=agg_func,
                num_points_per_agg_freq=num_points_per_agg_freq,
                drop_incomplete=True,
                index=index
            )
            past_df = past_df[past_df[self.time_col_] < train_df[self.time_col_].min()]

        return {
            "train_df": train_df,
            "fit_df": fit_df,
            "past_df": past_df,
            "fit_df_has_incomplete_period": fit_df_has_incomplete_period
        }

    def _get_silverkite_ar_max_order(self):
        """Gets the AR order so that the model can use ``past_df`` to generate AR terms instead of imputation.

        This function only applies to the Silverkite family.
        This function is called after ``freq`` and ``forecast_horizon`` parameters have been added to model instances.

        Returns
        -------
        max_ar_orders : `list` [`int` or None]
            The maximum AR orders needed in each model.
            The value is 0 if the model does not belong to the Silverkite family or the autoregression
            parameter is not configured.
        """
        max_ar_orders = []
        for freq, model in zip(self.agg_freqs, self.models):
            try:
                # All Silverkite family estimators have the method ``get_max_ar_order``.
                max_ar_order = model.get_max_ar_order()
            except (AttributeError, TypeError):
                max_ar_order = 0
            max_ar_orders.append(max_ar_order)
        return max_ar_orders

    def _train(
            self,
            df: pd.DataFrame):
        """Trains the Multistage Forecast model with the given configurations.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The input dataframe.

        Returns
        -------
        fit_df : `pandas.DataFrame`
            The dataframe with aggregated time columns and predictions.
        """
        # Sorts the models by training data length from long to short.
        (self.agg_freqs, self.agg_funcs, self.train_lengths_in_seconds,
         self.fit_lengths_in_seconds, self.models, self.forecast_horizons) = zip(*sorted(
            zip(self.agg_freqs, self.agg_funcs, self.train_lengths_in_seconds,
                self.fit_lengths_in_seconds, self.models, self.forecast_horizons),
            key=lambda x: x[2],  # key is ``train_lengths_in_seconds``
            reverse=True))

        # Here we add the ``forecast_horizon`` and ``freq`` attribute regardless of what model it is.
        # This doesn't affect the model if it does not expect the ``forecast_horizon``
        # or ``freq`` attribute before fitting.
        # If the forecast horizon parameter varies due to different periods in the fit
        # and predict input, users can leave them as None and let the estimator automatically fill them.
        # If the entry point is template, it's possible that "forecast_horizon" is a missing parameter.
        # We add it here when it's missing.
        # If either of these parameter is already set, we won't modify it.
        for model, forecast_horizon, agg_freq in zip(self.models, self.forecast_horizons, self.agg_freqs):
            if getattr(model, "forecast_horizon", None) is None:
                model.forecast_horizon = forecast_horizon
            if getattr(model, "freq", None) is None:
                model.freq = agg_freq

        self.num_points_per_agg_freqs = self._get_num_points_per_agg_freq(
            data_freq=self.freq,
            agg_freqs=self.agg_freqs
        )

        self.max_ar_orders = self._get_silverkite_ar_max_order()

        # Adds resampled timestamps and aggregated columns to df.
        # This is done after the sorting, so the index of model steps will be correct.
        df_with_freq = self._add_agg_freq_cols(df=df)

        # Makes a copy. This is used to store the results.
        fit_result_df = df_with_freq.copy()
        # Adds a column to track cumulative fitted values.
        # At each stage this column will be subtracted from the original time series
        # to obtain the current residual to be fitted on the next model.
        fit_result_df["cum_fitted_values"] = 0

        for index, (freq, func, train_length_in_seconds, fit_length_in_seconds,
                    model, num_points_per_agg_freq, max_order) in enumerate(
            zip(self.agg_freqs, self.agg_funcs, self.train_lengths_in_seconds,
                self.fit_lengths_in_seconds, self.models, self.num_points_per_agg_freqs,
                self.max_ar_orders)):

            df_with_freq_copy = df_with_freq.copy()  # makes a copy since we want to calculate the residuals.
            df_with_freq_copy[self.value_col_] -= fit_result_df["cum_fitted_values"]

            # Gets the train_df and fit_df.
            # The dfs will be subset and aggregated.
            # The fitted values are to be calculated on fit_df.
            # fit_df has length as the maximum of train_length and fit_length.
            agg_dfs = self._get_agg_dfs(
                df=df_with_freq_copy,
                agg_freq=freq,
                agg_func=func,
                train_length_in_seconds=train_length_in_seconds,
                fit_length_in_seconds=fit_length_in_seconds,
                num_points_per_agg_freq=num_points_per_agg_freq,
                max_ar_order=max_order,
                index=index
            )
            train_df = agg_dfs["train_df"]
            fit_df = agg_dfs["fit_df"]
            # Adds ``past_df`` in case the model expects extra data to calculate autoregression terms.
            if agg_dfs["past_df"] is not None:
                model.past_df = agg_dfs["past_df"]
            if agg_dfs["fit_df_has_incomplete_period"]:
                regressor_cols = getattr(model, "regressor_cols", None)
                if regressor_cols is not None and regressor_cols != []:
                    log_message(
                        message="There are incomplete periods in `fit_df`, thus the regressor "
                                "values are biased after aggregation.",
                        level=LoggingLevelEnum.WARNING
                    )
            # Adds the actual values to the result df.
            fit_result_df = fit_result_df.merge(
                train_df.rename(columns={
                    self.time_col_: self._get_freq_col(index, freq),
                    self.value_col_: f"{self.value_col_}__{index}__{freq}"}),
                on=self._get_freq_col(index, freq),
                how="left")

            # Fits the model.
            model.fit(
                train_df,
                time_col=self.time_col_,
                value_col=self.value_col_)

            # Calculates fitted values.
            y_fitted = model.predict(fit_df)[[cst.TIME_COL, cst.PREDICTED_COL]].rename(
                columns={
                    cst.TIME_COL: f"{cst.TIME_COL}__{index}__{freq}",
                    cst.PREDICTED_COL: f"{cst.PREDICTED_COL}__{index}__{freq}"})

            # Joins the fitted values with the original ``fit_result_df``.
            # This is a left join since the fitted values is a subset of the entire period.
            fit_result_df = fit_result_df.merge(
                y_fitted,
                how="left",
                left_on=self._get_freq_col(index, freq),
                right_on=f"{cst.TIME_COL}__{index}__{freq}")
            # Adds the current fitted values to the previous fitted values to get cumulated fitted values.
            fit_result_df["cum_fitted_values"] += fit_result_df[f"{cst.PREDICTED_COL}__{index}__{freq}"]

        fit_result_df = fit_result_df.rename(columns={
            "cum_fitted_values": cst.PREDICTED_COL
        })

        return fit_result_df

    def _predict(
            self,
            df: pd.DataFrame):
        """The prediction function.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The input dataframe, covering the prediction phase.

        Returns
        -------
        pred : `pandas.DataFrame`
            The predicted dataframe.
        """
        # Since Multistage Forecast partitions the data and does not use the
        # entire period to train the models, we do not allow predictions
        # going beyond the earliest fit period.
        # The predictions before the allowed periods will be marked as 0 for compatibility
        # in calculating error metrics.
        # This only affects the training evaluation in pipeline, not the validation/test/forecast.
        fit_starts = [self.train_end - relativedelta(seconds=fit_length) for fit_length in self.fit_lengths_in_seconds]

        # Adds resampled timestamps and aggregated columns to df.
        df_with_freq = self._add_agg_freq_cols(df=df)

        # Generates predictions for each freq/model.
        for index, (freq, func, model, start) in enumerate(
                zip(self.agg_freqs, self.agg_funcs, self.models, fit_starts)):
            freq_col = self._get_freq_col(index, freq)
            current_df = df_with_freq[df_with_freq[self.time_col_] >= start]
            current_df = current_df[
                [freq_col]
                + [col for col in self._get_non_time_cols(current_df.columns) if cst.PREDICTED_COL not in col]]
            # We do this aggregation for the concern that there are regressors in ``df``.
            # Uses ``set_index`` instead of resample "on" because the aggregation functions are
            # defined with ``partial`` and there's some incompatibility between partial
            # and ``np.nanmean`` with the "on" column.
            current_df = current_df.set_index(freq_col).resample(freq).apply(func)
            current_df[self.time_col_] = current_df.index
            current_df = current_df.reset_index(drop=True)
            predicted_df = model.predict(current_df)  # ``past_df`` has been added to model instance.
            df_with_freq = df_with_freq.merge(
                predicted_df[[cst.TIME_COL, cst.PREDICTED_COL]].rename(columns={
                    cst.TIME_COL: freq_col,
                    cst.PREDICTED_COL: f"{cst.PREDICTED_COL}__{index}__{freq}"
                }),
                how="left",
                on=freq_col
            )

        df_with_freq[cst.PREDICTED_COL] = 0
        for index, freq in enumerate(self.agg_freqs):
            df_with_freq[cst.PREDICTED_COL] += df_with_freq[f"{cst.PREDICTED_COL}__{index}__{freq}"]

        return df_with_freq

    def _fit_uncertainty(self):
        """Fits the uncertainty model."""
        fit_df_dropna = self.fit_df.dropna(
            subset=[cst.PREDICTED_COL]).rename(
            columns={cst.VALUE_COL: self.value_col_})  # Estimator predictions have standard value column.

        self.fit_uncertainty(
            df=fit_df_dropna,
            uncertainty_dict=self.uncertainty_dict,
        )
        if self.uncertainty_model is not None:
            fit_df_with_uncertainty = self.predict_uncertainty(
                df=fit_df_dropna
            )
            if fit_df_with_uncertainty is not None:
                fit_df_with_uncertainty = self.fit_df.merge(
                    fit_df_with_uncertainty[[self.time_col_] + [
                        col for col in fit_df_with_uncertainty if col not in self.fit_df.columns]],
                    on=self.time_col_,
                    how="left"
                )
                self.fit_df = fit_df_with_uncertainty

    def plot_components(self):
        """Makes component plots.

        Returns
        -------
        figs : `list` [`plotly.graph_objects.Figure` or None]
            A list of figures from each model.
        """
        if self.fit_df is None:
            raise ValueError("Please call `fit` before calling `plot_components`.")
        figs = []
        for model in self.models:
            try:
                fig = model.plot_components()
            except AttributeError:
                fig = None
            figs.append(fig)
        return figs

    def summary(self):
        """Gets model summaries.

        Returns
        -------
        summaries : `list` [`~greykite.algo.common.model_summary.ModelSummary` or None]
            A list of model summaries from each model.
        """
        if self.fit_df is None:
            raise ValueError("Please call `fit` before calling `summary`.")
        summaries = []
        for model in self.models:
            try:
                summary = model.summary()
            except AttributeError:
                summary = None
            summaries.append(summary)
        return summaries
