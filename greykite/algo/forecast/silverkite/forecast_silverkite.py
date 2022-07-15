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
# original author: Reza Hosseini
import re
import warnings
from typing import Type

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from pandas.tseries.frequencies import to_offset

from greykite.algo.changepoint.adalasso.changepoint_detector import get_changepoints_dict
from greykite.algo.changepoint.adalasso.changepoint_detector import get_seasonality_changepoints
from greykite.algo.changepoint.adalasso.changepoints_utils import build_seasonality_feature_df_from_detection_result
from greykite.algo.changepoint.adalasso.changepoints_utils import get_seasonality_changepoint_df_cols
from greykite.algo.common.ml_models import fit_ml_model_with_evaluation
from greykite.algo.common.ml_models import predict_ml
from greykite.algo.common.ml_models import predict_ml_with_uncertainty
from greykite.algo.forecast.silverkite.constants.silverkite_constant import default_silverkite_constant
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonalityEnum
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonalityEnumMixin
from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_fourier_feature_col_names
from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_similar_lag
from greykite.common.constants import CHANGEPOINT_COL_PREFIX
from greykite.common.constants import ERR_STD_COL
from greykite.common.constants import QUANTILE_SUMMARY_COL
from greykite.common.constants import SEASONALITY_REGEX
from greykite.common.constants import TimeFeaturesEnum
from greykite.common.enums import TimeEnum
from greykite.common.features.timeseries_features import add_daily_events
from greykite.common.features.timeseries_features import add_time_features_df
from greykite.common.features.timeseries_features import fourier_series_multi_fcn
from greykite.common.features.timeseries_features import get_changepoint_dates_from_changepoints_dict
from greykite.common.features.timeseries_features import get_changepoint_features
from greykite.common.features.timeseries_features import get_changepoint_features_and_values_from_config
from greykite.common.features.timeseries_features import get_default_origin_for_time_vars
from greykite.common.features.timeseries_lags import build_autoreg_df
from greykite.common.features.timeseries_lags import build_autoreg_df_multi
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import get_pattern_cols
from greykite.common.python_utils import unique_elements_in_list
from greykite.common.time_properties import describe_timeseries
from greykite.common.time_properties import fill_missing_dates
from greykite.common.time_properties import min_gap_in_seconds
from greykite.common.time_properties_forecast import get_default_horizon_from_period


register_matplotlib_converters()

"""
Defines a class that provides the silverkite forecast functionality for training and prediction.
Currently provides the following functionality for users of the class
TODO : List all public functions here
This class can be extended and the user can customize the following functions
TODO : List all the protected functions here
"""


class SilverkiteForecast():
    def __init__(
            self,
            constants: SilverkiteSeasonalityEnumMixin = default_silverkite_constant):
        self._silverkite_seasonality_enum: Type[SilverkiteSeasonalityEnum] = constants.get_silverkite_seasonality_enum()

    def forecast(
            self,
            df,
            time_col,
            value_col,
            freq=None,
            origin_for_time_vars=None,
            extra_pred_cols=None,
            drop_pred_cols=None,
            explicit_pred_cols=None,
            train_test_thresh=None,
            training_fraction=0.9,  # This is for internal ML models validation. The final returned model will be trained on all data.
            fit_algorithm="linear",
            fit_algorithm_params=None,
            daily_event_df_dict=None,
            fs_components_df=pd.DataFrame({
                "name": [
                    TimeFeaturesEnum.tod.value,
                    TimeFeaturesEnum.tow.value,
                    TimeFeaturesEnum.toy.value],
                "period": [24.0, 7.0, 1.0],
                "order": [3, 3, 5],
                "seas_names": ["daily", "weekly", "yearly"]}),
            autoreg_dict=None,
            past_df=None,
            lagged_regressor_dict=None,
            changepoints_dict=None,
            seasonality_changepoints_dict=None,
            changepoint_detector=None,
            min_admissible_value=None,
            max_admissible_value=None,
            uncertainty_dict=None,
            normalize_method=None,
            adjust_anomalous_dict=None,
            impute_dict=None,
            regression_weight_col=None,
            forecast_horizon=None,
            simulation_based=False,
            simulation_num=10,
            fast_simulation=False):
        """A function for forecasting.
        It captures growth, seasonality, holidays and other patterns.
        See "Capturing the time-dependence in the precipitation process for
        weather risk assessment" as a reference:
        https://link.springer.com/article/10.1007/s00477-016-1285-8

        Parameters
        ----------
        df : `pandas.DataFrame`
            A data frame which includes the timestamp column
            as well as the value column.
        time_col : `str`
            The column name in ``df`` representing time for the time series data.
            The time column can be anything that can be parsed by pandas DatetimeIndex.
        value_col: `str`
            The column name which has the value of interest to be forecasted.
        freq: `str`, optional, default None
            The intended timeseries frequency, DateOffset alias.
            See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            If None automatically inferred. This frequency will be passed through
            this function as a part of the trained model and used at predict time
            if needed.
            If data include missing timestamps, and frequency is monthly/annual,
            user should pass this parameter, as it cannot be inferred.
        origin_for_time_vars : `float`, optional, default None
            The time origin used to create continuous variables for time.
            If None, uses the first record in ``df``.
        extra_pred_cols : `list` of `str`, default None
            Names of the extra predictor columns.

            If None, uses ["ct1"], a simple linear growth term.

            It can leverage regressors included in ``df`` and those generated
            by the other parameters. The following effects will not be modeled
            unless specified in ``extra_pred_cols``:

                - included in ``df``: e.g. macro-economic factors, related timeseries
                - from `~greykite.common.features.timeseries_features.build_time_features_df`:
                  e.g. ct1, ct_sqrt, dow, ...
                - from ``daily_event_df_dict``: e.g. "events_India", ...

            The columns corresponding to the following parameters are included
            in the model without specification in ``extra_pred_cols``.
            ``extra_pred_cols`` can be used to add interactions with these terms.

                changepoints_dict: e.g. changepoint0, changepoint1, ...
                fs_components_df: e.g. sin2_dow, cos4_dow_weekly
                autoreg_dict: e.g. x_lag1, x_avglag_2_3_4, y_avglag_1_to_5

            If a regressor is passed in ``df``, it needs to be provided to
            the associated predict function:

                ``predict_silverkite``: via ``fut_df`` or ``new_external_regressor_df``
                ``silverkite.predict_n(_no_sim``: via ``new_external_regressor_df``
        drop_pred_cols : `list` [`str`] or None, default None
            Names of predictor columns to be dropped from the final model.
            Ignored if None
        explicit_pred_cols : `list` [`str`] or None, default None
            Names of the explicit predictor columns which will be
            the only variables in the final model. Note that this overwrites
            the generated predictors in the model and may include new
            terms not appearing in the predictors (e.g. interaction terms).
            Ignored if None
        train_test_thresh : `datetime.datetime`, optional
            e.g. datetime.datetime(2019, 6, 30)
            The threshold for training and testing split.
            Note that the final returned model is trained using all data.
            If None, training split is based on ``training_fraction``
        training_fraction : `float`, optional
            The fraction of data used for training (0.0 to 1.0)
            Used only if ``train_test_thresh`` is None.
            If this is also None or 1.0, then we skip testing
            and train on the entire dataset.
        fit_algorithm : `str`, optional, default "linear"
            The type of predictive model used in fitting.

            See `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
            for available options and their parameters.
        fit_algorithm_params : `dict` or None, optional, default None
            Parameters passed to the requested fit_algorithm.
            If None, uses the defaults in `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
        daily_event_df_dict : `dict` or None, optional, default None
            A dictionary of data frames, each representing events data for the corresponding key.
            The DataFrame has two columns:

                - The first column contains event dates. Must be in a format
                  recognized by `pandas.to_datetime`. Must be at daily
                  frequency for proper join. It is joined against the time
                  in ``df``, converted to a day:
                  ``pd.to_datetime(pd.DatetimeIndex(df[time_col]).date)``.
                - the second column contains the event label for each date

            The column order is important; column names are ignored.
            The event dates must span their occurrences in both the training
            and future prediction period.

            During modeling, each key in the dictionary is mapped to a categorical variable
            named ``f"{EVENT_PREFIX}_{key}"``, whose value at each timestamp is specified
            by the corresponding DataFrame.

            For example, to manually specify a yearly event on September 1
            during a training/forecast period that spans 2020-2022::

                daily_event_df_dict = {
                    "custom_event": pd.DataFrame({
                        "date": ["2020-09-01", "2021-09-01", "2022-09-01"],
                        "label": ["is_event", "is_event", "is_event"]
                    })
                }

            It's possible to specify multiple events in the same df. Two events,
            ``"sep"`` and ``"oct"`` are specified below for 2020-2021::

                daily_event_df_dict = {
                    "custom_event": pd.DataFrame({
                        "date": ["2020-09-01", "2020-10-01", "2021-09-01", "2021-10-01"],
                        "event_name": ["sep", "oct", "sep", "oct"]
                    })
                }

            Use multiple keys if two events may fall on the same date. These events
            must be in separate DataFrames::

                daily_event_df_dict = {
                    "fixed_event": pd.DataFrame({
                        "date": ["2020-09-01", "2021-09-01", "2022-09-01"],
                        "event_name": "fixed_event"
                    }),
                    "moving_event": pd.DataFrame({
                        "date": ["2020-09-01", "2021-08-28", "2022-09-03"],
                        "event_name": "moving_event"
                    }),
                }

            The multiple event specification can be used even if events never overlap. An
            equivalent specification to the second example::

                daily_event_df_dict = {
                    "sep": pd.DataFrame({
                        "date": ["2020-09-01", "2021-09-01"],
                        "event_name": "is_event"
                    }),
                    "oct": pd.DataFrame({
                        "date": ["2020-10-01", "2021-10-01"],
                        "event_name": "is_event"
                    }),
                }

            .. note::

                The events you want to use must be specified in ``extra_pred_cols``.
                These take the form: ``f"{EVENT_PREFIX}_{key}"``, where
                `~greykite.common.constants.EVENT_PREFIX` is the constant.

                Do not use `~greykite.common.constants.EVENT_DEFAULT`
                in the second column. This is reserved to indicate dates that do not
                correspond to an event.
        fs_components_df : `pandas.DataFrame` or None, optional
            A dataframe with information about fourier series generation.
            Must contain columns with following names:

                "name": name of the timeseries feature e.g. "tod", "tow" etc.
                "period": Period of the fourier series, optional, default 1.0
                "order": Order of the fourier series, optional, default 1.0
                "seas_names": season names corresponding to the name
                (e.g. "daily", "weekly" etc.), optional.

            Default includes daily, weekly , yearly seasonality.
        autoreg_dict : `dict` or `str` or None, optional, default `None`
            If a `dict`: A dictionary with arguments for `~greykite.common.features.timeseries_lags.build_autoreg_df`.
            That function's parameter ``value_col`` is inferred from the input of
            current function ``self.forecast``. Other keys are:

                ``"lag_dict"`` : `dict` or None
                ``"agg_lag_dict"`` : `dict` or None
                ``"series_na_fill_func"`` : callable

            If a `str`: The string will represent a method and a dictionary will be
            constructed using that `str`.
            Currently only implemented method is "auto" which uses
            `~greykite.algo.forecast.silverkite.SilverkiteForecast.__get_default_autoreg_dict`
            to create a dictionary.
            See more details for above parameters in
            `~greykite.common.features.timeseries_lags.build_autoreg_df`.
        past_df : `pandas.DataFrame` or None, default None
            The past df used for building autoregression features.
            This is not necessarily needed since imputation is possible.
            However, it is recommended to provide ``past_df`` for more accurate
            autoregression features and faster training (by skipping imputation).
            The columns are:

                time_col : `pandas.Timestamp` or `str`
                    The timestamps.
                value_col : `float`
                    The past values.
                addition_regressor_cols : `float`
                    Any additional regressors.

            Note that this ``past_df`` is assumed to immediately precede ``df`` without gaps,
            otherwise an error will be raised.
        lagged_regressor_dict : `dict` or None, default None
            A dictionary with arguments for `~greykite.common.features.timeseries_lags.build_autoreg_df_multi`.
            The keys of the dictionary are the target lagged regressor column names.
            It can leverage the regressors included in ``df``.
            The value of each key is either a `dict` or `str`.
            If `dict`, it has the following keys:

                ``"lag_dict"`` : `dict` or None
                ``"agg_lag_dict"`` : `dict` or None
                ``"series_na_fill_func"`` : callable

            If `str`, it represents a method and a dictionary will be constructed using that `str`.
            Currently the only implemented method is "auto" which uses
            `~greykite.algo.forecast.silverkite.SilverkiteForecast.__get_default_lagged_regressor_dict`
            to create a dictionary for each lagged regressor.
            An example::

                lagged_regressor_dict = {
                    "regressor1": {
                        "lag_dict": {"orders": [1, 2, 3]},
                        "agg_lag_dict": {
                            "orders_list": [[7, 7 * 2, 7 * 3]],
                            "interval_list": [(8, 7 * 2)]},
                        "series_na_fill_func": lambda s: s.bfill().ffill()},
                    "regressor2": "auto"}

            Check the docstring of `~greykite.common.features.timeseries_lags.build_autoreg_df_multi`
            for more details for each argument.
        changepoints_dict : `dict` or None, optional, default None
            Specifies the changepoint configuration.

            "method": `str`
                The method to locate changepoints.
                Valid options:

                    - "uniform". Places n_changepoints evenly spaced changepoints to allow growth to change.
                    - "custom". Places changepoints at the specified dates.
                    - "auto". Automatically detects change points. For configuration, see
                      `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_trend_changepoints`

                Additional keys to provide parameters for each particular method are described below.
            "continuous_time_col": `str`, optional
                Column to apply ``growth_func`` to, to generate changepoint features
                Typically, this should match the growth term in the model
            "growth_func": Optional[func]
                Growth function (scalar -> scalar). Changepoint features are created
                by applying ``growth_func`` to ``continuous_time_col`` with offsets.
                If None, uses identity function to use ``continuous_time_col`` directly
                as growth term
                If changepoints_dict["method"] == "uniform", this other key is required:

                    ``"n_changepoints"``: int
                        number of changepoints to evenly space across training period

                If changepoints_dict["method"] == "custom", this other key is required:

                    ``"dates"``: Iterable[Union[int, float, str, datetime]]
                        Changepoint dates. Must be parsable by pd.to_datetime.
                        Changepoints are set at the closest time on or after these dates
                        in the dataset.

                If changepoints_dict["method"] == "auto", the keys that matches the parameters in
                `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_trend_changepoints`,
                except ``df``, ``time_col`` and ``value_col``, are optional.
                Extra keys also include "dates", "combine_changepoint_min_distance" and "keep_detected" to specify
                additional custom trend changepoints. These three parameters correspond to the three parameters
                "custom_changepoint_dates", "min_distance" and "keep_detected" in
                `~greykite.algo.changepoint.adalasso.changepoints_utils.combine_detected_and_custom_trend_changepoints`.

        seasonality_changepoints_dict : `dict` or None, default `None`
            The parameter dictionary for seasonality change point detection. Parameters are in
            `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`.
            Note ``df``, ``time_col``, ``value_col`` and ``trend_changepoints`` are auto populated,
            and do not need to be provided.
        changepoint_detector : `ChangepointDetector` or `None`, default `None`
            The ChangepointDetector class
            :class:`~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector`.
            This is specifically for
            `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.forecast_simple_silverkite`
            to pass the ChangepointDetector class for plotting purposes, in case that users use
            ``forecast_simple_silverkite`` with ``changepoints_dict["method"] == "auto"``. The
            trend change point detection has to be run there to include possible interaction terms,
            so we need to pass the detection result from there to include in the output.
        min_admissible_value : `float` or None, optional, default `None`
            The minimum admissible value to return during prediction.
            If None, no limit is applied.
        max_admissible_value : `float` or None, optional, default `None`
            The maximum admissible value to return during prediction.
            If None, no limit is applied.
        uncertainty_dict : `dict` or None, optional, default `None`
            How to fit the uncertainty model. A dictionary with keys:
                ``"uncertainty_method"`` : `str`
                    The title of the method.
                    Only "simple_conditional_residuals" is implemented
                    in ``fit_ml_model`` which calculates CIs using residuals
                ``"params"`` : `dict`
                    A dictionary of parameters needed for
                    the requested ``uncertainty_method``. For example, for
                    ``uncertainty_method="simple_conditional_residuals"``, see
                    parameters of `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`:

                        ``"conditional_cols"``
                        ``"quantiles"``
                        ``"quantile_estimation_method"``
                        ``"sample_size_thresh"``
                        ``"small_sample_size_method"``
                        ``"small_sample_size_quantile"``

            If None, no uncertainty intervals are calculated.
        normalize_method : `str` or None, default None
            If a string is provided, it will be used as the normalization method
            in `~greykite.common.features.normalize.normalize_df`, passed via
            the argument ``method``.
            Available options are: "zero_to_one", "statistical", "minus_half_to_half", "zero_at_origin".
            If None, no normalization will be performed.
            See that function for more details.
        adjust_anomalous_dict : `dict` or None, default None
            If not None, a dictionary with following items:

                - "func" : `callable`
                    A function to perform adjustment of anomalous data with following
                    signature::

                        adjust_anomalous_dict["func"](
                            df=df,
                            time_col=time_col,
                            value_col=value_col,
                            **params) ->
                        {"adjusted_df": adjusted_df, ...}

                - "params" : `dict`
                    The extra parameters to be passed to the function above.
        impute_dict : `dict` or None, default None
            If not None, a dictionary with following items:

            - "func" : `callable`
                A function to perform imputations with following
                signature::

                    impute_dict["func"](
                        df=df,
                        value_col=value_col,
                        **impute_dict["params"] ->
                    {"df": imputed_df, ...}

            - "params" : `dict`
                The extra parameters to be passed to the function above.

        regression_weight_col : `str` or None, default None
            The column name for the weights to be used in weighted regression version
            of applicable machine-learning models.
        forecast_horizon : `int` or None, default None
            The number of periods for which forecast is needed.
            Note that this is only used in deciding what parameters should be
            used for certain components e.g. autoregression, if automatic methods
            are requested. While, the prediction time forecast horizon could be different
            from this variable, ideally they should be the same.
        simulation_based : `bool`, default False
            Boolean to specify if the future predictions are to be using simulations
            or not.
            Note that this is only used in deciding what parameters should be
            used for certain components e.g. autoregression, if automatic methods
            are requested. However, the auto-settings and the prediction settings
            regarding using simulations should match.
        simulation_num : `int`, default 10
            The number of simulations for when simulations are used for generating
            forecasts and prediction intervals.
        fast_simulation: `bool`, default False
            Deterimes if fast simulations are to be used. This only impacts models
            which include auto-regression. This method will only generate one simulation
            without any error being added and then add the error using the volatility
            model. The advantage is a major boost in speed during inference and the
            disadvantage is potentially less accurate prediction intervals.


        Returns
        -------
        trained_model : `dict`
            A dictionary that includes the fitted model from the function
            :func:`~greykite.algo.common.ml_models.fit_ml_model_with_evaluation`.
            The keys are:

                df_dropna: `pandas.DataFrame`
                    The ``df`` with NAs dropped.
                df: `pandas.DataFrame`
                    The original ``df``.
                num_training_points: `int`
                    The number of training points.
                features_df: `pandas.DataFrame`
                    The ``df`` with augmented time features.
                min_timestamp: `pandas.Timestamp`
                    The minimum timestamp in data.
                max_timestamp: `pandas.Timestamp`
                    The maximum timestamp in data.
                freq: `str`
                    The data frequency.
                inferred_freq: `str`
                    The data freqency inferred from data.
                inferred_freq_in_secs : `float`
                    The data frequency inferred from data in seconds.
                inferred_freq_in_days: `float`
                    The data frequency inferred from data in days.
                time_col: `str`
                    The time column name.
                value_col: `str`
                    The value column name.
                origin_for_time_vars: `float`
                    The first time stamp converted to a float number.
                fs_components_df: `pandas.DataFrame`
                    The dataframe that specifies the seasonality Fourier configuration.
                autoreg_dict: `dict`
                    The dictionary that specifies the autoregression configuration.
                lagged_regressor_dict: `dict`
                    The dictionary that specifies the lagged regressors configuration.
                lagged_regressor_cols: `list` [`str`]
                    List of regressor column names used for lagged regressor
                normalize_method: `str`
                    The normalization method.
                    See the function input parameter ``normalize_method``.
                daily_event_df_dict: `dict`
                    The dictionary that specifies daily events configuration.
                changepoints_dict: `dict`
                    The dictionary that specifies changepoints configuration.
                changepoint_values: `list` [`float`]
                    The list of changepoints in continuous time values.
                normalized_changepoint_values : `list` [`float`]
                    The list of changepoints in continuous time values normalized to 0 to 1.
                continuous_time_col: `str`
                    The continuous time column name in ``features_df``.
                growth_func: `func`
                    The growth function used in changepoints, None is linear function.
                fs_func: `func`
                    The function used to generate Fourier series for seasonality.
                has_autoreg_structure: `bool`
                    Whether the model has autoregression structure.
                autoreg_func: `func`
                    The function to generate autoregression columns.
                min_lag_order: `int`
                    Minimal lag order in autoregression.
                max_lag_order: `int`
                    Maximal lag order in autoregression.
                has_lagged_regressor_structure: `bool`
                    Whether the model has lagged regressor structure.
                lagged_regressor_func: `func`
                    The function to generate lagged regressor columns.
                min_lagged_regressor_order: `int`
                    Minimal lag order in lagged regressors.
                max_lagged_regressor_order: `int`
                    Maximal lag order in lagged regressors.
                uncertainty_dict: `dict`
                    The dictionary that specifies uncertainty model configuration.
                pred_cols: `list` [`str`]
                    List of predictor names.
                last_date_for_fit: `str` or `pandas.Timestamp`
                    The last timestamp used for fitting.
                trend_changepoint_dates: `list` [`pandas.Timestamp`]
                    List of trend changepoints.
                changepoint_detector: `class`
                    The `ChangepointDetector` class used to detected trend changepoints.
                seasonality_changepoint_dates: `list` [`pandas.Timestamp`]
                    List of seasonality changepoints.
                seasonality_changepoint_result: `dict`
                    The seasonality changepoint detection results.
                fit_algorithm: `str`
                    The algorithm used to fit the model.
                fit_algorithm_params: `dict`
                    The dictionary of parameters for ``fit_algorithm``.
                adjust_anomalous_info: `dict`
                    A dictionary that has anomaly adjustment results.
                impute_info: `dict`
                    A dictionary that has the imputation results.
                forecast_horizon: `int`
                    The forecast horizon in steps.
                forecast_horizon_in_days: `float`
                    The forecast horizon in days.
                forecast_horizon_in_timedelta: `datetime.timmdelta`
                    The forecast horizon in timedelta.
                simulation_based: `bool`
                    Whether to use simulation in prediction with autoregression terms.
                simulation_num : `int`, default 10
                    The number of simulations for when simulations are used for generating
                    forecasts and prediction intervals.
                train_df : `pandas.DataFrame`
                    The past dataframe used to generate AR terms.
                    It includes the concatenation of ``past_df`` and ``df`` if ``past_df`` is provided,
                    otherwise it is the ``df`` itself.

        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        num_training_points = df.shape[0]
        adjust_anomalous_info = None

        if simulation_num is not None:
            assert simulation_num > 0, "simulation number must be a natural number"

        if past_df is not None:
            if past_df.shape[0] == 0:
                past_df = None
            else:
                past_df = past_df.copy()
                past_df[time_col] = pd.to_datetime(past_df[time_col])
                past_df = past_df.sort_values(by=time_col)

        # Adjusts anomalies if requested
        if adjust_anomalous_dict is not None:
            adjust_anomalous_info = adjust_anomalous_dict["func"](
                df=df,
                time_col=time_col,
                value_col=value_col,
                **adjust_anomalous_dict["params"])
            df = adjust_anomalous_info["adjusted_df"]

        impute_info = None
        if impute_dict is not None:
            impute_info = impute_dict["func"](
                df=df,
                value_col=value_col,
                **impute_dict["params"])
            df = impute_info["df"]

        # Calculates time properties of the series
        # We include these properties in the returned `trained_model` object
        time_stats = describe_timeseries(df, time_col=time_col)
        max_timestamp = time_stats["max_timestamp"]
        min_timestamp = time_stats["min_timestamp"]
        # This infers a constant length freq (in seconds units or days) from data.
        # Note that in some cases e.g. monthly or annual data the real frequency can
        # be non-constant in length.
        # This inferred freq is of `pandas._libs.tslibs.timedeltas.Timedelta` type.
        inferred_freq = time_stats["freq_in_timedelta"]
        inferred_freq_in_secs = time_stats["freq_in_secs"]
        inferred_freq_in_days = time_stats["freq_in_days"]
        # However in some use cases user might provide more complex
        # freq to `predict_n` functions.
        # As an example `freq='W-SUN'` which can be passed by user.
        # If such freq is not passed, we can attempt to infer it.
        # Note that if there are data with gaps, this freq cannot be inferred.
        # E.g. if hourly data only include 9am-5pm.
        if freq is None:
            freq = pd.infer_freq(df[time_col])

        # Calculates forecast horizon (as a number of observations)
        if forecast_horizon is None:
            # expected to be kept in sync with default value set in ``get_default_time_parameters``
            forecast_horizon = get_default_horizon_from_period(
                period=inferred_freq_in_secs,
                num_observations=num_training_points)
        forecast_horizon_in_timedelta = inferred_freq * forecast_horizon
        forecast_horizon_in_days = inferred_freq_in_days * forecast_horizon

        if extra_pred_cols is None:
            extra_pred_cols = [TimeFeaturesEnum.ct1.value]  # linear in time

        # Makes sure the ``train_test_thresh`` is within the data
        last_time_available = max(df[time_col])
        if train_test_thresh is not None and train_test_thresh >= last_time_available:
            raise ValueError(
                f"Input timestamp for the parameter 'train_test_threshold' "
                f"({train_test_thresh}) exceeds the maximum available timestamp "
                f"of the time series ({last_time_available})."
                f"Please pass a value within the range.")

        # Sets default origin so that "ct1" feature from `build_time_features_df`
        # Starts at 0 on train start date
        if origin_for_time_vars is None:
            origin_for_time_vars = get_default_origin_for_time_vars(df, time_col)
        # Updates `changepoints_dict`, unchanged if not "method" == "auto"
        changepoints_dict, changepoint_detector_class = get_changepoints_dict(
            df=df,
            time_col=time_col,
            value_col=value_col,
            changepoints_dict=changepoints_dict)
        if changepoint_detector_class is None:
            # Handles the case that user uses `forecast_simple_silverkite` with automatic
            # trend change point detection. In that case, the `changepoints_dict` is already
            # transformed to "method" = "custom", thus no changepoint detector is returned
            # by `get_changepoints_dict`, so we need the original `ChangepointDetector` class
            # to include in the output for plotting purpose.
            changepoint_detector_class = changepoint_detector
        # Defines trend changepoints.
        # `df` contains all dates in the training period, including those
        # where `value_col` is np.nan and therefore not used in training
        # by `fit_ml_model_with_evaluation`.
        # Thus, when changepoint "method" = "uniform", all dates are used to uniformly
        # place the changepoints. When changepoint "method" = "auto", only dates without
        # missing values are used to place potential changepoints, after resampling
        # according to `resample_freq`. Seasonality changepoints are also placed using
        # resampled dates after excluding the missing values.
        trend_changepoint_dates = get_changepoint_dates_from_changepoints_dict(
            changepoints_dict=changepoints_dict,
            df=df,
            time_col=time_col
        )
        changepoints = get_changepoint_features_and_values_from_config(
            df=df,
            time_col=time_col,
            changepoints_dict=changepoints_dict,
            origin_for_time_vars=origin_for_time_vars)
        # Checks the provided `extra_pred_cols`. If it contains a feature involving a changepoint,
        # the changepoint must be valid
        keep_extra_pred_cols = []
        for col in extra_pred_cols:
            if CHANGEPOINT_COL_PREFIX in col:
                for changepoint_col in changepoints["changepoint_cols"]:
                    if changepoint_col in col:
                        keep_extra_pred_cols.append(col)
                        break
            else:
                keep_extra_pred_cols.append(col)
        if len(keep_extra_pred_cols) < len(extra_pred_cols):
            removed_pred_cols = set(extra_pred_cols) - set(keep_extra_pred_cols)
            extra_pred_cols = keep_extra_pred_cols
            warnings.warn(f"The following features in extra_pred_cols are removed for this"
                          f" training set: {removed_pred_cols}. This is possible if running backtest"
                          f" or cross validation, but you are fitting on the entire training set,"
                          f" double check `extra_pred_cols` and other configuration.")

        changepoint_values = changepoints["changepoint_values"]
        continuous_time_col = changepoints["continuous_time_col"]
        changepoint_cols = changepoints["changepoint_cols"]
        growth_func = changepoints["growth_func"]

        # Adds fourier series for seasonality
        # Initializes fourier series function with None
        # and alters if fourier components are input
        fs_func = None
        fs_cols = []
        if fs_components_df is not None:
            fs_components_df = fs_components_df[fs_components_df["order"] != 0]
            fs_components_df = fs_components_df.reset_index()
            if len(fs_components_df.index) > 0:
                fs_func = fourier_series_multi_fcn(
                    col_names=fs_components_df["name"],  # looks for corresponding column name in input df
                    periods=fs_components_df.get("period"),
                    orders=fs_components_df.get("order"),
                    seas_names=fs_components_df.get("seas_names")
                )
                # Determines fourier series column names for use in "build_features"
                fs_cols = get_fourier_feature_col_names(
                    df=df,
                    time_col=time_col,
                    fs_func=fs_func,
                    conti_year_origin=origin_for_time_vars
                )
        # Removes fs_cols with perfect or almost perfect collinearity for OLS.
        # For example, yearly seasonality with order 4 and quarterly seasonality with order 1, and etc.
        if fit_algorithm in ["linear", "statsmodels_wls", "statsmodels_gls"]:
            # Removes fourier columns with perfect or almost perfect collinearity.
            fs_cols = self.__remove_fourier_col_with_collinearity(
                fs_cols)
            # Also removes these terms from interactions.
            extra_pred_cols = self.__remove_fourier_col_with_collinearity_and_interaction(
                extra_pred_cols, fs_cols)

        # Adds seasonality change point features
        seasonality_changepoint_result = None
        seasonality_changepoints = None
        seasonality_changepoint_cols = []
        if seasonality_changepoints_dict is not None:
            seasonality_changepoint_result = get_seasonality_changepoints(
                df=df,
                time_col=time_col,
                value_col=value_col,
                trend_changepoint_dates=trend_changepoint_dates,
                seasonality_changepoints_dict=seasonality_changepoints_dict
            )
            seasonality_changepoints = seasonality_changepoint_result["seasonality_changepoints"]
            seasonality_available = list(set([x.split("_")[-1] for x in fs_cols]))
            seasonality_changepoint_cols = get_seasonality_changepoint_df_cols(
                df=df,
                time_col=time_col,
                seasonality_changepoints=seasonality_changepoints,
                seasonality_components_df=seasonality_changepoint_result["seasonality_components_df"],
                include_original_block=False,
                include_components=seasonality_available
            )

        features_df = self.__build_silverkite_features(
            df=df,
            time_col=time_col,
            origin_for_time_vars=origin_for_time_vars,
            daily_event_df_dict=daily_event_df_dict,
            changepoint_values=changepoint_values,
            continuous_time_col=continuous_time_col,
            growth_func=growth_func,
            fs_func=fs_func,
            seasonality_changepoint_result=seasonality_changepoint_result,
            changepoint_dates=trend_changepoint_dates)

        # Adds autoregression columns to feature matrix
        autoreg_func = None
        lag_col_names = []
        agg_lag_col_names = []
        min_lag_order = None
        max_lag_order = None
        if autoreg_dict is not None and isinstance(autoreg_dict, str):
            if autoreg_dict.lower() == "auto":
                autoreg_info = self.__get_default_autoreg_dict(
                    freq_in_days=inferred_freq_in_days,
                    forecast_horizon=forecast_horizon,
                    simulation_based=simulation_based)
                autoreg_dict = autoreg_info["autoreg_dict"]
            else:
                raise ValueError(f"The method {autoreg_dict} is not implemented.")

        has_autoreg_structure = False
        if autoreg_dict is not None:
            has_autoreg_structure = True
            autoreg_components = build_autoreg_df(
                value_col=value_col,
                **autoreg_dict)
            autoreg_func = autoreg_components["build_lags_func"]
            lag_col_names = autoreg_components["lag_col_names"]
            agg_lag_col_names = autoreg_components["agg_lag_col_names"]
            min_lag_order = autoreg_components["min_order"]
            max_lag_order = autoreg_components["max_order"]

            if autoreg_func is not None:
                if past_df is not None:
                    # Fills in the gaps for imputation.
                    expected_last_timestamp = df[time_col].min() - to_offset(freq)
                    if past_df[time_col].iloc[-1] < expected_last_timestamp:  # ``past_df`` is already sorted.
                        # If ``past_df`` and ``df`` have gap in between, adds the last timestamp before ``df``.
                        # Then the rest will be filled with NA.
                        log_message(
                            message="There is gaps between ``past_df`` and ``df``. "
                                    "Filling the missing timestamps.",
                            level=LoggingLevelEnum.DEBUG
                        )
                        last_timestamp_df = pd.DataFrame({
                            col: [np.nan] if col != time_col else [expected_last_timestamp] for col in past_df.columns
                        })
                        past_df = past_df.append(last_timestamp_df).reset_index(drop=True)
                    past_df = fill_missing_dates(
                        df=past_df,
                        time_col=time_col,
                        freq=freq)[0]  # `fill_missing_dates` returns a tuple where the first one is the df.
                    # Only takes ``past_df`` that are before ``df``.
                    past_df = past_df[past_df[time_col] <= expected_last_timestamp]
                autoreg_df = self.__build_autoreg_features(
                    df=df,
                    value_col=value_col,
                    autoreg_func=autoreg_func,
                    phase="fit",
                    past_df=past_df)
                features_df = pd.concat([features_df, autoreg_df], axis=1, sort=False)

        # Adds lagged regressor columns to feature matrix
        lagged_regressor_func = None
        lagged_regressor_col_names = []
        lagged_regressor_cols = []
        min_lagged_regressor_order = None
        max_lagged_regressor_order = None
        if lagged_regressor_dict is not None:
            key_remove = []
            for key, value in lagged_regressor_dict.items():
                if isinstance(value, str):
                    if value.lower() != "auto":
                        raise ValueError(f"The method {value} is not implemented.")
                    lag_reg_dict_info = self.__get_default_lagged_regressor_dict(
                        freq_in_days=inferred_freq_in_days,
                        forecast_horizon=forecast_horizon)
                    # If "auto" determines that no lag is needed, remove the key
                    if lag_reg_dict_info["lag_reg_dict"] is None:
                        key_remove += [key]
                    else:
                        lagged_regressor_dict[key] = lag_reg_dict_info["lag_reg_dict"]
            for key in key_remove:
                lagged_regressor_dict.pop(key, None)
                log_message(f"Column {key} has been dropped from `lagged_regressor_dict` and was not "
                            f"used for lagged regressor as determined by 'auto' option.", LoggingLevelEnum.INFO)
        # Converts empty dictionary to None if all keys are removed
        if lagged_regressor_dict == {}:
            lagged_regressor_dict = None

        has_lagged_regressor_structure = False
        if lagged_regressor_dict is not None:
            has_lagged_regressor_structure = True
            lagged_regressor_components = build_autoreg_df_multi(value_lag_info_dict=lagged_regressor_dict)
            lagged_regressor_func = lagged_regressor_components["autoreg_func"]
            lagged_regressor_col_names = lagged_regressor_components["autoreg_col_names"]
            lagged_regressor_cols = lagged_regressor_components["autoreg_orig_col_names"]
            min_lagged_regressor_order = lagged_regressor_components["min_order"]
            max_lagged_regressor_order = lagged_regressor_components["max_order"]

            lagged_regressor_df = self.__build_lagged_regressor_features(
                df=df,
                lagged_regressor_cols=lagged_regressor_cols,
                lagged_regressor_func=lagged_regressor_func,
                phase="fit",
                past_df=None)
            features_df = pd.concat([features_df, lagged_regressor_df], axis=1, sort=False)

        features_df[value_col] = df[value_col].values

        # prediction cols
        # (Includes growth, interactions, if specified in extra_pred_cols)
        pred_cols = extra_pred_cols + fs_cols
        if changepoint_cols is not None:
            pred_cols = pred_cols + changepoint_cols
        if seasonality_changepoint_cols:
            pred_cols = pred_cols + seasonality_changepoint_cols
        if lag_col_names is not None:
            pred_cols = pred_cols + lag_col_names
        if agg_lag_col_names is not None:
            pred_cols = pred_cols + agg_lag_col_names
        if lagged_regressor_col_names is not None:
            pred_cols = pred_cols + lagged_regressor_col_names

        pred_cols = unique_elements_in_list(pred_cols)
        # Drops un-desired predictors
        if drop_pred_cols is not None:
            pred_cols = [col for col in pred_cols if col not in drop_pred_cols]

        # Only uses predictors appearing in ``explicit_pred_cols``
        if explicit_pred_cols is not None:
            pred_cols = explicit_pred_cols

        # Makes sure we don't have an empty regressor string, which will cause patsy formula error.
        if not pred_cols:
            pred_cols = ["1"]
        explan_str = "+".join(pred_cols)
        model_formula_str = value_col + "~" + explan_str
        ind_train = None
        ind_test = None

        if train_test_thresh is not None:
            ind_train = np.where(df[time_col] < train_test_thresh)[0].tolist()
            ind_test = np.where(df[time_col] >= train_test_thresh)[0].tolist()

        trained_model = fit_ml_model_with_evaluation(
            df=features_df,
            model_formula_str=model_formula_str,
            fit_algorithm=fit_algorithm,
            fit_algorithm_params=fit_algorithm_params,
            ind_train=ind_train,
            ind_test=ind_test,
            training_fraction=training_fraction,
            randomize_training=False,
            min_admissible_value=min_admissible_value,
            max_admissible_value=max_admissible_value,
            uncertainty_dict=uncertainty_dict,
            normalize_method=normalize_method,
            regression_weight_col=regression_weight_col)

        # Normalizes the changepoint_values
        normalized_changepoint_values = self.__normalize_changepoint_values(
            changepoint_values=changepoint_values,
            pred_cols=pred_cols,
            continuous_time_col=continuous_time_col,
            normalize_df_func=trained_model["normalize_df_func"]
        )

        # Excludes points with NA that are not used in fitting, similar to "y" and "x_mat".
        trained_model["df_dropna"] = df.loc[trained_model["y"].index]
        # Includes points with NA
        trained_model["df"] = df
        trained_model["num_training_points"] = num_training_points
        trained_model["features_df"] = features_df
        trained_model["min_timestamp"] = min_timestamp
        trained_model["max_timestamp"] = max_timestamp
        trained_model["freq"] = freq
        trained_model["inferred_freq"] = inferred_freq
        trained_model["inferred_freq_in_secs"] = inferred_freq_in_secs
        trained_model["inferred_freq_in_days"] = inferred_freq_in_days
        trained_model["time_col"] = time_col
        trained_model["value_col"] = value_col
        trained_model["origin_for_time_vars"] = origin_for_time_vars
        trained_model["fs_components_df"] = fs_components_df
        trained_model["autoreg_dict"] = autoreg_dict
        trained_model["lagged_regressor_dict"] = lagged_regressor_dict
        trained_model["lagged_regressor_cols"] = lagged_regressor_cols
        trained_model["normalize_method"] = normalize_method
        trained_model["daily_event_df_dict"] = daily_event_df_dict
        trained_model["changepoints_dict"] = changepoints_dict
        trained_model["changepoint_values"] = changepoint_values
        trained_model["normalized_changepoint_values"] = normalized_changepoint_values
        trained_model["continuous_time_col"] = continuous_time_col
        trained_model["growth_func"] = growth_func
        trained_model["fs_func"] = fs_func
        trained_model["has_autoreg_structure"] = has_autoreg_structure
        trained_model["autoreg_func"] = autoreg_func
        # ``past_df`` has been manipulated to have all timestamps (could be with NA) and immediately
        # precedes ``df``. If ``past_df`` is not None, the stored ``past_df`` will be the concatenation of
        # ``past_df`` and ``df``. Otherwise it will be ``df``.
        trained_model["train_df"] = pd.concat([past_df, df], axis=0).reset_index(drop=True)
        trained_model["min_lag_order"] = min_lag_order
        trained_model["max_lag_order"] = max_lag_order
        trained_model["has_lagged_regressor_structure"] = has_lagged_regressor_structure
        trained_model["lagged_regressor_func"] = lagged_regressor_func
        trained_model["min_lagged_regressor_order"] = min_lagged_regressor_order
        trained_model["max_lagged_regressor_order"] = max_lagged_regressor_order
        trained_model["uncertainty_dict"] = uncertainty_dict
        trained_model["pred_cols"] = pred_cols  # predictor column names
        trained_model["last_date_for_fit"] = max(df[time_col])
        trained_model["trend_changepoint_dates"] = trend_changepoint_dates
        trained_model["changepoint_detector"] = changepoint_detector_class  # the ChangepointDetector class with detection results
        trained_model["seasonality_changepoint_dates"] = seasonality_changepoints
        trained_model["seasonality_changepoint_result"] = seasonality_changepoint_result
        trained_model["fit_algorithm"] = fit_algorithm
        trained_model["fit_algorithm_params"] = fit_algorithm_params
        trained_model["adjust_anomalous_info"] = adjust_anomalous_info
        trained_model["impute_info"] = impute_info
        trained_model["forecast_horizon"] = forecast_horizon
        trained_model["forecast_horizon_in_days"] = forecast_horizon_in_days
        trained_model["forecast_horizon_in_timedelta"] = forecast_horizon_in_timedelta
        trained_model["simulation_based"] = simulation_based
        trained_model["simulation_num"] = simulation_num
        trained_model["fast_simulation"] = fast_simulation

        return trained_model

    def predict_no_sim(
            self,
            fut_df,
            trained_model,
            past_df=None,
            new_external_regressor_df=None,
            time_features_ready=False,
            regressors_ready=False):
        """Performs predictions for the dates in ``fut_df``.
        If ``extra_pred_cols`` refers to a column in ``df``, either ``fut_df``
        or ``new_external_regressor_df`` must contain the regressors and the columns needed for lagged regressors.

        Parameters
        ----------

        fut_df: `pandas.DataFrame`
            The data frame which includes the timestamps.
            for prediction and any regressors.
        trained_model : `dict`
            A fitted silverkite model which is the output of ``self.forecast``.
        past_df: `pandas.DataFrame`, optional
            A data frame with past values if autoregressive methods are called
            via autoreg_dict parameter of ``greykite.algo.forecast.silverkite.SilverkiteForecast.py``.
        new_external_regressor_df : `pandas.DataFrame`, optional
            Contains the regressors not already included in `fut_df`.
        time_features_ready : `bool`
            Boolean to denote if time features are already given in df or not.
        regressors_ready : `bool`
            Boolean to denote if regressors are already added to data (``fut_df``).

        Return
        --------
        result: `dict`
            A dictionary with following items

            - "fut_df": `pandas.DataFrame`
                The same as input dataframe with an added column for the response.
                If value_col already appears in ``fut_df``, it will be over-written.
                If ``uncertainty_dict`` is provided as input,
                it will also contain a ``QUANTILE_SUMMARY_COL`` column.
            - "x_mat": `pandas.DataFrame`
                Design matrix of the predictive machine-learning model
            - "features_df": `pandas.DataFrame`
                The features dataframe used for prediction.

        """
        fut_df = fut_df.copy()
        time_col = trained_model["time_col"]
        value_col = trained_model["value_col"]
        max_lag_order = trained_model["max_lag_order"]
        max_lagged_regressor_order = trained_model["max_lagged_regressor_order"]
        min_lagged_regressor_order = trained_model["min_lagged_regressor_order"]
        lagged_regressor_cols = trained_model["lagged_regressor_cols"]

        if max_lag_order is not None and (past_df is None or past_df.shape[0] < max_lag_order):
            warnings.warn(
                "The autoregression lags data had to be interpolated at predict time."
                "`past_df` was either not passed to `predict_silverkite` "
                "or it was not long enough to calculate all necessery lags "
                f"which is equal to `max_lag_order`={max_lag_order}")

        if max_lagged_regressor_order is not None and (
                past_df is None or past_df.shape[0] < max_lagged_regressor_order):
            warnings.warn(
                "The lagged regressor data had to be interpolated at predict time."
                "`past_df` was either not passed to `predict_silverkite` "
                "or it was not long enough to calculate all necessery lags "
                f"which is equal to `max_lagged_regressor_order`={max_lagged_regressor_order}")

        # This is the overall maximum lag order
        max_order = None
        if max_lag_order is not None and max_lagged_regressor_order is not None:
            max_order = np.max([
                max_lag_order,
                max_lagged_regressor_order])
        elif max_lag_order is not None:
            max_order = max_lag_order
        elif max_lagged_regressor_order is not None:
            max_order = max_lagged_regressor_order
        # We only keep the rows needed in ``past_df``
        if past_df is not None and max_order is not None and len(past_df) > max_order:
            past_df = past_df.tail(max_order).reset_index(drop=True)

        # adds extra regressors if provided
        if new_external_regressor_df is None or regressors_ready:
            features_df_fut = fut_df
        else:
            new_external_regressor_df = new_external_regressor_df.reset_index(
                drop=True)
            features_df_fut = pd.concat(
                [fut_df, new_external_regressor_df],
                axis=1,
                sort=False)

        # adds the other features
        if time_features_ready is not True:
            features_df_fut = self.__build_silverkite_features(
                df=features_df_fut,
                time_col=trained_model["time_col"],
                origin_for_time_vars=trained_model["origin_for_time_vars"],
                daily_event_df_dict=trained_model["daily_event_df_dict"],
                changepoint_values=trained_model["changepoint_values"],
                continuous_time_col=trained_model["continuous_time_col"],
                growth_func=trained_model["growth_func"],
                fs_func=trained_model["fs_func"],
                seasonality_changepoint_result=trained_model["seasonality_changepoint_result"],
                changepoint_dates=trained_model["trend_changepoint_dates"])

        # adds autoregression columns to future feature matrix
        if trained_model["autoreg_func"] is not None:
            if past_df is None:
                raise ValueError(
                    "Autoregression was used but no past_df was passed to "
                    "`predict_no_sim`")
            else:
                # If the timestamps in ``fut_df`` are all before ``train_end_timestamp``,
                # then the phase is to calculate fitted values.
                # In this case if there are any values in ``fut_df``,
                # they can be used since the information is known by ``train_end_timestamp``.
                train_end_timestamp = trained_model["max_timestamp"]
                fut_df_max_timestamp = pd.to_datetime(fut_df[time_col]).max()
                phase = "predict" if train_end_timestamp < fut_df_max_timestamp else "fit"
                if phase == "predict":
                    # If phase is predict, we do not allow using ``value_col``.
                    # The AR lags should be enough since otherwise one would use ``predict_via_sim``.
                    df = pd.DataFrame({value_col: [np.nan] * fut_df.shape[0]})
                    df.index = fut_df.index
                else:
                    # If phase is fit, we keep the values in ``value_col``.
                    df = fut_df[[value_col]].copy()
                autoreg_df = self.__build_autoreg_features(
                    df=df,
                    value_col=trained_model["value_col"],
                    autoreg_func=trained_model["autoreg_func"],
                    phase=phase,
                    past_df=past_df[[value_col]])
                features_df_fut = pd.concat(
                    [features_df_fut, autoreg_df],
                    axis=1,
                    sort=False)

        # adds lagged regressor columns to future feature matrix
        if trained_model["lagged_regressor_func"] is not None:
            if past_df is None:
                raise ValueError(
                    "Lagged regressor(s) were used but no past_df was passed to "
                    "`predict_no_sim`")
            else:
                # `build_lagged_regressor_features` requires both ``df`` and ``past_df``
                # to contain the columns needed for lagged regressors
                # case 1: ``min_lagged_regressor_order`` >= ``fut_df.shape[0]``
                # In this case we do not need ``fut_df`` to contain any values for
                # those columns, but we do need to make sure these columns are included
                # as required by `build_lagged_regressor_features`.
                if min_lagged_regressor_order >= fut_df.shape[0]:
                    for col in lagged_regressor_cols:
                        if col not in fut_df.columns:
                            features_df_fut[col] = np.nan
                # case 2: ``min_lagged_regressor_order`` < ``fut_df.shape[0]``
                # In this case ``fut_df`` has to contain those columns, and an error
                # will be raised when `build_lagged_regressor_features` is called.
                lagged_regressor_df = self.__build_lagged_regressor_features(
                    df=features_df_fut.copy(),
                    lagged_regressor_cols=trained_model["lagged_regressor_cols"],
                    lagged_regressor_func=trained_model["lagged_regressor_func"],
                    phase="predict",
                    past_df=past_df[trained_model["lagged_regressor_cols"]])
                features_df_fut = pd.concat(
                    [features_df_fut, lagged_regressor_df],
                    axis=1,
                    sort=False)

        if value_col in features_df_fut.columns:
            # This is to remove duplicate ``value_col`` generated by building features.
            # The duplicates happen during calculating extended fitted values
            # when we intentionally include ``value_col``.
            del features_df_fut[value_col]
        features_df_fut[value_col] = 0.0
        if trained_model["uncertainty_dict"] is None:
            # predictions are stored to ``value_col``
            pred_res = predict_ml(
                fut_df=features_df_fut,
                trained_model=trained_model)
            fut_df = pred_res["fut_df"]
            x_mat = pred_res["x_mat"]
        else:
            # predictions are stored to ``value_col``
            # quantiles are stored to ``QUANTILE_SUMMARY_COL``
            pred_res = predict_ml_with_uncertainty(
                fut_df=features_df_fut,
                trained_model=trained_model)
            fut_df = pred_res["fut_df"]
            x_mat = pred_res["x_mat"]

        # Makes sure to return only necessary columns
        potential_forecast_cols = [time_col, value_col, QUANTILE_SUMMARY_COL, ERR_STD_COL]
        existing_forecast_cols = [col for col in potential_forecast_cols if col in fut_df.columns]
        fut_df = fut_df[existing_forecast_cols]

        return {
            "fut_df": fut_df,
            "x_mat": x_mat,
            "features_df": features_df_fut}

    def predict_n_no_sim(
            self,
            fut_time_num,
            trained_model,
            freq,
            new_external_regressor_df=None,
            time_features_ready=False,
            regressors_ready=False):
        """This is the forecast function which can be used to forecast.
        It accepts extra regressors (``extra_pred_cols``) originally in
        ``df`` via ``new_external_regressor_df``.

        Parameters
        ----------
        fut_time_num : `int`
            number of needed future values
        trained_model : `dict`
            A fitted silverkite model which is the output of ``self.forecast``
        freq : `str`
            Frequency of future predictions.
            Accepts any valid frequency for ``pd.date_range``.
        new_external_regressor_df : `pandas.DataFrame` or None
            Contains the extra regressors if specified.
        time_features_ready : `bool`
            Boolean to denote if time features are already given in df or not.
        regressors_ready : `bool`
            Boolean to denote if regressors are already added to data (``fut_df``).

        Returns
        -------
        result: `dict`
            A dictionary with following items

            - "fut_df": `pandas.DataFrame`
                The same as input dataframe with an added column for the response.
                If value_col already appears in ``fut_df``, it will be over-written.
                If ``uncertainty_dict`` is provided as input,
                it will also contain a ``QUANTILE_SUMMARY_COL`` column.
            - "x_mat": `pandas.DataFrame`
                Design matrix of the predictive machine-learning model

        """
        # creates the future time grid
        dates = pd.date_range(
            start=trained_model["last_date_for_fit"],
            periods=fut_time_num + 1,
            freq=freq)
        dates = dates[dates > trained_model["last_date_for_fit"]]  # drops values up to last_date_for_fit
        fut_df = pd.DataFrame({trained_model["time_col"]: dates.tolist()})

        return self.predict_no_sim(
            fut_df=fut_df,
            trained_model=trained_model,
            past_df=trained_model["df"].copy(),  # observed data used for training the model
            new_external_regressor_df=new_external_regressor_df,
            time_features_ready=time_features_ready,
            regressors_ready=regressors_ready)

    def simulate(
            self,
            fut_df,
            trained_model,
            past_df=None,
            new_external_regressor_df=None,
            include_err=True,
            time_features_ready=False,
            regressors_ready=False):
        """A function to simulate future series.
        If the fitted model supports uncertainty e.g. via ``uncertainty_dict``,
        errors are incorporated into the simulations.

        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The data frame which includes the timestamps
            for prediction and any regressors.
        trained_model : `dict`
            A fitted silverkite model which is the output of ``self.forecast``.
        past_df : `pandas.DataFrame`, optional
            A data frame with past values if autoregressive methods are called
            via ``autoreg_dict`` parameter of ``greykite.algo.forecast.silverkite.SilverkiteForecast.py``
        new_external_regressor_df: `pandas.DataFrame`, optional
            Contains the regressors not already included in ``fut_df``.
        include_err : `bool`
            Boolean to determine if errors are to be incorporated in the simulations.
        time_features_ready : `bool`
            Boolean to denote if time features are already given in df or not.
        regressors_ready : `bool`
            Boolean to denote if regressors are already added to data (``fut_df``).

        Returns
        -------
        result: `dict`
            A dictionary with following items

            - "fut_df": `pandas.DataFrame`
                The same as input dataframe with an added column for the response.
                If value_col already appears in ``fut_df``, it will be over-written.
                If ``uncertainty_dict`` is provided as input,
                it will also contain a ``QUANTILE_SUMMARY_COL`` column.
                Here are the expected columns:

                (1) A time column with the column name being ``trained_model["time_col"]``
                (2) The predicted response in ``value_col`` column.
                (3) Quantile summary response in ``QUANTILE_SUMMARY_COL`` column.
                    This column only appears if the model includes uncertainty.
                (4) Error std in `ERR_STD_COL` column.
                    This column only appears if the model includes uncertainty.

            - "x_mat": `pandas.DataFrame`
                Design matrix of the predictive machine-learning model
            - "features_df": `pandas.DataFrame`
                The features dataframe used for prediction.

        """
        n = len(fut_df)
        past_df_sim = None if past_df is None else past_df.copy()
        fut_df = fut_df.reset_index(drop=True)
        fut_df_sim = fut_df.copy()
        time_col = trained_model["time_col"]
        value_col = trained_model["value_col"]
        fut_df_sim[value_col] = np.nan
        fut_df_sim = fut_df_sim.astype({value_col: "float64"})
        max_lag_order = trained_model["max_lag_order"]
        max_lagged_regressor_order = trained_model["max_lagged_regressor_order"]
        # overall maximum lag order
        max_order = None
        if max_lag_order is not None and max_lagged_regressor_order is not None:
            max_order = np.max([
                max_lag_order,
                max_lagged_regressor_order])
        elif max_lag_order is not None:
            max_order = max_lag_order
        elif max_lagged_regressor_order is not None:
            max_order = max_lagged_regressor_order

        # Only need to keep the last relevant rows to calculate AR terms
        if past_df_sim is not None and max_order is not None and len(past_df_sim) > max_order:
            past_df_sim = past_df_sim.tail(max_order)

        # adds the other features
        if time_features_ready is not True:
            fut_df = self.__build_silverkite_features(
                df=fut_df,
                time_col=time_col,
                origin_for_time_vars=trained_model["origin_for_time_vars"],
                daily_event_df_dict=trained_model["daily_event_df_dict"],
                changepoint_values=trained_model["changepoint_values"],
                continuous_time_col=trained_model["continuous_time_col"],
                growth_func=trained_model["growth_func"],
                fs_func=trained_model["fs_func"],
                seasonality_changepoint_result=trained_model["seasonality_changepoint_result"],
                changepoint_dates=trained_model["trend_changepoint_dates"])

        if new_external_regressor_df is not None and not regressors_ready:
            new_external_regressor_df = new_external_regressor_df.reset_index(
                drop=True)
            fut_df = pd.concat(
                [fut_df, new_external_regressor_df],
                axis=1,
                sort=False)

        x_mat_list = []
        features_df_list = []
        for i in range(n):
            fut_df_sub = fut_df.iloc[[i]].reset_index(drop=True)
            assert len(fut_df_sub) == 1, "the subset dataframe must have only one row"

            pred_res = self.predict_no_sim(
                fut_df=fut_df_sub,
                trained_model=trained_model,
                past_df=past_df_sim,
                new_external_regressor_df=None,
                time_features_ready=True,
                regressors_ready=True)

            fut_df_sub = pred_res["fut_df"]
            # we expect the returned prediction will have only one row
            assert len(fut_df_sub) == 1

            x_mat = pred_res["x_mat"]
            features_df = pred_res["features_df"]
            x_mat_list.append(x_mat)
            features_df_list.append(features_df)

            fut_df_sim.at[i, value_col] = fut_df_sub[value_col].values[0]

            if include_err:
                if ERR_STD_COL in list(fut_df_sub.columns):
                    scale = fut_df_sub[ERR_STD_COL].values[0]
                    err = np.random.normal(
                        loc=0.0,
                        scale=scale)
                    fut_df_sim.at[i, value_col] = (
                            fut_df_sub[value_col].values[0]
                            + err)
                else:
                    raise ValueError(
                        "Error is requested via ``include_err = True``. "
                        f"However the std column ({ERR_STD_COL}) "
                        "does not appear in the prediction")

            # we get the last prediction value and concat that to the end of
            # ``past_df``
            past_df_increment = fut_df_sub[[value_col]]
            assert len(past_df_increment) == 1
            if past_df_sim is None:
                past_df_sim = past_df_increment
            else:
                past_df_sim = pd.concat(
                    [past_df_sim, past_df_increment],
                    axis=0,
                    sort=False)

            # Only need to keep the last relevant rows to calculate AR terms
            if past_df_sim is not None and max_order is not None and len(past_df_sim) > max_order:
                past_df_sim = past_df_sim.tail(max_order)
            past_df_sim = past_df_sim.reset_index(drop=True)

        x_mat = pd.concat(
            x_mat_list,
            axis=0,
            ignore_index=True,  # The index does not matter as we simply want to stack up the data
            sort=False)
        assert len(x_mat) == len(fut_df), "The design matrix size (number of rows) used in simulation must have same size as the input"

        features_df = pd.concat(
            features_df_list,
            axis=0,
            ignore_index=True,  # The index does not matter as we simply want to stack up the data
            sort=False)
        assert len(features_df) == len(fut_df), "The features data size (number of rows) used in simulation must have same size as the input"

        return {
            "sim_df": fut_df_sim[[time_col, value_col]],
            "x_mat": x_mat,
            "features_df": features_df}

    def simulate_multi(
            self,
            fut_df,
            trained_model,
            simulation_num=10,
            past_df=None,
            new_external_regressor_df=None,
            include_err=None):
        """A function to simulate future series.
        If the fitted model supports uncertainty e.g. via ``uncertainty_dict``,
        errors are incorporated into the simulations.

        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The data frame which includes the timestamps
            for prediction and any regressors.
        trained_model : `dict`
            A fitted silverkite model which is the output of ``self.forecast``.
        simulation_num : `int`
            The number of simulated series,
            (each of which have the same number of rows as ``fut_df``)
            to be stacked up row-wise. This number must be larger than zero.
        past_df : `pandas.DataFrame`, optional
            A data frame with past values if autoregressive methods are called
            via ``autoreg_dict`` parameter of ``greykite.algo.forecast.silverkite.SilverkiteForecast.py``.
        new_external_regressor_df: `pandas.DataFrame`, optional
            Contains the regressors not already included in ``fut_df``.
        include_err : `bool`, optional, default None
            Boolean to determine if errors are to be incorporated in the simulations.
            If None, it will be set to True if uncertainty is passed to the model and
            otherwise will be set to False.

        Returns
        -------
        result: `dict`
            A dictionary with follwing items

            - "fut_df_sim" : `pandas.DataFrame`
                Row-wise concatenation of dataframes each being the same as
                input dataframe (``fut_df``) with an added column for the response
                and a new column: "sim_label" to differentiate various simulations.
                The row number of the returned dataframe is:
                    ``simulation_num`` times the row number of ``fut_df``.
                If ``value_col`` already appears in ``fut_df``, it will be over-written.
            - "x_mat": `pandas.DataFrame`
                ``simulation_num`` copies of design matrix of the predictive machine-learning model
                concatenated. An extra index column ("original_row_index")  is also added
                for aggregation when needed.
                Note that the all copies will be the same except for the case where
                auto-regression is utilized.
        """
        assert simulation_num > 0, "simulation number has to be a natural number."

        if include_err is None:
            include_err = trained_model["uncertainty_dict"] is not None

        if trained_model["uncertainty_dict"] is None and include_err:
            raise ValueError(
                "`include_err=True` was passed. "
                "However model does not support uncertainty. "
                "To support uncertainty pass `uncertainty_dict` to the model.")

        value_col = trained_model["value_col"]
        fut_df = fut_df.reset_index(drop=True)  # reset_index returns a copy

        fut_df = self.__build_silverkite_features(
            df=fut_df,
            time_col=trained_model["time_col"],
            origin_for_time_vars=trained_model["origin_for_time_vars"],
            daily_event_df_dict=trained_model["daily_event_df_dict"],
            changepoint_values=trained_model["changepoint_values"],
            continuous_time_col=trained_model["continuous_time_col"],
            growth_func=trained_model["growth_func"],
            fs_func=trained_model["fs_func"],
            seasonality_changepoint_result=trained_model["seasonality_changepoint_result"],
            changepoint_dates=trained_model["trend_changepoint_dates"])

        if new_external_regressor_df is not None:
            new_external_regressor_df = new_external_regressor_df.reset_index(
                drop=True)
            fut_df = pd.concat(
                [fut_df, new_external_regressor_df],
                axis=1)

        def one_sim_func(label):
            """Creates one simulation and labels it with ``label`` in an added
            column : "sim_label"
            """
            sim_res = self.simulate(
                fut_df=fut_df,
                trained_model=trained_model,
                past_df=past_df,
                new_external_regressor_df=None,
                include_err=include_err,
                time_features_ready=True,
                regressors_ready=True)
            sim_df = sim_res["sim_df"]
            x_mat = sim_res["x_mat"]
            sim_df["sim_label"] = label
            # ``x_mat`` does not necessarily have an index column.
            # We keep track of the original index, to be able to aggregate
            # across simulations later.
            x_mat["original_row_index"] = range(len(fut_df))

            return {
                "sim_df": sim_df,
                "x_mat": x_mat}

        sim_res_list = [one_sim_func(i) for i in range(simulation_num)]
        sim_df_list = [sim_res_list[i]["sim_df"] for i in range(simulation_num)]
        x_mat_list = [sim_res_list[i]["x_mat"] for i in range(simulation_num)]

        sim_df = pd.concat(
            sim_df_list,
            axis=0,
            ignore_index=True,  # The index does not matter as we simply want to stack up the data
            sort=False)
        sim_df[value_col] = sim_df[value_col].astype(float)

        x_mat = pd.concat(
            x_mat_list,
            axis=0,
            ignore_index=True,  # The index does not matter as we simply want to stack up the data
            sort=False)
        sim_df[value_col] = sim_df[value_col].astype(float)

        assert len(sim_df) == len(fut_df) * simulation_num
        assert len(x_mat) == len(fut_df) * simulation_num

        return {
            "sim_df": sim_df,
            "x_mat": x_mat}

    def predict_via_sim(
            self,
            fut_df,
            trained_model,
            past_df=None,
            new_external_regressor_df=None,
            simulation_num=10,
            include_err=None):
        """Performs predictions and calculate uncertainty using
        multiple simulations.

        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The data frame which includes the timestamps for prediction
            and possibly regressors.
        trained_model : `dict`
            A fitted silverkite model which is the output of ``self.forecast``
        past_df : `pandas.DataFrame`, optional
            A data frame with past values if autoregressive methods are called
            via autoreg_dict parameter of ``greykite.algo.forecast.silverkite.SilverkiteForecast.py``
        new_external_regressor_df: `pandas.DataFrame`, optional
            Contains the regressors not already included in ``fut_df``.
        simulation_num : `int`, optional, default 10
            The number of simulated series to be used in prediction.
        include_err : `bool`, optional, default None
            Boolean to determine if errors are to be incorporated in the simulations.
            If None, it will be set to True if uncertainty is passed to the model and
            otherwise will be set to False

        Returns
        -------
        result: `dict`
            A dictionary with following items

            - "fut_df": `pandas.DataFrame`
                The same as input dataframe with an added column for the response.
                If value_col already appears in ``fut_df``, it will be over-written.
                If ``uncertainty_dict`` is provided as input,
                it will also contain a ``QUANTILE_SUMMARY_COL`` column.
                Here are the expected columns:

                (1) A time column with the column name being ``trained_model["time_col"]``
                (2) The predicted response in ``value_col`` column.
                (3) Quantile summary response in ``QUANTILE_SUMMARY_COL`` column.
                    This column only appears if the model includes uncertainty.
                (4) Error std in `ERR_STD_COL` column.
                    This column only appears if the model includes uncertainty.

            - "x_mat": `pandas.DataFrame`
                Design matrix of the predictive machine-learning model

        """
        fut_df = fut_df.copy()
        if include_err is None:
            include_err = trained_model["uncertainty_dict"] is not None

        if trained_model["uncertainty_dict"] is None and include_err:
            raise ValueError(
                "`include_err=True` was passed. "
                "However model does not support uncertainty. "
                "To support uncertainty pass `uncertainty_dict` to the model.")

        sim_res = self.simulate_multi(
            fut_df=fut_df,
            trained_model=trained_model,
            simulation_num=simulation_num,
            past_df=past_df,
            new_external_regressor_df=new_external_regressor_df,
            include_err=include_err)
        sim_df = sim_res["sim_df"]
        x_mat = sim_res["x_mat"]

        try:
            quantiles = trained_model["uncertainty_dict"].get(
                "params").get("quantiles")
        except AttributeError:
            quantiles = [0.025, 0.975]

        def quantile_summary(x):
            return tuple(np.quantile(a=x, q=quantiles))

        value_col = trained_model["value_col"]
        time_col = trained_model["time_col"]
        agg_dict = {value_col: ["mean", quantile_summary, "std"]}
        agg_df = sim_df.groupby([time_col], as_index=False).agg(agg_dict)
        # we flatten multi-index (result of aggregation)
        agg_df.columns = [f"{a}_{b}" if b else a for (a, b) in agg_df.columns]
        agg_df.columns = [
            time_col,
            value_col,
            QUANTILE_SUMMARY_COL,
            ERR_STD_COL]

        # When there is no uncertainty dict, the uncertainty columns are NA.
        # In this case, we only keep the other two columns.
        if trained_model["uncertainty_dict"] is None:
            agg_df = agg_df[[time_col, value_col]]

        x_mat = x_mat.groupby(
            ["original_row_index"], as_index=False).agg(np.mean)

        del x_mat["original_row_index"]
        # Checks to see if ``x_mat`` has the right number of rows
        assert len(x_mat) == len(agg_df)
        # Checks to see if predict ``x_mat`` has the same columns as fitted ``x_mat``
        assert list(trained_model["x_mat"].columns) == list(x_mat.columns)

        return {
            "fut_df": agg_df,
            "x_mat": x_mat}

    def predict_via_sim_fast(
            self,
            fut_df,
            trained_model,
            past_df=None,
            new_external_regressor_df=None):
        """Performs predictions and calculates uncertainty using
        one simulation of future and calculate the error separately
        (not relying on multiple simulations). Due to this the prediction
        intervals well into future will be narrower than ``predict_via_sim``
        and therefore less accurate. However there will be a major speed gain
        which might be important in some use cases.

        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The data frame which includes the timestamps for prediction
            and possibly regressors.
        trained_model : `dict`
            A fitted silverkite model which is the output of ``self.forecast``.
        past_df : `pandas.DataFrame` or None, default None
            A data frame with past values if autoregressive methods are called
            via ``autoreg_dict`` parameter of ``greykite.algo.forecast.silverkite.SilverkiteForecast.py``
        new_external_regressor_df: `pandas.DataFrame` or None, default None
            Contains the regressors not already included in ``fut_df``.

        Returns
        -------
        result: `dict`
            A dictionary with following items

            - "fut_df": `pandas.DataFrame`
                The same as input dataframe with an added column for the response.
                If value_col already appears in ``fut_df``, it will be over-written.
                If ``uncertainty_dict`` is provided as input,
                it will also contain a ``QUANTILE_SUMMARY_COL`` column.
                Here are the expected columns:

                (1) A time column with the column name being ``trained_model["time_col"]``
                (2) The predicted response in ``value_col`` column.
                (3) Quantile summary response in ``QUANTILE_SUMMARY_COL`` column.
                    This column only appears if the model includes uncertainty.
                (4) Error std in `ERR_STD_COL` column.
                    This column only appears if the model includes uncertainty.

            - "x_mat": `pandas.DataFrame`
                Design matrix of the predictive machine-learning model
            - "features_df": `pandas.DataFrame`
                The features dataframe used for prediction.

        """
        fut_df = fut_df.copy()
        time_col = trained_model["time_col"]
        value_col = trained_model["value_col"]

        # We only simulate one series without using any error during simulations
        sim_res = self.simulate(
            fut_df=fut_df,
            trained_model=trained_model,
            past_df=past_df,
            new_external_regressor_df=new_external_regressor_df,
            include_err=False)
        x_mat = sim_res["x_mat"]
        features_df = sim_res["features_df"]

        if trained_model["uncertainty_dict"] is None:
            # predictions are stored to ``value_col``
            pred_res = predict_ml(
                fut_df=features_df,
                trained_model=trained_model)
            fut_df = pred_res["fut_df"]
            x_mat = pred_res["x_mat"]
        else:
            # predictions are stored to ``value_col``
            # quantiles are stored to ``QUANTILE_SUMMARY_COL``
            pred_res = predict_ml_with_uncertainty(
                fut_df=features_df,
                trained_model=trained_model)
            fut_df = pred_res["fut_df"]
            x_mat = pred_res["x_mat"]

        # Makes sure to return only necessary columns
        potential_forecast_cols = [time_col, value_col, QUANTILE_SUMMARY_COL, ERR_STD_COL]
        existing_forecast_cols = [col for col in potential_forecast_cols if col in fut_df.columns]
        fut_df = fut_df[existing_forecast_cols]

        return {
            "fut_df": fut_df,
            "x_mat": x_mat,
            "features_df": features_df}

    def predict_n_via_sim(
            self,
            fut_time_num,
            trained_model,
            freq,
            new_external_regressor_df=None,
            simulation_num=10,
            fast_simulation=False,
            include_err=None):
        """This is the forecast function which can be used to forecast.
        This function's predictions are constructed using simulations
        from the fitted series. This supports both ``predict_silverkite_via_sim``
        and ````predict_silverkite_via_sim_fast`` depending on value of the
        passed argument ``fast_simulation``.

        The ``past_df`` is set to be the training data which is available
        in ``trained_model``.
        It accepts extra regressors (``extra_pred_cols``) originally in
        ``df`` via ``new_external_regressor_df``.

        Parameters
        ----------
        fut_time_num : `int`
            number of needed future values
        trained_model : `dict`
            A fitted silverkite model which is the output of ``self.forecast``
        freq : `str`
            Frequency of future predictions.
            Accepts any valid frequency for ``pd.date_range``.
        new_external_regressor_df : `pandas.DataFrame` or None
            Contains the extra regressors if specified.
        simulation_num : `int`, optional, default 10
            The number of simulated series to be used in prediction.
        fast_simulation: `bool`, default False
            Deterimes if fast simulations are to be used. This only impacts models
            which include auto-regression. This method will only generate one simulation
            without any error being added and then add the error using the volatility
            model. The advantage is a major boost in speed during inference and the
            disadvantage is potentially less accurate prediction intervals.
        include_err : `bool`, optional, default None
            Boolean to determine if errors are to be incorporated in the simulations.
            If None, it will be set to True if uncertainty is passed to the model and
            otherwise will be set to False

        Returns
        -------
        result: `dict`
            A dictionary with following items

            - "fut_df": `pandas.DataFrame`
                The same as input dataframe with an added column for the response.
                If value_col already appears in ``fut_df``, it will be over-written.
                If ``uncertainty_dict`` is provided as input,
                it will also contain a ``QUANTILE_SUMMARY_COL`` column.
                Here are the expected columns:

                (1) A time column with the column name being ``trained_model["time_col"]``
                (2) The predicted response in ``value_col`` column.
                (3) Quantile summary response in ``QUANTILE_SUMMARY_COL`` column.
                    This column only appears if the model includes uncertainty.
                (4) Error std in `ERR_STD_COL` column.
                    This column only appears if the model includes uncertainty.

            - "x_mat": `pandas.DataFrame`
                Design matrix of the predictive machine-learning model

        """
        if include_err is None:
            include_err = trained_model["uncertainty_dict"] is not None

        if trained_model["uncertainty_dict"] is None and include_err:
            raise ValueError(
                "`include_err=True` was passed. "
                "However model does not support uncertainty. "
                "To support uncertainty pass `uncertainty_dict` to the model.")

        time_col = trained_model["time_col"]
        value_col = trained_model["value_col"]
        # creates the future time grid
        dates = pd.date_range(
            start=trained_model["last_date_for_fit"],
            periods=fut_time_num + 1,
            freq=freq)
        dates = dates[dates > trained_model["last_date_for_fit"]]  # drops values up to last_date_for_fit
        fut_df = pd.DataFrame({time_col: dates.tolist()})

        past_df = trained_model["df"][[value_col]].reset_index(drop=True)

        if fast_simulation:
            return self.predict_via_sim_fast(
                fut_df=fut_df,
                trained_model=trained_model,
                past_df=past_df,  # observed data used for training the model
                new_external_regressor_df=new_external_regressor_df)

        return self.predict_via_sim(
            fut_df=fut_df,
            trained_model=trained_model,
            past_df=past_df,  # observed data used for training the model
            new_external_regressor_df=new_external_regressor_df,
            simulation_num=simulation_num,
            include_err=include_err)

    def predict(
            self,
            fut_df,
            trained_model,
            freq=None,
            past_df=None,
            new_external_regressor_df=None,
            include_err=None,
            force_no_sim=False,
            simulation_num=None,
            fast_simulation=None,
            na_fill_func=lambda s: s.interpolate().bfill().ffill()):
        """Performs predictions using silverkite model.
        It determines if the prediction should be simulation-based or not and then
        predicts using that setting.
        The function determines if it should use simulation-based predictions or
        that is not necessary.
        Here is the logic for determining if simulations are needed:

        - If the model is not autoregressive, then clearly no simulations are needed
        - If the model is autoregressive, however the minimum lag appearing in the model
          is larger than the forecast horizon, then simulations are not needed.
          This is because the lags can be calculated fully without predicting the future.

        User can overwrite the above behavior and force no simulations using
        ``force_no_sim`` argument, in which case some lags will be imputed.
        This option should not be used by most users.
        Some scenarios where advanced user might want to use
        this is (a) when ``min_lag_order >= forecast_horizon`` does not hold strictly
        but close to hold. (b) user want to predict fast, the autoregression
        lags are normalized. In that case the predictions returned could correspond
        to an approximation of a model without autoregression.


        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The data frame which includes the timestamps for prediction
            and possibly regressors.
        trained_model : `dict`
            A fitted silverkite model which is the output of ``self.forecast``
        freq : `str`, optional, default None
            Timeseries frequency, DateOffset alias.
            See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            for the allowed strings.
            If None, it is extracted from ``trained_model`` input.
        past_df : `pandas.DataFrame` or None, default None
            A data frame with past values if autoregressive methods are called
            via autoreg_dict parameter of ``greykite.algo.forecast.silverkite.SilverkiteForecast.py``.
            Note that this ``past_df`` can be anytime before the training end timestamp, but can not
            exceed it.
        new_external_regressor_df: `pandas.DataFrame` or None, default None
            Contains the regressors not already included in ``fut_df``.
        include_err : `bool`, optional, default None
            Boolean to determine if errors are to be incorporated in the simulations.
            If None, it will be set to True if uncertainty is passed to the model and
            otherwise will be set to False
        force_no_sim : `bool`, default False
            If True, prediction with no simulations is forced.
            This can be useful when speed is of concern or for validation purposes.
            In this case, the potential non-available lags will be imputed.
            Most users should not set this to True as the consequences could be
            hard to quantify.
        simulation_num : `int` or None, default None
            The number of simulations for when simulations are used for generating
            forecasts and prediction intervals. If None, it will be inferred from
            the model (``trained_model``).
        fast_simulation: `bool` or None, default None
            Deterimes if fast simulations are to be used. This only impacts models
            which include auto-regression. This method will only generate one simulation
            without any error being added and then add the error using the volatility
            model. The advantage is a major boost in speed during inference and the
            disadvantage is potentially less accurate prediction intervals.
            If None, it will be inferred from the model (``trained_model``).
        na_fill_func : callable (`pd.DataFrame` -> `pd.DataFrame`)
            default::

                lambda df: df.interpolate().bfill().ffill()

            A function which interpolates missing values in a dataframe.
            The main usage is invoked when there is a gap between the timestamps in ``fut_df``.
            The main use case is when the user wants to predict a period which is not an immediate period
            after training.
            In that case to fill in the gaps, the regressors need to be interpolated/filled.
            The default works by first interpolating the continuous variables.
            Then it uses back-filling and then forward-filling for categorical variables.


        Returns
        -------
        result: `dict`
            A dictionary with following items

            - "fut_df": `pandas.DataFrame`
                The same as input dataframe with an added column for the response.
                If value_col already appears in ``fut_df``, it will be over-written.
                If ``uncertainty_dict`` is provided as input,
                it will also contain a ``QUANTILE_SUMMARY_COL`` column.
                Here are the expected columns:

                (1) A time column with the column name being ``trained_model["time_col"]``
                (2) The predicted response in ``value_col`` column.
                (3) Quantile summary response in ``QUANTILE_SUMMARY_COL`` column.
                    This column only appears if the model includes uncertainty.
                (4) Error std in `ERR_STD_COL` column.
                    This column only appears if the model includes uncertainty.

            - "x_mat": `pandas.DataFrame`
                Design matrix of the predictive machine-learning model

        """
        fut_df = fut_df.copy()
        time_col = trained_model["time_col"]
        value_col = trained_model["value_col"]
        min_lag_order = trained_model["min_lag_order"]
        max_lag_order = trained_model["max_lag_order"]

        if simulation_num is None:
            simulation_num = trained_model["simulation_num"]

        if fast_simulation is None:
            fast_simulation = trained_model["fast_simulation"]

        if freq is None:
            freq = trained_model["freq"]

        if fut_df.shape[0] <= 0:
            raise ValueError("``fut_df`` must be a dataframe of non-zero size.")

        if time_col not in fut_df.columns:
            raise ValueError(
                f"``fut_df`` must include {time_col} as time column, "
                "which is what ``trained_model`` considers to be the time column.")
        fut_df[time_col] = pd.to_datetime(fut_df[time_col])

        # Handles ``past_df``.
        training_past_df = trained_model["train_df"].copy()
        if past_df is None or len(past_df) == 0:
            # In the case that we use ``train_df`` from the ``forecast`` method,
            # we don't check the quality since it's constructed by the method.
            log_message(
                message="``past_df`` not provided during prediction, use the ``train_df`` from training results.",
                level=LoggingLevelEnum.DEBUG
            )
            # The ``past_df`` has been manipulated in the training method to immediately precede the future periods.
            past_df = training_past_df
        else:
            # In the case that ``past_df`` is passed, we combine it with the known dfs.
            past_df[time_col] = pd.to_datetime(past_df[time_col])
            if past_df[time_col].max() > training_past_df[time_col].max():
                raise ValueError("``past_df`` can not have timestamps later than the training end timestamp.")
            # Combines ``past_df`` with ``training_past_df`` to get all available values.
            past_df = (past_df
                       .append(training_past_df)
                       .dropna(subset=[value_col])
                       # When there are duplicates, the value passed from ``past_df`` is kept.
                       .drop_duplicates(subset=time_col)
                       .reset_index(drop=True)
                       )
        # Fills any missing timestamps in ``past_df``. These values will be imputed.
        past_df = fill_missing_dates(
            df=past_df,
            time_col=time_col,
            freq=freq)[0]  # `fill_missing_dates` returns a tuple where the first one is the df.

        # If ``value_col`` appears in user provided ``fut_df``,
        # we remove it to avoid issues in merging
        # also note that such column is unnecessary
        if value_col in fut_df.columns:
            del fut_df[value_col]

        if include_err is None:
            include_err = trained_model["uncertainty_dict"] is not None

        if trained_model["uncertainty_dict"] is None and include_err:
            raise ValueError(
                "`include_err=True` was passed. "
                "However model does not support uncertainty. "
                "To support uncertainty pass `uncertainty_dict` to the model.")

        # If the minimal lag order for lagged regressors is less than the size of fut_df,
        # raise a warning of potential imputation of lagged regressor columns.
        # Note that all lagged regressor columns must be included in ``fut_df`` or ``new_external_regressor_df``
        min_lagged_regressor_order = trained_model["min_lagged_regressor_order"]
        lagged_regressor_dict = trained_model['lagged_regressor_dict']
        if min_lagged_regressor_order is not None and min_lagged_regressor_order < fut_df.shape[0]:
            warnings.warn(
                f"Trained model's `min_lagged_regressor_order` ({int(min_lagged_regressor_order)}) "
                f"is less than the size of `fut_df` ({fut_df.shape[0]}), "
                f"NaN values (if there are any) in lagged regressor columns have been imputed. "
                f"More info: {lagged_regressor_dict}.",
                UserWarning)

        has_autoreg_structure = trained_model["has_autoreg_structure"]

        # In absence of autoregression, we can return quickly.
        # Also note ``fut_df`` can overlap with training times without any issues,
        # ``past_df`` is not needed for autoregression, but may be needed for lagged regression
        # and we do not need to track the overlap.
        if not has_autoreg_structure:
            pred_res = self.predict_no_sim(
                fut_df=fut_df,
                trained_model=trained_model,
                past_df=past_df,
                new_external_regressor_df=new_external_regressor_df,
                time_features_ready=False,
                regressors_ready=False)
            fut_df = pred_res["fut_df"]
            x_mat = pred_res["x_mat"]

            return {
                "fut_df": fut_df,
                "x_mat": x_mat,
                "simulations_not_used": None,
                "fut_df_info": None,
                "min_lag_order": None}

        # From here we assume model has autoregression,
        # because otherwise we would have returned above.

        # Checks if imputation is needed.
        # Writes to log message if imputation is needed for debugging purposes.
        # This happens when
        # (1) ``past_df`` is too short and does not cover the earliest lag needed.
        # (2) ``past_df`` has missing values.
        past_df_sufficient = True
        # The check happens when ``freq`` is not None.
        # We made sure ``freq`` is not None before but wanna add a safeguard.
        if freq is not None:
            pred_min_ts = fut_df[time_col].min()  # the prediction period's minimum timestamp
            past_df_min_ts = past_df[time_col].min()  # the past df's minimum timestamp
            lag_min_ts_needed = pred_min_ts - to_offset(freq) * max_lag_order  # the minimum timestamp needed (max lag)
            # Checks (1) if ``past_df`` covers the period after ``lag_min_ts_needed``.
            past_df_sufficient = past_df_sufficient and (past_df_min_ts <= lag_min_ts_needed)
            if past_df_sufficient:
                # Checks (2) if ``past_df`` has any missing value after ``lag_min_ts_needed``
                past_df_after_min_ts = past_df[past_df[time_col] >= lag_min_ts_needed]
                past_df_sufficient = past_df_sufficient and (past_df_after_min_ts[value_col].isna().sum() == 0)
        if not past_df_sufficient:
            log_message(
                message="``past_df`` is not sufficient, imputation is performed when creating autoregression terms.",
                level=LoggingLevelEnum.DEBUG)

        if new_external_regressor_df is not None:
            fut_df = pd.concat(
                [fut_df, new_external_regressor_df],
                axis=1,
                ignore_index=False,
                sort=False)

        fut_df_info = self.partition_fut_df(
            fut_df=fut_df,
            trained_model=trained_model,
            na_fill_func=na_fill_func,
            freq=freq)

        fut_df_before_training = fut_df_info["fut_df_before_training"]
        fut_df_within_training = fut_df_info["fut_df_within_training"]
        fut_df_after_training_expanded = fut_df_info["fut_df_after_training_expanded"]
        index_after_training_original = fut_df_info["index_after_training_original"]

        inferred_forecast_horizon = fut_df_info["inferred_forecast_horizon"]

        fut_df_list = []
        x_mat_list = []

        # We allow calculating extended fitted values on a longer backward
        # history with imputation if ``past_df`` is not sufficient.
        if fut_df_before_training.shape[0] > 0:
            min_timestamp = fut_df_before_training[time_col].min()
            past_df_before_min_timestamp = past_df[past_df[time_col] < min_timestamp]
            # Since ``fut_df_before_training`` does not have ``value_col`` (dropped above),
            # but we need the actual values for ``fut_df_before_training`` in case the lags
            # are not enough and we don't have simulation, we try to find the values from
            # ``past_df``. If some values are still missing, those values will be imputed.
            fut_df_before_training = fut_df_before_training.merge(
                past_df[[time_col, value_col]],
                on=time_col,
                how="left"
            )
            # Imputation will be done during ``self.predict_no_sim`` if ``past_df_before_min_timestamp``
            # does not have sufficient AR terms.
            pred_res = self.predict_no_sim(
                fut_df=fut_df_before_training,
                trained_model=trained_model,
                past_df=past_df_before_min_timestamp,
                new_external_regressor_df=None,
                time_features_ready=False,
                regressors_ready=True)
            fut_df0 = pred_res["fut_df"]
            x_mat0 = pred_res["x_mat"]
            fut_df_list.append(fut_df0.reset_index(drop=True))
            x_mat_list.append(x_mat0)

        fitted_df = trained_model["fitted_df"]
        fitted_x_mat = trained_model["x_mat"]
        potential_forecast_cols = [time_col, value_col, QUANTILE_SUMMARY_COL, ERR_STD_COL]
        existing_forecast_cols = [col for col in potential_forecast_cols if col in fitted_df.columns]
        fitted_df = fitted_df[existing_forecast_cols]

        # For within training times, we simply use the fitted data
        if fut_df_within_training.shape[0] > 0:
            # Creates a dummy index to get the consistent index on ``fitted_x_mat``
            fut_df0 = pd.merge(
                fut_df_within_training.reset_index(drop=True),
                fitted_df.reset_index(drop=True),
                on=[time_col])

            # Finds out where ``fut_df_within_training`` intersects with ``fitted_df``
            # This is for edge cases where ```fut_df_within_training``` does not have all the
            # times appearing in ``fitted_df``
            fut_df_within_training["dummy_bool_index"] = True
            fut_df_index = pd.merge(
                fut_df_within_training.reset_index(drop=True)[[time_col, "dummy_bool_index"]],
                fitted_df.reset_index(drop=True)[[time_col]],
                on=[time_col],
                how="right")
            ind = fut_df_index["dummy_bool_index"].fillna(False)
            del fut_df_index
            fitted_x_mat = fitted_x_mat.reset_index(drop=True).loc[ind]
            del fut_df_within_training["dummy_bool_index"]

            assert fut_df0.shape[0] == fut_df_within_training.shape[0]

            fut_df_list.append(fut_df0.reset_index(drop=True))
            x_mat_list.append(fitted_x_mat)

        # The future timestamps need to be predicted
        # There are two cases: either simulations are needed or not
        # This is decided as follows:
        simulations_not_used = (not has_autoreg_structure) or force_no_sim or (
                inferred_forecast_horizon <= min_lag_order)

        # ``new_external_regressor_df`` will be passed as None
        # since it is already included in ``fut_df``.
        # ``past_df`` doesn't need to change because either (1) it is passed from
        # this ``predict`` method directly, in which case it should be immediately preceding the
        # ``fut_df_after_training_expanded``;
        # or (2) it is from the training model, where the last term is the last training timestamp,
        # which should also immediately precedes the ``fut_df_after_training_expanded``.
        if fut_df_after_training_expanded is not None and fut_df_after_training_expanded.shape[0] > 0:
            if simulations_not_used:
                pred_res = self.predict_no_sim(
                    fut_df=fut_df_after_training_expanded,
                    trained_model=trained_model,
                    past_df=past_df,
                    new_external_regressor_df=None,
                    time_features_ready=False,
                    regressors_ready=True)
                fut_df0 = pred_res["fut_df"]
                x_mat0 = pred_res["x_mat"]
            elif fast_simulation:
                pred_res = self.predict_via_sim_fast(
                    fut_df=fut_df_after_training_expanded,
                    trained_model=trained_model,
                    past_df=past_df,
                    new_external_regressor_df=None)
                fut_df0 = pred_res["fut_df"]
                x_mat0 = pred_res["x_mat"]
            else:
                pred_res = self.predict_via_sim(
                    fut_df=fut_df_after_training_expanded,
                    trained_model=trained_model,
                    past_df=past_df,
                    new_external_regressor_df=None,
                    simulation_num=simulation_num,
                    include_err=include_err)
                fut_df0 = pred_res["fut_df"]
                x_mat0 = pred_res["x_mat"]
            fut_df0 = fut_df0[index_after_training_original]
            x_mat0 = x_mat0[index_after_training_original]
            fut_df_list.append(fut_df0.reset_index(drop=True))
            x_mat_list.append(x_mat0)

        fut_df_final = pd.concat(
            fut_df_list,
            axis=0,
            ignore_index=True,
            sort=False)

        x_mat_final = pd.concat(
            x_mat_list,
            axis=0,
            ignore_index=True,
            sort=False)

        # Makes sure to return only necessary columns
        potential_forecast_cols = [time_col, value_col, QUANTILE_SUMMARY_COL, ERR_STD_COL]
        existing_forecast_cols = [col for col in potential_forecast_cols if col in fut_df_final.columns]
        fut_df_final = fut_df_final[existing_forecast_cols]

        # Expects the created data has same size as the passed ``fut_df``
        assert len(fut_df_final) == len(fut_df), "The generated data at predict phase must have same length as input ``fut_df``"
        assert len(x_mat_final) == len(fut_df), "The generated data at predict phase must have same length as input ``fut_df``"

        return {
            "fut_df": fut_df_final,
            "x_mat": x_mat_final,
            "simulations_not_used": simulations_not_used,
            "fut_df_info": fut_df_info,
            "min_lag_order": min_lag_order}

    def predict_n(
            self,
            fut_time_num,
            trained_model,
            freq=None,
            past_df=None,
            new_external_regressor_df=None,
            include_err=None,
            force_no_sim=False,
            simulation_num=None,
            fast_simulation=None,
            na_fill_func=lambda s: s.interpolate().bfill().ffill()):
        """This is the forecast function which can be used to forecast a number of
        periods into the future.
        It determines if the prediction should be simulation-based or not and then
        predicts using that setting. Currently if the silverkite model uses
        autoregression simulation-based prediction/CIs are used.

        Parameters
        ----------
        fut_time_num : `int`
            number of needed future values
        trained_model : `dict`
            A fitted silverkite model which is the output of ``self.forecast``
        freq : `str`, optional, default None
            Timeseries frequency, DateOffset alias.
            See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            for the allowed frequencies.
            If None, it is extracted from ``trained_model`` input.
        new_external_regressor_df : `pandas.DataFrame` or None
            Contains the extra regressors if specified.
        simulation_num : `int`, optional, default 10
            The number of simulated series to be used in prediction.
        fast_simulation: `bool` or None, default None
            Deterimes if fast simulations are to be used. This only impacts models
            which include auto-regression. This method will only generate one simulation
            without any error being added and then add the error using the volatility
            model. The advantage is a major boost in speed during inference and the
            disadvantage is potentially less accurate prediction intervals.
            If None, it will be inferred from the model (``trained_model``).
        include_err : `bool` or None, default None
            Boolean to determine if errors are to be incorporated in the simulations.
            If None, it will be set to True if uncertainty is passed to the model and
            otherwise will be set to False
        force_no_sim: `bool`, default False
            If True, prediction with no simulations is forced.
            This can be useful when speed is of concern or for validation purposes.
        na_fill_func : callable (`pd.DataFrame` -> `pd.DataFrame`)
            default::

                lambda df: df.interpolate().bfill().ffill()

            A function which interpolated missing values in a dataframe.
            The main usage is invoked when there is a gap between the timestamps.
            In that case to fill in the gaps, the regressors need to be interpolated/filled.
            The default works by first interpolating the continuous variables.
            Then it uses back-filling and then forward-filling for categorical variables.


        Returns
        -------
        result: `dict`
            A dictionary with following items

            - "fut_df": `pandas.DataFrame`
                The same as input dataframe with an added column for the response.
                If value_col already appears in ``fut_df``, it will be over-written.
                If ``uncertainty_dict`` is provided as input,
                it will also contain a ``QUANTILE_SUMMARY_COL`` column.
                Here are the expected columns:

                (1) A time column with the column name being ``trained_model["time_col"]``
                (2) The predicted response in ``value_col`` column.
                (3) Quantile summary response in ``QUANTILE_SUMMARY_COL`` column.
                    This column only appears if the model includes uncertainty.
                (4) Error std in `ERR_STD_COL` column.
                    This column only appears if the model includes uncertainty.

            - "x_mat": `pandas.DataFrame`
                Design matrix of the predictive machine-learning model

        """
        if freq is None:
            freq = trained_model["freq"]
        # Creates the future time grid
        dates = pd.date_range(
            start=trained_model["last_date_for_fit"],
            periods=fut_time_num + 1,
            freq=freq)
        dates = dates[dates > trained_model["last_date_for_fit"]]  # drops values up to last_date_for_fit
        fut_df = pd.DataFrame({trained_model["time_col"]: dates.tolist()})

        return self.predict(
            fut_df=fut_df,
            trained_model=trained_model,
            past_df=past_df,
            new_external_regressor_df=new_external_regressor_df,
            include_err=include_err,
            force_no_sim=force_no_sim,
            simulation_num=simulation_num,
            fast_simulation=fast_simulation,
            na_fill_func=na_fill_func)

    def partition_fut_df(
            self,
            fut_df,
            trained_model,
            freq,
            na_fill_func=lambda s: s.interpolate().bfill().ffill()):
        """This function takes a dataframe ``fut_df`` which includes the timestamps to forecast
        and a ``trained_model`` returned by
        `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`
        and decomposes
        ``fut_df`` to various dataframes which reflect if the timestamps are before,
        during or after the training periods.
        It also determines if: 'the future timestamps after the training period' are immediately
        after 'the last training period' or if there is some extra gap.
        In that case, this function creates an expanded dataframe which includes the missing
        timestamps as well.
        If ``fut_df`` also includes extra columns (they could be regressor columns),
        this function will interpolate the extra regressor columns.

        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The data frame which includes the timestamps for prediction
            and possibly regressors. Note that the timestamp column in ``fut_df``
            must be the same as ``trained_model["time_col"]``.
            We assume ``fut_df[time_col]`` is pandas.datetime64 type.
        trained_model : `dict`
            A fitted silverkite model which is the output of
            `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`
        freq : `str`
            Timeseries frequency, DateOffset alias.
            See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            for the allowed frequencies.
        na_fill_func : callable (`pd.DataFrame` -> `pd.DataFrame`)
            default::

                lambda df: df.interpolate().bfill().ffill()

            A function which interpolated missing values in a dataframe.
            The main usage is invoked when there is a gap between the timestamps.
            In that case to fill in the gaps, the regressors need to be interpolated/filled.
            The default works by first interpolating the continuous variables.
            Then it uses back-filling and then forward-filling for categorical variables.

        Returns
        -------
        result: `dict`
            A dictionary with following items:

            - ``"fut_freq_in_secs"``: `float`
                The inferred frequency in ``fut_df``
            - ``"training_freq_in_secs"``: `float`
                The inferred frequency in training data
            - ``"index_before_training"``: `list` [`bool`]
                A boolean list to determine which rows of ``fut_df`` include a time
                which is before the training start.
            - ``"index_within_training"``: `list` [`bool`]
                A boolean list to determine which rows of ``fut_df`` include a time
                which is during the training period.
            - ``"index_after_training"``: `list` [`bool`]
                A boolean list to determine which rows of ``fut_df`` include a time
                which is after the training end date.
            - ``"fut_df_before_training"``: `pandas.DataFrame`
                A partition of ``fut_df`` with timestamps before the training start date
            - ``"fut_df_within_training"``: `pandas.DataFrame`
                A partition of ``fut_df`` with timestamps during the training period
            - ``"fut_df_after_training"``: `pandas.DataFrame`
                A partition of ``fut_df`` with timestamps after the training start date
            - ``"fut_df_gap"``: `pandas.DataFrame` or None
                If there is a gap between training end date and the first timestamp
                after the training end date in ``fut_df``, this dataframe can fill the
                gap between the two. In case ``fut_df`` includes extra columns as well,
                the values for those columns will be filled using ``na_fill_func``.
            - ``"fut_df_after_training_expanded"``: `pandas.DataFrame`
                If there is a gap between training end date and the first timestamp
                after the training end date in ``fut_df``, this dataframe will include
                the data for the gaps (``fut_df_gap``) as well as ``fut_df_after_training``.
            - ``"index_after_training_original"``: `list` [`bool`]
                A boolean list to determine which rows of ``fut_df_after_training_expanded``
                correspond to raw data passed by user which are after training end date,
                appearing in ``fut_df``.
                Note that this partition corresponds to ``fut_df_after_training``
                which is the subset of data in ``fut_df`` provided by user and
                also returned by this function.
            - ``"missing_periods_num"``: `int`
                Number of missing timestamps between the last date of training and
                first date in ``fut_df`` appearing after the training end date
            - ``"inferred_forecast_horizon"``: `int`
                This is the inferred forecast horizon from ``fut_df``.
                This is defined to be the distance between the last training end date
                and last date appearing in ``fut_df``.
                Note that this value can be smaller or larger than the number of
                rows of ``fut_df``.
                This is calculated by adding the number of potentially missing timestamps
                and the number of time periods appearing after the training end point.
                Also note if there are no timestamps after the training end point in
                ``fut_df``, this value will be zero.
            - ``"forecast_partition_summary"``: `dict`
                A dictionary which includes the size of various partitions of ``fut_df``
                as well as the missing timestamps if needed. The dictionary keys are as
                follows:

                    - ``"len_before_training"``: the number of time periods before training start
                    - ``"len_within_training"``: the number of time periods within training
                    - ``"len_after_training"``: the number of time periods after training
                    - ``"len_gap"``: the number of missing time periods between training data and
                      future time stamps in ``fut_df``

        """
        fut_df = fut_df.copy()
        training_start_timestamp = trained_model["min_timestamp"]
        training_end_timestamp = trained_model["max_timestamp"]
        training_freq_in_secs = trained_model["inferred_freq_in_secs"]
        time_col = trained_model["time_col"]

        if len(fut_df) > 1:
            fut_df_time_stats = describe_timeseries(
                df=fut_df,
                time_col=time_col)

            if not fut_df_time_stats["regular_increments"]:
                warnings.warn(
                    "``fut_df`` does not have regular time increments")

            if not fut_df_time_stats["increasing"]:
                raise ValueError(f"``fut_df``'s time column {time_col} must be increasing in time")

            fut_freq_in_secs = fut_df_time_stats["freq_in_secs"]
        else:
            # When test_horizon/cv_horizon/forecast_horizon is 1, not all stats above
            # are available, thus it produces an error.
            # The "else" handles this case.
            fut_freq_in_secs = None

        index_before_training = (fut_df[time_col] < training_start_timestamp)
        index_within_training = (
                (fut_df[time_col] >= training_start_timestamp) &
                (fut_df[time_col] <= training_end_timestamp))
        index_after_training = (fut_df[time_col] > training_end_timestamp)

        fut_df_before_training = fut_df[index_before_training]
        fut_df_within_training = fut_df[index_within_training]
        fut_df_after_training = fut_df[index_after_training]

        fut_df_gap = None  # a dataframe which fills in the missing time periods
        missing_periods_num = 0  # the number of missing time periods

        if fut_df_after_training.shape[0] > 0:
            min_timestamp_after_training = min(
                fut_df_after_training[time_col])
            expected_timestamp_after_training = pd.date_range(
                start=training_end_timestamp,
                periods=2,
                freq=freq)[1]

            if min_timestamp_after_training < expected_timestamp_after_training:
                raise ValueError(
                    "The most immediate time in the future is off "
                    f"The last training date: {training_end_timestamp}. "
                    f"The first future period: {min_timestamp_after_training}. "
                    f"Expected first future period is {expected_timestamp_after_training}")
            elif min_timestamp_after_training > expected_timestamp_after_training:
                missing_dates = pd.date_range(
                    start=expected_timestamp_after_training,
                    end=min_timestamp_after_training,
                    freq=freq)
                # The last timestamp is already there, therefore we drop it
                missing_dates = missing_dates[:-1]
                missing_periods_num = len(missing_dates)
                # The length of missing dates is non-zero since there are missing timestamps
                # since ``min_timestamp_after_training > next_period_after_training``
                assert missing_periods_num > 0
                fut_df_gap = pd.DataFrame({time_col: missing_dates.tolist()})

        # `fut_df` might include other columns than `time_col`
        # Those extra columns might be the regressors passed through `fut_df`
        # Therefore we need to ensure `fut_df_gap` includes those columns
        # Also note that those extra columns need to be imputed in that case
        if fut_df_gap is not None and len(fut_df.columns) > 1:
            fut_df_expanded = pd.concat(
                [fut_df_within_training, fut_df_gap, fut_df_after_training],
                axis=0,
                ignore_index=True,
                sort=False)
            # Imputes the missing values
            fut_df_expanded = na_fill_func(fut_df_expanded)
            index = (
                    [False] * fut_df_within_training.shape[0] +
                    [True] * fut_df_gap.shape[0] +
                    [False] * fut_df_after_training.shape[0])
            fut_df_gap = fut_df_expanded[index].copy()

        inferred_forecast_horizon = fut_df_after_training.shape[0]
        if fut_df_gap is not None:
            inferred_forecast_horizon += fut_df_gap.shape[0]

        # Creates an expanded dataframe which includes the missing times
        # between the end of training and the forecast period
        fut_df_after_training_expanded = fut_df_after_training
        index_after_training_original = [True] * fut_df_after_training.shape[0]
        if fut_df_gap is not None:
            fut_df_after_training_expanded = pd.concat(
                [fut_df_gap, fut_df_after_training],
                axis=0,
                ignore_index=True,
                sort=False)
            index_after_training_original = (
                    [False] * fut_df_gap.shape[0] +
                    [True] * fut_df_after_training.shape[0])

        forecast_partition_summary = {
            "len_before_training": fut_df_before_training.shape[0],
            "len_within_training": fut_df_within_training.shape[0],
            "len_after_training": fut_df_after_training.shape[0],
            "len_gap": missing_periods_num
        }

        return {
            "fut_freq_in_secs": fut_freq_in_secs,
            "training_freq_in_secs": training_freq_in_secs,
            "index_before_training": index_before_training,
            "index_within_training": index_within_training,
            "index_after_training": index_after_training,
            "fut_df_before_training": fut_df_before_training,
            "fut_df_within_training": fut_df_within_training,
            "fut_df_after_training": fut_df_after_training,
            "fut_df_gap": fut_df_gap,
            "fut_df_after_training_expanded": fut_df_after_training_expanded,
            "index_after_training_original": index_after_training_original,
            "missing_periods_num": missing_periods_num,
            "inferred_forecast_horizon": inferred_forecast_horizon,
            "forecast_partition_summary": forecast_partition_summary}

    def __build_silverkite_features(
            self,
            df,
            time_col,
            origin_for_time_vars,
            daily_event_df_dict=None,
            changepoint_values=None,
            continuous_time_col=None,
            growth_func=None,
            fs_func=None,
            seasonality_changepoint_result=None,
            changepoint_dates=None):
        """This function adds the prediction model features in training and
        predict phase for ``self.forecast`` internal use but can be called
        outside that context if desired.
        The features are time related features such as seasonality, change points,
        holidays, ...

        Parameters
        ----------
        df : `pandas.DataFrame`
            input dataframe, which could be in training phase or predict phase
        time_col : `str`
            The column name in df representing time for the time series data
            The time column values can be anything that can be parsed by pandas DatetimeIndex
        origin_for_time_vars : `float`
            The time origin used to create continuous variables for time
        daily_event_df_dict : `dict` [`str`, `pandas.DataFrame`] or None, default None
            A dictionary of data frames, each representing events data for the corresponding key.
            The DataFrame has two columns:

                - The first column contains event dates. Must be in a format
                  recognized by `pandas.to_datetime`. Must be at daily
                  frequency for proper join. It is joined against the time
                  in ``df``, converted to a day:
                  ``pd.to_datetime(pd.DatetimeIndex(df[time_col]).date)``.
                - the second column contains the event label for each date

            The column order is important; column names are ignored.
            The event dates must span their occurrences in both the training
            and future prediction period.

            During modeling, each key in the dictionary is mapped to a categorical variable
            named ``f"{EVENT_PREFIX}_{key}"``, whose value at each timestamp is specified
            by the corresponding DataFrame.

            For example, to manually specify a yearly event on September 1
            during a training/forecast period that spans 2020-2022::

                daily_event_df_dict = {
                    "custom_event": pd.DataFrame({
                        "date": ["2020-09-01", "2021-09-01", "2022-09-01"],
                        "label": ["is_event", "is_event", "is_event"]
                    })
                }

            It's possible to specify multiple events in the same df. Two events,
            ``"sep"`` and ``"oct"`` are specified below for 2020-2021::

                daily_event_df_dict = {
                    "custom_event": pd.DataFrame({
                        "date": ["2020-09-01", "2020-10-01", "2021-09-01", "2021-10-01"],
                        "event_name": ["sep", "oct", "sep", "oct"]
                    })
                }

            Use multiple keys if two events may fall on the same date. These events
            must be in separate DataFrames::

                daily_event_df_dict = {
                    "fixed_event": pd.DataFrame({
                        "date": ["2020-09-01", "2021-09-01", "2022-09-01"],
                        "event_name": "fixed_event"
                    }),
                    "moving_event": pd.DataFrame({
                        "date": ["2020-09-01", "2021-08-28", "2022-09-03"],
                        "event_name": "moving_event"
                    }),
                }

            The multiple event specification can be used even if events never overlap. An
            equivalent specification to the second example::

                daily_event_df_dict = {
                    "sep": pd.DataFrame({
                        "date": ["2020-09-01", "2021-09-01"],
                        "event_name": "is_event"
                    }),
                    "oct": pd.DataFrame({
                        "date": ["2020-10-01", "2021-10-01"],
                        "event_name": "is_event"
                    }),
                }

            Note: The events you want to use must be specified in ``extra_pred_cols``.
            These take the form: ``f"{EVENT_PREFIX}_{key}"``, where
            `~greykite.common.constants.EVENT_PREFIX` is the constant.

            Note: Do not use `~greykite.common.constants.EVENT_DEFAULT`
            in the second column. This is reserved to indicate dates that do not
            correspond to an event.
        changepoint_values : `list` of Union[int, float, double]], optional
            The values of the growth term at the changepoints
            Can be generated by the ``get_evenly_spaced_changepoints``,
                `get_custom_changepoints` functions
        continuous_time_col : `str`, optional
            This parameter is used only if ``changepoint_values`` is not None.
            Column to apply growth_func to, to generate changepoint features
        growth_func: callable, optional
            Growth function (scalar -> scalar).
            This parameter is used only if ``changepoint_values`` is not None.
            Changepoint features are created by applying
            ``growth_func`` to ``continuous_time_col`` with offsets.
            If None, uses identity function to use ``continuous_time_col`` directly
            as growth term.
        fs_func: callable, optional
            A function which takes a df as input and returns an output df
            with fourier terms. ``fs_func`` is expected to be constructed using
            ``greykite.common.features.timeseries_features.fourier_series_multi_fcn``, but
            that is not a hard requirement.
        seasonality_changepoint_result: `dict`
            The detected seasonality change points result dictionary, returned by
            `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`.
        changepoint_dates : `list`
            List of change point dates with `strftime` attribute.

        Returns
        -------
        features_df : `pandas.DataFrame`
            a data frame with the added features as new columns
        """
        # adds time features
        features_df = add_time_features_df(
            df=df,
            time_col=time_col,
            conti_year_origin=origin_for_time_vars)

        # adds daily events (e.g. holidays)
        # if daily event data are given, we add them to temporal features data
        # ``date_col`` below is used to join with ``daily_events`` data given
        # in ``daily_event_df_dict``
        # Note: events must be provided for both train and forecast time range
        if daily_event_df_dict is not None:
            if (df.shape[0] > 1
                    and min_gap_in_seconds(df, time_col) > TimeEnum.ONE_DAY_IN_SECONDS.value):
                warnings.warn("The granularity of data is larger than daily. "
                              "Ensure the daily events data match the timestamps")
            features_df = add_daily_events(
                df=features_df,
                event_df_dict=daily_event_df_dict,
                date_col="date")

        # adds changepoints
        if changepoint_values is not None:
            changepoint_features_df = get_changepoint_features(
                features_df,
                changepoint_values,
                continuous_time_col=continuous_time_col,
                growth_func=growth_func,
                changepoint_dates=changepoint_dates)

            assert features_df.shape[0] == changepoint_features_df.shape[0]
            features_df = pd.concat(
                [features_df, changepoint_features_df],
                axis=1,
                sort=False)

        # adds seasonality
        if fs_func is not None:
            fs_features = fs_func(features_df)
            fs_df = fs_features["df"]
            features_df = pd.concat(
                [features_df, fs_df],
                axis=1,
                sort=False)

        # adds seasonality change points
        if seasonality_changepoint_result is not None:
            seasonality_available = list(set([x.split("_")[-1] for x in fs_df.columns])) if fs_func is not None else []
            seasonality_df = build_seasonality_feature_df_from_detection_result(
                df=df,
                time_col=time_col,
                seasonality_changepoints=seasonality_changepoint_result["seasonality_changepoints"],
                seasonality_components_df=seasonality_changepoint_result["seasonality_components_df"],
                include_original_block=False,
                include_components=seasonality_available
            )
            features_df = pd.concat(
                [features_df, seasonality_df.reset_index(drop=True)],
                axis=1,
                sort=False)
        features_df.index = df.index  # assigns a copy of original index
        return features_df

    def __build_autoreg_features(
            self,
            df,
            value_col,
            autoreg_func,
            phase="fit",
            past_df=None):
        """Builds autoregressive df to be used in forecast models.

        Parameters
        ----------
        df : `pandas.Dataframe`
            Dataframe to predict on, passed to ``autoreg_func``.
        value_col : `str`, optional
            This is the column name for the values of the time series.
            This parameter is only required if autoregressive methods are used.
            ``value_col`` is needed at the "predict" phase to add to the ``df``
            with NULL values so it can be appended to ``past_df``.
        autoreg_func : callable, optional
            A function constructed by
            `~greykite.common.features.timeseries_lags.build_autoreg_df`
            with the following signature::

                def autoreg_func(df: pd.DataFrame, past_df: pd.DataFrame) ->
                    dict(lag_df: pd.DataFrame, agg_lag_df: pd.DataFrame)

            See more details for above parameters in
            `~greykite.common.features.timeseries_lags.build_autoreg_df`.
        phase : `str`, optional, default "fit"
            It denotes the phase the features are being built. It can be either of

                - "fit": indicates the features are being built for the fitting phase

                - "predict": indicates the features are being built for predict phase
            This argument is used minimally inside the function.
            Currently only to throw an exception when ``phase = "predict"`` and
            ``autoreg_func`` is not None but ``past_df`` is None.
        past_df : `pandas.DataFrame`, optional
            If autoregressive methods are used by providing ``autoreg_func``,
            this parameter is used to append to ``df`` (from left)
            before calculating the lags

        Returns
        -------
        autoreg_df : `pandas.DataFrame`
            a data frame with autoregression columns
        """
        df = df.copy()
        # we raise an exception if we are in the 'predict' phase
        # and `autoreg_func` is not None
        # but either of ``past_df`` or ``value_col`` is not provided
        # This is because in that case `autoreg_func` will not be able to provide useful
        # lag-based predictors
        if phase == "predict":
            if value_col is None or past_df is None:
                raise ValueError(
                    "At 'predict' phase, if autoreg_func is not None,"
                    " 'past_df' and 'value_col' must be provided to "
                    "`build_autoreg_features`")
            else:
                # in the predict phase, we add the `value_col` to the df
                # to enable `past_df` to be appended
                df[value_col] = np.nan

        if past_df is not None and df is not None:
            assert list(df.columns) == list(past_df.columns), (
                "`autoreg_func(df, past_df)` expects "
                "`df` and `past_df` to have the same columns. "
                "This is not the case: "
                f"`df` columns: {list(df.columns)}; "
                f"`past_df` columns: {list(past_df.columns)}")

        autoreg_data = autoreg_func(df=df, past_df=past_df)
        autoreg_df = pd.concat(autoreg_data.values(), axis=1, sort=False)

        # Preserves the original index of `df`
        autoreg_df.index = df.index
        return autoreg_df

    def __build_lagged_regressor_features(
            self,
            df,
            lagged_regressor_cols,
            lagged_regressor_func,
            phase="fit",
            past_df=None):
        """Builds lagged regressor df to be used in forecast models.

        Parameters
        ----------
        df : `pandas.Dataframe`
            Dataframe to predict on, passed to ``lagged_regressor_func``.
        lagged_regressor_cols : `list` [`str`], optional
            This is the original column names for the lagged regressors.
            This parameter is only required if lagged regressor methods are used.
            ``lagged_regressor_cols`` is needed at the "predict" phase to add to the ``df``
            with NULL values so it can be appended to ``past_df``.
        lagged_regressor_func : callable, optional
            A function constructed by
            `~greykite.common.features.timeseries_lags.build_autoreg_df_multi`
            with the following signature::

                def build_autoreg_df_multi(df: pd.DataFrame, past_df: pd.DataFrame) -> autoreg_df: pd.DataFrame

            See more details for above parameters in
            `~greykite.common.features.timeseries_lags.build_autoreg_df_multi`.
        phase : `str`, optional, default "fit"
            It denotes the phase the features are being built. It can be either of

                - "fit": indicates the features are being built for the fitting phase

                - "predict": indicates the features are being built for predict phase

            This argument is used minimally inside the function.
            Currently only to throw an exception when ``phase = "predict"`` and
            ``lagged_regressor_func`` is not None but ``past_df`` is None.
        past_df : `pandas.DataFrame`, optional
            If lagged regressor methods are used by providing ``lagged_regressor_func``,
            this parameter is used to append to ``df`` (from left)
            before calculating the lags

        Returns
        -------
        lagged_regressor_df : `pandas.DataFrame`
            a data frame with lagged regressor columns
        """
        df = df.copy()
        # we raise an exception if we are in the 'predict' phase
        # and `lagged_regressor_func` is not None
        # but either of ``past_df`` or ``lagged_regressor_cols`` is not provided
        # This is because in that case `lagged_regressor_func` will not be able to provide useful
        # lag-based predictors
        if phase == "predict":
            if lagged_regressor_cols is None or past_df is None:
                raise ValueError(
                    "At 'predict' phase, if lagged_regressor_func is not None,"
                    " 'past_df' and 'lagged_regressor_cols' must be provided to "
                    "`build_lagged_regressor_features`")

        if df is not None:
            df_col_missing = set(lagged_regressor_cols).difference(set(df.columns))
            if len(df_col_missing) > 0:
                raise ValueError(
                    "All columns in `lagged_regressor_cols` must appear in `df`, "
                    f"but {df_col_missing} is missing in `df`.")

        if past_df is not None:
            past_df_col_missing = set(lagged_regressor_cols).difference(set(past_df.columns))
            if len(past_df_col_missing) > 0:
                raise ValueError(
                    "All columns in `lagged_regressor_cols` must appear in `past_df`, "
                    f"but {past_df_col_missing} is missing in `past_df`.")

        lagged_regressor_df = lagged_regressor_func(df=df, past_df=past_df)
        # Preserves the original index of `df`
        lagged_regressor_df.index = df.index
        return lagged_regressor_df

    def __get_default_autoreg_dict(
            self,
            freq_in_days,
            forecast_horizon,
            simulation_based=False):
        """Generates the autoregressive components for forecasting
        given the forecast horizon and time frequency.

        Only if ``forecast_horizon`` is less than or equal to 30 days
        auto-regression is used.
        If ``forecast_horizon`` is larger than 30, the function returns None.

        First, we calculate an integer called ``proper_order`` defined below:
        This will be the smallest integer which is
            (i) larger than ``forecast_horizon``
            (ii) multiple of number of observations per week
        For example, for daily data if ``forecast_horizon`` is 2,
        we let the ``proper_order`` to be 7.
        As another example, if ``forecast_horizon`` is 9, we let the ``proper_order``
        to be 14.
        This order is useful because often the same day of week is best
        correlated with the observed value.
        As an example, for daily data, one aggregated lag predictors
        can be constructed by averaging these lags:
            `[proper_order, proper_order+7, proper_order+7*2]`
        which is equal to `[7, 14, 21]` when `forecast_horizon = 1`.

        Parameters
        ----------
        freq_in_days : `float`
            The frequency of the timeseries in days. e.g. 7.0 for weekly data,
            1.0 for daily data, 0.04166... for hourly data.
        forecast_horizon : `int`
            The number of time intervals into the future which are to be forecasted.
        simulation_based : `bool`, default False
            A boolean to decide if the forecast is performed via simulations or
            without simulations.

        Returns
        -------
        autoreg_dict : `dict` or `None`
            A dictionary which can be passed to
            `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`
            to specify the autoregressive structure.
            See that function's definition for details.
        proper_order : `int` or None
            This will be the smallest integer which is
                (i) larger than ``forecast_horizon``
                (ii) multiple of 7
        """
        forecast_horizon_in_days = freq_in_days * forecast_horizon

        similar_lag = get_similar_lag(freq_in_days)
        proper_order = None
        if similar_lag is not None:
            proper_order = int(np.ceil(forecast_horizon / similar_lag) * similar_lag)

        autoreg_dict = None
        orders = None
        orders_list = []
        interval_list = []

        # Following considers two cases:
        # (i) simulation-based
        # (ii) non-simulation-based
        # In simulation-based we are able to use small orders
        # even for longer horizon ie we allow a orders for which
        # ``order < forecast_horizon``.
        # The above is not possible for non-simulation based approach.
        if simulation_based:
            orders = [1, 2, 3]  # 1st, 2nd, 3rd time lags
            if similar_lag is not None:
                interval_list = [(1, similar_lag), (
                    similar_lag + 1,
                    similar_lag * 2)]  # weekly average of last week, and weekly average of two weeks ago
                orders_list = [[
                    similar_lag,  # (i) same week day in a week which is 7 days prior
                    similar_lag * 2,  # (ii) same week day a week before (i)
                    similar_lag * 3]]  # (iii) same week day in a week before (ii)
        else:  # non-simulation-based case
            if forecast_horizon_in_days <= 30:
                orders = [forecast_horizon, forecast_horizon + 1, forecast_horizon + 2]

                if similar_lag is not None:
                    interval_list = [
                        (forecast_horizon, forecast_horizon + similar_lag - 1),
                        (forecast_horizon + similar_lag, forecast_horizon + similar_lag * 2 - 1)]

                    # The following will induce an average between three lags on the same time of week
                    orders_list = [[
                        proper_order,  # (i) same time in week, in a week which is ``proper_order`` times prior
                        proper_order + similar_lag,  # (ii) same time in a week before (i)
                        proper_order + similar_lag * 2]]  # (iii) same time in a week before (ii)

        if forecast_horizon_in_days <= 30:
            autoreg_dict = {}
            autoreg_dict["lag_dict"] = None
            autoreg_dict["agg_lag_dict"] = None
            if orders is not None:
                autoreg_dict["lag_dict"] = {"orders": orders}
            if len(orders_list) > 0 or len(interval_list) > 0:
                autoreg_dict["agg_lag_dict"] = {
                    "orders_list": orders_list,
                    "interval_list": interval_list}
            autoreg_dict["series_na_fill_func"] = (lambda s: s.bfill().ffill())

        return {
            "proper_order": proper_order,
            "autoreg_dict": autoreg_dict
        }

    def __get_default_lagged_regressor_dict(
            self,
            freq_in_days,
            forecast_horizon):
        """Generates the lagged regressor components for forecasting
        given the forecast horizon and time frequency.
        This applies to ONE lagged regressor column at a time.

        Only if ``forecast_horizon`` is less than or equal to 30 days
        lagged regressors are used.
        If ``forecast_horizon`` is larger than 30, the function returns None.

        First, we calculate an integer called ``proper_order`` defined below:
        This will be the smallest integer which is
            (i) larger than ``forecast_horizon``
            (ii) multiple of number of observations per week
        For example, for daily data if ``forecast_horizon`` is 2,
        we let the ``proper_order`` to be 7.
        As another example, if ``forecast_horizon`` is 9, we let the ``proper_order``
        to be 14.
        This order is useful because often the same day of week is best
        correlated with the observed response.
        As an example, for daily data, one aggregated lagged regressor
        can be constructed by averaging these lags:
            `[proper_order, proper_order+7, proper_order+7*2]`
        which is equal to `[7, 14, 21]` when `forecast_horizon = 1`.

        Parameters
        ----------
        freq_in_days : `float`
            The frequency of the timeseries in days. e.g. 7.0 for weekly data,
            1.0 for daily data, 0.04166... for hourly data.
        forecast_horizon : `int`
            The number of time intervals into the future which are to be forecasted.

        Returns
        -------
        lag_reg_dict : `dict` or `None`
            A dictionary which specifies the lagged regressor structure in ``lagged_regressor_dict``,
            which then can be passed to
            `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`.
            See that function's definition for details.
        proper_order : `int` or None
            This will be the smallest integer which is
                (i) larger than ``forecast_horizon``
                (ii) multiple of 7
        """
        forecast_horizon_in_days = freq_in_days * forecast_horizon

        similar_lag = get_similar_lag(freq_in_days)
        proper_order = None
        if similar_lag is not None:
            proper_order = int(np.ceil(forecast_horizon / similar_lag) * similar_lag)

        lag_reg_dict = None
        orders = None
        orders_list = []
        interval_list = []

        # Since simulation is not allowed for lagged regressors,
        # the minimal lag order has to be greater than or equal to the forecast horizon.
        if forecast_horizon_in_days <= 30:
            if forecast_horizon == 1:
                orders = [1]
            elif proper_order is not None:
                orders = [proper_order]
            else:
                orders = [forecast_horizon]

            if similar_lag is not None:
                interval_list = [
                    (forecast_horizon, forecast_horizon + similar_lag - 1)]

                # The following will induce an average between three lags on the same time of week
                orders_list = [[
                    proper_order,  # (i) same time in week, in a week which is ``proper_order`` times prior
                    proper_order + similar_lag,  # (ii) same time in a week before (i)
                    proper_order + similar_lag * 2]]  # (iii) same time in a week before (ii)

        if forecast_horizon_in_days <= 30:
            lag_reg_dict = {}
            lag_reg_dict["lag_dict"] = None
            lag_reg_dict["agg_lag_dict"] = None
            if orders is not None:
                lag_reg_dict["lag_dict"] = {"orders": orders}
            if len(orders_list) > 0 or len(interval_list) > 0:
                lag_reg_dict["agg_lag_dict"] = {
                    "orders_list": orders_list,
                    "interval_list": interval_list}
            lag_reg_dict["series_na_fill_func"] = (lambda s: s.bfill().ffill())

        return {
            "proper_order": proper_order,
            "lag_reg_dict": lag_reg_dict
        }

    def __normalize_changepoint_values(
            self,
            changepoint_values,
            pred_cols,
            continuous_time_col,
            normalize_df_func):
        """Normalizes the ``changepoint_values`` in
        `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`
        with the same normalize method specified in the model.

        Parameters
        ----------
        changepoint_values : `numpy.array` or `None`
            The trend change point values as returned by
            `~greykite.common.features.timeseries_features.get_changepoint_features_and_values_from_config`
        pred_cols : `list`
            List of names of predictors.
        continuous_time_col : `str`
            The name of continuous time column in ``pred_cols``.
        normalize_df_func : `function` or `None`
            The normalization function as returned by
            `~greykite.common.features.normalize.normalize_df`
            It should be compatible with ``pred_cols`` (generated on the same design matrix).

        Returns
        -------
        normalized_changepoint_values : `numpy.array`
            The normalized change points, on the same scale as the normalized continuous time column.
        """
        if changepoint_values is None:
            return None
        if normalize_df_func is None:
            return changepoint_values
        if continuous_time_col is None:
            continuous_time_col = TimeFeaturesEnum.ct1.value
        new_df = pd.DataFrame(np.zeros([len(changepoint_values), len(pred_cols)]))
        new_df.columns = pred_cols
        new_df[continuous_time_col] = changepoint_values
        normalized_df = normalize_df_func(new_df)
        return normalized_df[continuous_time_col].values.ravel()

    def __remove_fourier_col_with_collinearity(self, fs_cols):
        """Removes fourier series terms with perfect or almost perfect collinearity.
        This function is intended to be used when fitting algorithm is OLS.

        These terms include, for example, yearly seasonality with order 4 and quarterly
        seasonality with order 1; yearly seasonality with order 12, quarterly seasonality
        with order 3 and monthly seasonality with order 1; etc.
        Including these terms together is possible to lead to NaN coefficients in OLS models.

        Note: the function assumes the user includes ``seas_names`` in ``fs_components_df``
        and labels them: weekly, monthly, quarterly and yearly.

        Parameters
        ----------
        fs_cols : `list` [`str`]
            A list of Fourier series column names generated by
            `~greykite.common.features.timeseries_features.fourier_series_multi_fcn`

        Returns
        -------
        fs_cols : `list` [`str`]
            The ``fs_cols`` with collinear cols removed.
            The removed columns always have shorter periods.
        """
        yearly_cols = [col for col in fs_cols if "yearly" in col]
        quarterly_cols = [col for col in fs_cols if "quarterly" in col]
        monthly_cols = [col for col in fs_cols if "monthly" in col]
        weekly_cols = [col for col in fs_cols if "weekly" in col]

        # Assuming the provided seasonality orders are in reasonable ranges.
        # We need to deal with year/quarter, year/month, quarter/month for cos/sin
        # We need to deal with weekly for cos.
        # The Fourier series column names are generated by ``get_fourier_col_name``,
        # and the maximum order of a component can be parsed from the names.
        # For example, yearly seasonality has the form "sin12_ct1_yearly" or "cos12_ct1_yearly".
        # Parsing the number after sin/cos and before the first "_" gives the order.
        max_yearly_order = max([int(col.split("_")[0][3:]) for col in yearly_cols], default=0)
        max_quarterly_order = max([int(col.split("_")[0][3:]) for col in quarterly_cols], default=0)

        # Adds columns to be removed for year/quarter, year/month, quarter/month
        # These always include components with shorter periods.
        # For example, if we have both yearly seasonality with order 4 and quarterly seasonality with order 1,
        # quarterly seasonality with order 1 will be removed.
        removed_cols = []

        # Removes redundant quarterly seasonality with yearly seasonality.
        for i in range(4, max_yearly_order + 1, 4):
            removed_cols += [col for col in quarterly_cols if f"sin{i // 4}_" in col or f"cos{i // 4}_" in col]
        # Removes redundant monthly seasonality with yearly seasonality.
        for i in range(12, max_yearly_order + 1, 12):
            removed_cols += [col for col in monthly_cols if f"sin{i // 12}_" in col or f"cos{i // 12}_" in col]
        # Removes redundant monthly seasonality with quarterly seasonality.
        for i in range(3, max_quarterly_order + 1, 3):
            removed_cols += [col for col in monthly_cols if f"sin{i // 3}_" in col or f"cos{i // 3}_" in col]

        # Adds columns for weekly seasonality.
        # Removes higher order cosine terms because order k and order period - k have the same cosine columns.
        for i in range(int(self._silverkite_seasonality_enum.WEEKLY_SEASONALITY.value.period) // 2 + 1,
                       int(self._silverkite_seasonality_enum.WEEKLY_SEASONALITY.value.period) + 1):
            removed_cols += [col for col in weekly_cols if f"cos{i}_" in col]

        # Removes both sine and cosine terms if the order is greater than the period.
        # The reason is that for weekly order 1 is the same as order 8.
        # This concern only applies to weekly seasonality, because the period 7 is small.
        removed_cols += [col for col in weekly_cols
                         if (int(col.split("_")[0][3:])
                             > self._silverkite_seasonality_enum.WEEKLY_SEASONALITY.value.period)]

        final_cols = [col for col in fs_cols if col not in removed_cols]
        if len(removed_cols) > 0:
            warnings.warn(f"The following Fourier series terms are removed due to collinearity:\n{removed_cols}")
        return final_cols

    def __remove_fourier_col_with_collinearity_and_interaction(
            self,
            extra_pred_cols,
            fs_cols):
        """Removes interaction terms that include fourier series terms removed in
        `~greykite.algo.forecast.silverkite.SilverkiteForecast.__remove_fourier_col_with_collinearity`.
        This function is intended to be used when fitting algorithm is OLS.

        Parameters
        ----------
        extra_pred_cols : `list` [`str`]
            A list of features that include extra interaction terms in
            `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`.
        fs_cols : `list` [`str`]
            A list of Fourier series column names to keep from
            `~greykite.algo.forecast.silverkite.SilverkiteForecast.__remove_fourier_col_with_collinearity`.

        Returns
        -------
        extra_pred_cols : `list` [`str`]
            The ``extra_pred_cols`` with interaction terms including fourier series not in ``fs_col`` removed.
        """
        seas_cols = get_pattern_cols(extra_pred_cols, SEASONALITY_REGEX)
        seas_cols = get_pattern_cols(seas_cols, ":")
        removed_cols = []
        for term in seas_cols:
            if any([(x not in fs_cols) and (re.search(SEASONALITY_REGEX, x)) for x in term.split(":")]):
                removed_cols.append(term)
        extra_pred_cols = [x for x in extra_pred_cols if x not in removed_cols]
        if len(removed_cols) > 0:
            warnings.warn(f"The following interaction terms are removed:\n{removed_cols}\n"
                          f"due to the removal of the corresponding Fourier series terms.")
        return extra_pred_cols
