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
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Albert Chen


from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pandas as pd

from greykite.algo.changepoint.adalasso.changepoint_detector import get_changepoints_dict
from greykite.algo.forecast.silverkite.auto_config import get_auto_growth
from greykite.algo.forecast.silverkite.auto_config import get_auto_holidays
from greykite.algo.forecast.silverkite.auto_config import get_auto_seasonality
from greykite.algo.forecast.silverkite.constants.silverkite_column import SilverkiteColumn
from greykite.algo.forecast.silverkite.constants.silverkite_constant import SilverkiteConstant
from greykite.algo.forecast.silverkite.constants.silverkite_constant import default_silverkite_constant
from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.algo.forecast.silverkite.constants.silverkite_time_frequency import SilverkiteTimeFrequencyEnum
from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import cols_interact
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import generate_holiday_events
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import get_event_pred_cols
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import patsy_categorical_term
from greykite.common import constants as cst
from greykite.common.constants import GrowthColEnum
from greykite.common.enums import SimpleTimeFrequencyEnum
from greykite.common.enums import TimeEnum
from greykite.common.features.timeseries_features import get_available_holidays_across_countries
from greykite.common.features.timeseries_features import get_changepoint_features_and_values_from_config
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import unique_elements_in_list
from greykite.common.python_utils import update_dictionary
from greykite.common.time_properties_forecast import get_forecast_time_properties


class SimpleSilverkiteForecast(SilverkiteForecast):
    """A derived class of `~greykite.algo.forecast.silverkite.SilverkiteForecast`.
    Provides an alternative interface with simplified configuration parameters.
    Produces the same trained model output and uses the same predict functions.
    """

    def __init__(
            self,
            constants: SilverkiteConstant = default_silverkite_constant):
        super().__init__(constants=constants)
        self._silverkite_time_frequency_enum: Type[
            SilverkiteTimeFrequencyEnum] = constants.get_silverkite_time_frequency_enum()
        self._silverkite_holiday: Type[SilverkiteHoliday] = constants.get_silverkite_holiday()
        self._silverkite_column: Type[SilverkiteColumn] = constants.get_silverkite_column()

    def convert_params(
            self,
            df: pd.DataFrame,
            time_col: str,
            value_col: str,
            time_properties: Optional[Dict] = None,
            freq: Optional[str] = None,
            forecast_horizon: Optional[int] = None,
            origin_for_time_vars: Optional[float] = None,
            train_test_thresh: Optional[datetime] = None,
            training_fraction: Optional[float] = 0.9,
            fit_algorithm: str = "ridge",
            fit_algorithm_params: Optional[Dict] = None,
            auto_holiday: bool = False,
            holidays_to_model_separately: Optional[Union[str, List[str]]] = "auto",
            holiday_lookup_countries: Optional[Union[str, List[str]]] = "auto",
            holiday_pre_num_days: int = 2,
            holiday_post_num_days: int = 2,
            holiday_pre_post_num_dict: Optional[Dict] = None,
            daily_event_df_dict: Optional[Dict] = None,
            auto_growth: bool = False,
            changepoints_dict: Optional[Dict] = None,
            auto_seasonality: bool = False,
            yearly_seasonality: Union[bool, str, int] = "auto",
            quarterly_seasonality: Union[bool, str, int] = "auto",
            monthly_seasonality: Union[bool, str, int] = "auto",
            weekly_seasonality: Union[bool, str, int] = "auto",
            daily_seasonality: Union[bool, str, int] = "auto",
            max_daily_seas_interaction_order: Optional[int] = None,
            max_weekly_seas_interaction_order: Optional[int] = None,
            autoreg_dict: Optional[Dict] = None,
            past_df: Optional[pd.DataFrame] = None,
            lagged_regressor_dict: Optional[Dict] = None,
            seasonality_changepoints_dict: Optional[Dict] = None,
            min_admissible_value: Optional[float] = None,
            max_admissible_value: Optional[float] = None,
            uncertainty_dict: Optional[Dict] = None,
            normalize_method: Optional[str] = None,
            growth_term: Optional[str] = cst.GrowthColEnum.linear.name,
            regressor_cols: Optional[List[str]] = None,
            feature_sets_enabled: Optional[Union[bool, str, Dict[str, Optional[Union[bool, str]]]]] = "auto",
            extra_pred_cols: Optional[List[str]] = None,
            drop_pred_cols: Optional[List[str]] = None,
            explicit_pred_cols: Optional[List[str]] = None,
            regression_weight_col: Optional[str] = None,
            simulation_based: Optional[bool] = False,
            simulation_num: int = 10,
            fast_simulation: bool = False):
        """Converts parameters of
        :func:`~greykite.algo.forecast.silverkite.forecast_simple_silverkite` into those
        of :func:`~greykite.algo.forecast.forecast_silverkite.SilverkiteForecast::forecast`.

        Makes it easier to set parameters to ``SilverkiteForecast::forecast`` suitable for most forecasting problems.
        Provides data-aware defaults for seasonality and interaction terms. Provides a simple
        configuration of holidays from an internal holiday database, and user-friendly configuration
        for growth and regressors.

        These parameters can be set from a plain-text config (e.g. no pandas dataframes).
        The parameter list is intentionally flat to facilitate hyperparameter grid search. Every
        parameter is either a parameter of ``SilverkiteForecast::forecast`` or a tuning parameter.

        Notes
        -----
        The basic parameters are identical to ``SilverkiteForecast::forecast``.
        The more complex parameters are specified via config parameters:

        * ``daily_event_df_dict`` (via ``holiday*``)
        * ``fs_components_df`` (via `*_seasonality``)
        * ``extra_pred_cols`` (via ``holiday*``, ``*seas*``, ``growth_term``,
          ``regressor_cols``, ``feature_sets_enabled``, ``extra_pred_cols``)

        Parameters
        ----------
        df : `pandas.DataFrame`
            A data frame which includes the timestamp column
            as well as the value column. This is the ``df`` for
            training the model, not for future prediction.
        time_col : `str`
            The column name in `df` representing time for the time series data
            The time column can be anything that can be parsed by pandas DatetimeIndex
        value_col: `str`
            The column name which has the value of interest to be forecasted
        time_properties : `dict` [`str`, `any`] or None, optional
            Time properties dictionary (likely produced by
            `~greykite.common.time_properties_forecast.get_forecast_time_properties`)
            with keys:

                ``"ts"`` : `UnivariateTimeSeries` or None
                    ``df`` converted to a ``UnivariateTimeSeries``.
                ``"period"`` : `int`
                    Period of each observation (i.e. minimum time between observations, in seconds).
                ``"simple_freq"`` : `SimpleTimeFrequencyEnum`
                    ``SimpleTimeFrequencyEnum`` member corresponding to data frequency.
                ``"num_training_points"`` : `int`
                    Number of observations for training.
                ``"num_training_days"`` : `int`
                    Number of days for training.
                ``"start_year"`` : `int`
                    Start year of the training period.
                ``"end_year"`` : `int`
                    End year of the forecast period.
                ``"origin_for_time_vars"`` : `float`
                    Continuous time representation of the first date in ``df``.

            In this function,

                - ``start_year`` and ``end_year`` are used to define ``daily_event_df_dict``.
                - ``simple_freq`` and ``num_training_days`` are used to define ``fs_components_df``.
                - ``simple_freq`` and ``num_training_days`` are used to set default ``feature_sets_enabled``.
                - ``origin_for_time_vars`` is used to set default ``origin_for_time_vars``.
                - the other parameters are ignored

            It is okay if ``num_training_points``, ``num_training_days``, ``start_year``, ``end_year``
            are computed for a superset of ``df``. This allows CV splits and backtest, which train on
            partial data, to use the same data-aware model parameters as the forecast on all training data.

            If None, the values are computed for ``df``. This corresponds to using the same
            modeling *approach* on the CV splits and backtest from `forecast_pipeline`, without
            requiring the same parameters. In this case, make sure ``forecast_horizon`` is at
            least as large as the test period for the split, to ensure all holidays are captured.
        freq : `str` or None, optional, default `None`
            Frequency of input data.
            Used to compute ``time_properties`` only if ``time_properties is None``.
            Frequency strings can have multiples, e.g. '5H'.
            See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            for a list of frequency aliases.
            If None, inferred by `pandas.infer_freq`.
            Provide this parameter if ``df`` has missing timepoints.
        forecast_horizon : `int` or None, optional, default `None`
            Number of periods to forecast into the future. Must be > 0.
            Used to compute ``time_properties`` only if ``time_properties is None``.
            If None, default is determined by input data frequency.
            Used to determine forecast end date, to pull the appropriate holiday data.
            Should be at least as large as the prediction period (if this function
            is called from ``forecast_pipeline``, the prediction period for different
            splits is set via ``cv_horizon``, ``test_horizon``, ``forecast_horizon``).
        origin_for_time_vars : `float` or None, optional, default `None`
            The time origin used to create continuous variables for time.
            If None, uses the value from ``time_properties``.
        train_test_thresh : `datetime.datetime` or None, optional, default `None`
            e.g. datetime.datetime(2019, 6, 30)
            The threshold for training and testing split.
            Note that the final returned model is trained using all data.
            If None, training split is based on ``training_fraction``.
        training_fraction : `float` or None, optional, default 0.9
            The fraction of data used for training (0.0 to 1.0)
            Used only if ``train_test_thresh is None``.
            If this is also None or 1.0, then we skip testing
            and train on the entire dataset.
        fit_algorithm : `str`, optional, default "linear"
            The type of predictive model used in fitting.

            See `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
            for available options and their parameters.
        fit_algorithm_params : `dict` or None, optional, default None
            Parameters passed to the requested fit_algorithm.
            If None, uses the defaults in `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
        auto_holiday : `bool`, default False
            Whether to automatically infer holiday configuration based on the input timeseries.
            The candidate lookup countries are specified by ``holiday_lookup_countries``.
            If True, the following parameters will be ignored:

                * "holidays_to_model_separately"
                * "holiday_pre_num_days"
                * "holiday_post_num_days"
                * "holiday_pre_post_num_dict"

            For details, see `~greykite.algo.common.holiday_inferrer.HolidayInferrer`.
            Extra events specified in ``daily_event_df_dict`` will be added to the inferred holidays.
        holiday_lookup_countries : `list` [`str`] or "auto" or None, optional, default "auto"
            The countries that contain the holidays you intend to model
            (``holidays_to_model_separately``).

                * If "auto", uses a default list of countries
                  that contain the default ``holidays_to_model_separately``.
                  See `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO`.
                * If a list, must be a list of country names.
                * If None or an empty list, no holidays are modeled.

        holidays_to_model_separately : `list` [`str`] or "auto" or `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES` or None, optional, default "auto"  # noqa: E501
            Which holidays to include in the model.
            The model creates a separate key, value for each item in ``holidays_to_model_separately``.
            The other holidays in the countries are grouped together as a single effect.

                * If "auto", uses a default list of important holidays.
                  See `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO`.
                * If `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES`,
                  uses all available holidays in ``holiday_lookup_countries``. This can often
                  create a model that has too many parameters, and should typically be avoided.
                * If a list, must be a list of holiday names.
                * If None or an empty list, all holidays in ``holiday_lookup_countries`` are grouped together
                  as a single effect.

            Use ``holiday_lookup_countries`` to provide a list of countries where these holiday occur.
        holiday_pre_num_days : `int`, default 2
            Model holiday effects for ``holiday_pre_num_days`` days before the holiday.
        holiday_post_num_days : `int`, default 2
            Model holiday effects for ``holiday_post_num_days`` days after the holiday.
        holiday_pre_post_num_dict : `dict` [`str`, (`int`, `int`)] or None, default None
            Overrides ``pre_num`` and ``post_num`` for each holiday in
            ``holidays_to_model_separately``.
            For example, if ``holidays_to_model_separately`` contains "Thanksgiving" and "Labor Day",
            this parameter can be set to ``{"Thanksgiving": [1, 3], "Labor Day": [1, 2]}``,
            denoting that the "Thanksgiving" ``pre_num`` is 1 and ``post_num`` is 3, and "Labor Day"
            ``pre_num`` is 1 and ``post_num`` is 2.
            Holidays not specified use the default given by ``pre_num`` and ``post_num``.
        daily_event_df_dict : `dict` [`str`, `pandas.DataFrame`] or None, default None
            A dictionary of data frames, each representing events data for the corresponding key.
            Specifies additional events to include besides the holidays specified above. The format
            is the same as in `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`.
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

            Note: All these events are automatically added to the model. There is no need
            to specify them in ``extra_pred_cols`` as you would for
            `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`.

            Note: Do not use `~greykite.common.constants.EVENT_DEFAULT`
            in the second column. This is reserved to indicate dates that do not
            correspond to an event.
        auto_growth : `bool`, default False
            Whether to automatically infer growth configuration.
            If True, the growth term and automatically changepoint detection configuration
            will be inferred from input timeseries,
            and the following parameters will be ignored:

                * "growth_term"
                * "changepoints_dict" (Except parameters that controls how custom changepoint
                  are combined with automatically detected changepoints. These parameters include
                  "dates", "combine_changepoint_min_distance" and "keep_detected".)

            For detail, see
            `~greykite.algo.changepoint.adalasso.auto_changepoint_params.generate_trend_changepoint_detection_params`.
        changepoints_dict : `dict` or None, optional, default None
            Specifies the changepoint configuration.

            ``"method"``: `str`
                The method to locate changepoints.
                Valid options:

                    - "uniform". Places n_changepoints evenly spaced changepoints to allow growth to change.
                    - "custom". Places changepoints at the specified dates.
                    - "auto". Automatically detects change points. For configuration, see
                      `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_trend_changepoints`

                Additional keys to provide parameters for each particular method are described below.
            ``"continuous_time_col"``: `str`, optional
                Column to apply ``growth_func`` to, to generate changepoint features
                Typically, this should match the growth term in the model
            ``"growth_func"``: callable or None, optional
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

        auto_seasonality : `bool`, default False
            Whether to automatically infer seasonality orders.
            If True, the seasonality orders will be automatically inferred from input timeseries
            and the following parameters will be ignored unless the value is ``False``:

                * "yearly_seasonality"
                * "quarterly_seasonality"
                * "monthly_seasonality"
                * "weekly_seasonality"
                * "daily_seasonality"

            If any of the above parameter's value is ``False``,
            the corresponding seasonality order will be forced to be zero,
            regardless of the inferring result.
            For detail, see `~greykite.algo.common.seasonality_inferrer.SeasonalityInferrer`.
        yearly_seasonality : `str` or `bool` or `int`
            Determines the yearly seasonality.
            'auto', True, False, or a number for the Fourier order
        quarterly_seasonality : `str` or `bool` or `int`
            Determines the quarterly seasonality.
            'auto', True, False, or a number for the Fourier order
        monthly_seasonality : `str` or `bool` or `int`
            Determines the monthly seasonality.
            'auto', True, False, or a number for the Fourier order
        weekly_seasonality : `str` or `bool` or `int`
            Determines the weekly seasonality.
            'auto', True, False, or a number for the Fourier order
        daily_seasonality : `str` or `bool` or `int`
            Determines the daily seasonality.
            'auto', True, False, or a number for the Fourier order
        max_daily_seas_interaction_order : `int` or None, optional, default `None`
            Max fourier order for interaction terms with daily seasonality.
            If None, uses all available terms.
        max_weekly_seas_interaction_order : `int` or None, optional, default `None`
            Max fourier order for interaction terms with weekly seasonality.
            If None, uses all available terms.
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
            This is not necessarily needed since imputation is available,
            however, if such data is available but not used in training for speed purposes,
            they can be passed here to build more accurate autoregression features.
        lagged_regressor_dict : `dict` or None, default None
            A dictionary with arguments for `greykite.common.features.timeseries_lags.build_autoreg_df_multi`.
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
        seasonality_changepoints_dict : `dict` or None, optional, default `None`
            The parameter dictionary for seasonality change point detection. Parameters are in
            `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`.
            Note ``df``, ``time_col``, ``value_col`` and ``trend_changepoints`` are auto populated,
            and do not need to be provided.
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
                    in ``fit_prediction_model`` which calculates CIs using residuals
                ``"params"``: `dict`
                    A dictionary of parameters needed for
                    the requested ``uncertainty_method``. For example, for
                    ``uncertainty_method="simple_conditional_residuals"``, see
                    parameters of `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`,
                    listed briefly here:

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
        growth_term : `str` or None, optional, default "ct1"
            How to model the growth. Valid options are
            {"linear", "quadratic", "sqrt", "cuberoot"}.
            See `~greykite.common.constants.GrowthColEnum`.
        regressor_cols : `list` [`str`] or None, optional, default None
            The columns in ``df`` to use as regressors.
            These must be provided during prediction as well.
        feature_sets_enabled: `dict` [`str`, `bool` or "auto" or None] or `bool` or "auto" or None, default "auto"
            Whether to include interaction terms and categorical variables to increase model flexibility.

            If a `dict`, boolean values indicate whether include various sets of features in the model.
            The following keys are recognized
            (from `~greykite.algo.forecast.silverkite.constants.silverkite_column.SilverkiteColumn`):

                ``"COLS_HOUR_OF_WEEK"`` : `str`
                    Constant hour of week effect
                ``"COLS_WEEKEND_SEAS"`` : `str`
                    Daily seasonality interaction with is_weekend
                ``"COLS_DAY_OF_WEEK_SEAS"`` : `str`
                    Daily seasonality interaction with day of week
                ``"COLS_TREND_DAILY_SEAS"`` : `str`
                    Allow daily seasonality to change over time by is_weekend
                ``"COLS_EVENT_SEAS"`` : `str`
                    Allow sub-daily event effects
                ``"COLS_EVENT_WEEKEND_SEAS"`` : `str`
                    Allow sub-daily event effect to interact with is_weekend
                ``"COLS_DAY_OF_WEEK"`` : `str`
                    Constant day of week effect
                ``"COLS_TREND_WEEKEND"`` : `str`
                    Allow trend (growth, changepoints) to interact with is_weekend
                ``"COLS_TREND_DAY_OF_WEEK"`` : `str`
                    Allow trend to interact with day of week
                ``"COLS_TREND_WEEKLY_SEAS"`` : `str`
                    Allow weekly seasonality to change over time

            The following dictionary values are recognized:

                - True: include the feature set in the model
                - False: do not include the feature set in the model
                - None: do not include the feature set in the model
                - "auto" or not provided: use the default setting based on data frequency and size

            If not a `dict`:

                - if a boolean, equivalent to a dictionary with all values set to the boolean.
                - if None, equivalent to a dictionary with all values set to False.
                - if "auto", equivalent to a dictionary with all values set to "auto".

        extra_pred_cols : `list` [`str`] or None, optional, default `None`
            Columns to include in ``extra_pred_cols`` for ``SilverkiteForecast::forecast``.
            Other columns are added to ``extra_pred_cols`` by the other
            parameters of this function (i.e. ``holidays_*``, ``growth_term``,
            ``regressors``, ``feature_sets_enabled``).
            If `None`, treated is the same as [].
        drop_pred_cols : `list` [`str`] or None, default None
            Names of predictor columns to be dropped from the final model.
            Ignored if None.
        explicit_pred_cols : `list` [`str`] or None, default None
            Names of the explicit predictor columns which will be
            the only variables in the final model. Note that this overwrites
            the generated predictors in the model and may include new
            terms not appearing in the predictors (e.g. interaction terms).
            Ignored if None.
        regression_weight_col : `str` or None, default None
            The column name for the weights to be used in weighted regression version
            of applicable machine-learning models.
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
        parameters : `dict`
            Parameters to call :func:`~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`.
        """
        if extra_pred_cols is None:
            extra_pred_cols = []
        else:
            # Does not modify the input list
            extra_pred_cols = extra_pred_cols.copy()

        # Specifies regressors (via ``extra_pred_cols``)
        if regressor_cols is None:
            regressor_cols = []
        extra_pred_cols += regressor_cols

        if time_properties is None:
            # ``df`` only contains the dates for training,
            # so we can use ``use_univariate_ts=False``.
            # ``forecast_horizon`` must be at least as large as
            # the actual size of the test set / forecast set
            # in order to pull all holidays
            time_properties = get_forecast_time_properties(
                df=df,
                time_col=time_col,
                value_col=value_col,
                freq=freq,
                regressor_cols=regressor_cols,
                forecast_horizon=forecast_horizon)

        if time_properties is not None:
            forecast_horizon = forecast_horizon or time_properties.get("forecast_horizon")

        if origin_for_time_vars is None:
            origin_for_time_vars = time_properties["origin_for_time_vars"]

        # Specifies seasonality (added to ``extra_pred_cols`` by `SilverkiteForecast::forecast`)
        # Seasonality orders are automatically inferred if ``auto_seasonality`` is True,
        # and they are pulled from configuration if False.
        if auto_seasonality:
            seasonality_dict = get_auto_seasonality(
                df=df,
                time_col=time_col,
                value_col=value_col,
                yearly_seasonality=(yearly_seasonality is not False),
                quarterly_seasonality=(quarterly_seasonality is not False),
                monthly_seasonality=(monthly_seasonality is not False),
                weekly_seasonality=(weekly_seasonality is not False),
                daily_seasonality=(daily_seasonality is not False)
            )
        else:
            seasonality_dict = {
                "yearly_seasonality": yearly_seasonality,
                "quarterly_seasonality": quarterly_seasonality,
                "monthly_seasonality": monthly_seasonality,
                "weekly_seasonality": weekly_seasonality,
                "daily_seasonality": daily_seasonality,
            }

        fs_components_df = self.__get_silverkite_seasonality(
            simple_freq=time_properties["simple_freq"].name,
            num_days=time_properties["num_training_days"],
            seasonality=seasonality_dict)

        # Specifies growth (via ``extra_pred_cols``)
        # The ``growth_term`` and ``changepoints_dict`` are automatically inferred
        # and overridden if ``auto_growth`` is True.
        if auto_growth:
            growth_result = get_auto_growth(
                df=df,
                time_col=time_col,
                value_col=value_col,
                forecast_horizon=forecast_horizon,
                changepoints_dict_override=changepoints_dict
            )
            growth_term = growth_result["growth_term"]
            changepoints_dict = growth_result["changepoints_dict"]
        growth_term_formula = None
        if growth_term is not None:
            growth_term_formula = GrowthColEnum[growth_term].value
            extra_pred_cols += [growth_term_formula]

        # Specifies events (via ``daily_event_df_dict``, ``extra_pred_cols``).
        # Constant daily effect.
        holiday_df_dict = self.__get_silverkite_holidays(
            auto_holiday=auto_holiday,
            holiday_lookup_countries=holiday_lookup_countries,
            holidays_to_model_separately=holidays_to_model_separately,
            start_year=time_properties["start_year"],
            end_year=time_properties["end_year"],
            pre_num=holiday_pre_num_days,
            post_num=holiday_post_num_days,
            pre_post_num_dict=holiday_pre_post_num_dict,
            df=df,
            time_col=time_col,
            value_col=value_col,
            forecast_horizon=forecast_horizon)
        if holiday_df_dict is not None:
            # Adds holidays to the user-specified events,
            # giving preference to user events
            # if there are conflicts
            daily_event_df_dict = update_dictionary(
                holiday_df_dict,
                overwrite_dict=daily_event_df_dict)

        if not daily_event_df_dict:
            # Sets empty dictionary to None
            daily_event_df_dict = None

        extra_pred_cols += get_event_pred_cols(daily_event_df_dict)

        # Specifies ``extra_pred_cols`` (interactions and additional model terms).
        # Seasonality interaction order is limited by the available order and max requested.
        daily_seas_interaction_order = self.__get_seasonality_order_from_dataframe(
            seasonality=self._silverkite_seasonality_enum.DAILY_SEASONALITY.value,
            fs=fs_components_df,
            max_order=max_daily_seas_interaction_order
        )

        weekly_seas_interaction_order = self.__get_seasonality_order_from_dataframe(
            seasonality=self._silverkite_seasonality_enum.WEEKLY_SEASONALITY.value,
            fs=fs_components_df,
            max_order=max_weekly_seas_interaction_order
        )

        # updates `changepoints_dict`, unchanged if not "method" == "auto"
        changepoints_dict, changepoint_detector = get_changepoints_dict(
            df=df,
            time_col=time_col,
            value_col=value_col,
            changepoints_dict=changepoints_dict)

        # determines changepoint column names
        if changepoints_dict is not None:
            changepoints = get_changepoint_features_and_values_from_config(
                df=df,  # the training dataset
                time_col=time_col,
                changepoints_dict=changepoints_dict,
                origin_for_time_vars=origin_for_time_vars)
            changepoint_cols = changepoints["changepoint_cols"]
        else:
            changepoint_cols = []

        feature_sets_enabled = self.__get_feature_sets_enabled(
            simple_freq=time_properties["simple_freq"].name,
            num_days=time_properties["num_training_days"],
            feature_sets_enabled=feature_sets_enabled)

        model_feature_terms = self.__get_feature_sets_terms(
            daily_event_df_dict=daily_event_df_dict,
            daily_seas_interaction_order=daily_seas_interaction_order,
            weekly_seas_interaction_order=weekly_seas_interaction_order,
            growth_term=growth_term_formula,
            changepoint_cols=changepoint_cols)

        # extends ``extra_pred_cols`` by the requested feature sets from ``feature_sets_enabled``
        for feature_set_name, feature_set_terms in model_feature_terms.items():
            if feature_sets_enabled[feature_set_name]:
                extra_pred_cols += feature_set_terms
        extra_pred_cols = unique_elements_in_list(extra_pred_cols)

        # the parameters to call ``SilverkiteForecast::forecast``
        # parameters that are directly passed through are noted below
        parameters = dict(
            df=df,  # pass-through
            time_col=time_col,  # pass-through
            value_col=value_col,  # pass-through
            origin_for_time_vars=origin_for_time_vars,
            extra_pred_cols=extra_pred_cols,
            drop_pred_cols=drop_pred_cols,
            explicit_pred_cols=explicit_pred_cols,
            train_test_thresh=train_test_thresh,  # pass-through
            training_fraction=training_fraction,  # pass-through
            fit_algorithm=fit_algorithm,  # pass-through
            fit_algorithm_params=fit_algorithm_params,  # pass-through
            daily_event_df_dict=daily_event_df_dict,
            fs_components_df=fs_components_df,
            autoreg_dict=autoreg_dict,  # pass-through
            past_df=past_df,  # pass-through
            lagged_regressor_dict=lagged_regressor_dict,  # pass-through
            changepoints_dict=changepoints_dict,  # pass-through
            seasonality_changepoints_dict=seasonality_changepoints_dict,  # pass-through
            changepoint_detector=changepoint_detector,
            min_admissible_value=min_admissible_value,  # pass-through
            max_admissible_value=max_admissible_value,  # pass-through
            uncertainty_dict=uncertainty_dict,
            normalize_method=normalize_method,  # pass-through
            regression_weight_col=regression_weight_col,  # pass-through
            forecast_horizon=forecast_horizon,  # pass-through
            simulation_based=simulation_based,  # pass-through
            simulation_num=simulation_num,  # pass-through
            fast_simulation=fast_simulation  # pass-through
        )

        return parameters

    def forecast_simple(
            self,
            *args,
            **kwargs):
        """A wrapper around ``SilverkiteForecast::forecast`` that simplifies some of the input parameters.

        Parameters
        ----------
        args : positional args
            Positional args to pass to
            :func:`~greykite.algo.forecast.silverkite.forecast_simple_silverkite.convert_simple_silverkite_params`.
            See that function for details.

        kwargs : keyword args
            Keyword args to pass to
            :func:`~greykite.algo.forecast.silverkite.forecast_simple_silverkite.convert_simple_silverkite_params`.
            See that function for details.

        Returns
        -------
        trained_model : `dict`
            The return value of :func:`~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`
            A dictionary that includes the fitted model from the function
            :func:`~greykite.algo.common.ml_models.fit_ml_model_with_evaluation`.
        """
        parameters = self.convert_params(*args, **kwargs)
        trained_model = super().forecast(**parameters)
        return trained_model

    def __get_requested_seasonality_order(
            self,
            requested_seasonality="auto",
            default_order=5,
            is_enabled_auto=True):
        """Returns requested seasonality fourier series order.

        Parameters
        ----------
        requested_seasonality :  `str` or `bool` or `int`, default = 'auto'
            The requested seasonality.
            'auto', True, False, or a number for the Fourier order.
        default_order : `int`
            The default order to use for 'auto' and True.
        is_enabled_auto : `bool`
            Whether the seasonality should be modeled for 'auto' seasonality.

        Returns
        -------
        order : `int`
            Seasonality fourier series order.
        """
        if requested_seasonality is True or (requested_seasonality == 'auto' and is_enabled_auto):
            order = default_order
        elif requested_seasonality is False or (requested_seasonality == 'auto' and not is_enabled_auto):
            order = 0
        else:
            try:
                order = int(requested_seasonality)
            except ValueError as e:
                log_message(f"Requested seasonality order '{requested_seasonality}' must be one of:"
                            f" 'auto', True, False, integer", LoggingLevelEnum.ERROR)
                raise e
        return order

    def __get_silverkite_seasonality(
            self,
            simple_freq=SimpleTimeFrequencyEnum.DAY.name,
            num_days=1000,
            seasonality=None):
        """Generates `fs_components_df` parameter for `forecast_silverkite`
        for modeling seasonality.

        Parameters
        ----------
        simple_freq : `str`
            SimpleTimeFrequencyEnum member that best matches the input data frequency
            according to `get_simple_time_frequency_from_period`
        num_days : `int`
            Number of days of observations in the input data
        seasonality : `dict` or None
            Seasonality configuration dictionary, with the following optional keys.
            (keys are SilverkiteSeasonalityEnum members in lower case):

                - ``"yearly_seasonality"`` : `str` or `bool` or `int` or None, default = 'auto'
                    Determines the yearly seasonality
                    'auto', True, False, or a number for the Fourier order
                ``"quarterly_seasonality"`` : `str` or `bool` or `int` or None, default = 'auto'
                    Determines the quarterly seasonality
                    'auto', True, False, or a number for the Fourier order
                ``"monthly_seasonality"`` : `str` or `bool` or `int` or None, default = 'auto'
                    Determines the monthly seasonality
                    'auto', True, False, or a number for the Fourier order
                ``"weekly_seasonality"`` : `str` or `bool` or `int` or None, default = 'auto'
                    Determines the weekly seasonality
                    'auto', True, False, or a number for the Fourier order
                ``"daily_seasonality"`` : `str` or `bool` or `int` or None, default = 'auto'
                    Determines the daily seasonality
                    'auto', True, False, or a number for the Fourier order

            None is equivalent to 'auto'. If 'auto', seasonality components are based on input data
            (``num_days``, ``simple_freq``), according to
            `~greykite.algo.forecast.silverkite.constants.silverkite_seasonality.SilverkiteSeasonalityEnum`.
            and `~greykite.algo.forecast.silverkite.constants.silverkite_time_frequency.SilverkiteTimeFrequencyEnum`.

        Returns
        -------
        fs_components_df : `pandas.DataFrame`
            Contains fourier series specification. Columns:

                - "name"
                - "period"
                - "order"
                - "seas_names"
        """
        if seasonality is None:
            seasonality = {}

        # recognized seasonalities for silverkite
        silverkite_seasonalities = self._silverkite_seasonality_enum.__members__.copy()
        silverkite_seasonalities = {k.lower(): v for k, v in silverkite_seasonalities.items()}

        # valid seasonalities based on input data frequency
        freq_valid_seas_names = SimpleTimeFrequencyEnum[simple_freq].value.valid_seas
        freq_auto_seas_names = self._silverkite_time_frequency_enum[simple_freq].value.auto_fourier_seas

        for key in seasonality.keys():
            if key not in silverkite_seasonalities.keys():
                raise ValueError(f"{key} must be one of {silverkite_seasonalities.keys()}")

        seasonalities = []  # seasonalities to add to the model
        for seas in silverkite_seasonalities.values():
            # keys are SilverkiteSeasonalityEnum members in lower case
            seas_input = seasonality.get(seas.name.lower(), "auto")
            # under auto configuration, seasonality is added if it's recommended for both
            # the input frequency and data size
            is_enabled_auto = (
                    num_days >= seas.value.default_min_days
                    and seas.name in freq_auto_seas_names)
            order = self.__get_requested_seasonality_order(
                requested_seasonality=seas_input,
                default_order=seas.value.order,
                is_enabled_auto=is_enabled_auto)
            if order > 0:
                if seas.name not in freq_valid_seas_names:
                    log_message(f"'{seas.name.lower()}' is typically not valid for "
                                f"data with '{simple_freq}' frequency. Each seasonality period "
                                f"should cover multiple observations in the data. To remove "
                                f"these seasonality terms from the model, remove {seas.name.lower()}={seas_input} "
                                f"or set it to 'auto' or 0.", LoggingLevelEnum.WARNING)
                seasonalities.append({
                    "name": seas.value.name,
                    "period": seas.value.period,
                    "order": order,  # user is allowed to override default order
                    "seas_names": seas.value.seas_names
                })

        # constructs dataframe where each seasonality is a row
        if len(seasonalities) > 0:
            fs = pd.DataFrame(
                seasonalities,
                columns=["name", "period", "order", "seas_names"])
        else:
            fs = None
        return fs

    def __get_seasonality_order_from_dataframe(
            self,
            seasonality,
            fs=None,
            max_order=None):
        """Returns fourier series order from a `pandas.DataFrame`
        fourier series specification. Return value is capped by ``max_order``.

        Parameters
        ----------
        seasonality : `SilverkiteSeasonalityEnum.Seasonality` namedtuple
            Which seasonality to extract from ``fs``.
            Has attributes ``name``, ``period``, ``order``, ``seas_names``
            Can be a `SilverkiteSeasonalityEnum` member value.
        fs : `pandas.DataFrame` or None, optional, default `None`
            Columns: "name", "period", "order", "seas_names"
            Suitable for ``fs_components_df`` parameter for ``forecast_silverkite``
            for modeling seasonality.
            Could be returned by ``get_silverkite_seasonality``.
            Assumes that ``name`` and ``seas_names`` uniquely identify a row.
        max_order: `int` or None, optional, default `None`
            Upper limit on seasonality_order.

        Returns
        -------
        fs_order : `int`
            The Fourier series order of the row with the given `name` and `seas_names`
        """
        order = 0
        if fs is not None:
            name_match = (fs["name"] == seasonality.name)
            seas_match = ((fs["seas_names"] == seasonality.seas_names)
                          if seasonality.seas_names is not None
                          else pd.isna(fs["seas_names"]))

            if any(name_match & seas_match):
                order = fs.loc[(name_match & seas_match), "order"].values[0]
        if max_order is not None:
            order = min(order, max_order)
        return order

    def __get_feature_sets_enabled(
            self,
            simple_freq=SimpleTimeFrequencyEnum.DAY.name,
            num_days=1000,
            feature_sets_enabled="auto"):
        """Returns default feature sets based on training data frequency and size.

        Parameters
        ----------
        simple_freq: `str`, default SimpleTimeFrequencyEnum.DAY.name
            SimpleTimeFrequencyEnum member that best matches the input data frequency
            according to `get_simple_time_frequency_from_period`
        num_days: `int`, default 1000
            Number of days of observations in the input data
        feature_sets_enabled: `dict` [`str`, `bool` or "auto" or None] or `bool` or "auto" or None, default "auto"
            Whether to include interaction terms and categorical variables to increase model flexibility.

            If a `dict`, boolean values indicate whether include various sets of features in the model.
            The following keys are recognized
            (from `~greykite.algo.forecast.silverkite.constants.silverkite_column.SilverkiteColumn`):

                ``"COLS_HOUR_OF_WEEK"`` : `str`
                    Constant hour of week effect
                ``"COLS_WEEKEND_SEAS"`` : `str`
                    Daily seasonality interaction with is_weekend
                ``"COLS_DAY_OF_WEEK_SEAS"`` : `str`
                    Daily seasonality interaction with day of week
                ``"COLS_TREND_DAILY_SEAS"`` : `str`
                    Allow daily seasonality to change over time by is_weekend
                ``"COLS_EVENT_SEAS"`` : `str`
                    Allow sub-daily event effects
                ``"COLS_EVENT_WEEKEND_SEAS"`` : `str`
                    Allow sub-daily event effect to interact with is_weekend
                ``"COLS_DAY_OF_WEEK"`` : `str`
                    Constant day of week effect
                ``"COLS_TREND_WEEKEND"`` : `str`
                    Allow trend (growth, changepoints) to interact with is_weekend
                ``"COLS_TREND_DAY_OF_WEEK"`` : `str`
                    Allow trend to interact with day of week
                ``"COLS_TREND_WEEKLY_SEAS"`` : `str`
                    Allow weekly seasonality to change over time

            The following dictionary values are recognized:

                - True: include the feature set in the model
                - False: do not include the feature set in the model
                - None: do not include the feature set in the model
                - "auto" or not provided: use the default setting based on data frequency and size

            If not a `dict`:

                - if a boolean, equivalent to a dictionary with all values set to the boolean.
                - if None, equivalent to a dictionary with all values set to False.
                - if "auto", equivalent to a dictionary with all values set to "auto".

        Returns
        -------
        feature_sets_enabled : `dict` [`str`, `bool`]
            Indicates which feature sets will be added to the model. Feature sets are determined
            by `get_model_feature_terms` and may be empty (e.g. if there are no events,
            there is no event:seasonality interaction)
            Same valid options as `feature_sets_enabled` parameter.
        """
        feature_sets_enabled_default = {
            self._silverkite_column.COLS_HOUR_OF_WEEK: False,
            self._silverkite_column.COLS_WEEKEND_SEAS: False,
            self._silverkite_column.COLS_DAY_OF_WEEK_SEAS: False,
            self._silverkite_column.COLS_TREND_DAILY_SEAS: False,
            self._silverkite_column.COLS_EVENT_SEAS: False,
            self._silverkite_column.COLS_EVENT_WEEKEND_SEAS: False,
            self._silverkite_column.COLS_DAY_OF_WEEK: False,
            self._silverkite_column.COLS_TREND_WEEKEND: False,
            self._silverkite_column.COLS_TREND_DAY_OF_WEEK: False,
            self._silverkite_column.COLS_TREND_WEEKLY_SEAS: False,
        }
        frequency = SimpleTimeFrequencyEnum[simple_freq].value

        # for sub-daily data
        if (
                frequency.seconds_per_observation
                <= SimpleTimeFrequencyEnum.HOUR.value.seconds_per_observation):
            if num_days >= TimeEnum.ONE_MONTH_IN_DAYS.value:
                # hour of week offset, helps the fourier terms
                feature_sets_enabled_default[self._silverkite_column.COLS_HOUR_OF_WEEK] = True
                # daily seasonality on weekday vs weekend
                feature_sets_enabled_default[self._silverkite_column.COLS_WEEKEND_SEAS] = True
                # daily seasonality by day of week
                feature_sets_enabled_default[self._silverkite_column.COLS_DAY_OF_WEEK_SEAS] = True
                # daily seasonality trend on weekday, weekend
                feature_sets_enabled_default[self._silverkite_column.COLS_TREND_DAILY_SEAS] = True

            if num_days < 3 * TimeEnum.ONE_YEAR_IN_DAYS.value:
                # holiday daily seasonality
                feature_sets_enabled_default[self._silverkite_column.COLS_EVENT_SEAS] = True
            else:
                # holiday daily seasonality that depends on weekend/weekday
                #   By pigeonhole principle, with reasonable assumption that a holiday must fall on a different
                #   day of the week for any three consecutive years (or else always be on the same day of week),
                #   it takes at most 3 years of training data to observe all weekend/weekday possibilities.
                feature_sets_enabled_default[self._silverkite_column.COLS_EVENT_WEEKEND_SEAS] = True

        # for sub-weekly data
        if (
                frequency.seconds_per_observation
                <= SimpleTimeFrequencyEnum.DAY.value.seconds_per_observation):
            # day of week offset, helps the fourier terms
            feature_sets_enabled_default[self._silverkite_column.COLS_DAY_OF_WEEK] = True

            # allows different trend on weekday vs weekend
            if num_days >= TimeEnum.ONE_MONTH_IN_DAYS.value:
                feature_sets_enabled_default[self._silverkite_column.COLS_TREND_WEEKEND] = True

            # allows trend interaction with day of week
            if num_days >= TimeEnum.ONE_QUARTER_IN_DAYS.value:
                feature_sets_enabled_default[self._silverkite_column.COLS_TREND_DAY_OF_WEEK] = True

            if num_days >= TimeEnum.ONE_YEAR_IN_DAYS.value:
                # weekly seasonality trend over time
                feature_sets_enabled_default[self._silverkite_column.COLS_TREND_WEEKLY_SEAS] = True

        # None is treated the same as False.
        # Intuitively, feature_sets_enabled=None should
        #   mean no feature sets are enabled.
        if feature_sets_enabled is None:
            feature_sets_enabled = False

        # Overrides defaults with user provided dictionary
        if feature_sets_enabled == "auto":
            pass  # uses the automatic defaults directly
        elif isinstance(feature_sets_enabled, bool):
            # All values are set to the provided boolean value
            for k in feature_sets_enabled_default.keys():
                feature_sets_enabled_default[k] = feature_sets_enabled
        elif isinstance(feature_sets_enabled, dict):
            # Uses the boolean values in `feature_sets_enabled` to override `feature_sets_enabled_default`
            for setting, is_enabled in feature_sets_enabled.items():
                if setting not in feature_sets_enabled_default:
                    raise ValueError(f"Unrecognized feature set: '{setting}'. Value feature sets are "
                                     f"{list(feature_sets_enabled_default.keys())}")

                if is_enabled == "auto":
                    # "auto" values are considered not set by the user and fall back to the default
                    continue
                if is_enabled is True:
                    # User explicitly turned on this feature set.
                    feature_sets_enabled_default[setting] = True
                elif is_enabled is False or is_enabled is None:
                    # User explicitly turned off this feature set.
                    # None values are treated the same as False.
                    feature_sets_enabled_default[setting] = False
                else:
                    raise ValueError(
                        f"Unrecognized `feature_sets_enabled` dictionary value for key {setting}: "
                        f"expected bool or 'auto' or None. Found: {is_enabled}")
        else:
            raise ValueError(
                f"Unrecognized type for `feature_sets_enabled`: expected bool, dict, 'auto', or None. Found: {feature_sets_enabled}")
        return feature_sets_enabled_default

    def __get_feature_sets_terms(
            self,
            daily_event_df_dict=None,
            daily_seas_interaction_order=0,
            weekly_seas_interaction_order=0,
            growth_term=None,
            changepoint_cols=None):
        """Defines features sets for use in the `extra_pred_cols` parameter
        to `forecast_silverkite`.
        Derived from events, seasonality, and trend (growth + changepoints).

        :param daily_event_df_dict: Optional[Dict[str, pd.DataFrame("date", "event")]]
            suitable for use as `daily_event_df_dict` parameter in `forecast_silverkite`
            Each event is modeled as its own effect
        :param daily_seas_interaction_order: int
            Order on interaction terms with daily seasonality
        :param weekly_seas_interaction_order: int
            Order on interaction terms with weekly seasonality
        :param growth_term: Optional[str]
            How to model the growth. Valid options are "linear", "quadratic", "sqrt", "cubic", "cuberoot".
            See `~greykite.common.constants.GrowthColEnum`.
        :param changepoint_cols: Optional[List[str]]
            Names of the changepoint feature columns to be generated by `build_silverkite_features`
        :return: Dict[str, List[str]]
            The patsy model terms for each feature set
            key: feature set name
            value: list of patsy model terms
                If there are no valid patsy model terms according to the input configuration,
                the list is empty.
                For example, if there are no events, the event related effects will be empty
        """
        # enumerates all possible keys
        extra_pred_cols_grouped = {
            self._silverkite_column.COLS_HOUR_OF_WEEK: [],
            self._silverkite_column.COLS_WEEKEND_SEAS: [],
            self._silverkite_column.COLS_DAY_OF_WEEK_SEAS: [],
            self._silverkite_column.COLS_TREND_DAILY_SEAS: [],
            self._silverkite_column.COLS_EVENT_SEAS: [],
            self._silverkite_column.COLS_EVENT_WEEKEND_SEAS: [],
            self._silverkite_column.COLS_DAY_OF_WEEK: [],
            self._silverkite_column.COLS_TREND_WEEKEND: [],
            self._silverkite_column.COLS_TREND_DAY_OF_WEEK: [],
            self._silverkite_column.COLS_TREND_WEEKLY_SEAS: [],
        }

        # the columns which constitute the trend
        if changepoint_cols is None:
            changepoint_cols = []
        growth_col = [growth_term] if growth_term is not None else []
        trend_cols = growth_col + changepoint_cols

        # all possible values of `dow` and `dow_hr` from `build_time_features_df`
        dow_levels = ["1-Mon", "2-Tue", "3-Wed", "4-Thu", "5-Fri", "6-Sat", "7-Sun"]
        dow_hr_levels = [f"{day + 1}_{str(hour).zfill(2)}" for day in range(7) for hour in range(24)]
        day_of_week = patsy_categorical_term(term=cst.TimeFeaturesEnum.str_dow.value, levels=dow_levels)
        hour_of_week = patsy_categorical_term(term=cst.TimeFeaturesEnum.dow_hr.value, levels=dow_hr_levels)

        extra_pred_cols_grouped[self._silverkite_column.COLS_DAY_OF_WEEK] = [day_of_week]
        extra_pred_cols_grouped[self._silverkite_column.COLS_HOUR_OF_WEEK] = [hour_of_week]
        extra_pred_cols_grouped[self._silverkite_column.COLS_TREND_WEEKEND] = [
            f"{cst.TimeFeaturesEnum.is_weekend.value}:{col}" for col in trend_cols]
        extra_pred_cols_grouped[self._silverkite_column.COLS_TREND_DAY_OF_WEEK] = [
            f"{day_of_week}:{col}" for col in trend_cols]

        # allows major holidays to have different daily seasonality
        # interact with fourier series terms up to fs_daily_interaction_order
        daily_seasonality = self._silverkite_seasonality_enum.DAILY_SEASONALITY.value
        weekly_seasonality = self._silverkite_seasonality_enum.WEEKLY_SEASONALITY.value
        if daily_seas_interaction_order > 0:
            for holiday in self._silverkite_holiday.HOLIDAYS_TO_INTERACT:
                if daily_event_df_dict is not None and holiday in daily_event_df_dict.keys():
                    event_levels = [
                        cst.EVENT_DEFAULT]  # reference level for non-event days, added by `add_daily_events`
                    # This event's levels
                    event_levels += list(daily_event_df_dict[holiday][cst.EVENT_DF_LABEL_COL].unique())

                    # `term` matches new_col in `add_daily_events`
                    term = f"{cst.EVENT_PREFIX}_{holiday}"
                    extra_pred_cols_grouped[self._silverkite_column.COLS_EVENT_SEAS] += cols_interact(
                        static_col=f"{patsy_categorical_term(term=term, levels=event_levels)}",
                        fs_name=daily_seasonality.name,
                        fs_order=daily_seas_interaction_order,
                        fs_seas_name=daily_seasonality.seas_names)

                    extra_pred_cols_grouped[self._silverkite_column.COLS_EVENT_WEEKEND_SEAS] += cols_interact(
                        static_col=f"{cst.TimeFeaturesEnum.is_weekend.value}:"
                                   f"{patsy_categorical_term(term=term, levels=event_levels)}",
                        fs_name=daily_seasonality.name,
                        fs_order=daily_seas_interaction_order,
                        fs_seas_name=daily_seasonality.seas_names)

            extra_pred_cols_grouped[self._silverkite_column.COLS_WEEKEND_SEAS] = cols_interact(
                static_col=cst.TimeFeaturesEnum.is_weekend.value,
                fs_name=daily_seasonality.name,
                fs_order=daily_seas_interaction_order,
                fs_seas_name=daily_seasonality.seas_names)

            extra_pred_cols_grouped[self._silverkite_column.COLS_DAY_OF_WEEK_SEAS] = cols_interact(
                static_col=day_of_week,
                fs_name=daily_seasonality.name,
                fs_order=daily_seas_interaction_order,
                fs_seas_name=daily_seasonality.seas_names)

            for col in trend_cols:
                extra_pred_cols_grouped[self._silverkite_column.COLS_TREND_DAILY_SEAS] += cols_interact(
                    static_col=f"{cst.TimeFeaturesEnum.is_weekend.value}:{col}",
                    fs_name=daily_seasonality.name,
                    fs_order=daily_seas_interaction_order,
                    fs_seas_name=daily_seasonality.seas_names)

        if weekly_seas_interaction_order > 0:
            for col in trend_cols:
                extra_pred_cols_grouped[self._silverkite_column.COLS_TREND_WEEKLY_SEAS] += cols_interact(
                    static_col=col,
                    fs_name=weekly_seasonality.name,
                    fs_order=weekly_seas_interaction_order,
                    fs_seas_name=weekly_seasonality.seas_names)

        return extra_pred_cols_grouped

    def __get_silverkite_holidays(
            self,
            auto_holiday=False,
            holiday_lookup_countries="auto",
            holidays_to_model_separately="auto",
            start_year=2015,
            end_year=2030,
            pre_num=2,
            post_num=2,
            pre_post_num_dict=None,
            df=None,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            forecast_horizon=None):
        """Generates holidays dictionary for input to daily_event_df_dict parameter of silverkite model.
        The main purpose is to provide reasonable defaults for the holiday names and countries

        Parameters
        ----------
        auto_holiday : `bool`, default False
            If True, the other holiday configurations will be ignored.
            An algorithm is used to automatically infer the holiday configuration.
            For details, see `~greykite.algo.common.holiday_inferrer.HolidayInferrer`.
            If False, the specified holiday configuration will be used to generate holiday features.
        holiday_lookup_countries : `list` [`str`] or "auto" or None, optional, default "auto"
            The countries that contain the holidays you intend to model
            (``holidays_to_model_separately``).

            * If "auto", uses a default list of countries
              that contain the default ``holidays_to_model_separately``.
              See `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO`.
            * If a list, must be a list of country names.
            * If None or an empty list, no holidays are modeled.

        holidays_to_model_separately : `list` [`str`] or "auto" or `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES` or None, optional, default "auto"  # noqa: E501
            Which holidays to include in the model.
            The model creates a separate key, value for each item in ``holidays_to_model_separately``.
            The other holidays in the countries are grouped together as a single effect.

            * If "auto", uses a default list of important holidays.
              See `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO`.
            * If `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES`,
              uses all available holidays in ``holiday_lookup_countries``. This can often
              create a model that has too many parameters, and should typically be avoided.
            * If a list, must be a list of holiday names.
            * If None or an empty list, all holidays in ``holiday_lookup_countries`` are grouped together
              as a single effect.

            Use ``holiday_lookup_countries`` to provide a list of countries where these holiday occur.
        start_year : `int`
            Year of first training data point, used to generate holiday events.
        end_year : `int`
            Year of last forecast data point, used to generate holiday events.
        pre_num : `int`
            Model holiday effects for ``pre_num`` days before the holiday.
        post_num : `int`
            Model holiday effects for ``post_num`` days after the holiday.
        pre_post_num_dict : `dict` [`str`, (`int`, `int`)] or None, default None
            Overrides ``pre_num`` and ``post_num`` for each holiday in
            ``holidays_to_model_separately``.
            For example, if ``holidays_to_model_separately`` contains "Thanksgiving" and "Labor Day",
            this parameter can be set to ``{"Thanksgiving": [1, 3], "Labor Day": [1, 2]}``,
            denoting that the "Thanksgiving" ``pre_num`` is 1 and ``post_num`` is 3, and "Labor Day"
            ``pre_num`` is 1 and ``post_num`` is 2.
            Holidays not specified use the default given by ``pre_num`` and ``post_num``.
        df : `pandas.DataFrame` or None, default None.
            The timeseries data needed for automatically inferring holiday configuration.
            This is not used when ``auto_holiday`` is False.
        time_col : `str`, default `cst.TIME_COL`
            The column name for timestamps in ``df``.
            This is not used when ``auto_holiday`` is False.
        value_col : `str`, default `cst.VALUE_COL`
            The column name for values in ``df``.
            This is not used when ``auto_holiday`` is False.
        forecast_horizon : `int` or None, default None
            The forecast horizon, used to calculate the year list for "auto" option.

        Returns
        -------
        daily_event_df_dict : `dict` [`str`, `pandas.DataFrame` [EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL]]
            Suitable for use as `daily_event_df_dict` parameter in `forecast_silverkite`.
            Each holiday is modeled as its own effect (not specific to each country).

        See Also
        --------
        `~greykite.common.features.timeseries_features.get_available_holiday_lookup_countries`
        to list available countries for modeling.

        `~greykite.common.features.timeseries_features.get_available_holidays_across_countries`
        to see available holidays in those countries.
        """
        if holiday_lookup_countries is None:
            # `None` will not model any holidays
            holiday_lookup_countries = []
        elif holiday_lookup_countries == "auto":
            # countries that contain the default `holidays_to_model_separately`
            holiday_lookup_countries = self._silverkite_holiday.HOLIDAY_LOOKUP_COUNTRIES_AUTO
        elif not isinstance(holiday_lookup_countries, (list, tuple)):
            raise ValueError(
                f"`holiday_lookup_countries` should be a list, found {holiday_lookup_countries}")
        if auto_holiday:
            if df is None:
                raise ValueError("Automatically inferring holidays is turned on. Dataframe must be provided.")
            return get_auto_holidays(
                df=df,
                time_col=time_col,
                value_col=value_col,
                countries=holiday_lookup_countries,
                forecast_horizon=forecast_horizon,
                daily_event_dict_override=None)  # ``daily_event_dict`` is handled in `convert_params` with other cases.
        else:
            if holidays_to_model_separately is None:
                holidays_to_model_separately = []
            elif holidays_to_model_separately == "auto":
                # important holidays
                holidays_to_model_separately = self._silverkite_holiday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO
            elif holidays_to_model_separately == self._silverkite_holiday.ALL_HOLIDAYS_IN_COUNTRIES:
                holidays_to_model_separately = get_available_holidays_across_countries(
                    countries=holiday_lookup_countries,
                    year_start=start_year - 1,
                    year_end=end_year + 1)
            elif not isinstance(holidays_to_model_separately, (list, tuple)):
                raise ValueError(
                    f"`holidays_to_model_separately` should be a list, found {holidays_to_model_separately}")

            return generate_holiday_events(
                countries=holiday_lookup_countries,
                holidays_to_model_separately=holidays_to_model_separately,
                year_start=start_year - 1,  # subtract 1 just in case, to ensure coverage of all holidays
                year_end=end_year + 1,  # add 1 just in case, to ensure coverage of all holidays
                pre_num=pre_num,
                post_num=post_num,
                pre_post_num_dict=pre_post_num_dict)
