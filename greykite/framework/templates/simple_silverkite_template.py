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
"""Template for `greykite.sklearn.estimator.simple_silverkite_estimator`.
Takes input data and forecast config,
and returns parameters to call
:func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
"""

import dataclasses
import warnings
from typing import Type

import numpy as np

from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.common.features.timeseries_lags import build_autoreg_df_multi
from greykite.common.python_utils import dictionaries_values_to_lists
from greykite.common.python_utils import unique_dict_in_list
from greykite.common.python_utils import unique_in_list
from greykite.common.python_utils import update_dictionaries
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.base_template import BaseTemplate
from greykite.framework.templates.simple_silverkite_template_config import SimpleSilverkiteTemplateConstants
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator


class SimpleSilverkiteTemplate(BaseTemplate):
    """A template for :class:`~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`.

    Takes input data and optional configuration parameters
    to customize the model. Returns a set of parameters to call
    :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.

    Notes
    -----
    The attributes of a `~greykite.framework.templates.autogen.forecast_config.ForecastConfig` for
    :class:`~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator` are:

        computation_param: `ComputationParam` or None, default None
            How to compute the result. See
            :class:`~greykite.framework.templates.autogen.forecast_config.ComputationParam`.
        coverage: `float` or None, default None
            Intended coverage of the prediction bands (0.0 to 1.0).
            Same as coverage in ``forecast_pipeline``.
            You may tune how the uncertainty is computed
            via `model_components.uncertainty["uncertainty_dict"]`.
        evaluation_metric_param: `EvaluationMetricParam` or None, default None
            What metrics to evaluate. See
            :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam`.
        evaluation_period_param: `EvaluationPeriodParam` or None, default None
            How to split data for evaluation. See
            :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationPeriodParam`.
        forecast_horizon: `int` or None, default None
            Number of periods to forecast into the future. Must be > 0
            If None, default is determined from input data frequency
            Same as forecast_horizon in `forecast_pipeline`
        metadata_param: `MetadataParam` or None, default None
            Information about the input data. See
            :class:`~greykite.framework.templates.autogen.forecast_config.MetadataParam`.
        model_components_param: `ModelComponentsParam`, `list` [`ModelComponentsParam`] or None, default None
            Parameters to tune the model. See
            :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`.
            The fields are dictionaries with the following items.

            See inline comments on which values accept lists for grid search.

            seasonality: `dict` [`str`, `any`] or None, optional
                Seasonality configuration dictionary, with the following optional keys.
                (keys are SilverkiteSeasonalityEnum members in lower case).

                The keys are parameters of
                `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.forecast_simple_silverkite`.
                Refer to that function for more details.

                ``"auto_seasonality"`` : `bool`, default False
                        Whether to automatically infer seasonality orders.
                        If True, the seasonality orders will be automatically inferred from input timeseries
                        and the following parameters will be ignored:

                            * ``"yearly_seasonality"``
                            * ``"quarterly_seasonality"``
                            * ``"monthly_seasonality"``
                            * ``"weekly_seasonality"``
                            * ``"daily_seasonality"``

                        For detail, see `~greykite.algo.common.seasonality_inferrer.SeasonalityInferrer`.
                ``"yearly_seasonality"``: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
                    Determines the yearly seasonality
                    'auto', True, False, or a number for the Fourier order
                ``"quarterly_seasonality"``: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
                    Determines the quarterly seasonality
                    'auto', True, False, or a number for the Fourier order
                ``"monthly_seasonality"``: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
                    Determines the monthly seasonality
                    'auto', True, False, or a number for the Fourier order
                ``"weekly_seasonality"``: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
                    Determines the weekly seasonality
                    'auto', True, False, or a number for the Fourier order
                ``"daily_seasonality"``: `str` or `bool` or `int` or a list of such values for grid search, default 'auto'
                    Determines the daily seasonality
                    'auto', True, False, or a number for the Fourier order

            growth: `dict` [`str`, `any`] or None, optional
                Growth configuration dictionary with the following optional key:

                ``"growth_term"``: `str` or None or a list of such values for grid search
                    How to model the growth.
                    Valid options are "linear", "quadratic", "sqrt", "cubic", "cuberoot".
                    See `~greykite.common.constants.GrowthColEnum`.
                    All these terms have their origin at the train start date.

            events: `dict` [`str`, `any`] or None, optional
                Holiday/events configuration dictionary with the following optional keys:

                ``"auto_holiday"`` : `bool`, default False
                    Whether to automatically infer holiday configuration based on the input timeseries.
                    If True, the following keys will be ignored:

                        * ``"holiday_lookup_countries"``
                        * ``"holidays_to_model_separately"``
                        * ``"holiday_pre_num_days"``
                        * ``"holiday_post_num_days"``
                        * ``"holiday_pre_post_num_dict"``

                    For details, see `~greykite.algo.common.holiday_inferrer.HolidayInferrer`.
                    Extra events specified in ``daily_event_df_dict`` will be added to the inferred holidays.

                ``"holiday_lookup_countries"``: `list` [`str`] or "auto" or None or a list of such values for grid search, default "auto"
                    The countries that contain the holidays you intend to model
                    (``holidays_to_model_separately``).

                    * If "auto", uses a default list of countries
                      that contain the default ``holidays_to_model_separately``.
                      See `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO`.
                    * If a list, must be a list of country names.
                    * If None or an empty list, no holidays are modeled.

                ``"holidays_to_model_separately"``: `list` [`str`] or "auto" or `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES` or None or a list of such values for grid search, default "auto"  # noqa: E501
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

                ``"holiday_pre_num_days"``: `int` or a list of such values for grid search, default 2
                    model holiday effects for pre_num days before the holiday.
                    The unit is days, not periods. It does not depend on input data frequency.
                ``"holiday_post_num_days"``: `int` or a list of such values for grid search, default 2
                    model holiday effects for post_num days after the holiday.
                    The unit is days, not periods. It does not depend on input data frequency.

                ``"holiday_pre_post_num_dict"``: `dict` [`str`, (`int`, `int`)] or None, default None
                    Overrides ``pre_num`` and ``post_num`` for each holiday in
                    ``holidays_to_model_separately``.
                    For example, if ``holidays_to_model_separately`` contains "Thanksgiving" and "Labor Day",
                    this parameter can be set to ``{"Thanksgiving": [1, 3], "Labor Day": [1, 2]}``,
                    denoting that the "Thanksgiving" ``pre_num`` is 1 and ``post_num`` is 3, and "Labor Day"
                    ``pre_num`` is 1 and ``post_num`` is 2.
                    Holidays not specified use the default given by ``pre_num`` and ``post_num``.
                ``"daily_event_df_dict"``: `dict` [`str`, `pandas.DataFrame`] or None, default None
                    A dictionary of data frames, each representing events data for the corresponding key.
                    Specifies additional events to include besides the holidays specified above. The format
                    is the same as in `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
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
                    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

                    Note: Do not use `~greykite.common.constants.EVENT_DEFAULT`
                    in the second column. This is reserved to indicate dates that do not
                    correspond to an event.
            changepoints: `dict` [`str`, `dict`] or None, optional
                Specifies the changepoint configuration. Dictionary with the following
                optional key:

                ``"auto_growth"`` : `bool`, default False
                    Whether to automatically infer growth configuration.
                    If True, the growth term and automatically changepoint detection configuration
                    will be inferred from input timeseries,
                    and the following parameters will be ignored:

                        * ``"growth_term"`` in ``growth`` dictionary
                        * ``"changepoints_dict"`` (All parameters but custom changepoint parameters
                          to be combined with automatically detected changepoints.)

                    For detail, see
                    `~greykite.algo.changepoint.adalasso.auto_changepoint_params.generate_trend_changepoint_detection_params`.

                ``"changepoints_dict"``: `dict` or None or a list of such values for grid search
                    Changepoints dictionary passed to ``forecast_simple_silverkite``. A dictionary
                    with the following optional keys:

                    ``"method"``: `str`
                        The method to locate changepoints. Valid options:

                            - "uniform". Places n_changepoints evenly spaced changepoints to allow growth to change.
                            - "custom". Places changepoints at the specified dates.
                            - "auto". Automatically detects change points.

                        Additional keys to provide parameters for each particular method are described below.
                    ``"continuous_time_col"``: `str` or None
                        Column to apply `growth_func` to, to generate changepoint features
                        Typically, this should match the growth term in the model
                    ``"growth_func"``: callable or None
                        Growth function (`numeric` -> `numeric`). Changepoint features are created
                        by applying `growth_func` to "continuous_time_col" with offsets.
                        If None, uses identity function to use `continuous_time_col` directly
                        as growth term

                    If changepoints_dict["method"] == "uniform", this other key is required:

                        ``"n_changepoints"``: `int`
                            number of changepoints to evenly space across training period

                    If changepoints_dict["method"] == "custom", this other key is required:

                        ``"dates"``: `list` [`int` or `float` or `str` or `datetime`]
                            Changepoint dates. Must be parsable by pd.to_datetime.
                            Changepoints are set at the closest time on or after these dates
                            in the dataset.

                    If changepoints_dict["method"] == "auto", optional keys can be passed that match the parameters in
                    `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_trend_changepoints`
                    (except ``df``, ``time_col`` and ``value_col``, which are already known).
                    To add manually specified changepoints to the automatically detected ones, the keys ``dates``,
                    ``combine_changepoint_min_distance`` and ``keep_detected`` can be specified, which correspond to the
                    three parameters ``custom_changepoint_dates``, ``min_distance`` and ``keep_detected`` in
                    `~greykite.algo.changepoint.adalasso.changepoints_utils.combine_detected_and_custom_trend_changepoints`.

                ``"seasonality_changepoints_dict"``: `dict` or None or a list of such values for grid search
                        seasonality changepoints dictionary passed to ``forecast_simple_silverkite``. The optional
                        keys are the parameters in
                        `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`.
                        You don't need to provide ``df``, ``time_col``, ``value_col`` or ``trend_changepoints``, since they
                        are passed with the class automatically.

            autoregression: `dict` [`str`, `dict`] or None, optional
                Specifies the autoregression configuration. Dictionary with the following optional keys:

                ``"autoreg_dict"``: `dict` or `str` or None or a list of such values for grid search
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

                ``"simulation_num"`` :  `int`, default 10
                    The number of simulations to use. Applies only if any of the lags in ``autoreg_dict``
                    are smaller than ``forecast_horizon``. In that case, simulations are needed to generate
                    forecasts and prediction intervals.

                ``"fast_simulation"`` : `bool`, default False
                    Deterimes if fast simulations are to be used. This only impacts models
                    which include auto-regression. This method will only generate one simulation
                    without any error being added and then add the error using the volatility
                    model. The advantage is a major boost in speed during inference and the
                    disadvantage is potentially less accurate prediction intervals.

            regressors: `dict` [`str`, `any`] or None, optional
                Specifies the regressors to include in the model (e.g. macro-economic factors).
                Dictionary with the following optional keys:

                ``"regressor_cols"``: `list` [`str`] or None or a list of such values for grid search
                    The columns in ``df`` to use as regressors.
                    Note that regressor values must be available in ``df`` for all prediction dates.
                    Thus, ``df`` will contain timestamps for both training and future prediction.

                    * regressors must be available on all dates
                    * the response must be available for training dates (metadata["value_col"])

                Use ``extra_pred_cols`` to specify interactions of any model terms with the regressors.

            lagged_regressors: `dict` [`str`, `dict`] or None, optional
                Specifies the lagged regressors configuration. Dictionary with the following optional key:

                ``"lagged_regressor_dict"``: `dict` or None or a list of such values for grid search
                    A dictionary with arguments for `~greykite.common.features.timeseries_lags.build_autoreg_df_multi`.
                    The keys of the dictionary are the target lagged regressor column names.
                    It can leverage the regressors included in ``df``.
                    The value of each key is either a `dict` or `str`.
                    If `dict`, it has the following keys:

                        ``"lag_dict"`` : `dict` or None
                        ``"agg_lag_dict"`` : `dict` or None
                        ``"series_na_fill_func"`` : callable

                    If `str`, it represents a method and a dictionary will be constructed using that `str`.
                    Currently the only implemented method is "auto" which uses ``SilverkiteForecast``'s
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

            uncertainty: `dict` [`str`, `dict`] or None, optional
                Along with ``coverage``, specifies the uncertainty interval configuration. Use ``coverage``
                to set interval size. Use ``uncertainty`` to tune the calculation.

                ``"uncertainty_dict"``: `str` or `dict` or None or a list of such values for grid search
                    "auto" or a dictionary on how to fit the uncertainty model.
                    If a dictionary, valid keys are:

                        ``"uncertainty_method"``: `str`
                            The title of the method.
                            Only ``"simple_conditional_residuals"`` is implemented
                            in `~greykite.algo.common.ml_models.fit_ml_model`
                            which calculates intervals using residuals.

                        ``"params"``: `dict`
                            A dictionary of parameters needed for
                            the requested ``uncertainty_method``. For example, for
                            ``uncertainty_method="simple_conditional_residuals"``, see
                            parameters of `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`:

                                * ``"conditional_cols"``
                                * ``"quantiles"``
                                * ``"quantile_estimation_method"``
                                * ``"sample_size_thresh"``
                                * ``"small_sample_size_method"``
                                * ``"small_sample_size_quantile"``

                            The default value for ``quantiles`` is inferred from coverage.

                    If "auto", see
                    `~greykite.algo.forecast.silverkite.forecast_silverkite_helper.get_silverkite_uncertainty_dict`
                    for the default value. If ``coverage`` is not None and ``uncertainty_dict`` is not provided,
                    then the "auto" setting is used.

                    If ``coverage`` is None and ``uncertainty_dict`` is None, then no intervals are returned.

            custom: `dict` [`str`, `any`] or None, optional
                Custom parameters that don't fit the categories above. Dictionary
                with the following optional keys:

                ``"fit_algorithm_dict"``: `dict` or a list of such values for grid search
                    How to fit the model. A dictionary with the following optional keys.

                        ``"fit_algorithm"``: `str`, optional, default "ridge"
                            The type of predictive model used in fitting.

                            See `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
                            for available options and their parameters.
                        ``"fit_algorithm_params"``: `dict` or None, optional, default None
                            Parameters passed to the requested fit_algorithm.
                            If None, uses the defaults in `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.

                ``"feature_sets_enabled"``: `dict` [`str`, `bool` or "auto" or None] or `bool` or "auto" or None; or a list of such values for grid search
                    Whether to include interaction terms and categorical variables to increase model flexibility.

                    If a `dict`, boolean values indicate whether include various sets of features in the model.
                    The following keys are recognized
                    (from `~greykite.algo.forecast.silverkite.constants.silverkite_column.SilverkiteColumn`):

                        ``"COLS_HOUR_OF_WEEK"``: `str`
                            Constant hour of week effect
                        ``"COLS_WEEKEND_SEAS"``: `str`
                            Daily seasonality interaction with is_weekend
                        ``"COLS_DAY_OF_WEEK_SEAS"``: `str`
                            Daily seasonality interaction with day of week
                        ``"COLS_TREND_DAILY_SEAS"``: `str`
                            Allow daily seasonality to change over time by is_weekend
                        ``"COLS_EVENT_SEAS"``: `str`
                            Allow sub-daily event effects
                        ``"COLS_EVENT_WEEKEND_SEAS"``: `str`
                            Allow sub-daily event effect to interact with is_weekend
                        ``"COLS_DAY_OF_WEEK"``: `str`
                            Constant day of week effect
                        ``"COLS_TREND_WEEKEND"``: `str`
                            Allow trend (growth, changepoints) to interact with is_weekend
                        ``"COLS_TREND_DAY_OF_WEEK"``: `str`
                            Allow trend to interact with day of week
                        ``"COLS_TREND_WEEKLY_SEAS"``: `str`
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

                ``"max_daily_seas_interaction_order"``: `int` or None or a list of such values for grid search, default 5
                    Max fourier order to use for interactions with daily seasonality.
                    (COLS_EVENT_SEAS, COLS_EVENT_WEEKEND_SEAS, COLS_WEEKEND_SEAS, COLS_DAY_OF_WEEK_SEAS, COLS_TREND_DAILY_SEAS).

                    Model includes interactions terms specified by ``feature_sets_enabled``
                    up to the order limited by this value and the available order from ``seasonality``.
                ``"max_weekly_seas_interaction_order"`` : `int` or None or a list of such values for grid search, default 2
                    Max fourier order to use for interactions with weekly seasonality (COLS_TREND_WEEKLY_SEAS).

                    Model includes interactions terms specified by ``feature_sets_enabled``
                    up to the order limited by this value and the available order from ``seasonality``.
                ``"extra_pred_cols"``: `list` [`str`] or None or a list of such values for grid search, default None
                    Names of extra predictor columns to pass to ``forecast_silverkite``.
                    The standard interactions can be controlled via ``feature_sets_enabled`` parameter.
                    Accepts any valid patsy model formula term. Can be used to model complex interactions
                    of time features, events, seasonality, changepoints, regressors. Columns should be
                    generated by ``build_silverkite_features`` or included with input data.
                    These are added to any features already included by ``feature_sets_enabled`` and
                    terms specified by ``model``.
                ``"drop_pred_cols"`` : `list` [`str`] or None, default None
                    Names of predictor columns to be dropped from the final model.
                    Ignored if None.
                ``"explicit_pred_cols"`` : `list` [`str`] or None, default None
                    Names of the explicit predictor columns which will be
                    the only variables in the final model. Note that this overwrites
                    the generated predictors in the model and may include new
                    terms not appearing in the predictors (e.g. interaction terms).
                    Ignored if None.
                ``"min_admissible_value"``: `float` or `double` or `int` or None, default None
                    The lowest admissible value for the forecasts and prediction
                    intervals. Any value below this will be mapped back to this value.
                    If None, there is no lower bound.
                ``"max_admissible_value"``: `float` or `double` or `int` or None, default None
                    The highest admissible value for the forecasts and prediction
                    intervals. Any value above this will be mapped back to this value.
                    If None, there is no upper bound.
                ``"normalize_method"``: `str` or None, default None
                    The normalization method for feature matrix.
                    If a string is provided, it will be used as the normalization method
                    in `~greykite.common.features.normalize.normalize_df`,
                    passed via the argument ``method``.
                    Available values are "statistical", "zero_to_one", "minus_half_to_half"
                    and "zero_at_origin". See that function for more details.

            hyperparameter_override: `dict` [`str`, `any`] or None or `list` [`dict` [`str`, `any`] or None], optional
                After the above model components are used to create a hyperparameter grid, the result is
                updated by this dictionary, to create new keys or override existing ones.
                Allows for complete customization of the grid search.

                Keys should have format ``{named_step}__{parameter_name}`` for the named steps of the
                `sklearn.pipeline.Pipeline` returned by this function. See `sklearn.pipeline.Pipeline`.

                For example::

                    hyperparameter_override={
                        "estimator__silverkite": SimpleSilverkiteForecast(),
                        "estimator__silverkite_diagnostics": SilverkiteDiagnostics(),
                        "estimator__growth_term": "linear",
                        "input__response__null__impute_algorithm": "ts_interpolate",
                        "input__response__null__impute_params": {"orders": [7, 14]},
                        "input__regressors_numeric__normalize__normalize_algorithm": "RobustScaler",
                    }

                If a list of dictionaries, grid search will be done for each dictionary in the list.
                Each dictionary in the list override the defaults. This enables grid search
                over specific combinations of parameters to reduce the search space.

                * For example, the first dictionary could define combinations of parameters for a
                  "complex" model, and the second dictionary could define combinations of parameters
                  for a "simple" model, to prevent mixed combinations of simple and complex.
                * Or the first dictionary could grid search over fit algorithm, and the second dictionary
                  could use a single fit algorithm and grid search over seasonality.

                The result is passed as the ``param_distributions`` parameter
                to `sklearn.model_selection.RandomizedSearchCV`.

        model_template: `str`, `list`[`str`] or None, default None
            The simple silverkite template support single templates, multi templates or a list of single/multi templates.
            A valid single template must be one of ``SILVERKITE``, ``SILVERKITE_MONTHLY``,
            ``SILVERKITE_DAILY_1_CONFIG_1``, ``SILVERKITE_DAILY_1_CONFIG_2``, ``SILVERKITE_DAILY_1_CONFIG_3``,
            ``SILVERKITE_EMPTY``, or that consists of

                {FREQ}_SEAS_{VAL}_GR_{VAL}_CP_{VAL}_HOL_{VAL}_FEASET_{VAL}_ALGO_{VAL}_AR_{VAL}

            For example, we have DAILY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_ON_ALGO_RIDGE_AR_ON. The valid FREQ and VAL
            can be found at `~greykite.framework.templates.simple_silverkite_template_config`. The components stand for seasonality,
            growth, changepoints_dict, events, feature_sets_enabled, fit_algorithm and autoregression in
            `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`, which is used in
            `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`. Users are allowed to

                - Omit any number of component-value pairs, and the omitted will be filled with default values.
                - Switch the order of different component-value pairs.

            A valid multi template must belong to
            `~greykite.framework.templates.simple_silverkite_template_config.MULTI_TEMPLATES`
            or must be a list of single or multi template names.
    """
    DEFAULT_MODEL_TEMPLATE = "SILVERKITE"
    """The default model template. See `~greykite.framework.templates.model_templates.ModelTemplateEnum`.
    Uses a string to avoid circular imports.
    Overrides the value from `~greykite.framework.templates.forecast_config_defaults.ForecastConfigDefaults`.
    """
    def __init__(
            self,
            constants: SimpleSilverkiteTemplateConstants = SimpleSilverkiteTemplateConstants(),
            estimator: BaseForecastEstimator = SimpleSilverkiteEstimator()):
        super().__init__(estimator=estimator)
        self._constants = constants
        """Constants used by the template class. Includes the model templates and their default values."""

    @property
    def allow_model_template_list(self) -> bool:
        """SimpleSilverkiteTemplate allows config.model_template to be a list."""
        return True

    @property
    def allow_model_components_param_list(self) -> bool:
        """SilverkiteTemplate allows `config.model_components_param` to be a list."""
        return True

    @property
    def constants(self) -> SimpleSilverkiteTemplateConstants:
        """Constants used by the template class. Includes the model templates and their default values.
        """
        return self._constants

    @property
    def _silverkite_holiday(self) -> Type[SilverkiteHoliday]:
        """Holiday constants used by the estimator."""
        return self.estimator.silverkite._silverkite_holiday

    def get_regressor_cols(self):
        """Returns regressor column names from the model components.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Uses these attributes:

            model_components: :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`,
            `list` [`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`] or None, default None
                Configuration of model growth, seasonality, holidays, etc.
                See :func:`~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`
                for details.

        Returns
        -------
        regressor_cols : `list` [`str`] or None
            The names of regressor columns used in any hyperparameter set
            requested by ``model_components``.
            None if there are no regressors.
        """
        model_components = self.config.model_components_param
        if model_components is not None and isinstance(model_components, ModelComponentsParam) and model_components.regressors is not None:
            # ``regressor_cols`` is a list of strings to initialize
            # SimpleSilverkiteEstimator.regressor_cols, or a list of
            # such lists.
            regressor_cols = model_components.regressors.get("regressor_cols", [])
        elif model_components is not None and isinstance(model_components, list) and any([item.regressors is not None for item in model_components]):
            # Provides a list of ``model_components_param``.
            # ``regressor_cols`` is the union of each single ``regressor_cols``.
            regressor_cols = []
            for item in model_components:
                if item.regressors is not None:
                    regressor_cols += item.regressors.get("regressor_cols", [])
        else:
            regressor_cols = []
        return unique_in_list(
            array=regressor_cols,
            ignored_elements=(None,))

    def get_lagged_regressor_info(self):
        """Returns lagged regressor column names and minimal/maximal lag order. The lag order
        can be used to check potential imputation in the computation of lags.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Returns
        -------
        lagged_regressor_info : `dict`
            A dictionary that includes the lagged regressor column names and maximal/minimal lag order
            The keys are:

                lagged_regressor_cols : `list` [`str`] or None
                    See `~greykite.framework.pipeline.pipeline.forecast_pipeline`.
                overall_min_lag_order : `int` or None
                overall_max_lag_order : `int` or None

            For example::

                self.config.model_components_param.lagged_regressors["lagged_regressor_dict"] = [
                    {"regressor1": {
                        "lag_dict": {"orders": [7]},
                        "agg_lag_dict": {
                            "orders_list": [[7, 7 * 2, 7 * 3]],
                            "interval_list": [(8, 7 * 2)]},
                        "series_na_fill_func": lambda s: s.bfill().ffill()}
                    },
                    {"regressor2": {
                        "lag_dict": {"orders": [2]},
                        "agg_lag_dict": {
                            "orders_list": [[7, 7 * 2]],
                            "interval_list": [(8, 7 * 2)]},
                        "series_na_fill_func": lambda s: s.bfill().ffill()}
                    },
                    {"regressor3": "auto"}
                ]

            Then the function returns::

                lagged_regressor_info = {
                    "lagged_regressor_cols": ["regressor1", "regressor2", "regressor3"],
                    "overall_min_lag_order": 2,
                    "overall_max_lag_order": 21
                }

            Note that "regressor3" is skipped as the "auto" option makes sure the lag order is proper.
        """
        lagged_regressor_info = {
            "lagged_regressor_cols": None,
            "overall_min_lag_order": None,
            "overall_max_lag_order": None
        }
        if (self.config is None or self.config.model_components_param is None or
                self.config.model_components_param.lagged_regressors is None):
            return lagged_regressor_info

        lag_reg_dict = self.config.model_components_param.lagged_regressors.get("lagged_regressor_dict", None)
        if lag_reg_dict is None or lag_reg_dict == [None]:
            return lagged_regressor_info

        lag_reg_dict_list = [lag_reg_dict] if isinstance(lag_reg_dict, dict) else lag_reg_dict
        lagged_regressor_cols = []
        overall_min_lag_order = np.inf
        overall_max_lag_order = -np.inf
        for d in lag_reg_dict_list:
            if isinstance(d, dict):
                lagged_regressor_cols += list(d.keys())
                # Also gets the minimal lag order for each lagged_regressor_dict.
                # Looks at each individual regressor column, "auto" is skipped because
                # "auto" always makes sure that minimal lag order is at least forecast horizon.
                for key, value in d.items():
                    if isinstance(value, dict):
                        d_tmp = {key: value}
                        lag_reg_components = build_autoreg_df_multi(value_lag_info_dict=d_tmp)
                        overall_min_lag_order = min(
                            lag_reg_components["min_order"],
                            overall_min_lag_order)
                        overall_max_lag_order = max(
                            lag_reg_components["max_order"],
                            overall_max_lag_order)
        lagged_regressor_cols = list(set(lagged_regressor_cols))

        lagged_regressor_info["lagged_regressor_cols"] = lagged_regressor_cols
        lagged_regressor_info["overall_min_lag_order"] = overall_min_lag_order
        lagged_regressor_info["overall_max_lag_order"] = overall_max_lag_order
        return lagged_regressor_info

    def get_hyperparameter_grid(self):
        """Returns hyperparameter grid.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Converts model components, time properties, and model template into
        :class:`~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`
        hyperparameters.

        Uses these attributes:

            model_components: :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`,
            `list` [`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`] or None, default None
                Configuration of model growth, seasonality, events, etc.
                See :func:`~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`
                for details.

            time_properties: `dict` [`str`, `any`] or None, default None
                Time properties dictionary (likely produced by
                `~greykite.common.time_properties_forecast.get_forecast_time_properties`)
                with keys:

                ``"period"``: `int`
                    Period of each observation (i.e. minimum time between observations, in seconds).
                ``"simple_freq"``: `SimpleTimeFrequencyEnum`
                    ``SimpleTimeFrequencyEnum`` member corresponding to data frequency.
                ``"num_training_points"``: `int`
                    Number of observations for training.
                ``"num_training_days"``: `int`
                    Number of days for training.
                ``"start_year"``: `int`
                    Start year of the training period.
                ``"end_year"``: `int`
                    End year of the forecast period.
                ``"origin_for_time_vars"``: `float`
                    Continuous time representation of the first date in ``df``.

            model_template: `str`, default "SILVERKITE"
                The name of model template, must be one of the valid templates defined in
                `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`.

        Notes
        -----
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`
        handles the train/test splits according to ``EvaluationPeriodParam``,
        so ``estimator__train_test_thresh`` and ``estimator__training_fraction`` are always None.

        Similarly, ``estimator__origin_for_time_vars`` is set to None.

        Returns
        -------
        hyperparameter_grid : `dict` [`str`, `list` [`any`]] or `list` [ `dict` [`str`, `list` [`any`]] ]
            hyperparameter_grid for grid search in :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
            The output dictionary values are lists, combined in grid search.
        """
        if self.config.model_components_param is None or isinstance(self.config.model_components_param, ModelComponentsParam):
            model_components_list = [self.config.model_components_param]
        else:
            model_components_list = self.config.model_components_param
        model_components_list = self.__get_model_components_and_override_from_model_template(
            template=self.config.model_template,
            model_components=model_components_list)
        hyperparameter_grid_result = []
        for model_components in model_components_list:
            hyperparameter_grid = self.__get_hyperparameter_grid_from_model_components(model_components)
            # Adds time/training information to `hyperparameter_grid`.
            hyperparameter_grid = update_dictionaries(
                default_dict=hyperparameter_grid,
                overwrite_dicts={
                    "estimator__time_properties": [self.time_properties],
                    "estimator__origin_for_time_vars": [None],
                    "estimator__train_test_thresh": [None],
                    "estimator__training_fraction": [None]})
            # Update values with `hyperparameter_override`.
            hyperparameter_grid = update_dictionaries(
                default_dict=hyperparameter_grid,
                overwrite_dicts=model_components.hyperparameter_override)
            if isinstance(hyperparameter_grid, dict):
                hyperparameter_grid = [hyperparameter_grid]
            hyperparameter_grid_result += hyperparameter_grid
        # Ensures all items have the proper type for
        # `sklearn.model_selection.RandomizedSearchCV`.
        # List-type hyperparameters are specified below
        # with their accepted non-list type values.
        hyperparameter_grid_result = dictionaries_values_to_lists(
            hyperparameter_grid_result,
            hyperparameters_list_type={
                "estimator__holidays_to_model_separately": [None, "auto", self._silverkite_holiday.ALL_HOLIDAYS_IN_COUNTRIES],
                "estimator__holiday_lookup_countries": [None, "auto"],
                "estimator__regressor_cols": [None],
                "estimator__extra_pred_cols": [None]}
        )
        hyperparameter_grid_result = unique_dict_in_list(hyperparameter_grid_result)
        if len(hyperparameter_grid_result) == 1:
            hyperparameter_grid_result = hyperparameter_grid_result[0]
        return hyperparameter_grid_result

    # Additional helper functions for `get_hyperparameter_grid` are below.
    # Their behavior is customized by `self.constants`.

    # Uses `self.constants`.
    def __template_name_from_dataclass(self, name_dataclass=None):
        """Decodes the string format single template name from the ``name_dataclass`` dataclass.

        Parameters
        ----------
        name_dataclass : `SimpleSilverkiteTemplateName`, default None.
            The dataclass that contains the simple silverkite template string-type name values.
            If None, the default values will be used.

        Returns
        -------
        full_name : `str`
            The processed template name. For example, the input above will become
            `DAILY_SEAS_LT_GR_LINEAR_CP_LT_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_ON`.
        """
        if name_dataclass is None:
            name_dataclass = self.constants.SimpleSilverkiteTemplateOptions()
        return f"{name_dataclass.freq.name}_" \
               f"{self.constants.SILVERKITE_COMPONENT_KEYWORDS.SEAS.name}_{name_dataclass.seas.name}_" \
               f"{self.constants.SILVERKITE_COMPONENT_KEYWORDS.GR.name}_{name_dataclass.gr.name}_" \
               f"{self.constants.SILVERKITE_COMPONENT_KEYWORDS.CP.name}_{name_dataclass.cp.name}_" \
               f"{self.constants.SILVERKITE_COMPONENT_KEYWORDS.HOL.name}_{name_dataclass.hol.name}_" \
               f"{self.constants.SILVERKITE_COMPONENT_KEYWORDS.FEASET.name}_{name_dataclass.feaset.name}_" \
               f"{self.constants.SILVERKITE_COMPONENT_KEYWORDS.ALGO.name}_{name_dataclass.algo.name}_" \
               f"{self.constants.SILVERKITE_COMPONENT_KEYWORDS.AR.name}_{name_dataclass.ar.name}_" \
               f"{self.constants.SILVERKITE_COMPONENT_KEYWORDS.DSI.name}_{name_dataclass.dsi.name}_" \
               f"{self.constants.SILVERKITE_COMPONENT_KEYWORDS.WSI.name}_{name_dataclass.wsi.name}"

    # Uses `self.constants`.
    def __decode_single_template(self, template):
        """Given a single simple silverkite template's name, preprocess the name order and fill in default values.

        The naming of a simple silverkite template is either ``SILVERKITE`` or consists of

            {FREQ}_SEAS_{VAL}_GR_{VAL}_CP_{VAL}_HOL_{VAL}_FEASET_{VAL}_ALGO_{VAL}_AR_{VAL}

        For example, we have DAILY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_ON_ALGO_RIDGE_AR_ON. The valid FREQ and VAL
        can be found at `~greykite.framework.templates.simple_silverkite_template_config`. The components stand for seasonality,
        growth, changepoints_dict, events, feature_sets_enabled, fit_algorithm and autoregression in
        `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`, which is used in
        `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`. Users are allowed to

            - Omit any number of component-value pairs, and the omitted will be filled with default values.
            - Switch the order of different component-value pairs.

        Parameters
        ----------
        template : `str` or `SimpleSilverkiteTemplateName`
            A template name. For example, `DAILY_CP_LT_HOL_NONE_AR_ON`.
            You could also define the name throught the `SimpleSilverkiteTemplateName` class.

        Returns
        -------
        full_name : `str`
            The processed template name. For example, the input above will become
            `DAILY_SEAS_LT_GR_LINEAR_CP_LT_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_ON`.
        """
        if template in ("SILVERKITE",
                        "SILVERKITE_MONTHLY",
                        "SILVERKITE_DAILY_1_CONFIG_1",
                        "SILVERKITE_DAILY_1_CONFIG_2",
                        "SILVERKITE_DAILY_1_CONFIG_3"):
            return template
        if template == "SILVERKITE_EMPTY":
            return self.constants.SILVERKITE_EMPTY
        if isinstance(template, str):
            names = template.split("_")
            freq, components = names[0], names[1:]
            if freq not in self.constants.VALID_FREQ:
                raise ValueError(f"Template not recognized. Please make sure the frequency belongs to {self.constants.VALID_FREQ}.")
            new_name = [freq]
            # No keyword in the template name, try to decode it with the default order.
            # If `components` is empty, then all values are filled with default.
            if not any([component in self.constants.SILVERKITE_COMPONENT_KEYWORDS.__members__ for component in components]) and components:
                raise ValueError("No keyword found in the template name. Please make sure your template name "
                                 "consists of keyword + value pairs.")
            # At least one keyword in the template name. Try to fill the others with default values.
            # In this case the order of the components does not matter, so we iterate the OrderedDict.
            used_components = []
            default_components = []
            # Fills component names and values in the default order.
            # Uses the input value if exists.
            for component in self.constants.SILVERKITE_COMPONENT_KEYWORDS.__members__:
                if component == "FREQ":
                    continue
                if component in components:
                    i = components.index(component)
                    try:
                        value = components[i + 1]
                        if value not in self.constants.SILVERKITE_COMPONENT_KEYWORDS[component].value.__members__:
                            raise ValueError(f"{value} is not recognized for {component}. "
                                             f"The valid values are {list(self.constants.SILVERKITE_COMPONENT_KEYWORDS[component].value.__members__)}.")
                    except IndexError:
                        raise ValueError("Template name is not valid. It must be either keyword + value or purely values.")
                    used_components.append(i)
                    used_components.append(i + 1)
                else:
                    value = self.constants.SILVERKITE_COMPONENT_KEYWORDS[component].value.DEFAULT.name
                    default_components.append(component)
                new_name.append(component)
                new_name.append(value)
            unused_components = [component for i, component in enumerate(components) if i not in used_components]
            if unused_components:
                warnings.warn("The following words are not used because they are neither a component keyword"
                              f" nor following a component keyword, or because you have duplicate keywords. {unused_components}")
            default_components = [item for item in default_components if item not in ["DSI", "WSI"]]  # These two keywords are not exposed to users.
            if default_components:
                warnings.warn("The following component keywords are not found in the template name, "
                              "thus the default values will be used. For default values, please check the doc. "
                              f"{default_components}")
            return "_".join(new_name)
        elif isinstance(template, self.constants.SimpleSilverkiteTemplateOptions):
            return self.__template_name_from_dataclass(template)
        else:
            raise ValueError(f"The template type {type(template)} is not recognized. It must be str or `SimpleSilverkiteTemplateStringName`.")

    # Uses `self.constants`, and called by `forecaster.py`
    def check_template_type(self, template):
        """Checks the template name is valid and whether it is single or multi template.
        Raises an error if the template is not recognized.

        A valid single template must be one of ``SILVERKITE``, ``SILVERKITE_MONTHLY``,
        ``SILVERKITE_DAILY_1_CONFIG_1``, ``SILVERKITE_DAILY_1_CONFIG_2``, ``SILVERKITE_DAILY_1_CONFIG_3``,
        ``SILVERKITE_EMPTY``, or that consists of

            {FREQ}_SEAS_{VAL}_GR_{VAL}_CP_{VAL}_HOL_{VAL}_FEASET_{VAL}_ALGO_{VAL}_AR_{VAL}

        For example, we have DAILY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_ON_ALGO_RIDGE_AR_ON. The valid FREQ and VAL
        can be found at `~greykite.framework.templates.simple_silverkite_template_config`. The components stand for seasonality,
        growth, changepoints_dict, events, feature_sets_enabled, fit_algorithm and autoregression in
        `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`, which is used in
        `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`. Users are allowed to

            - Omit any number of component-value pairs, and the omitted will be filled with default values.
            - Switch the order of different component-value pairs.

        A valid multi template must belong to
        `~greykite.framework.templates.simple_silverkite_template_config.MULTI_TEMPLATES`
        or must be a list of single or multi template names.

        Parameters
        ----------
        template : `str`, `SimpleSilverkiteTemplateName` or `list`[`str`, `SimpleSilverkiteTemplateName`]
            The ``model_template`` parameter fed into
            `~greykite.framework.templates.autogen.forecast_config.ForecastConfig`.
            for simple silverkite templates.

        Returns
        -------
        template_type : `str`
            "single" or "multi".
        """
        if (template in ["SILVERKITE",
                         "SILVERKITE_MONTHLY",
                         "SILVERKITE_DAILY_1_CONFIG_1",
                         "SILVERKITE_DAILY_1_CONFIG_2",
                         "SILVERKITE_DAILY_1_CONFIG_3",
                         "SILVERKITE_EMPTY"]
                or isinstance(template, self.constants.SimpleSilverkiteTemplateOptions)
                or template.split("_")[0] in self.constants.VALID_FREQ):
            return "single"
        if template in self.constants.MULTI_TEMPLATES:
            return "multi"
        raise ValueError(f"The template name {template} is not recognized. It must be 'SILVERKITE', "
                         f"'SILVERKITE_DAILY_1_CONFIG_1', 'SILVERKITE_DAILY_1_CONFIG_2', 'SILVERKITE_DAILY_1_CONFIG_3', 'SILVERKITE_EMPTY', "
                         f"a `SimpleSilverkiteTemplateOptions` data class, of the type"
                         " '{FREQ}_SEAS_{VAL}_GR_{VAL}_CP_{VAL}_HOL_{VAL}_FEASET_{VAL}_ALGO_{VAL}_AR_{VAL}' or"
                         f" belong to {list(self.constants.MULTI_TEMPLATES.keys())}.")

    # Depends on a function that uses `self.constants`.
    def __get_name_string_from_model_template(self, template):
        """Gets the name string(s) from model template.

        The template could be a name string, a `SimpleSilverkiteTemplateOptions` dataclass,
        or a list of such strings and/or dataclasses.
        If a list is given, a list of `ModelComponentsParam` is returned.

        The naming of a simple silverkite template is either ``SILVERKITE`` or consists of

            {FREQ}_SEAS_{VAL}_GR_{VAL}_CP_{VAL}_HOL_{VAL}_FEASET_{VAL}_ALGO_{VAL}_AR_{VAL}

        For example, we have DAILY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_ON_ALGO_RIDGE_AR_ON. The valid FREQ and VAL
        can be found at `~greykite.framework.templates.simple_silverkite_template_config`. The components stand for seasonality,
        growth, changepoints_dict, events, feature_sets_enabled, fit_algorithm and autoregression in
        `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`, which is used in
        `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`. Users are allowed to

            - Omit any number of component-value pairs, and the omitted will be filled with default values.
            - Switch the order of different component-value pairs.

        Parameters
        ----------
        template : `str`, `SimpleSilverkiteTemplateOptions` or `list` [`str`, `SimpleSilverkiteTemplateOptions`]
            The ``model_template`` in `ForecastConfig`, could be a name string, a `SimpleSilverkiteTemplateOptions` dataclass,
            or a list of such strings and/or dataclasses.

        Returns
        -------
        name_string: `list` [`str`]
            The list of name string(s) that corresponds to ``template``.
        """
        def get_name_string_from_single_or_multi_template(model_template):
            """Gets the name string when model template type is single or multi."""
            template_type = self.check_template_type(model_template)
            if template_type == "single":
                return [self.__decode_single_template(model_template)]
            if template_type == "multi":
                return [self.__decode_single_template(single_template)
                        for single_template in self.constants.MULTI_TEMPLATES[model_template]]

        if isinstance(template, list):
            name_strings = []
            for item in template:
                if isinstance(item, list):
                    raise ValueError(f"{item} is not recognized as a valid single or multi template name.")
                current_item_name_string = get_name_string_from_single_or_multi_template(item)
                name_strings += current_item_name_string
            return name_strings
        else:
            name_strings = get_name_string_from_single_or_multi_template(template)
            return name_strings

    # Uses `self.constants`.
    def __get_single_model_components_param_from_template(self, template):
        """Gets the `ModelComponentsParam` class from a single simple silverkite template name.

        Parameters
        ----------
        template : `str`
            The full template name output by
            `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate.__decode_single_template`.

        Returns
        -------
        model_components_param : `ModelComponentsParam`
            The corresponding `ModelComponentsParam`, see `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`.
        """
        if template == "SILVERKITE":
            return self.constants.SILVERKITE
        if template == "SILVERKITE_MONTHLY":
            return self.constants.SILVERKITE_MONTHLY
        if template == "SILVERKITE_DAILY_1_CONFIG_1":
            return self.constants.SILVERKITE_DAILY_1_CONFIG_1
        if template == "SILVERKITE_DAILY_1_CONFIG_2":
            return self.constants.SILVERKITE_DAILY_1_CONFIG_2
        if template == "SILVERKITE_DAILY_1_CONFIG_3":
            return self.constants.SILVERKITE_DAILY_1_CONFIG_3
        names = template.split("_")
        freq, components = names[0], names[1:]
        # The value of a keyword is supposed to follow it.
        # For each keyword, we extract the next substring (index + 1) as its value.
        model_components_param = ModelComponentsParam(
            seasonality=self.constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS["SEAS"][freq][components[components.index("SEAS")+1]],
            growth=self.constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS["GR"][components[components.index("GR")+1]],
            changepoints={
                "auto_growth": False,
                "changepoints_dict": self.constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS["CP"][freq][components[components.index("CP")+1]],
                "seasonality_changepoints_dict": None
            },
            events=self.constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS["HOL"][components[components.index("HOL")+1]],
            custom={
                "feature_sets_enabled": self.constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS["FEASET"][components[components.index("FEASET")+1]],
                "fit_algorithm_dict": self.constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS["ALGO"][components[components.index("ALGO")+1]],
                "max_daily_seas_interaction_order": self.constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS["DSI"][freq][components[components.index("DSI")+1]],
                "max_weekly_seas_interaction_order": self.constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS["WSI"][freq][components[components.index("WSI")+1]],
                "extra_pred_cols": [],
                "drop_pred_cols": None,
                "explicit_pred_cols": None,
                "min_admissible_value": None,
                "max_admissible_value": None,
                "regression_weight_col": None,
                "normalize_method": "zero_to_one"
            },
            autoregression=self.constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS["AR"][components[components.index("AR")+1]],
            regressors={
                "regressor_cols": []
            },
            lagged_regressors={
                "lagged_regressor_dict": None
            },
            uncertainty={
                "uncertainty_dict": None
            })
        return model_components_param

    # Public interface, useful for debugging.
    def get_model_components_from_model_template(self, template):
        """Gets the `ModelComponentsParam` class from model template.

        The template could be a name string, a `SimpleSilverkiteTemplateOptions` dataclass,
        or a list of such strings and/or dataclasses.
        If a list is given, a list of `ModelComponentsParam` is returned.
        If a single element is given, a list of length 1 is returned.

        Parameters
        ----------
        template : `str`, `SimpleSilverkiteTemplateOptions` or `list` [`str`, `SimpleSilverkiteTemplateOptions`]
            The ``model_template`` in `ForecastConfig`, could be a name string, a `SimpleSilverkiteTemplateOptions` dataclass,
            or a list of such strings and/or dataclasses.

        Returns
        -------
        model_components_param: `list` [`ModelComponentsParam`]
            The list of `ModelComponentsParam` class(es) that correspond to ``template``.
        """
        name_strings = self.__get_name_string_from_model_template(template)
        model_components = [self.__get_single_model_components_param_from_template(single_template) for single_template in name_strings]
        return model_components

    # Depends on a function that uses `self.constants`.
    def __override_model_components(
            self,
            default_model_components,
            model_components=None):
        """Uses the values in ``model_components`` to update the corresponding values in ``default_model_components``.

        If a value is not set in model_components, i.e., the value is None, then the default value is not updated.

        Parameters
        ----------
        default_model_components : `ModelComponentsParam`
            The default model components to be updated.
        model_components : `ModelComponentsParam`, default None
            The new model components used to update ``default_model_components``.

        Returns
        -------
        updated_model_components : `ModelComponentsParam`
            The updated model components.
        """
        default_model_components = dataclasses.replace(default_model_components)  # Returns a copy.
        if default_model_components.hyperparameter_override is None:
            default_model_components.hyperparameter_override = {}
        if model_components is None:
            return default_model_components
        components = ["seasonality", "growth", "changepoints", "events", "custom", "autoregression",
                      "regressors", "lagged_regressors", "uncertainty", "hyperparameter_override"]
        for component in components:
            allow_unknown_keys = (component == "hyperparameter_override")  # Allows unknown keys for hyperparameter_override.
            updated_value = update_dictionaries(
                default_dict=getattr(default_model_components, component, {}),
                overwrite_dicts=getattr(model_components, component),
                allow_unknown_keys=allow_unknown_keys)
            setattr(default_model_components, component, updated_value)
        return default_model_components

    # Depends on a function that uses `self.constants`.
    def __get_model_components_and_override_from_model_template(
            self,
            template,
            model_components=None):
        """Gets the `ModelComponentsParam` class from model template, and overridden by ``model_components``.

        The template could be a name string, a `SimpleSilverkiteTemplateOptions` dataclass,
        or a list of such strings and/or dataclasses.
        If a list is given, a list of `ModelComponentsParam` is returned.
        If a single element is give, a list of length 1 is returned.

        The full cross product of all templates overridden by all provided model components is returned.

        Parameters
        ----------
        template : `str`, `SimpleSilverkiteTemplateOptions` or `list` [`str`, `SimpleSilverkiteTemplateOptions`]
            The ``model_template`` in `ForecastConfig`, could be a name string, a `SimpleSilverkiteTemplateOptions` dataclass,
            or a list of such strings and/or dataclasses.

        model_components : `ModelComponentsParam` or `list` [`ModelComponentsParam`], default None.
            The `ModelComponentsParam` in `ForecastConfig`, used to override the default values provided by ``template``.

        Returns
        -------
        model_components_param: `list` [`ModelComponentsParam`]
            The list of `ModelComponentsParam` class(es) that correspond to ``template``, overridden by ``model_components``.
        """
        if model_components is None:
            model_components = []
        if isinstance(model_components, ModelComponentsParam):
            model_components = [model_components]
        default_model_components = self.get_model_components_from_model_template(template)
        new_model_components = [
            self.__override_model_components(single_default_model_components, single_model_components)
            for single_model_components in model_components for single_default_model_components in default_model_components]
        return new_model_components

    @staticmethod
    def __get_hyperparameter_grid_from_model_components(model_components):
        """Gets the hyperparameter grid from the `ModelComponentsParam` class.

        Parameters
        ----------
        model_components : `class`
            The `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` class.
            Configuration of model growth, seasonality, events, etc.
            See :func:`~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`
            for details.

        Returns
        -------
        hyperparameter_grid : `dict`
            hyperparameter_grid for hyperparameter_override in
             `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam.hyperparameter_override`.
        """
        hyperparameter_grid = {
            "estimator__auto_seasonality": model_components.seasonality["auto_seasonality"],
            "estimator__yearly_seasonality": model_components.seasonality["yearly_seasonality"],
            "estimator__quarterly_seasonality": model_components.seasonality["quarterly_seasonality"],
            "estimator__monthly_seasonality": model_components.seasonality["monthly_seasonality"],
            "estimator__weekly_seasonality": model_components.seasonality["weekly_seasonality"],
            "estimator__daily_seasonality": model_components.seasonality["daily_seasonality"],
            "estimator__auto_growth": model_components.changepoints["auto_growth"],
            "estimator__growth_term": model_components.growth["growth_term"],
            "estimator__changepoints_dict": model_components.changepoints["changepoints_dict"],
            "estimator__seasonality_changepoints_dict": model_components.changepoints["seasonality_changepoints_dict"],
            "estimator__auto_holiday": model_components.events["auto_holiday"],
            "estimator__holidays_to_model_separately": model_components.events["holidays_to_model_separately"],
            "estimator__holiday_lookup_countries": model_components.events["holiday_lookup_countries"],
            "estimator__holiday_pre_num_days": model_components.events["holiday_pre_num_days"],
            "estimator__holiday_post_num_days": model_components.events["holiday_post_num_days"],
            "estimator__holiday_pre_post_num_dict": model_components.events["holiday_pre_post_num_dict"],
            "estimator__daily_event_df_dict": model_components.events["daily_event_df_dict"],
            "estimator__feature_sets_enabled": model_components.custom["feature_sets_enabled"],
            "estimator__fit_algorithm_dict": model_components.custom["fit_algorithm_dict"],
            "estimator__max_daily_seas_interaction_order": model_components.custom["max_daily_seas_interaction_order"],
            "estimator__max_weekly_seas_interaction_order": model_components.custom["max_weekly_seas_interaction_order"],
            "estimator__extra_pred_cols": model_components.custom["extra_pred_cols"],
            "estimator__drop_pred_cols": model_components.custom["drop_pred_cols"],
            "estimator__explicit_pred_cols": model_components.custom["explicit_pred_cols"],
            "estimator__min_admissible_value": model_components.custom["min_admissible_value"],
            "estimator__max_admissible_value": model_components.custom["max_admissible_value"],
            "estimator__normalize_method": model_components.custom["normalize_method"],
            "estimator__autoreg_dict": model_components.autoregression["autoreg_dict"],
            "estimator__simulation_num": model_components.autoregression["simulation_num"],
            "estimator__fast_simulation": model_components.autoregression["fast_simulation"],
            "estimator__regressor_cols": model_components.regressors["regressor_cols"],
            "estimator__lagged_regressor_dict": model_components.lagged_regressors["lagged_regressor_dict"],
            "estimator__regression_weight_col": model_components.custom["regression_weight_col"],
            "estimator__uncertainty_dict": model_components.uncertainty["uncertainty_dict"]
        }
        return hyperparameter_grid
