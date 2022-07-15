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
"""Changepoint detection via adaptive lasso."""

import warnings
from datetime import timedelta
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from greykite.algo.changepoint.adalasso.changepoints_utils import build_seasonality_feature_df_with_changes
from greykite.algo.changepoint.adalasso.changepoints_utils import build_trend_feature_df_with_changes
from greykite.algo.changepoint.adalasso.changepoints_utils import check_freq_unit_at_most_day
from greykite.algo.changepoint.adalasso.changepoints_utils import combine_detected_and_custom_trend_changepoints
from greykite.algo.changepoint.adalasso.changepoints_utils import compute_fitted_components
from greykite.algo.changepoint.adalasso.changepoints_utils import compute_min_changepoint_index_distance
from greykite.algo.changepoint.adalasso.changepoints_utils import estimate_seasonality_with_detected_changepoints
from greykite.algo.changepoint.adalasso.changepoints_utils import estimate_trend_with_detected_changepoints
from greykite.algo.changepoint.adalasso.changepoints_utils import get_changepoint_dates_from_changepoints_dict
from greykite.algo.changepoint.adalasso.changepoints_utils import get_seasonality_changes_from_adaptive_lasso
from greykite.algo.changepoint.adalasso.changepoints_utils import get_trend_changes_from_adaptive_lasso
from greykite.algo.changepoint.adalasso.changepoints_utils import get_yearly_seasonality_changepoint_dates_from_freq
from greykite.algo.changepoint.adalasso.changepoints_utils import plot_change
from greykite.common.constants import TimeFeaturesEnum
from greykite.common.features.timeseries_features import get_evenly_spaced_changepoints_dates
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.logging import pprint
from greykite.common.python_utils import ignore_warnings


class ChangepointDetector:
    """A class to implement change point detection.

    Currently supports long-term change point detection only. Input is a dataframe with time_col
    indicating the column of time info (the format should be able to be parsed by pd.to_datetime),
    and value_col indicating the column of observed time series values.

    Attributes
    ----------
    original_df : `pandas.DataFrame`
        The original data df, used to retrieve original observations, if aggregation is used in
        fitting change points.
    time_col : `str`
        The column name for time column.
    value_col : `str`
        The column name for value column.
    trend_potential_changepoint_n: `int`
        The number of change points that are evenly distributed over the time period.
    yearly_seasonality_order : `int`
        The yearly seasonality order used when fitting trend.
    y : `pandas.Series`
        The observations after aggregation.
    trend_df : `pandas.DataFrame`
        The augmented df of the original_df, including regressors of trend change points and
        Fourier series for yearly seasonality.
    trend_model : `sklearn.base.RegressionMixin`
        The fitted trend model.
    trend_coef : `numpy.array`
        The estimated trend coefficients.
    trend_intercept : `float`
        The estimated trend intercept.
    adaptive_lasso_coef : `list`
        The list of length two, first element is estimated trend coefficients, and second element
        is intercept, both estimated by adaptive lasso.
    trend_changepoints : `list`
        The list of detected trend change points, parsable by pd.to_datetime
    trend_estimation : `pd.Series`
        The estimated trend with detected trend change points.
    seasonality_df : `pandas.DataFrame`
        The augmented df of ``original_df``, including regressors of seasonality change points with
        different Fourier series frequencies.
    seasonality_changepoints : `dict`
        The dictionary of detected seasonality change points for each component.
        Keys are component names, and values are list of change points.
    seasonality_estimation : `pandas.Series`
        The estimated seasonality with detected seasonality change points.
        The series has the same length as ``original_df``. Index is timestamp, and values
        are the estimated seasonality at each timestamp.
        The seasonality estimation is the estimated of seasonality effect with trend estimated
        by `~greykite.algo.changepoint.adalasso.changepoints_utils.estimate_trend_with_detected_changepoints`
        removed.

    Methods
    -------
    find_trend_changepoints : callable
        Finds the potential trend change points for a given time series df.
    plot : callable
        Plot the results after implementing find_trend_changepoints.
    """

    def __init__(self):
        self.original_df: Optional[pd.DataFrame] = None
        self.time_col: Optional[str] = None
        self.value_col: Optional[str] = None
        self.trend_potential_changepoint_n: Optional[int] = None
        self.yearly_seasonality_order: Optional[int] = None
        self.y: Optional[pd.Series, pd.DataFrame] = None
        self.trend_df: Optional[pd.Series, pd.DataFrame] = None
        self.trend_model: Optional[RegressorMixin] = None
        self.trend_coef: Optional[np.ndarray] = None
        self.trend_intercept: Optional[float] = None
        self.adaptive_lasso_coef: Optional[List] = None
        self.trend_changepoints: Optional[List] = None
        self.trend_estimation: Optional[pd.Series] = None
        self.seasonality_df: Optional[pd.DataFrame] = None
        self.seasonality_changepoints: Optional[dict] = None
        self.seasonality_estimation: Optional[pd.Series] = None

    @ignore_warnings(category=ConvergenceWarning)
    def find_trend_changepoints(
            self,
            df,
            time_col,
            value_col,
            yearly_seasonality_order=8,
            yearly_seasonality_change_freq=None,
            resample_freq="D",
            trend_estimator="ridge",
            adaptive_lasso_initial_estimator="ridge",
            regularization_strength=None,
            actual_changepoint_min_distance="30D",
            potential_changepoint_distance=None,
            potential_changepoint_n=100,
            potential_changepoint_n_max=None,
            no_changepoint_distance_from_begin=None,
            no_changepoint_proportion_from_begin=0.0,
            no_changepoint_distance_from_end=None,
            no_changepoint_proportion_from_end=0.0,
            fast_trend_estimation=True):
        """Finds trend change points automatically by adaptive lasso.

        The algorithm does an aggregation with a user-defined frequency, defaults daily.

        If ``potential_changepoint_distance`` is not given,  ``potential_changepoint_n``
        potential change points are evenly distributed over the time period, else
        ``potential_changepoint_n`` is overridden by::

                total_time_length / ``potential_changepoint_distance``

        Users can specify either ``no_changepoint_proportion_from_end`` to specify what proportion
        from the end of data they do not want changepoints, or ``no_changepoint_distance_from_end``
        (overrides ``no_changepoint_proportion_from_end``) to specify how long from the end they
        do not want change points.

        Then all potential change points will be selected by adaptive lasso, with the initial
        estimator specified by ``adaptive_lasso_initial_estimator``. If user specifies
        ``regularization_strength``, then the adaptive lasso will be run with a single tuning
        parameter calculated based on user provided prior, else a cross-validation will be run to
        automatically select the tuning parameter.

        A yearly seasonality is also fitted at the same time, preventing trend from catching
        yearly periodical changes.

        A rule-based guard function is applied at the end to ensure change points are not
        too close, as specified by ``actual_changepoint_min_distance``.

        Parameters
        ----------
        df: `pandas.DataFrame`
            The data df
        time_col : `str`
            Time column name in ``df``
        value_col : `str`
            Value column name in ``df``
        yearly_seasonality_order : `int`, default 8
            Fourier series order to capture yearly seasonality.
        yearly_seasonality_change_freq : `DateOffset`, `Timedelta` or `str` or `None`, default `None`
            How often to change the yearly seasonality model. Set to `None` to disable this feature.

            This is useful if you have more than 2.5 years of data and the detected trend without this
            feature is inaccurate because yearly seasonality changes over the training period.
            Modeling yearly seasonality separately over the each period can prevent trend changepoints
            from fitting changes in yearly seasonality. For example, if you have 2.5 years of data and
            yearly seasonality increases in magnitude after the first year, setting this parameter to
            "365D" will model each year's yearly seasonality differently and capture both shapes.
            However, without this feature, both years will have the same yearly seasonality, roughly
            the average effect across the training set.

            Note that if you use `str` as input, the maximal supported unit is day, i.e.,
            you might use "200D" but not "12M" or "1Y".
        resample_freq : `DateOffset`, `Timedelta`, `str` or None, default "D".
            The frequency to aggregate data.
            Coarser aggregation leads to fitting longer term trends.
            If None, no aggregation will be done.
        trend_estimator : `str` in ["ridge", "lasso" or "ols"], default "ridge".
            The estimator to estimate trend. The estimated trend is only for plotting purposes.
            'ols' is not recommended when ``yearly_seasonality_order`` is specified other than 0,
            because significant over-fitting will happen.
            In this case, the given value is overridden by "ridge".
        adaptive_lasso_initial_estimator : `str` in ["ridge", "lasso" or "ols"], default "ridge".
            The initial estimator to compute adaptive lasso weights
        regularization_strength : `float` in [0, 1] or `None`
            The regularization for change points. Greater value implies fewer change points.
            0 indicates all change points, and 1 indicates no change point.
            If `None`, the turning parameter will be selected by cross-validation.
            If a value is given, it will be used as the tuning parameter.
        actual_changepoint_min_distance : `DateOffset`, `Timedelta` or `str`, default "30D"
            The minimal distance allowed between detected change points. If consecutive change points
            are within this minimal distance, the one with smaller absolute change coefficient will
            be dropped.
            Note: maximal unit is 'D', i.e., you may use units no more than 'D' such as
            '10D', '5H', '100T', '200S'. The reason is that 'W', 'M' or higher has either
            cycles or indefinite number of days, thus is not parsable by pandas as timedelta.
        potential_changepoint_distance : `DateOffset`, `Timedelta`, `str` or None, default None
            The distance between potential change points.
            If provided, will override the parameter ``potential_changepoint_n``.
            Note: maximal unit is 'D', i.e., you may only use units no more than 'D' such as
            '10D', '5H', '100T', '200S'. The reason is that 'W', 'M' or higher has either
            cycles or indefinite number of days, thus is not parsable by pandas as timedelta.
        potential_changepoint_n : `int`, default 100
            Number of change points to be evenly distributed, recommended 1-2 per month, based
            on the training data length.
        potential_changepoint_n_max : `int` or None, default None
            The maximum number of potential changepoints.
            This parameter is effective when user specifies ``potential_changepoint_distance``,
            and the number of potential changepoints in the training data is more than ``potential_changepoint_n_max``,
            then it is equivalent to specifying ``potential_changepoint_n = potential_changepoint_n_max``,
            and ignoring ``potential_changepoint_distance``.
        no_changepoint_distance_from_begin : `DateOffset`, `Timedelta`, `str` or None, default None
            The length of time from the beginning of training data, within which no change point will be placed.
            If provided, will override the parameter ``no_changepoint_proportion_from_begin``.
            Note: maximal unit is 'D', i.e., you may only use units no more than 'D' such as
            '10D', '5H', '100T', '200S'. The reason is that 'W', 'M' or higher has either
            cycles or indefinite number of days, thus is not parsable by pandas as timedelta.
        no_changepoint_proportion_from_begin : `float` in [0, 1], default 0.0.
            ``potential_changepoint_n`` change points will be placed evenly over the whole training period,
            however, change points that are located within the first ``no_changepoint_proportion_from_begin``
            proportion of training period will not be used for change point detection.
        no_changepoint_distance_from_end : `DateOffset`, `Timedelta`, `str` or None, default None
            The length of time from the end of training data, within which no change point will be placed.
            If provided, will override the parameter ``no_changepoint_proportion_from_end``.
            Note: maximal unit is 'D', i.e., you may only use units no more than 'D' such as
            '10D', '5H', '100T', '200S'. The reason is that 'W', 'M' or higher has either
            cycles or indefinite number of days, thus is not parsable by pandas as timedelta.
        no_changepoint_proportion_from_end : `float` in [0, 1], default 0.0.
            ``potential_changepoint_n`` change points will be placed evenly over the whole training period,
            however, change points that are located within the last ``no_changepoint_proportion_from_end``
            proportion of training period will not be used for change point detection.
        fast_trend_estimation : `bool`, default True
            If True, the trend estimation is not refitted on the original data,
            but is a linear interpolation of the fitted trend from the resampled time series.
            If False, the trend estimation is refitted on the original data.

        Return
        ------
        result : `dict`
            result dictionary with keys:

            ``"trend_feature_df"`` : `pandas.DataFrame`
                The augmented df for change detection, in other words, the design matrix for
                the regression model. Columns:

                    - 'changepoint0': regressor for change point 0, equals the continuous time
                      of the observation minus the continuous time for time of origin.
                    - ...
                    - 'changepoint{potential_changepoint_n}': regressor for change point
                      {potential_changepoint_n}, equals the continuous time of the observation
                      minus the continuous time of the {potential_changepoint_n}th change point.
                    - 'cos1_conti_year_yearly': cosine yearly seasonality regressor of first order.
                    - 'sin1_conti_year_yearly': sine yearly seasonality regressor of first order.
                    - ...
                    - 'cos{yearly_seasonality_order}_conti_year_yearly' : cosine yearly seasonality
                      regressor of {yearly_seasonality_order}th order.
                    - 'sin{yearly_seasonality_order}_conti_year_yearly' : sine yearly seasonality
                      regressor of {yearly_seasonality_order}th order.

            ``"trend_changepoints"`` : `list`
                The list of detected change points.
            ``"changepoints_dict"`` : `dict`
                The change point dictionary that is compatible as an input with
                `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`
            ``"trend_estimation"`` : `pandas.Series`
                The estimated trend with detected trend change points.
        """
        # Checks parameter rationality
        if potential_changepoint_n < 0:
            raise ValueError("potential_changepoint_n can not be negative. "
                             "A large number such as 100 is recommended")
        if yearly_seasonality_order < 0:
            raise ValueError("year_seasonality_order can not be negative. "
                             "A number less than or equal to 10 is recommended")
        if df.dropna().shape[0] < 5:
            raise ValueError("Change point detector does not work for less than "
                             "5 observations. Please increase sample size.")
        if no_changepoint_proportion_from_begin < 0 or no_changepoint_proportion_from_begin > 1:
            raise ValueError("no_changepoint_proportion_from_begin needs to be between 0 and 1.")
        if no_changepoint_proportion_from_end < 0 or no_changepoint_proportion_from_end > 1:
            raise ValueError("no_changepoint_proportion_from_end needs to be between 0 and 1.")
        if no_changepoint_distance_from_begin is not None:
            check_freq_unit_at_most_day(no_changepoint_distance_from_begin, "no_changepoint_distance_from_begin")
            data_length = pd.to_datetime(df[time_col].iloc[-1]) - pd.to_datetime(df[time_col].iloc[0])
            no_changepoint_proportion_from_begin = to_offset(no_changepoint_distance_from_begin).delta / data_length
            no_changepoint_proportion_from_begin = min(no_changepoint_proportion_from_begin, 1)
        if no_changepoint_distance_from_end is not None:
            check_freq_unit_at_most_day(no_changepoint_distance_from_end, "no_changepoint_distance_from_end")
            data_length = pd.to_datetime(df[time_col].iloc[-1]) - pd.to_datetime(df[time_col].iloc[0])
            no_changepoint_proportion_from_end = to_offset(no_changepoint_distance_from_end).delta / data_length
            no_changepoint_proportion_from_end = min(no_changepoint_proportion_from_end, 1)
        if potential_changepoint_distance is not None:
            check_freq_unit_at_most_day(potential_changepoint_distance, "potential_changepoint_distance")
            data_length = pd.to_datetime(df[time_col].iloc[-1]) - pd.to_datetime(df[time_col].iloc[0])
            potential_changepoint_n = data_length // to_offset(potential_changepoint_distance).delta
            if potential_changepoint_n_max is not None:
                if potential_changepoint_n_max <= 0:
                    raise ValueError("potential_changepoint_n_max must be a positive integer.")
                if potential_changepoint_n > potential_changepoint_n_max:
                    log_message(
                        message=f"Number of potential changepoints is capped by 'potential_changepoint_n_max' "
                                f"as {potential_changepoint_n_max}. The 'potential_changepoint_distance' "
                                f"{potential_changepoint_distance} is ignored. "
                                f"The original number of changepoints was {potential_changepoint_n}.",
                        level=LoggingLevelEnum.INFO
                    )
                    potential_changepoint_n = potential_changepoint_n_max
        if regularization_strength is not None and (regularization_strength < 0 or regularization_strength > 1):
            raise ValueError("regularization_strength must be between 0.0 and 1.0.")
        df = df.copy()
        self.trend_potential_changepoint_n = potential_changepoint_n
        self.time_col = time_col
        self.value_col = value_col
        self.original_df = df
        # Resamples df to get a coarser granularity to get rid of shorter seasonality.
        # The try except below speeds up unnecessary datetime transformation.
        if resample_freq is not None:
            try:
                df_resample = df.resample(resample_freq, on=time_col).mean().reset_index()
            except TypeError:
                df[time_col] = pd.to_datetime(df[time_col])
                df_resample = df.resample(resample_freq, on=time_col).mean().reset_index()
        else:
            df[time_col] = pd.to_datetime(df[time_col])
            df_resample = df.copy()
        # The ``df.resample`` function creates NA when the original df has a missing observation
        # or its value is NA.
        # The estimation algorithm does not allow NA, so we drop those rows.
        df_resample = df_resample.dropna()
        self.original_df[time_col] = df[time_col]
        # Prepares response df.
        y = df_resample[value_col]
        y.index = df_resample[time_col]
        self.y = y
        # Prepares trend feature df.
        # Potential changepoints are placed uniformly among rows without a missing value, after resampling.
        trend_df = build_trend_feature_df_with_changes(
            df=df_resample,
            time_col=time_col,
            changepoints_dict={
                "method": "uniform",
                "n_changepoints": potential_changepoint_n
            }
        )
        # Gets changepoint features only in range filtered by ``no_changepoint_proportion_from_begin`` and
        # ``no_changepoint_proportion_from_end`` of time period.
        n_changepoints_within_range_begin = int(potential_changepoint_n * no_changepoint_proportion_from_begin)
        n_changepoints_within_range_end = int(potential_changepoint_n * (1 - no_changepoint_proportion_from_end))
        if n_changepoints_within_range_begin < n_changepoints_within_range_end:
            trend_df = trend_df.iloc[:, [0] + list(range(n_changepoints_within_range_begin + 1, n_changepoints_within_range_end + 1))]
        else:
            # Linear growth term only.
            trend_df = trend_df.iloc[:, [0]]
        # Builds yearly seasonality feature df
        if yearly_seasonality_order is not None and yearly_seasonality_order > 0:
            self.yearly_seasonality_order = yearly_seasonality_order
            # Gets yearly seasonality changepoints, allowing varying yearly seasonality coefficients
            # to capture yearly seasonality shape change.
            yearly_seasonality_changepoint_dates = get_yearly_seasonality_changepoint_dates_from_freq(
                df=df,
                time_col=time_col,
                yearly_seasonality_change_freq=yearly_seasonality_change_freq)
            long_seasonality_df = build_seasonality_feature_df_with_changes(
                df=df_resample,
                time_col=time_col,
                changepoints_dict=dict(
                    method="custom",
                    dates=yearly_seasonality_changepoint_dates),
                fs_components_df=pd.DataFrame({
                    "name": [TimeFeaturesEnum.conti_year.value],
                    "period": [1.0],
                    "order": [yearly_seasonality_order],
                    "seas_names": ["yearly"]})
            )
            trend_df = pd.concat([trend_df, long_seasonality_df], axis=1)
        trend_df.index = df_resample[time_col]
        self.trend_df = trend_df
        # Estimates trend.
        if trend_estimator not in ["ridge", "lasso", "ols"]:
            warnings.warn("trend_estimator not in ['ridge', 'lasso', 'ols'], "
                          "estimating using ridge")
            trend_estimator = 'ridge'
        if trend_estimator == 'ols' and yearly_seasonality_order > 0:
            warnings.warn("trend_estimator = 'ols' with year_seasonality_order > 0 may create "
                          "over-fitting, trend_estimator has been set to 'ridge'.")
            trend_estimator = 'ridge'
        fit_algorithm_dict = {
            "ridge": RidgeCV,
            "lasso": LassoCV,
            "ols": LinearRegression
        }
        trend_model = fit_algorithm_dict[trend_estimator]().fit(trend_df.values, y.values)
        self.trend_model = trend_model
        self.trend_coef, self.trend_intercept = trend_model.coef_.ravel(), trend_model.intercept_.ravel()
        # Fetches change point dates for reference as datetime format.
        changepoint_dates = get_evenly_spaced_changepoints_dates(
            df=df_resample,
            time_col=time_col,
            n_changepoints=potential_changepoint_n
        )
        # Gets the changepoint dates filtered by ``no_changepoint_proportion_from_begin`` and ``no_changepoint_proportion_from_end``.
        if n_changepoints_within_range_begin < n_changepoints_within_range_end:
            changepoint_dates = changepoint_dates.iloc[[0] + list(range(n_changepoints_within_range_begin + 1, n_changepoints_within_range_end + 1))]
        else:
            # Linear growth term only.
            changepoint_dates = changepoint_dates.iloc[[0]]
        # Calculates the minimal allowed change point index distance.
        min_changepoint_index_distance = compute_min_changepoint_index_distance(
            df=df_resample,
            time_col=time_col,
            n_changepoints=potential_changepoint_n,
            min_distance_between_changepoints=actual_changepoint_min_distance
        )
        # Uses adaptive lasso to select change points.
        if adaptive_lasso_initial_estimator not in ['ridge', 'lasso', 'ols']:
            warnings.warn("adaptive_lasso_initial_estimator not in ['ridge', 'lasso', 'ols'], "
                          "estimating with ridge")
            adaptive_lasso_initial_estimator = "ridge"
        if adaptive_lasso_initial_estimator == trend_estimator:
            # When ``adaptive_lasso_initial_estimator`` is the same as ``trend_estimator``, the
            # estimated trend coefficients will be used to calculate the weights. The
            # ``get_trend_changes_from_adaptive_lasso`` function recognizes ``initial_coef`` as
            # `numpy.array` and calculates the weights directly.
            trend_changepoints, self.adaptive_lasso_coef = get_trend_changes_from_adaptive_lasso(
                x=trend_df.values,
                y=y.values,
                changepoint_dates=changepoint_dates,
                initial_coef=self.trend_coef,
                min_index_distance=min_changepoint_index_distance,
                regularization_strength=regularization_strength
            )
        else:
            # When ``adaptive_lasso_initial_estimator`` is different from ``trend_estimator``, the
            # ``adaptive_lasso_initial_estimator`` as a `str` will be passed. The
            # ``get_trend_changes_from_adaptive_lasso`` function recognizes ``initial_coef`` as
            # `str` and calculates the initial estimator with the corresponding estimator first
            # then calculates the weights.
            trend_changepoints, self.adaptive_lasso_coef = get_trend_changes_from_adaptive_lasso(
                x=trend_df.values,
                y=y.values,
                changepoint_dates=changepoint_dates,
                initial_coef=adaptive_lasso_initial_estimator,
                min_index_distance=min_changepoint_index_distance,
                regularization_strength=regularization_strength
            )
        # Checks if the beginning date is picked as a change point. If yes, drop it, because we
        # always include the growth term in our model.
        trend_changepoints = [cp for cp in trend_changepoints if cp > max(df_resample[time_col][0], df[time_col][0])]
        self.trend_changepoints = trend_changepoints
        # logging
        log_message(f"The detected trend change points are\n{trend_changepoints}", LoggingLevelEnum.INFO)
        # Creates changepoints_dict for silverkite to use.
        changepoints_dict = {
            "method": "custom",
            "dates": trend_changepoints
        }
        # Computes trend estimates for seasonality use.
        if fast_trend_estimation:
            # Fast calculation of trend estimation.
            # Do not fit trend again on the original df.
            # This is much faster when the original df has small frequencies.
            # Uses linear interpolation on the trend fitted with the resampled df.
            trend_estimation = np.matmul(
                trend_df.values[:, :(len(trend_changepoints) + 1)],
                trend_model.coef_[:(len(trend_changepoints) + 1)]
            ) + trend_model.intercept_
            trend_estimation = pd.DataFrame({
                time_col: df_resample[time_col],
                "trend": trend_estimation
            })
            trend_estimation = trend_estimation.merge(
                df[[time_col]],
                on=time_col,
                how="right"
            )
            trend_estimation["trend"].interpolate(inplace=True)
            trend_estimation.index = df[time_col]
            trend_estimation = trend_estimation["trend"]
        else:
            trend_estimation = estimate_trend_with_detected_changepoints(
                df=df,
                time_col=time_col,
                value_col=value_col,
                changepoints=trend_changepoints
            )
        self.trend_estimation = trend_estimation
        result = {
            "trend_feature_df": trend_df,
            "trend_changepoints": trend_changepoints,
            "changepoints_dict": changepoints_dict,
            "trend_estimation": trend_estimation
        }
        return result

    @ignore_warnings(category=ConvergenceWarning)
    def find_seasonality_changepoints(
            self,
            df,
            time_col,
            value_col,
            seasonality_components_df=pd.DataFrame({
                "name": [
                    TimeFeaturesEnum.tod.value,
                    TimeFeaturesEnum.tow.value,
                    TimeFeaturesEnum.conti_year.value],
                "period": [24.0, 7.0, 1.0],
                "order": [3, 3, 5],
                "seas_names": ["daily", "weekly", "yearly"]}),
            resample_freq="H",
            regularization_strength=0.6,
            actual_changepoint_min_distance="30D",
            potential_changepoint_distance=None,
            potential_changepoint_n=50,
            no_changepoint_distance_from_end=None,
            no_changepoint_proportion_from_end=0.0,
            trend_changepoints=None):
        """Finds the seasonality change points (defined as the time points where seasonality
        magnitude changes, i.e., the time series becomes "fatter" or "thinner".)

        Subtracts the estimated trend from the original time series first,
        then uses regression-based regularization methods to select important seasonality
        change points. Regressors are built from truncated Fourier series.

        If you have run ``find_trend_changepoints`` before running ``find_seasonality_changepoints``
        with the same df, the estimated trend will be automatically used for removing trend in
        ``find_seasonality_changepoints``.
        Otherwise, ``find_trend_changepoints`` will be run automatically with the same parameters
        as you passed to ``find_seasonality_changepoints``. If you do not want to use the same
        parameters, run ``find_trend_changepoints`` with your desired parameter before calling
        ``find_seasonality_changepoints``.

        The algorithm does an aggregation with a user-defined frequency, default hourly.

        The regression features consists of ``potential_changepoint_n`` + 1 blocks of
        predictors. The first block consists of Fourier series according to
        ``seasonality_components_df``, and other blocks are a copy of the first block
        truncated at the corresponding potential change point.

        If ``potential_changepoint_distance`` is not given,  ``potential_changepoint_n``
        potential change points are evenly distributed over the time period, else
        ``potential_changepoint_n`` is overridden by::

                total_time_length / ``potential_changepoint_distance``

        Users can specify either ``no_changepoint_proportion_from_end`` to specify what proportion
        from the end of data they do not want changepoints, or ``no_changepoint_distance_from_end``
        (overrides ``no_changepoint_proportion_from_end``) to specify how long from the end they
        do not want change points.

        Then all potential change points will be selected by adaptive lasso, with the initial
        estimator specified by ``adaptive_lasso_initial_estimator``. The regularization strength
        is specified by ``regularization_strength``, which lies between 0 and 1.

        A rule-based guard function is applied at the end to ensure change points are not
        too close, as specified by ``actual_changepoint_min_distance``.

        Parameters
        ----------
        df: `pandas.DataFrame`
            The data df
        time_col : `str`
            Time column name in ``df``
        value_col : `str`
            Value column name in ``df``
        seasonality_components_df : `pandas.DataFrame`
            The df to generate seasonality design matrix, which is compatible with
            ``seasonality_components_df`` in
            `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`
        resample_freq : `DateOffset, Timedelta or str`, default "H".
            The frequency to aggregate data.
            Coarser aggregation leads to fitting longer term trends.
        regularization_strength : `float` in [0, 1] or `None`, default 0.6.
            The regularization for change points. Greater value implies fewer change points.
            0 indicates all change points, and 1 indicates no change point.
            If `None`, the turning parameter will be selected by cross-validation.
            If a value is given, it will be used as the tuning parameter.
            Here "None" is not recommended, because seasonality change has different levels,
            and automatic selection by cross-validation may produce more change points than
            desired. Practically, 0.6 is a good choice for most cases. Tuning around
            0.6 is recommended.
        actual_changepoint_min_distance : `DateOffset`, `Timedelta` or `str`, default "30D"
            The minimal distance allowed between detected change points. If consecutive change points
            are within this minimal distance, the one with smaller absolute change coefficient will
            be dropped.
            Note: maximal unit is 'D', i.e., you may use units no more than 'D' such as
            '10D', '5H', '100T', '200S'. The reason is that 'W', 'M' or higher has either
            cycles or indefinite number of days, thus is not parsable by pandas as timedelta.
        potential_changepoint_distance : `DateOffset`, `Timedelta`, `str` or None, default None
            The distance between potential change points.
            If provided, will override the parameter ``potential_changepoint_n``.
            Note: maximal unit is 'D', i.e., you may only use units no more than 'D' such as
            '10D', '5H', '100T', '200S'. The reason is that 'W', 'M' or higher has either
            cycles or indefinite number of days, thus is not parsable by pandas as timedelta.
        potential_changepoint_n : `int`, default 50
            Number of change points to be evenly distributed, recommended 1 per month, based
            on the training data length.
        no_changepoint_distance_from_end : `DateOffset`, `Timedelta`, `str` or None, default None
            The length of time from the end of training data, within which no change point will be placed.
            If provided, will override the parameter ``no_changepoint_proportion_from_end``.
            Note: maximal unit is 'D', i.e., you may only use units no more than 'D' such as
            '10D', '5H', '100T', '200S'. The reason is that 'W', 'M' or higher has either
            cycles or indefinite number of days, thus is not parsable by pandas as timedelta.
        no_changepoint_proportion_from_end : `float` in [0, 1], default 0.0.
            ``potential_changepoint_n`` change points will be placed evenly over the whole training period,
            however, only change points that are not located within the last ``no_changepoint_proportion_from_end``
            proportion of training period will be used for change point detection.
        trend_changepoints : `list` or None
            A list of user specified trend change points, used to estimated the trend to be removed
            from the time series before detecting seasonality change points. If provided, the algorithm
            will not check existence of detected trend change points or run ``find_trend_changepoints``,
            but will use these change points directly for trend estimation.

        Return
        ------
        result : `dict`
            result dictionary with keys:

            ``"seasonality_feature_df"`` : `pandas.DataFrame`
                The augmented df for seasonality changepoint detection, in other words, the design matrix for
                the regression model. Columns:

                    - "cos1_tod_daily": cosine daily seasonality regressor of first order at change point 0.
                    - "sin1_tod_daily": sine daily seasonality regressor of first order at change point 0.
                    - ...
                    - "cos1_conti_year_yearly": cosine yearly seasonality regressor of first order at
                      change point 0.
                    - "sin1_conti_year_yearly": sine yearly seasonality regressor of first order at
                      change point 0.
                    - ...
                    - "cos{daily_seasonality_order}_tod_daily_cp{potential_changepoint_n}" : cosine
                      daily seasonality regressor of {yearly_seasonality_order}th order at change point
                      {potential_changepoint_n}.
                    - "sin{daily_seasonality_order}_tod_daily_cp{potential_changepoint_n}" : sine
                      daily seasonality regressor of {yearly_seasonality_order}th order at change point
                      {potential_changepoint_n}.
                    - ...
                    - "cos{yearly_seasonality_order}_conti_year_yearly_cp{potential_changepoint_n}" : cosine
                      yearly seasonality regressor of {yearly_seasonality_order}th order at change point
                      {potential_changepoint_n}.
                    - "sin{yearly_seasonality_order}_conti_year_yearly_cp{potential_changepoint_n}" : sine
                      yearly seasonality regressor of {yearly_seasonality_order}th order at change point
                      {potential_changepoint_n}.

            ``"seasonality_changepoints"`` : `dict`[`list`[`datetime`]]
                The dictionary of detected seasonality change points for each component.
                Keys are component names, and values are list of change points.
            ``"seasonality_estimation"`` : `pandas.Series`
                The estimated seasonality with detected seasonality change points.
                    The series has the same length as ``original_df``. Index is timestamp, and values
                    are the estimated seasonality at each timestamp.
                    The seasonality estimation is the estimated of seasonality effect with trend estimated
                    by `~greykite.algo.changepoint.adalasso.changepoints_utils.estimate_trend_with_detected_changepoints`
                    removed.
            ``"seasonality_components_df`` : `pandas.DataFrame`
                The processed ``seasonality_components_df``. Daily component row is removed if
                inferred frequency or aggregation frequency is at least one day.
        """
        # Checks parameter rationality.
        if potential_changepoint_n < 0:
            raise ValueError("potential_changepoint_n can not be negative. "
                             "A large number such as 50 is recommended")
        if df.dropna().shape[0] < 5:
            raise ValueError("Change point detector does not work for less than "
                             "5 observations. Please increase sample size.")
        if no_changepoint_proportion_from_end < 0 or no_changepoint_proportion_from_end > 1:
            raise ValueError("``no_changepoint_proportion_from_end`` needs to be between 0 and 1.")
        if no_changepoint_distance_from_end is not None:
            check_freq_unit_at_most_day(no_changepoint_distance_from_end, "no_changepoint_distance_from_end")
            data_length = pd.to_datetime(df[time_col].iloc[-1]) - pd.to_datetime(df[time_col].iloc[0])
            no_changepoint_proportion_from_end = to_offset(no_changepoint_distance_from_end).delta / data_length
        if potential_changepoint_distance is not None:
            check_freq_unit_at_most_day(potential_changepoint_distance, "potential_changepoint_distance")
            data_length = pd.to_datetime(df[time_col].iloc[-1]) - pd.to_datetime(df[time_col].iloc[0])
            potential_changepoint_n = data_length // to_offset(potential_changepoint_distance).delta
        if regularization_strength is None:
            warnings.warn("regularization_strength is set to None. This will trigger cross-validation to "
                          "select the tuning parameter which might result in too many change points. "
                          "Keep the default value or tuning around it is recommended.")
        if regularization_strength is not None and (regularization_strength < 0 or regularization_strength > 1):
            raise ValueError("regularization_strength must be between 0.0 and 1.0.")
        df[time_col] = pd.to_datetime(df[time_col])
        # If user provides a list of trend change points, these points will be used to estimate trend.
        if trend_changepoints is not None:
            trend_estimation = estimate_trend_with_detected_changepoints(
                df=df,
                time_col=time_col,
                value_col=value_col,
                changepoints=trend_changepoints
            )
            self.trend_changepoints = trend_changepoints
            self.trend_estimation = trend_estimation
            self.original_df = df
            self.time_col = time_col
            self.value_col = value_col
            self.y = df[value_col]
            self.y.index = df[time_col]
        # If user doesn't provide trend change points, the trend change points will be found automatically.
        else:
            # Checks if trend change point is available.
            # Runs trend change point detection with default value if not.
            compare_df = df.copy()
            if time_col != self.time_col or value_col != self.value_col:
                compare_df.rename({time_col: self.time_col, value_col: self.value_col}, axis=1, inplace=True)
            if (self.original_df is not None
                    and self.original_df[[self.time_col, self.value_col]].equals(
                        compare_df[[self.time_col, self.value_col]])
                    and self.trend_estimation is not None):
                # If the passed df is the same as ``self.original_df``, then the previous
                # ``self.trend_estimation`` is to be subtracted from the time series.
                trend_estimation = self.trend_estimation
                warnings.warn("Trend changepoints are already identified, using past trend estimation. "
                              "If you would like to run trend change point detection again, "
                              "please call ``find_trend_changepoints`` with desired parameters "
                              "before calling ``find_seasonality_changepoints``.")
            else:
                # If the passed df is different from ``self.original_df``, then trend change point
                # detection algorithm is run first, and trend estimation is calculated afterward.
                # In this case, the parameters passed to ``find_seasonality_changepoints`` are also
                # passed to ``find_trend_changepoint``.
                # If you do not want the parameters passed, run ``find_trend_changepoints`` with
                # desired parameters before calling ``find_seasonality_changepoints``.
                trend_result = self.find_trend_changepoints(
                    df=df,
                    time_col=time_col,
                    value_col=value_col,
                    actual_changepoint_min_distance=actual_changepoint_min_distance,
                    no_changepoint_distance_from_end=no_changepoint_distance_from_end,
                    no_changepoint_proportion_from_end=no_changepoint_proportion_from_end
                )
                warnings.warn(f"Trend changepoints are not identified for the input dataframe, "
                              f"triggering trend change point detection with parameters"
                              f"actual_changepoint_min_distance={actual_changepoint_min_distance}\n"
                              f"no_changepoint_proportion_from_end={no_changepoint_proportion_from_end}\n"
                              f"no_changepoint_distance_from_end={no_changepoint_distance_from_end}\n"
                              f" Found trend change points\n{self.trend_changepoints}\n"
                              "If you would like to run trend change point detection with customized "
                              "parameters, please call ``find_trend_changepoints`` with desired parameters "
                              "before calling ``find_seasonality_changepoints``.")
                trend_estimation = trend_result["trend_estimation"]
        # Splits trend effects from time series.
        df_without_trend = df.copy()
        df_without_trend[value_col] -= trend_estimation.values
        # Aggregates df.
        df_resample = df_without_trend.resample(resample_freq, on=time_col).mean().reset_index()
        df_resample = df_resample.dropna()
        # Removes daily component from seasonality_components_df if data has minimum freq daily.
        freq_at_least_day = (min(np.diff(df_resample[time_col]).astype("timedelta64[s]")) >= timedelta(days=1))
        if (freq_at_least_day
                and "daily" in seasonality_components_df["seas_names"].tolist()):
            warnings.warn("Inferred minimum data frequency is at least 1 day, daily component is "
                          "removed from seasonality_components_df.")
            seasonality_components_df = seasonality_components_df.loc[
                seasonality_components_df["seas_names"] != "daily"]
        # Builds seasonality feature df.
        seasonality_df = build_seasonality_feature_df_with_changes(
            df=df_resample,
            time_col=time_col,
            fs_components_df=seasonality_components_df,
            changepoints_dict={
                "method": "uniform",
                "n_changepoints": potential_changepoint_n
            }
        )
        # Eliminates change points from the end
        # the generated seasonality_df has {``potential_changepoint_n`` + 1} blocks, where there
        # are {sum_i(order of component i) * 2} columns consisting of the cosine and sine functions
        # for each order for each component.
        # The selection below selects the first {``n_changepoints_within_range`` + 1} columns,
        # which corresponds to the regular block (first block) and the blocks that correspond
        # to the change points that are within range.
        n_changepoints_within_range = int(potential_changepoint_n * (1 - no_changepoint_proportion_from_end))
        orders = seasonality_components_df["order"].tolist()
        seasonality_df = seasonality_df.iloc[:, :(n_changepoints_within_range + 1) * sum(orders) * 2]
        self.seasonality_df = seasonality_df
        # Fetches change point dates for reference as datetime format.
        changepoint_dates = get_evenly_spaced_changepoints_dates(
            df=df,
            time_col=time_col,
            n_changepoints=potential_changepoint_n
        )
        # Gets the changepoint dates that are not within ``no_changepoint_proportion_from_end``.
        changepoint_dates = changepoint_dates.iloc[0: n_changepoints_within_range + 1]
        # Calculates the minimal allowed change point index distance.
        min_changepoint_index_distance = compute_min_changepoint_index_distance(
            df=df_resample,
            time_col=time_col,
            n_changepoints=potential_changepoint_n,
            min_distance_between_changepoints=actual_changepoint_min_distance
        )
        seasonality_changepoints = get_seasonality_changes_from_adaptive_lasso(
            x=seasonality_df.values,
            y=df_resample[value_col].values,
            changepoint_dates=changepoint_dates,
            initial_coef="lasso",
            seasonality_components_df=seasonality_components_df,
            min_index_distance=min_changepoint_index_distance,
            regularization_strength=regularization_strength
        )
        # Checks if the beginning date is picked as a change point. If yes, drop it, because we
        # always include the overall seasonality term in our model.
        for key in seasonality_changepoints.keys():
            if df_resample[time_col][0] in seasonality_changepoints[key]:
                seasonality_changepoints[key] = seasonality_changepoints[key][1:]
        self.seasonality_changepoints = seasonality_changepoints
        # logging
        log_message(f"The detected seasonality changepoints are\n"
                    f"{pprint(seasonality_changepoints)}", LoggingLevelEnum.INFO)
        # Performs a seasonality estimation for plotting purposes.
        seasonality_estimation = estimate_seasonality_with_detected_changepoints(
            df=df_without_trend,
            time_col=time_col,
            value_col=value_col,
            seasonality_changepoints=seasonality_changepoints,
            seasonality_components_df=seasonality_components_df
        )
        self.seasonality_estimation = seasonality_estimation
        result = {
            "seasonality_feature_df": seasonality_df,
            "seasonality_changepoints": seasonality_changepoints,
            "seasonality_estimation": seasonality_estimation,
            "seasonality_components_df": seasonality_components_df
        }
        return result

    def plot(
            self,
            observation=True,
            observation_original=True,
            trend_estimate=True,
            trend_change=True,
            yearly_seasonality_estimate=False,
            adaptive_lasso_estimate=False,
            seasonality_change=False,
            seasonality_change_by_component=True,
            seasonality_estimate=False,
            plot=True):
        """Makes a plot to show the observations/estimations/change points.

        In this function, component parameters specify if each component in the plot is
        included or not. These are `bool` variables.
        For those components that are set to True, their values will be replaced by the
        corresponding data. Other components values will be set to None. Then these variables
        will be fed into
        `~greykite.algo.changepoint.adalasso.changepoints_utils.plot_change`

        Parameters
        ----------
        observation : `bool`
            Whether to include observation
        observation_original : `bool`
            Set True to plot original observations, and False to plot aggregated observations.
            No effect is ``observation`` is False
        trend_estimate : `bool`
            Set True to add trend estimation.
        trend_change : `bool`
            Set True to add change points.
        yearly_seasonality_estimate : `bool`
            Set True to add estimated yearly seasonality.
        adaptive_lasso_estimate : `bool`
            Set True to add adaptive lasso estimated trend.
        seasonality_change : `bool`
            Set True to add seasonality change points.
        seasonality_change_by_component : `bool`
            If true, seasonality changes will be plotted separately for different components,
            else all will be in the same symbol.
            No effect if ``seasonality_change`` is False
        seasonality_estimate : `bool`
            Set True to add estimated seasonality.
            The seasonality if plotted around trend, so the actual seasonality shown is
            trend estimation + seasonality estimation.
        plot : `bool`, default True
            Set to True to display the plot, and set to False to return the plotly figure object.

        Returns
        -------
        None (if ``plot`` == True)
            The function shows a plot.
        fig : `plotly.graph_objects.Figure`
            The plot object.
        """
        # Adds observation
        if observation:
            if observation_original:
                observation = self.original_df[self.value_col]
                observation.index = self.original_df[self.time_col]
            else:
                observation = self.y
        else:
            observation = None
        # Adds trend estimation
        if trend_estimate:
            trend_estimate = compute_fitted_components(
                x=self.trend_df,
                coef=self.trend_coef,
                regex='^changepoint',
                include_intercept=True,
                intercept=self.trend_intercept
            )
        else:
            trend_estimate = None
        # Adds trend change points
        if trend_change:
            if self.trend_changepoints is None:
                warnings.warn("You haven't run trend change point detection algorithm yet. "
                              "Please call find_trend_changepoints first.")
            trend_change = self.trend_changepoints
        else:
            trend_change = None
        # Adds yearly seasonality estimates
        if yearly_seasonality_estimate:
            yearly_seasonality_estimate = compute_fitted_components(
                x=self.trend_df,
                coef=self.trend_coef,
                regex='^.*yearly.*$',
                include_intercept=False
            )
        else:
            yearly_seasonality_estimate = None
        # Adds adaptive lasso trend estimates
        if adaptive_lasso_estimate and self.adaptive_lasso_coef is not None:
            adaptive_lasso_estimate = compute_fitted_components(
                x=self.trend_df,
                coef=self.adaptive_lasso_coef[1],
                regex='^changepoint',
                include_intercept=True,
                intercept=self.adaptive_lasso_coef[0])
        else:
            adaptive_lasso_estimate = None
        # Adds seasonality change points
        if seasonality_change:
            if self.seasonality_changepoints is None:
                warnings.warn("You haven't run seasonality change point detection algorithm yet. "
                              "Please call find_seasonality_changepoints first.")
            if seasonality_change_by_component:
                seasonality_change = self.seasonality_changepoints
            else:
                seasonality_change = []
                for key in self.seasonality_changepoints.keys():
                    seasonality_change += self.seasonality_changepoints[key]
        else:
            seasonality_change = None
        # Adds seasonality estimates
        if seasonality_estimate:
            if self.seasonality_estimation is None:
                warnings.warn("You haven't run seasonality change point detection algorithm yet. "
                              "Please call find_seasonality_changepoints first.")
                seasonality_estimate = None
            else:
                seasonality_estimate = self.seasonality_estimation + self.trend_estimation
        else:
            seasonality_estimate = None
        fig = plot_change(
            observation=observation,
            trend_estimate=trend_estimate,
            trend_change=trend_change,
            year_seasonality_estimate=yearly_seasonality_estimate,
            adaptive_lasso_estimate=adaptive_lasso_estimate,
            seasonality_change=seasonality_change,
            seasonality_estimate=seasonality_estimate,
            yaxis=self.value_col
        )
        if fig is not None and len(fig.data) > 0:
            if plot:
                fig.show()
            else:
                return fig
        else:
            warnings.warn("Figure is empty, at least one component has to be true.")
            return None


def get_changepoints_dict(df, time_col, value_col, changepoints_dict):
    """The functions takes the ``changepoints_dict`` dictionary and returns the processed
    ``changepoints_dict``.

    If "method" == "auto", the change point detection algorithm is run and the function returns
    the ``changepoints_dict`` with "method"="custom" and automatically detected change points.
    If "method" == "custom" or "uniform", the original ``changepoints_dict`` is returned.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The dataframe used to do change point detection.
    time_col : `str`
        The column name of time in ``df``.
    value_col : `str`
        The column name of values in ``df``.
    changepoints_dict : `dict`
        The ``changepoints_dict`` parameter that is fed into
        `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast` or
        `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.forecast_simple_silverkite`
        It must have keys:

            `"method"` : `str`, equals "custom", "uniform" or "auto"

        Depending on `"method"`, it must have keys:

            `"n_changepoints"` : `int`, when "method" == "uniform".
            `"dates"` : `Iterable[Union[int, float, str, datetime]]`, when "method" == "custom"

        When `"method"` == "auto", it can have optional keys that matches the parameters in
        `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_trend_changepoints`,
        except ``df``, ``time_col`` and ``value_col``, and extra keys "dates", "combine_changepoint_min_distance"
        and "keep_detected", which correspond to the three parameters "custom_changepoint_dates", "min_distance"
        and "keep_detected" in `~greykite.algo.changepoint.adalasso.changepoints_utils.combine_detected_and_custom_trend_changepoints`.
        In all three "method" cases, it can have optional keys:

            "continuous_time_col": ``str`` or ``None``
                Column to apply `growth_func` to, to generate changepoint features
                Typically, this should match the growth term in the model
            "growth_func": ``callable`` or ``None``
                Growth function (scalar -> scalar). Changepoint features are created
                by applying `growth_func` to "continuous_time_col" with offsets.
                If None, uses identity function to use `continuous_time_col` directly
                as growth term.

    Returns
    -------
    changepoints_dict : `dict`
        If "method" == "custom" or "method" == "uniform", the return is the original dictionary.
        If "method" == "auto", the return is the dictionary with "method" = "custom" and
        "dates" = detected_changepoints. Change point detection related keys and values are used
        for change point detection, other keys and values will be included in the return dictionary.

    changepoint_detector : `ChangepointDetector` or `None`
        The ChangepointDetector class used for automatically trend changepoint detection if "method" == "auto",
        otherwise None.
    """
    if (changepoints_dict is not None
            and "method" in changepoints_dict.keys()
            and changepoints_dict["method"] == "auto"):
        changepoint_detection_args = {
            "df": df,
            "time_col": time_col,
            "value_col": value_col
        }
        changepoint_detection_keys = [
            "yearly_seasonality_order",
            "yearly_seasonality_change_freq",
            "resample_freq",
            "trend_estimator",
            "adaptive_lasso_initial_estimator",
            "regularization_strength",
            "actual_changepoint_min_distance",
            "potential_changepoint_distance",
            "potential_changepoint_n",
            "potential_changepoint_n_max",
            "no_changepoint_distance_from_begin",
            "no_changepoint_proportion_from_end",
            "no_changepoint_distance_from_end",
            "no_changepoint_proportion_from_end"
        ]
        changepoints_dict_keys = [
            "continuous_time_col",
            "growth_func"
        ]
        for key in changepoint_detection_keys:
            if key in changepoints_dict.keys():
                changepoint_detection_args[key] = changepoints_dict[key]
        model = ChangepointDetector()
        result = model.find_trend_changepoints(**changepoint_detection_args)
        new_changepoints_dict = result["changepoints_dict"]
        custom_changepoint_keys = []
        if "dates" in changepoints_dict:
            # Here we reuse the key "dates" which is used in "method"=="custom" as additional custom dates
            # when "method"=="auto" to avoid having too many keys.
            # Gets the custom changepoints.
            custom_changepoints = changepoints_dict["dates"]
            df_dates = pd.to_datetime(df[time_col])
            min_date = min(df_dates)
            max_date = max(df_dates)
            custom_changepoints = pd.to_datetime(custom_changepoints)
            custom_changepoints = [cp for cp in custom_changepoints if min_date < cp < max_date]
            min_distance = changepoints_dict.get("combine_changepoint_min_distance", None)
            # If ``combine_changepoint_min_distance`` is not set, try to set is with ``actual_changepoint_min_distance``.
            if min_distance is None:
                min_distance = changepoints_dict.get("actual_changepoint_min_distance", None)
            # If ``keep_detected`` is not set, the default is False to keep custom changepoints.
            keep_detected = changepoints_dict.get("keep_detected", False)
            # Checks if custom changepoints are provided.
            # If provided, combines the detected changepoints and custom changepoints.
            custom_changepoint_keys = ["dates"]
            if custom_changepoints is not None:
                # Keys used in adding custom changepoints
                custom_changepoint_keys = [
                    "dates",
                    "combine_changepoint_min_distance",
                    "keep_detected"
                ]
                combined_changepoints = combine_detected_and_custom_trend_changepoints(
                    detected_changepoint_dates=new_changepoints_dict["dates"],
                    custom_changepoint_dates=custom_changepoints,
                    min_distance=min_distance,
                    keep_detected=keep_detected
                )
                # Logs info if the detected changepoints are altered.
                if not set(combined_changepoints) == set(new_changepoints_dict["dates"]):
                    log_message(f"Custom trend changepoints have been added to the detected trend changepoints."
                                f" The final trend changepoints are {combined_changepoints}",
                                LoggingLevelEnum.INFO)
                new_changepoints_dict["dates"] = combined_changepoints
        for key in changepoints_dict_keys:
            if key in changepoints_dict.keys():
                new_changepoints_dict[key] = changepoints_dict[key]
        unused_keys = [key for key in changepoints_dict.keys()
                       if key not in ["method"] + changepoints_dict_keys + changepoint_detection_keys + custom_changepoint_keys]
        if unused_keys:
            warnings.warn(f"The following keys in ``changepoints_dict`` are not recognized\n"
                          f"{unused_keys}")
        return new_changepoints_dict, model
    else:
        return changepoints_dict, None


def get_seasonality_changepoints(
        df,
        time_col,
        value_col,
        trend_changepoints_dict=None,
        trend_changepoint_dates=None,
        seasonality_changepoints_dict=None):
    """Automatically detects seasonality change points.

    The function first converts changepoints_dict if "method" == "auto", then extracts trend
    changepoint dates from the dictionary and feeds them into ``find_seasonality_changepoints``.
    With the detected seasonality change points, the detection result dictionary is returned.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The data df.
    time_col : `str`
        The column name for time in ``df``.
    value_col : `str`
        The column name for value in ``df``.
    trend_changepoints_dict : `dict` or `None`, default `None`
        The ``changepoints_dict`` parameter in
        `~greykite.algo.changepoint.adalasso.changepoint_detector.get_changepoints_dict`
    trend_changepoint_dates : `list` or `None`, default `None`
        List of trend change point dates. The dates need to be parsable ty `pandas.to_datetime`.
        If given, trend change point detection will not be run and ``trend_changepoints_dict``
        will have no effect.
    seasonality_changepoints_dict : `dict` or `None`, default `None`
        The keys are the parameter names of
        `~greykite.algo.changepoint.adalasso.changepoint_detector.find_seasonality_changepoints`
        The values are the corresponding desired values.
        Note ``df``, ``time_col``, ``value_col`` and ``trend_changepoints`` are auto populated,
        and do not need to be provided.

    Returns
    -------
    result : `dict`
        The detected seasonality change points result dictionary as returned by
        `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`.
    """
    if trend_changepoint_dates is None:
        # Runs trend change point detection algorithm if "method" == "auto", and returns
        # changepoints_dict with "method" == "custom";
        # Returns the original changepoints_dict if "method" == "uniform" or "custom";
        # Returns None if the original changepoints_dict is None.
        trend_changepoints_dict, _ = get_changepoints_dict(
            df=df,
            time_col=time_col,
            value_col=value_col,
            changepoints_dict=trend_changepoints_dict
        )
        # Extracts change point dates from the changepoints_dict.
        # Returns None if the original changepoints_dict is None.
        trend_changepoint_dates = get_changepoint_dates_from_changepoints_dict(
            changepoints_dict=trend_changepoints_dict,
            df=df,
            time_col=time_col
        )
    # Prepares arguments for ``find_seasonality_changepoints`` function.
    seasonality_changepoint_detection_args = {
        "df": df,
        "time_col": time_col,
        "value_col": value_col
    }
    seasonality_changepoint_detection_keys = [
        "seasonality_components_df",
        "resample_freq",
        "regularization_strength",
        "actual_changepoint_min_distance",
        "potential_changepoint_distance",
        "potential_changepoint_n",
        "no_changepoint_distance_from_end",
        "no_changepoint_proportion_from_end"
    ]
    if seasonality_changepoints_dict is not None:
        for key in seasonality_changepoint_detection_keys:
            if key in seasonality_changepoints_dict.keys():
                seasonality_changepoint_detection_args[key] = seasonality_changepoints_dict[key]
    seasonality_changepoint_detection_args["trend_changepoints"] = trend_changepoint_dates
    # Runs ``find_seasonality_changepoints``.
    cd = ChangepointDetector()
    result = cd.find_seasonality_changepoints(**seasonality_changepoint_detection_args)
    # Builds seasonality features with change points.
    return result
