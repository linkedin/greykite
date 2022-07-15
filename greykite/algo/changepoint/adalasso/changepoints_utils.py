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
"""Utilities for changepoint detection via adaptive lasso."""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.plotting import register_matplotlib_converters
from pandas.tseries.frequencies import to_offset
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from greykite.common.constants import CHANGEPOINT_COL_PREFIX
from greykite.common.constants import TimeFeaturesEnum
from greykite.common.features.timeseries_features import add_time_features_df
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.features.timeseries_features import fourier_series_multi_fcn
from greykite.common.features.timeseries_features import get_changepoint_dates_from_changepoints_dict
from greykite.common.features.timeseries_features import get_changepoint_features
from greykite.common.features.timeseries_features import get_changepoint_features_and_values_from_config
from greykite.common.features.timeseries_features import get_default_origin_for_time_vars
from greykite.common.python_utils import get_pattern_cols
from greykite.common.python_utils import unique_elements_in_list


np.seterr(divide="ignore")  # np.where evaluates values before selecting by conditions, set this to suppress divide_by_zero error


register_matplotlib_converters()


def check_freq_unit_at_most_day(freq, name):
    """Checks if the ``freq`` parameter passed to a function has unit at most "D"

    Parameters
    ----------
    freq : `DateOffset`, `Timedelta` or `str`
        The parameter passed as ``freq``.
    name : `str`
        The name of the parameter ``freq``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Because the input frequency has unit greater than "D".
    """
    if (isinstance(freq, str) and
            any(char in freq for char in ["W", "M", "Y"])):
        raise ValueError(f"In {name}, the maximal unit is 'D', "
                         "i.e., you may use units no more than 'D' such as"
                         "'10D', '5H', '100T', '200S'. The reason is that 'W', 'M' "
                         "or higher has either cycles or indefinite number of days, "
                         "thus is not parsable by pandas as timedelta.")


def build_trend_feature_df_with_changes(
        df,
        time_col,
        origin_for_time_vars=None,
        changepoints_dict="auto"):
    """A function to generate trend features from a given time series df.

    The trend features include columns of the format max(0, x-c_i), where c_i is the i-th change point value.
    If changepoints_dict has "method": "uniform", then n_changepoints change points plus the normal growth term are
    generated uniformly over the whole time period. This is recommended.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The data df.
    time_col : `str'
        The column name of time column in ``df``, entries can be parsed with pd.to_datetime.
    origin_for_time_vars : `float` or `None`, default `None`
        Original continuous time value, if not provided, will be parsed from data.
    changepoints_dict: `str`: "auto" or `dict`, default "auto"
        Change point dictionary, compatible with
        `~greykite.common.features.timeseries_features.get_changepoint_features_and_values_from_config`
        If not provided, default is 100 change points evenly
        distributed over the whole time period.

    Returns
    -------
    df :  `pandas.DataFrame`
        Change point feature df with n_changepoints + 1 columns generated.
    """
    df = df.copy()
    # Gets changepoints features from config. Evenly distributed with n_changepoints specified is recommended.
    if origin_for_time_vars is None:
        origin_for_time_vars = get_default_origin_for_time_vars(df, time_col)
    if changepoints_dict == "auto":
        changepoints_dict = {
            "method": "uniform",
            "n_changepoints": 100
        }
    changepoints = get_changepoint_features_and_values_from_config(
        df=df,
        time_col=time_col,
        changepoints_dict=changepoints_dict,
        origin_for_time_vars=origin_for_time_vars)
    if changepoints["changepoint_values"] is None:
        # allows n_changepoints = 0
        changepoint_values = np.array([0])
    else:
        changepoint_values = np.concatenate([[0], changepoints["changepoint_values"]])
    growth_func = changepoints["growth_func"]
    features_df = add_time_features_df(
        df=df,
        time_col=time_col,
        conti_year_origin=origin_for_time_vars)
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(
        changepoints_dict=changepoints_dict,
        df=df,
        time_col=time_col)
    changepoint_dates = [pd.to_datetime(df[time_col].iloc[0])] + changepoint_dates
    changepoint_features_df = get_changepoint_features(
        features_df,
        changepoint_values,
        continuous_time_col=TimeFeaturesEnum.ct1.value,
        growth_func=growth_func,
        changepoint_dates=changepoint_dates)
    changepoint_features_df.index = pd.to_datetime(df[time_col])
    return changepoint_features_df


def build_seasonality_feature_df_with_changes(
        df,
        time_col,
        origin_for_time_vars=None,
        changepoints_dict=None,
        fs_components_df=pd.DataFrame({
            "name": [
                TimeFeaturesEnum.tod.value,
                TimeFeaturesEnum.tow.value,
                TimeFeaturesEnum.toy.value],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 3, 5],
            "seas_names": ["daily", "weekly", "yearly"]})):
    """A function to generate yearly seasonality features from a given time series df.

    The seasonality features include n_changepoints * 2 columns of the format
    1{x > c_i} * sin(2 * pi / period * order * x) and 1{x > c_i} * cos(2 * pi / period * order * x),
    where c_i is the i-th change point value.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The data df.
    time_col : `str'
        The column name of time column in ``df``, entries can be parsed with pd.to_datetime.
    origin_for_time_vars : `float` or `None`, default `None`
        Original continuous time value, if not provided, will be parsed from data.
    changepoints_dict: `dict` or `None`, default `None`
        Change point dictionary, compatible with
        `~greykite.common.features.timeseries_features.get_changepoint_features_and_values_from_config`
        If not provided, default is no change points.
    fs_components_df: `pandas.DataFrame`
        fs_components config df, compatible with
        `~greykite.common.features.timeseries_features.fourier_series_multi_fcn`

    Returns
    -------
    df : `pandas.DataFrame`
        Seasonality feature df with [sum_{component_i} (2 * order_of_component_i)] * (num_changepoints + 1)
        columns.
        Each sum_{component_i} (2 * order_of_component_i) columns form a block. The first block contains the
        original seasonality features, which accounts for the overall seasonality magnitudes. Each of the
        rest num_changepoints block is a copy of the first block, with rows whose timestamps are before the
        corresponding changepoint replaced by zeros. Such blocks only have effect on the seasonality
        magnitudes after the corresponding changepoints.
    """
    df = df.copy()
    # Gets changepoints features from config. Evenly distributed with n_changepoints specified is recommended.
    if origin_for_time_vars is None:
        origin_for_time_vars = get_default_origin_for_time_vars(df, time_col)
    features_df = add_time_features_df(
        df=df,
        time_col=time_col,
        conti_year_origin=origin_for_time_vars)
    fs_func = None
    fs_cols = []
    if fs_components_df is not None:
        fs_components_df = fs_components_df[fs_components_df["order"] != 0]
        fs_components_df = fs_components_df.reset_index()
        if fs_components_df.shape[0] > 0:
            fs_func = fourier_series_multi_fcn(
                col_names=fs_components_df["name"],
                periods=fs_components_df.get("period"),
                orders=fs_components_df.get("order"),
                seas_names=fs_components_df.get("seas_names")
            )
            time_features_example_df = build_time_features_df(
                df[time_col][:2],  # only needs the first two rows to get the column info
                conti_year_origin=origin_for_time_vars)
            fs = fs_func(time_features_example_df)
            fs_cols = fs["cols"]
    fs_features = fs_func(features_df)
    fs_df = fs_features["df"]
    fs_df.index = pd.to_datetime(df[time_col])
    # Augments regular Fourier columns to truncated Fourier columns.
    if changepoints_dict is not None:
        changepoint_dates = get_changepoint_dates_from_changepoints_dict(
            changepoints_dict=changepoints_dict,
            df=df,
            time_col=time_col
        )
        changepoint_dates = unique_elements_in_list(changepoint_dates)
        # The following lines truncates the fourier series at each change point
        # For each change point, the values of the fourier series before the change point are
        # set to zero. The column name is simply appending `_%Y_%m_%d_%H` after the original column names.
        col_names = fs_cols + [f"{col}{date.strftime('_%Y_%m_%d_%H')}" for date in changepoint_dates for col in fs_cols]
        fs_truncated_df = pd.concat([fs_df] * (len(changepoint_dates) + 1), axis=1)
        fs_truncated_df.columns = col_names
        for i, date in enumerate(changepoint_dates):
            cols = fs_truncated_df.columns[fs_df.shape[1] * (i + 1):]
            fs_truncated_df.loc[(features_df["datetime"] < date).values, cols] = 0
        fs_df = fs_truncated_df
    return fs_df


def build_seasonality_feature_df_from_detection_result(
        df,
        time_col,
        seasonality_changepoints,
        seasonality_components_df,
        include_original_block=True,
        include_components=None):
    """Builds seasonality feature df from detected seasonality change points.

    The function is an extension of ``build_seasonality_feature_df_with_changes``. It can generate
    a seasonality feature df with different change points for different components. These changepoints
    are passed through a dictionary, which can be the output of
    `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`

    Parameters
    __________
    df : `pandas.DataFrame`
        The dataframe used to build seasonality feature df.
    time_col : `str`
        The name of the time column in ``df``.
    seasonality_changepoints : `dict`
        The seasonality change point dictionary. The keys are seasonality components, and the values
        are the corresponding change points given in lists.
        For example

            "weekly": [Timestamp('2020-01-01 00:00:00'), Timestamp('2021-04-05 00:00:00')]
            "yearly": [Timestamp('2020-08-06 00:00:00')]

    seasonality_components_df : `pandas.DataFrame`
        The seasonality components dataframe that is compatible with
        `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`
        The values in "seas_names" must equal to the keys in ``seasonality_changepoints``.
    include_original_block : `bool`, default True
        Whether to include the original untruncated block of sin or cos columns for each component. If set to
        False, the original seasonality block for each component will be dropped.
    include_components : `list` [`str`] or None, default None
        The components to be included from the result. If None, all components will be included.

    Returns
    -------
    seasonality_feature_df : `pandas.DataFrame`
        The seasonality feature dataframe similar to the output of `build_seasonality_feature_df_with_changes``
        but possibly with different changepoints on different components.
    """
    seasonality_df = pd.DataFrame()
    for component in seasonality_changepoints.keys():
        if include_components is None or component in include_components:
            seasonality_df_cp = build_seasonality_feature_df_with_changes(
                df=df,
                time_col=time_col,
                changepoints_dict={
                    "method": "custom",
                    "dates": seasonality_changepoints[component]
                },
                fs_components_df=seasonality_components_df[seasonality_components_df["seas_names"] == component]
            )
            if not include_original_block:
                original_block_size = int(2 * seasonality_components_df.loc[
                    seasonality_components_df["seas_names"] == component, "order"].values[0])
                seasonality_df_cp = seasonality_df_cp.iloc[:, original_block_size:]
            seasonality_df = pd.concat([seasonality_df, seasonality_df_cp], axis=1)
    extra_components = [component for component in seasonality_changepoints if component not in include_components] if include_components is not None else []
    if extra_components:
        warnings.warn(f"The following seasonality components have detected seasonality changepoints"
                      f" but these changepoints are not included in the model,"
                      f" because the seasonality component is not included in the model. {extra_components}")
    return seasonality_df


def compute_fitted_components(
        x,
        coef,
        regex,
        include_intercept,
        intercept=None):
    """Computes the fitted values with selected regressors indicated by ``regex``

    Parameters
    ----------
    x : `pandas.DataFrame`
        The design matrix df with conventional column names.
    coef : `numpy.array`
        Estimated coefficients.
    regex : regular expression
        Pattern of the names of the columns to be used.
    include_intercept : bool
        Whether to include intercept.
    intercept : `float`
        The estimated intercept, must be provided if ``include_intercept`` == True.

    Returns
    -------
    result: `pandas.Series`
        The estimated component from selected regressors.
    """
    if include_intercept and intercept is None:
        raise ValueError("``intercept`` must be provided when ``include_intercept`` is True.")
    columns = list(x.columns)
    selected_columns = get_pattern_cols(columns, regex)
    selected_columns_idx = [columns.index(col) for col in selected_columns]
    selected_coef = coef[selected_columns_idx]
    component = x[selected_columns].dot(selected_coef)
    if include_intercept:
        component += intercept
    return component


def plot_change(
        observation=None,
        adaptive_lasso_estimate=None,
        trend_change=None,
        trend_estimate=None,
        year_seasonality_estimate=None,
        seasonality_change=None,
        seasonality_estimate=None,
        title=None,
        xaxis="Dates",
        yaxis="Values"):
    """Makes a plot of the observed data and estimated components, as well as detected changes

    The function currently allows five different components to be plotted together. Specifically,
    ``trend_change`` can be plotted with at least one of ``observations``, ``trend_estimate`` and
    ``adaptive_lasso_estimate`` is provided.

    Parameters
    ----------
    observation : `pandas.Series` with time index or `None`
        The observed values, leave None to omit it from plot.
    adaptive_lasso_estimate : `pandas.Series` with time index or `None`
        The adaptive lasso estimated trend, leave None to omit it from plot.
    trend_change : `pandas.Series` with time index or `None`
        The detected trend change points, leave None to omit it from plot.
        Plotted as vertical lines if observation is provided, otherwise plotted as markers.
    trend_estimate : `pandas.Series` with time index or `None`
        The estimated trend, leave None to omit it from plot.
    year_seasonality_estimate : `pandas.Series` with time index or `None`
        The estimated yearly seasonality, leave None to omit it from plot.
    seasonality_change : `list` or `dict` or `None`
        The detected seasonality change points, leave None to omit it from plot.
        If the type is `list`, it should be a list of change points of all components.
        If the type is `dict`, its keys should be the name of components, and values
        should be the corresponding list of change points.
    seasonality_estimate : `pandas.Series` or `None`
        The estimated seasonality, leave None to omit it from plot.
    title : `str` or `None`
        Plot title.
    xaxis : `str`
        Plot x axis label.
    yaxis : `str`
        Plot y axis label.

    Returns
    -------
    fig : `plotly.graph_objects.Figure`
        The plotted plotly object, can be shown with `fig.show()`.
    """
    if title is None:
        if trend_change is None and seasonality_change is None:
            title = "Timeseries Plot"
        elif trend_change is None:
            title = "Timeseries Plot with detected seasonality change points"
        elif seasonality_change is None:
            title = "Timeseries Plot with detected trend change points"
        else:
            title = "Timeseries Plot with detected trend and seasonality change points"
    fig = go.Figure()
    # shows the true observation
    if observation is not None:
        fig.add_trace(
            go.Scatter(
                x=observation.index,
                y=observation,
                name="true",
                mode="lines",
                line=dict(color="#42A5F5"),  # blue 400
                opacity=1)
        )
    # shows the seasonality estimate
    if seasonality_estimate is not None:
        fig.add_trace(
            go.Scatter(
                name="seasonality+trend",
                mode="lines",
                x=seasonality_estimate.index,
                y=seasonality_estimate,
                line=dict(color="#B2FF59"),  # light green A200
                opacity=0.7,
                showlegend=True)
        )
    # shows the adaptive lasso estimated trend
    if adaptive_lasso_estimate is not None:
        fig.add_trace(
            go.Scatter(
                name="adaptive lasso estimated trend",
                mode="lines",
                x=adaptive_lasso_estimate.index,
                y=adaptive_lasso_estimate,
                line=dict(color="#FFA726"),  # orange 400
                opacity=1,
                showlegend=True)
        )
    # shows the detected trend change points
    # shown as vertical lines if ``observation`` is provided, otherwise as markers
    if trend_change is not None:
        if observation is not None or seasonality_estimate is not None:
            if observation is not None:
                min_y = min(0, observation.min())
                max_y = max(0, observation.max())
            else:
                min_y = min(0, seasonality_estimate.min())
                max_y = max(0, seasonality_estimate.max())
            for i, cp in enumerate(trend_change):
                showlegend = (i == 0)
                fig.add_trace(
                    go.Scatter(
                        name="trend change point",
                        mode="lines",
                        x=[pd.to_datetime(cp), pd.to_datetime(cp)],
                        y=[min_y, max_y],
                        line=go.scatter.Line(
                            color="#F44336",  # red 500
                            width=1.5,
                            dash="dash"
                        ),
                        opacity=1,
                        showlegend=showlegend,
                        legendgroup="trend"
                    )
                )
        elif trend_estimate is not None or adaptive_lasso_estimate is not None:
            trend = trend_estimate if trend_estimate is not None else adaptive_lasso_estimate
            for cp in trend_change:
                value = trend[cp]
                fig.add_trace(
                    go.Scatter(
                        name="trend change points",
                        mode="markers",
                        x=[pd.to_datetime(cp)],
                        y=[value],
                        marker=dict(
                            color="#F44336",  # red 500
                            size=15,
                            line=dict(
                                color="Black",
                                width=1.5
                            )),
                        opacity=0.2,
                        showlegend=False
                    )
                )
        else:
            warnings.warn("trend_change is not shown. Must provide observations, trend_estimate, "
                          "adaptive_lasso_estimate or seasonality_estimate to plot trend_change.")
            return None
    # shows the detected seasonality change points
    if seasonality_change is not None:
        if observation is not None or seasonality_estimate is not None:
            if observation is not None:
                min_y = min(0, observation.min())
                max_y = max(0, observation.max())
            else:
                min_y = min(0, seasonality_estimate.min())
                max_y = max(0, seasonality_estimate.max())
            if isinstance(seasonality_change, list):
                for i, cp in enumerate(seasonality_change):
                    showlegend = (i == 0)
                    fig.add_trace(
                        go.Scatter(
                            name="seasonality change point",
                            mode="lines",
                            x=[pd.to_datetime(cp), pd.to_datetime(cp)],
                            y=[min_y, max_y],
                            line=go.scatter.Line(
                                color="#1E88E5",  # blue 600
                                width=1.5,
                                dash="dash"
                            ),
                            opacity=1,
                            showlegend=showlegend,
                            legendgroup="seasonality"
                        )
                    )
            elif isinstance(seasonality_change, dict):
                colors = ["#2196F3", "#0D47A1"]  # [blue 500, blue 900]
                dashes = ["dash", "dot", "dashdot", None]
                types = [(color, dash) for color in colors for dash in dashes]
                quota = len(types)
                for i, (key, changepoints) in enumerate(seasonality_change.items()):
                    if quota == 0:
                        warnings.warn("Only the first 8 components with detected change points"
                                      "are plotted.")
                        break
                    if changepoints:
                        for j, cp in enumerate(changepoints):
                            showlegend = (j == 0)
                            fig.add_trace(
                                go.Scatter(
                                    name="seasonality change point " + key,
                                    mode="lines",
                                    x=[pd.to_datetime(cp), pd.to_datetime(cp)],
                                    y=[min_y, max_y],
                                    line=go.scatter.Line(
                                        color=types[i][0],
                                        width=1.5,
                                        dash=types[i][1]
                                    ),
                                    opacity=1,
                                    showlegend=showlegend,
                                    legendgroup=key
                                )
                            )
                        quota -= 1
            else:
                raise ValueError("seasonality_change must be either list or dict.")
        else:
            warnings.warn("seasonality_change is not shown. Must provide observations or"
                          " seasonality_estimate to plot seasonality_change.")
    # shows the estimated trend
    if trend_estimate is not None:
        fig.add_trace(
            go.Scatter(
                name="trend",
                mode="lines",
                x=trend_estimate.index,
                y=trend_estimate,
                line=dict(
                    color="#FFFF00",  # yellow A200
                    width=2),
                opacity=1,
                showlegend=True)
        )
    # shows the estimated yearly seasonality
    if year_seasonality_estimate is not None:
        fig.add_trace(
            go.Scatter(
                name="yearly seasonality",
                mode="lines",
                x=year_seasonality_estimate.index,
                y=year_seasonality_estimate,
                line=dict(color="#BF360C"),  # deep orange 900
                opacity=0.6,
                showlegend=True)
        )
    fig.update_layout(dict(
        xaxis=dict(title=xaxis),
        yaxis=dict(title=yaxis),
        title=title,
        title_x=0.5
    ))
    return fig


def adaptive_lasso_cv(x,
                      y,
                      initial_coef,
                      regularization_strength=None,
                      max_min_ratio=1e6):
    """Performs the adaptive lasso cross-validation.

     Algorithm is based on a transformation of `sklearn.linear_model.LassoCV()`.
     If initial_coef is not available, a lasso estimator is computed.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    initial_coef : `str` in ["ridge", "lasso", "old"] or `numpy.array`
        How to obtain the initial estimator. If a `str` is provided, the corresponding model is trained
        to obtain the initial estimator. If a `numpy.array` is provided, it is used as the initial
        estimator.
    regularization_strength : `float` in [0, 1] or `None`
        The regularization strength for change points. Greater values imply fewer change points.
        0 indicates all change points, and 1 indicates no change point.
        If `None`, cross-validation will be used to select tuning parameter,
        else the value will be used as the tuning parameters.
    max_min_ratio : `float`
        defines the min lambda by defining the ratio of lambda_max / lambda_min.
        `sklearn.linear_model.lasso_path` uses 1e3, but 1e6 seems better here.

    Returns
    -------
    intercept : `float`
        The estimated intercept.
    coef : `numpy.array`
        The estimated coefficients.
    """
    if regularization_strength is not None and (regularization_strength < 0 or regularization_strength > 1):
        raise ValueError("regularization_strength must be between 0.0 and 1.0.")
    if regularization_strength == 0:
        # regularization_strength == 0 implies linear regression
        model = LinearRegression().fit(x, y)
        return model.intercept_, model.coef_
    if regularization_strength == 1:
        # regularization_strength == 1 implies no change point selected
        # handle this case here separately for algorithm convergence and rounding concerns
        intercept = y.mean()
        coef = np.zeros(x.shape[1])
        return intercept, coef
    if type(initial_coef) == str:
        model = {"lasso": LassoCV(), "ols": LinearRegression(), "ridge": RidgeCV()}[initial_coef]
        model.fit(x, y)
        initial_coef = model.coef_
    else:
        assert x.shape[1] == len(initial_coef), (
            "the number of columns in x should equal to the length of weights"
            f"but got {x.shape[1]} and {len(initial_coef)}."
        )
    weights = np.where(initial_coef != 0, 1 / abs(initial_coef), 1e16)  # 1e16 is big enough for most cases.
    x_t = x / weights
    if regularization_strength is None:
        model = LassoCV().fit(x_t, y)
    else:
        # finds the minimum lambda that corresponds to no selection, formula derived from KKT condition.
        # this is the max lambda we need to consider
        max_lam = max(abs(np.matmul(x_t.T, y - y.mean()))) / x.shape[0]
        # the lambda we choose is ``regularization_strength``th log-percentile of [lambda_min, lambda_max]
        lam = 10 ** (np.log10(max_lam / max_min_ratio) + np.log10(max_min_ratio) * regularization_strength)
        model = Lasso(alpha=lam).fit(x_t, y)
    intercept = model.intercept_
    coef = model.coef_ / weights
    return intercept, coef


def find_neighbor_changepoints(cp_idx, min_index_distance=2):
    """Finds neighbor change points given their indices

    For example
        x = [1, 2, 3, 7, 8, 10, 15, 20]
    The return with `min_index_distance`=2 is
        result = [[1, 2, 3], [7, 8], [10], [15], [20]]
    The return with `min_index_distance`=3 is
        result = [[1, 2, 3], [7, 8, 10], [15], [20]]

    Parameters
    ----------
    cp_idx : `list`
        A list of the change point indices.
    min_index_distance : `int`
        The minimal index distance to be considered separate

    Returns
    -------
    neighbor_cps : `list`
        A list of neighbor change points lists. Single change points are put in a list as well.
    """
    if min_index_distance <= 0:
        raise ValueError("`min_index_distance` must be positive.")
    if len(cp_idx) <= 1:
        return [cp_idx]
    # check if sorted
    is_sorted = True
    for i in range(1, len(cp_idx)):
        if cp_idx[i] < cp_idx[i - 1]:
            is_sorted = False
            break
    # sort if `cp_idx` is not sorted
    if not is_sorted:
        cp_idx.sort()
        warnings.warn("The given `cp_idx` is not sorted. It has been sorted.")
    neighbor_cps = []
    i = 0
    while i < len(cp_idx):
        neighbor_cps.append([])
        neighbor_cps[-1].append(cp_idx[i])
        i += 1
        while i < len(cp_idx) and cp_idx[i] < neighbor_cps[-1][-1] + min_index_distance:
            neighbor_cps[-1].append(cp_idx[i])
            i += 1
    return neighbor_cps


def get_trend_changes_from_adaptive_lasso(x,
                                          y,
                                          changepoint_dates,
                                          initial_coef,
                                          min_index_distance=2,
                                          regularization_strength=None):
    """Parses the adaptive lasso estimator to get change point dates.

    The functions calls ``adaptive_lasso_cv`` to selected potential trend change points.
    Then a filter is applied to eliminate change points that are too close.
    Specifically, in a set of close change points, the one with the largest absolute
    coefficient will be kept.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    changepoint_dates : `pandas.Series`
        A pandas Series of all potential change point dates that were used to generate ``x``.
    initial_coef : `str` in ["ridge", "lasso", "old"] or `numpy.array'
        How to obtain the initial estimator. If a `str` is provided, the corresponding model is trained
        to obtain the initial estimator. If a `numpy.array` is provided, it is used as the initial
        estimator.
    min_index_distance : `int`
        The minimal index distance that is allowed between two change points.
    regularization_strength : `float` in [0, 1] or `None`
        The regularization power for change points. Greater values imply fewer change points.
        0 indicates all change points, and 1 indicates no change point.
        If `None`, cross-validation will be used to select tuning parameter,
        else the value will be used as the tuning parameters.

    Returns
    -------
    changepoints : `list`
        Detected trend change points.
    coefs : `list`
        Adaptive lasso estimated coefficients.
        First element is intercept, and second element is coefficients.
    """
    intercept, coef = adaptive_lasso_cv(x, y, initial_coef, regularization_strength)
    # x may contain yearly seasonality regressors, so we need i < len(changepoint_dates)
    nonzero_idx = [i for i, val in enumerate(coef) if val != 0 and i < len(changepoint_dates)]
    cp_blocks = find_neighbor_changepoints(
        cp_idx=nonzero_idx,
        min_index_distance=min_index_distance,
    )
    cp_idx = filter_changepoints(cp_blocks, coef, min_index_distance)
    return changepoint_dates.iloc[cp_idx].tolist(), [intercept, coef]


def compute_min_changepoint_index_distance(
        df,
        time_col,
        n_changepoints,
        min_distance_between_changepoints):
    """Computes the minimal index distance between two consecutive detected change points.

    Given a df, its time column, the number of change points that are evenly distributed,
    and the min_distance_between_changepoints in `DateOffset, Timedelta or str`, gets the
    min distance between change point indices.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The dataframe that has a time column.
    time_col : `str`
        The column name of time in ``df``.
    n_changepoints : `int`
        Number of change points that are uniformly placed over the time period.
    min_distance_between_changepoints : `DateOffset`, `Timedelta` or `str`
        The minimal distance that is allowed between two detected change points.
        Note: maximal unit is 'D', i.e., you may only use units no more than 'D' such as
        '10D', '5H', '100T', '200S'. The reason is that 'W', 'M' or higher has either
        cycles or indefinite number of days, thus is not parsable by pandas as timedelta.

    Returns
    -------
    min_changepoint_index_distance : `int`
        The minimal index distance that is allowed between two detected change points.
    """
    if n_changepoints == 0:
        return df.shape[0]
    check_freq_unit_at_most_day(min_distance_between_changepoints, "min_distance_between_changepoints")
    # `to_offset` function handles string frequency without numbers such as 'D' instead of '1D',
    # so we use `to_offset` here rather than `to_timedelta`.
    min_dist = to_offset(min_distance_between_changepoints)
    # There are `n_changepoints` change points and 1 growth term, so there is `n_changepoints` gaps
    # Therefore the following is divided by `n_changepoints`
    try:
        changepoint_dist = (df[time_col].iloc[-1] - df[time_col].iloc[0]) / n_changepoints
    except TypeError:
        changepoint_dist = (pd.to_datetime(df[time_col].iloc[-1]) - pd.to_datetime(df[time_col].iloc[0])) \
                           / n_changepoints
    return int(np.ceil(min_dist.delta.total_seconds() / changepoint_dist.total_seconds()))


def filter_changepoints(cp_blocks, coef, min_index_distance):
    """Filters change points that are too close.

    Given the ``cp_block`` from the output of ``find_neighbor_changepoints`` and the corresponding
    coefficients, finds the list of change points, one in each neighborhood list.

    The algorithm keeps all individual change points. If more than one change points are in the
    same block, with the maximum distance less than ``min_index_distance``, then only the one
    with the maximum absolute coefficient is retained. If more than one change points are in the
    same block, and the block covers a few ``max_index_distance`` length, then a greedy method
    is used to select change points. The principle is that, we perform one pass of the change
    points, within each ``max_index_distance``, we select one change point with the maximum
    absolute coefficients. If the next change point is with in ``max_index_distance`` of the
    previous one, the previous one is dropped. A back-tracking is also used to fill possible
    change points when a change point is dropped.

    Parameters
    ----------
    cp_blocks : `list`
        A list of list of change points, output from ``find_neighbor_changepoints``.
    coef : `numpy.array`
        The estimated coefficients for all potential change points.
        Note: ``coef`` is a 1-D array, which is the output from regression model.
    min_index_distance : `int`
        The minimum index distance between two selected change points.
        Note that this parameter is added to safe guard the extreme cases: if all
        significant change points are included in the same block, with insignificant
        change points connecting them, then we do not want to drop all and just leave
        one change points. With this parameter, we can leave more than one change point
        from each block, and keep them at least ``min_index_distance`` away.

    Returns
    -------
    changepoint_indices : `list`
        The indices of selected change points.
    """
    if min_index_distance <= 0:
        raise ValueError("`min_index_distance` is the minimum distance between change point"
                         "indices to consider them separate, and must be positive.")
    if min_index_distance == 1:
        return [cp for block in cp_blocks for cp in block]
    coef = coef.ravel()  # makes sure ``coef`` is 1-D array
    coef = coef[[i for block in cp_blocks for i in block]]  # Only needs the coefs that correspond to selected cp's.
    selected_changepoints = []
    if cp_blocks == [[]]:
        return []
    for i, block in enumerate(cp_blocks):
        if len(block) == 1:
            # if only one cp in a block, leaves it
            selected_changepoints.append(block[0])
        else:
            # needs coefs to decide
            coef_start = sum([len(block) for block in cp_blocks[:i]])
            coef_block = coef[coef_start: coef_start + len(block)]
            if block[-1] - block[0] < min_index_distance:
                # If block max distance is less than `min_index_distance`,
                # only one cp can be selected,
                # select the one with the greatest absolute coef
                selected_changepoints.append(block[abs(coef_block).argmax()])
            else:
                # For longer blocks, the change points with the maximum
                # absolute coefficients are selected, while keeping the
                # distance beyond ``min_index_distance``.
                # This prevents dropping too many change points in a long block.
                block_cps = []
                start = block[0]
                last_coef = 0
                while start <= block[-1]:
                    # gets the current ``min_index_distance`` sized sub-block of cps and coefs
                    current_sub_block = [cp for cp in block if start <= cp < start + min_index_distance]
                    if not current_sub_block:
                        start += min_index_distance
                        continue
                    current_cp = current_sub_block[
                        abs(coef_block[[block.index(cp) for cp in current_sub_block]]).argmax()]
                    current_cp_idx = block.index(current_cp)
                    current_coef = coef_block[current_cp_idx]
                    # the first change point, doesn't have to look back
                    if start == block[0]:
                        block_cps.append(current_cp)
                    else:
                        # check the distance from the last change point
                        # if distance is enough, we can fit the new change point
                        last_cp = block_cps[-1]
                        if current_cp - last_cp >= min_index_distance:
                            block_cps.append(current_cp)
                        # if the distance is not enough and the new cp has a greater absolute coef,
                        # we have to remove the previous one with less absolute coef
                        # however, we would like to see if another cp fits in between
                        elif abs(current_coef) > abs(last_coef):
                            # checks if an extra change point can fit between last_last_cp and current_cp
                            if len(block_cps) >= 2:
                                last_last_cp = block_cps[-2]
                            else:
                                # if no last_last_cp, there's not lower bound
                                last_last_cp = -min_index_distance
                            # gets the cps that can fit between last_last_cp and current_cp
                            back_fill_potential_cp = [
                                cp for cp in block if
                                last_last_cp + min_index_distance <= cp <= current_cp - min_index_distance]
                            if back_fill_potential_cp:
                                # if there is cp that can fit in between, gets the one with max absolute coef
                                potential_coef = coef[[block.index(cp) for cp in back_fill_potential_cp]]
                                back_fill_cp = back_fill_potential_cp[abs(potential_coef).argmax()]
                                # append the cp in between
                                block_cps.append(back_fill_cp)
                            # removes the last one and append the new one
                            block_cps.remove(last_cp)
                            block_cps.append(current_cp)
                    # updates last_coef for comparison
                    last_coef = current_coef
                    # search ahead
                    start += min_index_distance
                selected_changepoints += block_cps
    return selected_changepoints


def get_changes_from_beta(
        beta,
        seasonality_components_df,
        magnitude_only=False):
    """Gets the seasonality magnitude change arrays for each seasonality component from the
    estimated regression coefficients.

    In total there are::

        {sum_{i=1}^{number of components}(2 * order of component i)} * {number of changepoints + 1}

    coefficients in ``beta``, where the + 1 counts the overall seasonality. These coefficients
    indicate the change in magnitude for each term in each component at each change point.
    We can't work on these directly, since the values are positive or negative.

    Two options are available to calcumated the change metric, controlled by the parameter
    ``magnitude_only``.

    If ``magnitude_only`` == True, the cumulative sum is taken for every single cos or sin term
    (from all components) along the change points. The results are the magnitudes for each cos
    or sin term (from all components) at each change point. An L2 norm is taken over the cos and sin
    terms' magnitudes within the same component at the same change point. Now for each component
    at every change point, we have one L2 norm value that represents the magnitude (no longer
    cos or sin level). Then the magnitude changes for the same component between consecutive
    change points are computed. This option captures total magnitude changes only.

    If ``magnitude_only`` == False, an L2 norm is taken over the cos or sin magnitude changes
    in the same component directly. This option captures shape changes as well.

    A dictionary is returned with keys equal to the component names, and values are the corresponding
    arrays of changes.

    Parameters
    ----------
    beta : `np.array`
        The estimated regression coefficients.
    seasonality_components_df : `pandas.DataFrame`
        The df to generate seasonality design matrix, which is compatible with
        ``seasonality_components_df`` in
        `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`
    magnitude_only : `bool`, default False
        Set to True to compute the L2 norms on the seasonality magnitudes, and set to False to compute the
        L2 norms on the seasonailty changes.

    Returns
    -------
    seasonality_magnitude : `dict`
        Keys are seasonality components, and values are the arrays of changes at each changepoint for the
        components.

    Examples
    --------
    # 1 change point, 2 seasonality components: "weekly" has order 1, and "yearly" has order 2.
    # so (1 + 2) * 2 * (1 + 1) = 12 coefficients in total.
    # the terms above are
    # [(weekly_order + yearly_order) * 2_terms_sin_cos] * (overall_seasonality + num_changepoints)
    >>> # the initial coefficients are [1, 1, 1, 1, 1, 1], and changes to [2, 0, 2, 2, 2, 0] at the change point
    >>> # the change is [1, -1, 1, 1, 1, -1].
    >>> beta = np.array([1, 1,     1, 1, 1, 1,          # initial coefficients
    ...                  1, -1,    1, 1, 1, -1])        # changes to [2, 0,     2, 2, 2, 0]
    >>> seasonality_components_df = pd.DataFrame({
    ...     "name": ["tow", "conti_year"],
    ...     "period": [7.0, 1.0],
    ...     "order": [1, 2],
    ...     "seas_names": ["weekly", "yearly"]})
    >>> result = get_changes_from_beta(beta, seasonality_components_df, True)
    >>> # when ``magnitude_only`` is set to True, the norm is calculated on the magnitude coefficients
    >>> # we have norms = [[sqrt(2), 2], [2, 2 * sqrt(3)]]
    >>> # hence the changes at the change point are [2 - sqrt(2), 2 * sqrt(3) - 2]
    >>> # the first coefficients is always prepended.
    >>> for key, value in result.items():
    >>>     print(key, value)
    weekly [1.41421356 0.58578644]
    yearly [2.         1.46410162]

    >>> # when ``magnitude_only`` is set to False, the norm is calculated directly on the change coefficients
    >>> # we have changes [sqrt(2), 2] at the change point
    >>> result = get_changes_from_beta(beta, seasonality_components_df, False)
    >>> for key, value in result.items():
    >>>     print(key, value)
    weekly [1.41421356 1.41421356]
    yearly [2. 2.]
    """
    # gets the number of terms in each component
    num_terms = (seasonality_components_df["order"] * 2).tolist()
    # gets the names of components
    components = seasonality_components_df["seas_names"].tolist()
    # reshapes the regression coefficients and computes the magnitude coefficients
    # the columns are the terms of different orders for each component
    # each row is one change point
    beta_mat = beta.reshape(-1, sum(num_terms))
    if magnitude_only:
        # cumsum gets the coefficient for the term between this changepoint and the next one.
        coef_mat = np.cumsum(beta_mat, axis=0)
    # computes the change magnitudes of each component at each time point, metric is l2 norm
    result = {}
    for i in range(len(num_terms)):
        start = np.sum(num_terms[:i]).astype(int)
        end = np.sum(num_terms[:(i + 1)]).astype(int)
        if magnitude_only:
            result[components[i]] = np.diff(np.linalg.norm(coef_mat[:, start: end], axis=1), prepend=0)
        else:
            result[components[i]] = np.linalg.norm(beta_mat[:, start: end], axis=1)
    return result


def get_seasonality_changes_from_adaptive_lasso(
        x,
        y,
        changepoint_dates,
        initial_coef,
        seasonality_components_df,
        min_index_distance=2,
        regularization_strength=0.6):
    """Parses the adaptive lasso estimator to get change point dates.

    The functions calls ``adaptive_lasso_cv`` to selected potential seasonality change points.
    Then a filter is applied to eliminate change points that are too close.
    Specifically, in a set of close change points, the one with the largest absolute
    coefficient will be kept.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    changepoint_dates : `pandas.Series`
        A pandas Series of all potential change point dates that were used to generate ``x``.
    initial_coef : `str` in ["ridge", "lasso", "old"] or `numpy.array'
        How to obtain the initial estimator. If a `str` is provided, the corresponding model is trained
        to obtain the initial estimator. If a `numpy.array` is provided, it is used as the initial
        estimator.
    seasonality_components_df : `pandas.DataFrame`
        The df to generate seasonality design matrix, which is compatible with
        ``seasonality_components_df`` in
        `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`
    min_index_distance : `int`
        The minimal index distance that is allowed between two change points.
    regularization_strength : `float` in [0, 1]
        The regularization power for change points. Greater values imply fewer change points.
        0 indicates all change points, and 1 indicates no change point.

    Returns
    -------
    result : `dict`
        The detected seasonality change points result dictionary. Keys are the component names,
        and values are the corresponding detected change points.
    """
    intercept, beta = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef=initial_coef,
        regularization_strength=regularization_strength,
        # Typically, seasonality has fewer changepoints than trend. Lower ratio results in higher lambda.
        max_min_ratio=1e4)
    change_result = get_changes_from_beta(
        beta=beta,
        seasonality_components_df=seasonality_components_df)
    result = dict()
    for component, changes in change_result.items():
        nonzero_idx = [i for i, val in enumerate(changes) if val != 0]
        cp_blocks = find_neighbor_changepoints(
            cp_idx=nonzero_idx,
            min_index_distance=min_index_distance,
        )
        cp_idx = filter_changepoints(cp_blocks, changes, min_index_distance)
        result[component] = changepoint_dates.iloc[cp_idx].tolist()
    return result


def estimate_trend_with_detected_changepoints(
        df,
        time_col,
        value_col,
        changepoints,
        yearly_seasonality_order=8,
        estimator="ridge"):
    """Estimates the trend effect with detected change points.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The data df.
    time_col : `str`
        The column name of the time column in ``df``.
    value_col : `str`
        The column name of the value column in ``df``.
    changepoints : `list`
        A list of detected trend change points.
    yearly_seasonality_order : `int`
        The yearly seasonality order.
    estimator : `str`, default "ridge"
        "ols" or "ridge", the estimation model for trend estimation.

    Returns
    -------
    trend_estimate : `pandas.Series`
        The estimated trend.
    """
    trend_df = build_trend_feature_df_with_changes(
        df=df,
        time_col=time_col,
        changepoints_dict={
            "method": "custom",
            "dates": changepoints
        }
    )
    if yearly_seasonality_order > 0:
        long_seasonality_df = build_seasonality_feature_df_with_changes(
            df=df,
            time_col=time_col,
            fs_components_df=pd.DataFrame({
                "name": [TimeFeaturesEnum.conti_year.value],
                "period": [1.0],
                "order": [yearly_seasonality_order],
                "seas_names": ["yearly"]})
        )
        trend_df = pd.concat([trend_df, long_seasonality_df], axis=1)
    estimators = {
        "ridge": RidgeCV,
        "ols": LinearRegression
    }
    estimator = estimators.get(estimator)
    if estimator is None:
        raise ValueError("estimator can only be either 'ridge' or 'ols'.")
    non_na_index = ~df[value_col].isnull().values
    model = estimator().fit(trend_df.values[non_na_index], df[value_col].values[non_na_index])
    # can't simply do .predict, because need to exclude the seasonality terms.
    trend_estimate = np.matmul(
        trend_df.values[:, :(len(changepoints) + 1)], model.coef_[:(len(changepoints) + 1)]) \
        + model.intercept_
    trend_estimate = pd.Series(trend_estimate)
    trend_estimate.index = pd.to_datetime(df[time_col])
    return trend_estimate


def estimate_seasonality_with_detected_changepoints(
        df,
        time_col,
        value_col,
        seasonality_changepoints,
        seasonality_components_df,
        estimator="ols"):
    """Estimates the seasonality effect with detected change points.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The data df.
    time_col : `str`
        The column name of the time column in ``df``.
    value_col : `str`
        The column name of the value column in ``df``.
    seasonality_changepoints : `dict`
        The detected seasonality change points dictionary, output from
        `~greykite.algo.changepoint.adalasso.changepoints_utils.get_seasonality_changes_from_adaptive_lasso`
    seasonality_components_df : `pandas.DataFrame`
        The df to generate seasonality design matrix, which is compatible with
        ``seasonality_components_df`` in
        `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`
    estimator : `str`, default "ols"
        "ols" or "ridge", the estimation model for seasonality estimation.

    Returns
    -------
    seasonality_estimate : `pandas.Series`
        The estimated seasonality.
    """
    seasonality_df = build_seasonality_feature_df_from_detection_result(
        df=df,
        time_col=time_col,
        seasonality_changepoints=seasonality_changepoints,
        seasonality_components_df=seasonality_components_df
    )
    estimators = {
        "ridge": RidgeCV,
        "ols": LinearRegression
    }
    estimator = estimators.get(estimator)
    if estimator is None:
        raise ValueError("estimator can only be either 'ridge' or 'ols'.")
    non_na_index = ~df[value_col].isnull().values
    model = estimator().fit(seasonality_df.values[non_na_index], df[value_col].values[non_na_index])
    seasonality_estimate = model.predict(seasonality_df.values)
    seasonality_estimate = pd.Series(seasonality_estimate)
    seasonality_estimate.index = pd.to_datetime(df[time_col])
    return seasonality_estimate


def get_seasonality_changepoint_df_cols(
        df,
        time_col,
        seasonality_changepoints,
        seasonality_components_df,
        include_original_block=True,
        include_components=None):
    """Gets the seasonality change point feature df column names.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The dataframe used to build seasonality feature df.
    time_col : `str`
        The name of the time column in ``df``.
    seasonality_changepoints : `dict`
        The seasonality change point dictionary. The keys are seasonality components, and the values
        are the corresponding change points given in lists.
        For example

            "weekly": [Timestamp('2020-01-01 00:00:00'), Timestamp('2021-04-05 00:00:00')]
            "yearly": [Timestamp('2020-08-06 00:00:00')]

    seasonality_components_df : `pandas.DataFrame`
        The seasonality components dataframe that is compatible with
        `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_seasonality_changepoints`
        The values in "seas_names" must equal to the keys in ``seasonality_changepoints``.
    include_original_block : `bool`, default True
        Whether to include the original untruncated block of sin or cos columns for each component. If set to
        False, the original seasonality block for each component will be dropped.
    include_components : `list` [`str`] or None, default None
        The components to be included from the result. If None, all components will be included.

    Returns
    -------
    cols : `list`
        List of column names for seasonality change points df.
    """
    # only needs two rows to get the column names, reduces running time.
    df = df.iloc[:2, :]
    seasonality_cols = []
    for component, dates in seasonality_changepoints.items():
        if include_components is None or component in include_components:
            regular_seasonality_df = build_seasonality_feature_df_with_changes(
                df=df,
                time_col=time_col,
                fs_components_df=seasonality_components_df.loc[seasonality_components_df["seas_names"] == component, :]
            )
            temp_cols = list(regular_seasonality_df.columns) if include_original_block else []
            for date in dates:
                temp_cols += [f"{col}{date.strftime('_%Y_%m_%d_%H')}" for col in list(regular_seasonality_df.columns)]
            seasonality_cols += temp_cols
    return seasonality_cols


def get_trend_changepoint_dates_from_cols(trend_cols):
    """Gets the trend changepoint dates from trend changepoint column names.

    Parameters
    ----------
    trend_cols : `list[`str`]`
        List of trend changepoint column names. EX. "changepoint2_2018_01_05_00".

    Returns
    -------
    trend_changepoint_dates : `list[`timestamp`]`
        List of trend changepoint dates.
    """
    trend_changepoint_dates = []
    trend_cols = get_pattern_cols(trend_cols, f"^{CHANGEPOINT_COL_PREFIX}")
    if trend_cols:
        for col in trend_cols:
            date = col.split("_")[1:]
            date = date + ['00'] * (6 - len(date))  # 6 means Y m d H M S
            date = f'{"-".join(date[:3])} {":".join(date[3:])}'  # Y-m-d H:M:S
            trend_changepoint_dates.append(pd.to_datetime(date))
    return trend_changepoint_dates


def get_yearly_seasonality_changepoint_dates_from_freq(
        df,
        time_col,
        yearly_seasonality_change_freq,
        min_training_length="183D"):
    """Gets the yearly seasonality changepoint dates from change frequency.

    This is an internal function used for varying yearly seasonality effects in
    `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.find_trend_changepoints`
    For a given ``yearly_seasonality_change_freq``, for example "365D", it generates changepoint
    dates according to this frequency within the timeframe of ``df``. The length of data after the last
    changepoint date can not be less than ``least_training_length``.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The data df.
    time_col : `str`
        The column name for time column in ``df``.
    yearly_seasonality_change_freq : `DateOffset`, `Timedelta` or `str` or `None`
        How often to change the yearly seasonality model, i.e., how often to place yearly seasonality
        changepoints.
        Note that if you use `str` as input, the maximal supported unit is day, i.e.,
        you might use "200D" but not "12M" or "1Y".
    min_training_length : `DateOffset`, `Timedelta` or `str` or `None`, default "183D"
        The minimum length between the last changepoint date and the last date of ``df[time_col]``.
        Changepoints too close to the end will be omitted. Recommended at least half a year.
        Note that if you use `str` as input, the maximal supported unit is day, i.e.,
        you might use "200D" but not "12M" or "1Y".

    Returns
    -------
    yearly_seasonality_changepoint_dates : `list` [ `pandas._libs.tslibs.timestamps.Timestamp` ]
        A list of yearly seasonality changepoint dates.
    """
    if yearly_seasonality_change_freq is None:
        return []
    check_freq_unit_at_most_day(yearly_seasonality_change_freq, "yearly_seasonality_change_freq")
    yearly_seasonality_change_freq = to_offset(yearly_seasonality_change_freq)
    if yearly_seasonality_change_freq.delta < timedelta(days=365):
        warnings.warn("yearly_seasonality_change_freq is less than a year. It might be too short "
                      "to fit accurate yearly seasonality.")
    check_freq_unit_at_most_day(min_training_length, "least_training_length")
    if min_training_length is not None:
        min_training_length = to_offset(min_training_length)
    else:
        min_training_length = to_offset("0D")
    first_day_in_df = pd.to_datetime(df[time_col].iloc[0])
    last_day_in_df = pd.to_datetime(df[time_col].iloc[-1])
    yearly_seasonality_changepoint_dates = list(pd.date_range(
        start=first_day_in_df,
        end=last_day_in_df - min_training_length,
        freq=yearly_seasonality_change_freq))[1:]  # do not include the start date
    if len(yearly_seasonality_changepoint_dates) == 0:
        warnings.warn("No yearly seasonality changepoint added. Either data length is too short "
                      "or yearly_seasonality_change_freq is too long.")
    return yearly_seasonality_changepoint_dates


def combine_detected_and_custom_trend_changepoints(
        detected_changepoint_dates,
        custom_changepoint_dates,
        min_distance=None,
        keep_detected=False):
    """Adds custom trend changepoints to detected trend changepoints.

    Compares the distance between custom changepoints and detected changepoints,
    and drops a detected changepoint or a custom changepoint depending on ``keep_detected``
    if their distance is less than ``min_distance``.

    Parameters
    ----------
    detected_changepoint_dates : `list`
        A list of detected trend changepoints, parsable by `pandas.to_datetime`.
    custom_changepoint_dates : `list`
        A list of additional custom trend changepoints, parsable by `pandas.to_datetime`.
    min_distance : `DateOffset`, `Timedelta`, `str` or None, default None
        The minimum distance between detected changepoints and custom changepoints.
        If a detected changepoint and a custom changepoint have distance less than ``min_distance``,
        either the detected changepoint or the custom changepoint will be dropped according to ``keep_detected``.
        Does not compare the distance within detected changepoints or custom changepoints.
        Note: maximal unit is 'D', i.e., you may only use units no more than 'D' such as
        '10D', '5H', '100T', '200S'. The reason is that 'W', 'M' or higher has either
        cycles or indefinite number of days, thus is not parsable by pandas as timedelta.
        For example, see `pandas.tseries.frequencies.to_offset`.
    keep_detected : `bool`, default False
        When the distance of a detected changepoint and a custom changepoint is less than ``min_distance``,
        whether to keep the detected changepoint or the custom changepoint.

    Returns
    -------
    combined_changepoint_dates : `list`
        A list of combined changepoints in ascending order.
    """
    check_freq_unit_at_most_day(min_distance, "min_distance")
    detected_changepoint_dates = pd.to_datetime(detected_changepoint_dates)
    custom_changepoint_dates = pd.to_datetime(custom_changepoint_dates)
    if len(detected_changepoint_dates) == 0:
        return custom_changepoint_dates
    if len(custom_changepoint_dates) == 0:
        return detected_changepoint_dates
    if keep_detected:
        combined_changepoints = [cp for cp in detected_changepoint_dates]
        addition_changepoints = [cp for cp in custom_changepoint_dates]
    else:
        combined_changepoints = [cp for cp in custom_changepoint_dates]
        addition_changepoints = [cp for cp in detected_changepoint_dates]
    for addition_cp in addition_changepoints:
        if min_distance is not None:
            current_min_distance = min([abs(addition_cp - cp) for cp in combined_changepoints])
            if current_min_distance >= to_offset(min_distance):
                combined_changepoints.append(addition_cp)
        else:
            combined_changepoints.append(addition_cp)
    return sorted(combined_changepoints)
