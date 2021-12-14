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
# original author: Albert Chen, Sayan Patra
"""Plotting functions in plotly."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS

from greykite.common import constants as cst
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import update_dictionary
from greykite.common.viz.colors_utils import get_color_palette


def plot_multivariate(
        df,
        x_col,
        y_col_style_dict="plotly",
        default_color="rgba(0, 145, 202, 1.0)",
        xlabel=None,
        ylabel=cst.VALUE_COL,
        title=None,
        showlegend=True):
    """Plots one or more lines against the same x-axis values.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data frame with ``x_col`` and columns named by the keys in ``y_col_style_dict``.
    x_col: `str`
        Which column to plot on the x-axis.
    y_col_style_dict: `dict` [`str`, `dict` or None] or "plotly" or "auto" or "auto-fill", default "plotly"
        The column(s) to plot on the y-axis, and how to style them.

        If a dictionary:

            - key : `str`
                column name in ``df``
            - value : `dict` or None
                Optional styling options, passed as kwargs to `go.Scatter`.
                If None, uses the default: line labeled by the column name.
                See reference page for `plotly.graph_objects.Scatter` for options
                (e.g. color, mode, width/size, opacity).
                https://plotly.com/python/reference/#scatter.

        If a string, plots all columns in ``df`` besides ``x_col`` against ``x_col``:

            - "plotly": plot lines with default plotly styling
            - "auto": plot lines with color ``default_color``, sorted by value (ascending)
            - "auto-fill": plot lines with color ``default_color``, sorted by value (ascending), and fills between lines

    default_color: `str`, default "rgba(0, 145, 202, 1.0)" (blue)
        Default line color when ``y_col_style_dict`` is one of "auto", "auto-fill".
    xlabel : `str` or None, default None
        x-axis label. If None, default is ``x_col``.
    ylabel : `str` or None, default ``VALUE_COL``
        y-axis label
    title : `str` or None, default None
        Plot title. If None, default is based on axis labels.
    showlegend : `bool`, default True
        Whether to show the legend.

    Returns
    -------
    fig : `plotly.graph_objects.Figure`
        Interactive plotly graph of one or more columns
        in ``df`` against ``x_col``.

        See `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`
        return value for how to plot the figure and add customization.
    """

    if xlabel is None:
        xlabel = x_col
    if title is None and ylabel is not None:
        title = f"{ylabel} vs {xlabel}"

    auto_style = {"line": {"color": default_color}}
    if y_col_style_dict == "plotly":
        # Uses plotly default style
        y_col_style_dict = {col: None for col in df.columns if col != x_col}
    elif y_col_style_dict in ["auto", "auto-fill"]:
        # Columns ordered from low to high
        means = df.drop(columns=x_col).mean()
        column_order = list(means.sort_values().index)
        if y_col_style_dict == "auto":
            # Lines with color `default_color`
            y_col_style_dict = {col: auto_style for col in column_order}
        elif y_col_style_dict == "auto-fill":
            # Lines with color `default_color`, with fill between lines
            y_col_style_dict = {column_order[0]: auto_style}
            y_col_style_dict.update({
                col: {
                    "line": {"color": default_color},
                    "fill": "tonexty"
                } for col in column_order[1:]
            })

    data = []
    default_style = dict(mode="lines")
    for column, style_dict in y_col_style_dict.items():
        # By default, column name in ``df`` is used to label the line
        default_col_style = update_dictionary(default_style, overwrite_dict={"name": column})
        # User can overwrite any of the default values, or remove them by setting key value to None
        style_dict = update_dictionary(default_col_style, overwrite_dict=style_dict)
        line = go.Scatter(
            x=df[x_col],
            y=df[column],
            **style_dict)
        data.append(line)

    layout = go.Layout(
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        title=title,
        title_x=0.5,
        showlegend=showlegend,
        legend={'traceorder': 'reversed'}  # Matches the order of ``y_col_style_dict`` (bottom to top)
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def plot_multivariate_grouped(
        df,
        x_col,
        y_col_style_dict,
        grouping_x_col,
        grouping_x_col_values,
        grouping_y_col_style_dict,
        colors=DEFAULT_PLOTLY_COLORS,
        xlabel=None,
        ylabel=cst.VALUE_COL,
        title=None,
        showlegend=True):
    """Plots multiple lines against the same x-axis values. The lines can
    partially share the x-axis values.

    See parameter descriptions for a running example.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data frame with ``x_col`` and columns named by the keys in ``y_col_style_dict``,
        ``grouping_x_col``, ``grouping_y_col_style_dict``.

        For example::

            df = pd.DataFrame({
                time: [dt(2018, 1, 1),
                        dt(2018, 1, 2),
                        dt(2018, 1, 3)],
                "y1": [8.5, 2.0, 3.0],
                "y2": [1.4, 2.1, 3.4],
                "y3": [4.2, 3.1, 3.0],
                "y4": [0, 1, 2],
                "y5": [10, 9, 8],
                "group": [1, 2, 1],
            })
        This will be our running example.
    x_col: `str`
        Which column to plot on the x-axis.
        "time" in our example.
    y_col_style_dict: `dict` [`str`, `dict` or None]
        The column(s) to plot on the y-axis, and how to style them.
        These columns are plotted against the complete x-axis.

        - key : `str`
            column name in ``df``
        - value : `dict` or None
            Optional styling options, passed as kwargs to `go.Scatter`.
            If None, uses the default: line labeled by the column name.
            If line color is not given, it is added according to ``colors``.
            See reference page for `plotly.graph_objects.Scatter` for options
            (e.g. color, mode, width/size, opacity).
            https://plotly.com/python/reference/#scatter.

        For example::

            y_col_style_dict={
                "y1": {
                    "name": "y1_name",
                    "legendgroup": "one",
                    "mode": "markers",
                    "line": None  # Remove line params since we use mode="markers"
                },
                "y2": None,
            }

        The function will add a line color to "y1" and "y2" based on the ``colors`` parameter.
        It will also add a name to "y2", since none was given. The "name" of "y1" will be preserved.

        The output ``fig`` will have one line each for each of "y1" and "y2", each plot against
        the entire "time" column.
    grouping_x_col: `str`
        Which column to use to group columns in ``grouping_y_col_style_dict``.
        "group" in our example.
    grouping_x_col_values: `list` [`int`] or None
        Which values to use for grouping. If None, uses all the unique values in
        ``df`` [``grouping_x_col``].
        In our example, specifying ``grouping_x_col_values == [1, 2]`` would plot
        separate lines corresponding to ``group==1`` and ``group==2``.
    grouping_y_col_style_dict: `dict` [`str`, `dict` or None]
        The column(s) to plot on the y-axis, and how to style them.
        These columns are plotted against partial x-axis.
        For each ``grouping_x_col_values`` an element in this dictionary produces
        one line.

        - key : `str`
            column name in ``df``
        - value : `dict` or None
            Optional styling options, passed as kwargs to `go.Scatter`.
            If None, uses the default: line labeled by the ``grouping_x_col_values``,
            ``grouping_x_col`` and column name.
            If a name is given, it is augmented with the ``grouping_x_col_values``.
            If line color is not given, it is added according to ``colors``.
            All the lines sharing same ``grouping_x_col_values`` have the same color.
            See reference page for `plotly.graph_objects.Scatter` for options
            (e.g. color, mode, width/size, opacity).
            https://plotly.com/python/reference/#scatter.

        For example::

            grouping_y_col_style_dict={
                "y3": {
                    "line": {
                        "color": "blue"
                    }
                },
                "y4": {
                    "name": "y4_name",
                    "line": {
                        "width": 2,
                        "dash": "dot"
                    }
                },
                "y5": None,
            }

        The function will add a line color to "y4" and "y5" based on the ``colors`` parameter.
        The line color of "y3" will be "blue" as specified. We also preserve the given line
        properties of "y4".

    `   The function adds a name to "y3" and "y5", since none was given. The given "name" of "y4"
        will be augmented with ``grouping_x_col_values``.

        Each element of ``grouping_y_col_style_dict`` gets one line for each ``grouping_x_col_values``.
        In our example, there will be 2 lines corresponding to "y3", named "1_y3" and "2_y3".
        "1_y3" is plotted against "time = [dt(2018, 1, 1), dt(2018, 1, 3)]", corresponding to ``group==1``.
        "2_y3" is plotted against "time = [dt(2018, 1, 2)", corresponding to ``group==2``.
    colors: [`str`, `list` [`str`]], default ``DEFAULT_PLOTLY_COLORS``
        Which colors to use to build a color palette for plotting.
        This can be a list of RGB colors or a `str` from ``PLOTLY_SCALES``.
        Required number of colors equals sum of the length of ``y_col_style_dict``
        and length of ``grouping_x_col_values``.
        See `~greykite.common.viz.colors_utils.get_color_palette` for details.
    xlabel : `str` or None, default None
        x-axis label. If None, default is ``x_col``.
    ylabel : `str` or None, default ``VALUE_COL``
        y-axis label
    title : `str` or None, default None
        Plot title. If None, default is based on axis labels.
    showlegend : `bool`, default True
        Whether to show the legend.

    Returns
    -------
    fig : `plotly.graph_objects.Figure`
    Interactive plotly graph of one or more columns
    in ``df`` against ``x_col``.

    See `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`
    return value for how to plot the figure and add customization.
    """

    available_grouping_x_col_values = np.unique(df[grouping_x_col])
    if grouping_x_col_values is None:
        grouping_x_col_values = available_grouping_x_col_values
    else:
        missing_grouping_x_col_values = set(grouping_x_col_values) - set(available_grouping_x_col_values)
        if len(missing_grouping_x_col_values) > 0:
            raise ValueError(f"Following 'grouping_x_col_values' are missing in '{grouping_x_col}' column: "
                             f"{missing_grouping_x_col_values}")

    # Chooses the color palette
    n_color = len(y_col_style_dict) + len(grouping_x_col_values)
    color_palette = get_color_palette(num=n_color, colors=colors)

    # Updates colors for y_col_style_dict if it is not specified
    for color_num, (column, style_dict) in enumerate(y_col_style_dict.items()):
        if style_dict is None:
            style_dict = {}
        default_color = {"color": color_palette[color_num]}
        style_dict["line"] = update_dictionary(default_color, overwrite_dict=style_dict.get("line"))
        y_col_style_dict[column] = style_dict

    # Standardizes dataset for the next figure
    df_standardized = df.copy().drop_duplicates(subset=[x_col]).sort_values(by=x_col)

    # This figure plots the whole xaxis vs yaxis values
    fig = plot_multivariate(
        df=df_standardized,
        x_col=x_col,
        y_col_style_dict=y_col_style_dict,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        showlegend=showlegend)
    data = fig.data
    layout = fig.layout

    # These figures plot the sliced xaxis vs yaxis values
    for color_num, grouping_x_col_value in enumerate(grouping_x_col_values, len(y_col_style_dict)):
        default_color = {"color": color_palette[color_num]}

        sliced_y_col_style_dict = grouping_y_col_style_dict.copy()

        for column, style_dict in sliced_y_col_style_dict.items():
            # Updates colors if it is not specified
            if style_dict is None:
                style_dict = {}
            line_dict = update_dictionary(default_color, overwrite_dict=style_dict.get("line"))

            # Augments names with grouping_x_col_value
            name = style_dict.get("name")
            if name is None:
                updated_name = f"{grouping_x_col_value}_{grouping_x_col}_{column}"
            else:
                updated_name = f"{grouping_x_col_value}_{name}"

            overwrite_dict = {
                "name": updated_name,
                "line": line_dict
            }
            style_dict = update_dictionary(style_dict, overwrite_dict=overwrite_dict)
            sliced_y_col_style_dict[column] = style_dict

        df_sliced = df[df[grouping_x_col] == grouping_x_col_value]
        fig = plot_multivariate(
            df=df_sliced,
            x_col=x_col,
            y_col_style_dict=sliced_y_col_style_dict)
        data = data + fig.data

    fig = go.Figure(data=data, layout=layout)

    return fig


def plot_univariate(
        df,
        x_col,
        y_col,
        xlabel=None,
        ylabel=None,
        title=None,
        color="rgb(32, 149, 212)",  # light blue
        showlegend=True):
    """Simple plot of univariate timeseries.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data frame with ``x_col`` and ``y_col``
    x_col: `str`
        x-axis column name, usually the time column
    y_col: `str`
        y-axis column name, the value the plot
    xlabel : `str` or None, default None
        x-axis label
    ylabel : `str` or None, default None
        y-axis label
    title : `str` or None, default None
        Plot title. If None, default is based on axis labels.
    color : `str`, default "rgb(32, 149, 212)" (light blue)
        Line color
    showlegend : `bool`, default True
        Whether to show the legend

    Returns
    -------
    fig : `plotly.graph_objects.Figure`
        Interactive plotly graph of the value against time.

        See `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`
        return value for how to plot the figure and add customization.

    See Also
    --------
    `~greykite.common.viz.timeseries_plotting.plot_multivariate`
        Provides more styling options. Also consider using plotly's `go.Scatter` and `go.Layout` directly.
    """
    # sets default x and y-axis names based on column names
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col

    y_col_style_dict = {
        y_col: dict(
            name=y_col,
            mode="lines",
            line=dict(
                color=color
            ),
            opacity=0.8
        )
    }
    return plot_multivariate(
        df,
        x_col,
        y_col_style_dict,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        showlegend=showlegend,
    )


def plot_forecast_vs_actual(
        df,
        time_col=cst.TIME_COL,
        actual_col=cst.ACTUAL_COL,
        predicted_col=cst.PREDICTED_COL,
        predicted_lower_col=cst.PREDICTED_LOWER_COL,
        predicted_upper_col=cst.PREDICTED_UPPER_COL,
        xlabel=cst.TIME_COL,
        ylabel=cst.VALUE_COL,
        train_end_date=None,
        title=None,
        showlegend=True,
        actual_mode="lines+markers",
        actual_points_color="rgba(250, 43, 20, 0.7)",  # red
        actual_points_size=2.0,
        actual_color_opacity=1.0,
        forecast_curve_color="rgba(0, 90, 181, 0.7)",  # blue
        forecast_curve_dash="solid",
        ci_band_color="rgba(0, 90, 181, 0.15)",  # light blue
        ci_boundary_curve_color="rgba(0, 90, 181, 0.5)",  # light blue
        ci_boundary_curve_width=0.0,  # no line
        vertical_line_color="rgba(100, 100, 100, 0.9)",  # black color with opacity of 0.9
        vertical_line_width=1.0):
    """Plots forecast with prediction intervals, against actuals
    Adapted from plotly user guide:
    https://plot.ly/python/v3/continuous-error-bars/#basic-continuous-error-bars

    Parameters
    ----------
    df : `pandas.DataFrame`
        Timestamp, predicted, and actual values
    time_col : `str`, default `~greykite.common.constants.TIME_COL`
        Column in df with timestamp (x-axis)
    actual_col : `str`, default `~greykite.common.constants.ACTUAL_COL`
        Column in df with actual values
    predicted_col : `str`, default `~greykite.common.constants.PREDICTED_COL`
        Column in df with predicted values
    predicted_lower_col : `str` or None, default `~greykite.common.constants.PREDICTED_LOWER_COL`
        Column in df with predicted lower bound
    predicted_upper_col : `str` or None, default `~greykite.common.constants.PREDICTED_UPPER_COL`
        Column in df with predicted upper bound
    xlabel : `str`, default `~greykite.common.constants.TIME_COL`
        x-axis label.
    ylabel : `str`, default `~greykite.common.constants.VALUE_COL`
        y-axis label.
    train_end_date : `datetime.datetime` or None, default None
        Train end date.
        Must be a value in ``df[time_col]``.
    title : `str` or None, default None
        Plot title.
    showlegend : `bool`, default True
        Whether to show a plot legend.
    actual_mode : `str`, default "lines+markers"
        How to show the actuals.
        Options: ``markers``, ``lines``, ``lines+markers``
    actual_points_color : `str`, default "rgba(99, 114, 218, 1.0)"
        Color of actual line/marker.
    actual_points_size : `float`, default 2.0
        Size of actual markers.
        Only used if "markers" is in ``actual_mode``.
    actual_color_opacity : `float` or None, default 1.0
        Opacity of actual values points.
    forecast_curve_color : `str`, default "rgba(0, 145, 202, 1.0)"
        Color of forecasted values.
    forecast_curve_dash : `str`, default "solid"
        'dash' property of forecast ``scatter.line``.
        One of: ``['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']``
        or a string containing a dash length list in pixels or percentages
        (e.g. ``'5px 10px 2px 2px'``, ``'5, 10, 2, 2'``, ``'10% 20% 40%'``)
    ci_band_color : `str`, default "rgba(0, 145, 202, 0.15)"
        Fill color of the prediction bands.
    ci_boundary_curve_color : `str`, default "rgba(0, 145, 202, 0.15)"
        Color of the prediction upper/lower lines.
    ci_boundary_curve_width : `float`, default 0.0
        Width of the prediction upper/lower lines.
        default 0.0 (hidden)
    vertical_line_color : `str`, default "rgba(100, 100, 100, 0.9)"
        Color of the vertical line indicating train end date.
        Default is black with opacity of 0.9.
    vertical_line_width : `float`, default 1.0
        width of the vertical line indicating train end date

    Returns
    -------
    fig : `plotly.graph_objects.Figure`
        Plotly figure of forecast against actuals, with prediction
        intervals if available.

        Can show, convert to HTML, update::

            # show figure
            fig.show()

            # get HTML string, write to file
            fig.to_html(include_plotlyjs=False, full_html=True)
            fig.write_html("figure.html", include_plotlyjs=False, full_html=True)

            # customize layout (https://plot.ly/python/v3/user-guide/)
            update_layout = dict(
                yaxis=dict(title="new ylabel"),
                title_text="new title",
                title_x=0.5,
                title_font_size=30)
            fig.update_layout(update_layout)
    """
    if title is None:
        title = "Forecast vs Actual"
    if train_end_date is not None and not all(pd.Series(train_end_date).isin(df[time_col])):
        raise Exception(
            f"train_end_date {train_end_date} is not found in df['{time_col}']")

    fill_dict = {
        "mode": "lines",
        "fillcolor": ci_band_color,
        "fill": "tonexty"
    }
    data = []
    if predicted_lower_col is not None:
        lower_bound = go.Scatter(
            name="Lower Bound",
            x=df[time_col],
            y=df[predicted_lower_col],
            mode="lines",
            line=dict(
                width=ci_boundary_curve_width,
                color=ci_boundary_curve_color),
            legendgroup="interval"  # show/hide with the upper bound
        )
        data.append(lower_bound)

    # plotly fills between current and previous element in `data`.
    # Only fill if lower bound exists.
    forecast_fill_dict = fill_dict if predicted_lower_col is not None else {}
    if predicted_upper_col is not None:
        upper_bound = go.Scatter(
            name="Upper Bound",
            x=df[time_col],
            y=df[predicted_upper_col],
            line=dict(
                width=ci_boundary_curve_width,
                color=ci_boundary_curve_color),
            legendgroup="interval",  # show/hide with the lower bound
            **forecast_fill_dict)
        data.append(upper_bound)

    # If `predicted_lower_col` and `predicted_upper_col`, then the full range
    # has been filled in. If only one of them, then fill in between that line
    # and forecast.
    actual_params = {}
    if "lines" in actual_mode:
        actual_params.update(line=dict(color=actual_points_color))
    if "markers" in actual_mode:
        actual_params.update(marker=dict(color=actual_points_color, size=actual_points_size))
    actual = go.Scatter(
        name="Actual",
        x=df[time_col],
        y=df[actual_col],
        mode=actual_mode,
        opacity=actual_color_opacity,
        **actual_params
    )
    data.append(actual)

    forecast_fill_dict = fill_dict if (predicted_lower_col is None) != (predicted_upper_col is None) else {}
    forecast = go.Scatter(
        name="Forecast",
        x=df[time_col],
        y=df[predicted_col],
        line=dict(
            color=forecast_curve_color,
            dash=forecast_curve_dash),
        **forecast_fill_dict)
    data.append(forecast)

    layout = go.Layout(
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        title=title,
        title_x=0.5,
        showlegend=showlegend,
        # legend order from top to bottom: Actual, Forecast, Upper Bound, Lower Bound
        legend={'traceorder': 'reversed'}
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update()

    # adds a vertical line to separate training and testing phases
    if train_end_date is not None:
        new_layout = dict(
            # add vertical line
            shapes=[dict(
                type="line",
                xref="x",
                yref="paper",  # y-reference is assigned to the plot paper [0,1]
                x0=train_end_date,
                y0=0,
                x1=train_end_date,
                y1=1,
                line=dict(
                    color=vertical_line_color,
                    width=vertical_line_width)
            )],
            # add text annotation
            annotations=[dict(
                xref="x",
                x=train_end_date,
                yref="paper",
                y=.97,
                text="Train End Date",
                showarrow=True,
                arrowhead=0,
                ax=-60,
                ay=0
            )]
        )
        fig.update_layout(new_layout)
    return fig


def split_range_into_groups(
        n,
        group_size,
        which_group_complete="last"):
    """Partitions `n` elements into adjacent groups,
        each with `group_size` elements
        Group number starts from 0 and increments upward
        Can be used to generate groups for sliding window aggregation.
    :param n: int
        number of elemnts to split into groups
    :param group_size: int
        number of elements per group
    :param which_group_complete: str
        If n % group_size > 0, one group will have fewer than `group_size` elements
        if "first", the first group is full if possible, and last group may be incomplete
        if "last", (default) the last group is full if possible,
        and first group may be incomplete
    :return: np.array of length n
        values correspond to the element's group number

    Examples:
    >>> split_range_into_groups(10, 1, "last")
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> split_range_into_groups(10, 2, "last")
    array([0., 0., 1., 1., 2., 2., 3., 3., 4., 4.])
    >>> split_range_into_groups(10, 3, "last")
    array([0., 1., 1., 1., 2., 2., 2., 3., 3., 3.])
    >>> split_range_into_groups(10, 4, "last")
    array([0., 0., 1., 1., 1., 1., 2., 2., 2., 2.])
    >>> split_range_into_groups(10, 4, "first")
    array([0., 0., 0., 0., 1., 1., 1., 1., 2., 2.])
    >>> split_range_into_groups(10, 5, "last")
    array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
    >>> split_range_into_groups(10, 6, "last")
    array([0., 0., 0., 0., 1., 1., 1., 1., 1., 1.])
    >>> split_range_into_groups(10, 10, "last")
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> split_range_into_groups(10, 12, "last")
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    if which_group_complete.lower() == "first":
        offset = 0
    else:
        offset = group_size - n % group_size
        offset = offset % group_size  # sets offset to 0 if n % group_size == 0
    return np.floor(np.arange(offset, n + offset) / group_size)


def add_groupby_column(
        df,
        time_col,
        groupby_time_feature=None,
        groupby_sliding_window_size=None,
        groupby_custom_column=None):
    """Extracts a column to group by from ``df``.

    Exactly one of ``groupby_time_feature``, ``groupby_sliding_window_size``,
    `groupby_custom_column` must be provided.

    Parameters
    ----------
    df : 'pandas.DataFrame`
        Contains the univariate time series / forecast
    time_col : `str`
        The name of the time column of the univariate time series / forecast
    groupby_time_feature : `str` or None, optional
        If provided, groups by a column generated by
        `~greykite.common.features.timeseries_features.build_time_features_df`.
        See that function for valid values.
    groupby_sliding_window_size : `int` or None, optional
        If provided, sequentially partitions data into groups of size
        ``groupby_sliding_window_size``.
    groupby_custom_column : `pandas.Series` or None, optional
        If provided, groups by this column value.
        Should be same length as the ``df``.

    Returns
    -------
    result : `dict`
        Dictionary with two items:

        * ``"df"`` : `pandas.DataFrame`
            ``df`` with a grouping column added.
            The column can be used to group rows together.

        * ``"groupby_col"`` : `str`
            The name of the groupby column added to ``df``.
            The column name depends on the grouping method:

                - ``groupby_time_feature`` for ``groupby_time_feature``
                - ``{cst.TIME_COL}_downsample`` for ``groupby_sliding_window_size``
                - ``groupby_custom_column.name`` for ``groupby_custom_column``.
    """
    # Resets index to support indexing in groupby_sliding_window_size
    df = df.copy()
    dt = pd.Series(df[time_col].values)
    # Determines the groups
    is_groupby_time_feature = 1 if groupby_time_feature is not None else 0
    is_groupby_sliding_window_size = 1 if groupby_sliding_window_size is not None else 0
    is_groupby_custom_column = 1 if groupby_custom_column is not None else 0
    if is_groupby_time_feature + is_groupby_sliding_window_size + is_groupby_custom_column != 1:
        raise ValueError(
            "Exactly one of (groupby_time_feature, groupby_rolling_window_size, groupby_custom_column)"
            "must be specified")
    groups = None
    if is_groupby_time_feature == 1:
        # Group by a value derived from the time column
        time_features = build_time_features_df(dt, conti_year_origin=min(dt).year)
        groups = time_features[groupby_time_feature]
        groups.name = groupby_time_feature
    elif is_groupby_sliding_window_size == 1:
        # Group by sliding window for evaluation over time
        index_dates = split_range_into_groups(
            n=df.shape[0],
            group_size=groupby_sliding_window_size,
            which_group_complete="last")  # ensures the last group is complete (first group may be partial)
        groups = dt[index_dates * groupby_sliding_window_size]  # uses first date in each group as grouping value
        groups.name = f"{time_col}_downsample"
    elif is_groupby_custom_column == 1:
        # Group by custom column
        groups = groupby_custom_column

    groups_col_name = groups.name if groups.name is not None else "groups"
    df[groups_col_name] = groups.values
    if df.index.name in df.columns:
        # Removes ambiguity in case the index name is the same as the newly added column,
        # (or an existing column).
        df.index.name = None
    return {
        "df": df,
        "groupby_col": groups_col_name
    }


def grouping_evaluation(
        df,
        groupby_col,
        grouping_func,
        grouping_func_name):
    """Groups ``df`` and evaluates a function on each group.
    The function takes a `pandas.DataFrame` and returns a scalar.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Input data. For example, univariate time series, or forecast result.
        Contains ``groupby_col`` and columns to apply ``grouping_func`` on.
    groupby_col : `str`
        Column name in ``df`` to group by.
    grouping_func : `callable`
        Function that is applied to each group via `pandas.groupBy.apply`.
        Signature (grp: `pandas.DataFrame`) -> aggregated value: `float`.
    grouping_func_name : `str`
        What to call the output column generated by ``grouping_func``.

    Returns
    -------
    grouped_df : `pandas.DataFrame`
        Dataframe with ``grouping_func`` evaluated on each level of ``df[groupby_col]``.
        Contains two columns:

            - ``groupby_col``: The groupby value
            - ``grouping_func_name``: The output of ``grouping_func`` on the group
    """
    grouped_df = (df
                  .groupby(groupby_col)
                  .apply(grouping_func)
                  .reset_index()
                  .rename({0: grouping_func_name}, axis=1))

    return grouped_df


def flexible_grouping_evaluation(
        df,
        map_func_dict=None,
        groupby_col=None,
        agg_kwargs=None,
        extend_col_names=True,
        unpack_list=True,
        list_names_dict=None):
    """Flexible aggregation. Generates additional columns for evaluation via
    ``map_func_dict``, groups by ``groupby_col``, then aggregates according
    to ``agg_kwargs``.

    This function calls `pandas.DataFrame.apply` and
    `pandas.core.groupby.DataFrameGroupBy.agg` internally.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame to transform / aggregate
    map_func_dict : `dict` [`str`, `callable`] or None, default None
        Row-wise transformation functions to create new columns.
        If None, no new columns are added.

        key: new column name
        value: row-wise function to apply to ``df`` to generate the column value.
               Signature (row: `pandas.DataFrame`) -> transformed value: `float`.

        For example::

            map_func_dict = {
                "residual": lambda row: row["predicted"] - row["actual"],
                "squared_error": lambda row: (row["predicted"] - row["actual"])**2
            }

    groupby_col : `str` or None, default None
        Which column to group by.
        Can be in ``df`` or generated by ``map_func_dict``.
        If None, no grouping or aggregation is done.
    agg_kwargs : `dict` or None, default None
        Passed as keyword args to `pandas.core.groupby.DataFrameGroupBy.aggregate` after creating
        new columns and grouping by ``groupby_col``. Must be provided if ``groupby_col is not None``.
        To fully customize output column names, pass a dictionary as shown below.

        For example::

            # Example 1, named aggregation to explicitly name output columns.
            # Assume ``df`` contains ``abs_percent_err``, ``abs_err`` columns.
            # Output columns are "MedAPE", "MAPE", "MAE", etc. in a single level index.
            from functools import partial
            agg_kwargs = {
                # output column name: (column to aggregate, aggregation function)
                "MedAPE": pd.NamedAgg(column="abs_percent_err", aggfunc=np.nanmedian),
                "MAPE": pd.NamedAgg(column="abs_percent_err", aggfunc=np.nanmean),
                "MAE": pd.NamedAgg(column="abs_err", aggfunc=np.nanmean),
                "q95_abs_err": pd.NamedAgg(column="abs_err", aggfunc=partial(np.nanquantile, q=0.95)),
                "q05_abs_err": pd.NamedAgg(column="abs_err", aggfunc=partial(np.nanquantile, q=0.05)),
            }

            # Example 2, multi-level aggregation using `func` parameter
            # to `pandas.core.groupby.DataFrameGroupBy.aggregate`.
            # Assume ``df`` contains ``y1``, ``y2`` columns.
            agg_kwargs = {
                "func": {
                    "y1": [np.nanmedian, np.nanmean],
                    "y2": [np.nanmedian, np.nanmax],
                }
            }
            # `extend_col_names` controls the output column names
            extend_col_names = True  # output columns are "y1_nanmean", "y1_nanmedian", "y2_nanmean", "y2_nanmax"
            extend_col_names = False  # output columns are "nanmean", "nanmedian", "nanmean", "nanmax"

    extend_col_names : `bool` or None, default True
        How to flatten index after aggregation.
        In some cases, the column index after aggregation is a multi-index.
        This parameter controls how to flatten an index with 2 levels to 1 level.

            - If None, the index is not flattened.
            - If True, column name is a composite: ``{index0}_{index1}``
              Use this option if index1 is not unique.
            - If False, column name is simply ``{index1}``

        Ignored if the ColumnIndex after aggregation has only one level (e.g.
        if named aggregation is used in ``agg_kwargs``).

    unpack_list : `bool`, default True
        Whether to unpack (flatten) columns that contain list/tuple after aggregation,
        to create one column per element of the list/tuple.
        If True, ``list_names_dict`` can be used to rename the unpacked columns.

    list_names_dict : `dict` [`str`, `list` [`str`]] or None, default None
        If ``unpack_list`` is True, this dictionary can optionally be
        used to rename the unpacked columns.

            - Key = column name after aggregation, before upacking.
              E.g. ``{index0}_{index1}`` or ``{index1}`` depending on ``extend_col_names``.
            - Value = list of names to use for the unpacked columns. Length must match
              the length of the lists contained in the column.

        If a particular list/tuple column is not found in this dictionary, appends
        0, 1, 2, ..., n-1 to the original column name, where n = list length.

        For example, if the column contains a tuple of length 4 corresponding to
        quantiles 0.1, 0.25, 0.75, 0.9, then the following would be appropriate::

            aggfunc = lambda grp: partial(np.nanquantile, q=[0.1, 0.25, 0.75, 0.9])(grp).tolist()
            agg_kwargs = {
                "value_Q": pd.NamedAgg(column="value", aggfunc=aggfunc)
            }
            list_names_dict = {
                # the key is the name of the unpacked column
                "value_Q" : ["Q0.10", "Q0.25", "Q0.75", "Q0.90"]
            }
            # Output columns are "Q0.10", "Q0.25", "Q0.75", "Q0.90"

            # In this example, if list_names_dict=None, the default output column names
            # would be: "value_Q0", "value_Q1", "value_Q2", "value_Q3"

    Returns
    -------
    df_transformed : `pandas.DataFrame`
        df after transformation and optional aggregation.

        If ``groupby_col`` is None, returns ``df`` with additional columns as the keys in ``map_func_dict``.
        Otherwise, ``df`` is grouped by ``groupby_col`` and this becomes the index. Columns
        are determined by ``agg_kwargs`` and ``extend_col_names``.
    """
    if groupby_col and not agg_kwargs:
        raise ValueError("Must specify `agg_kwargs` if grouping is requested via `groupby_col`.")
    if agg_kwargs and not groupby_col:
        log_message(f"`agg_kwargs` is ignored because `groupby_col` is None. "
                    f"Specify `groupby_col` to allow aggregation.", LoggingLevelEnum.WARNING)

    df = df.copy()
    if map_func_dict is not None:
        for col_name, func in map_func_dict.items():
            df[col_name] = df.apply(func, axis=1)

    if groupby_col is not None:
        groups = df.groupby(groupby_col)
        with warnings.catch_warnings():
            # Ignores pandas FutureWarning. Use NamedAgg in pandas 0.25.+
            warnings.filterwarnings(
                "ignore",
                message="using a dict with renaming is deprecated",
                category=FutureWarning)
            df_transformed = groups.agg(**agg_kwargs)
        if extend_col_names is not None and df_transformed.columns.nlevels > 1:
            # Flattens multi-level column index
            if extend_col_names:
                # By concatenating names
                df_transformed.columns = ["_".join(col).strip("_") for col in df_transformed.columns]
            else:
                # By using level 1 names
                df_transformed.columns = list(df_transformed.columns.get_level_values(1))
                if np.any(df_transformed.columns.duplicated()):
                    warnings.warn("Column names are not unique. Use `extend_col_names=True` "
                                  "to uniquely identify every column.")
    else:
        # No grouping is requested
        df_transformed = df

    if unpack_list and df_transformed.shape[0] > 0:
        # Identifies the columns that contain list elements
        which_list_cols = df_transformed.iloc[0].apply(lambda x: isinstance(x, (list, tuple)))
        list_cols = list(which_list_cols[which_list_cols].index)
        for col in list_cols:
            if isinstance(df_transformed[col], pd.DataFrame):
                warnings.warn(f"Skipping list unpacking for `{col}`. There are multiple columns "
                              f"with this name. Make sure column names are unique to enable unpacking.")
                continue
            # Unpacks the column, creating one column for each list entry
            list_df = pd.DataFrame(df_transformed[col].to_list())
            n_cols = list_df.shape[1]
            # Adds column names
            if list_names_dict is not None and col in list_names_dict:
                found_length = len(list_names_dict[col])
                if found_length != n_cols:
                    raise ValueError(
                        f"list_names_dict['{col}'] has length {found_length}, "
                        f"but there are {n_cols} columns to name. Example row(s):\n"
                        f"{list_df.head(2)}")
                list_df.columns = [f"{list_names_dict.get(col)[i]}" for i in range(n_cols)]
            else:
                list_df.columns = [f"{col}{i}" for i in range(n_cols)]
            # replaces original column with new ones
            list_df.index = df_transformed.index
            del df_transformed[col]
            df_transformed = pd.concat([df_transformed, list_df], axis=1)

        if list_names_dict:
            unused_names = sorted(list(set(list_names_dict.keys()) - set(list_cols)))
            if len(unused_names) > 0:
                warnings.warn("These names from `list_names_dict` are not used, because the "
                              "column (key) is not found in the dataframe after aggregation:\n"
                              f"{unused_names}.\nAvailable columns are:\n"
                              f"{list_cols}.")

    return df_transformed
