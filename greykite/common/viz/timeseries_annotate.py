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
# original author: Reza Hosseini, Kaixu Yang
"""Plotting functions to add annotations to timseries"""
import pandas as pd
import plotly.graph_objects as go

from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import PREDICTED_ANOMALY_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.features.adjust_anomalous_data import label_anomalies_multi_metric
from greykite.common.viz.colors_utils import get_distinct_colors
from greykite.common.viz.timeseries_plotting import plot_forecast_vs_actual


def plt_annotate_series(
        df,
        x_col,
        value_col,
        label_col,
        annotate_labels=None,
        keep_cols=None,
        fill_cols=None,
        title=None):
    """A function which plots a values given in ``value_col`` of a dataframe
    ``df`` against x-axis values given in ``x_col`` and adds annotations based
    on labels in ``label_col``.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data frame with ``x_col``, ``value_col`` and ``label_col``.
        If ``keep_cols`` is not None ``df`` is supposed to have the columns
        in ``keep_cols`` as well.
    x_col: `str`
        Which column to plot on the x-axis.
    value_col : `str`
        The column with the values for the series.
    label_col : `str`
        The column which includes the labels for which we want to annotate the
        series given in ``value_col``.
    annotate_labels : `List` [`str`] or None, default None
        If not None, the annotations will only be added for the labels given in
        this list.
    keep_cols: `List` [`str`] or None, default None
        Extra columns to be plotted if not None.
    fill_cols : `List` [ `List` [`str`] ]
        Fill between each list of columns in ``fill_cols``.
    title : `str` or None, default None
        Plot title. If None, default is based on axis labels.

    Returns
    -------
    result : `dict` with following items:

        - "fig" : `plotly.graph_objects.Figure`
            Interactive plotly graph of ``value_col`` and potentially columns
            in specified in ``keep_cols`` (if not None)
            in ``df`` against ``x_col`` with added annotations.

        - "df": `pandas.DataFrame`
            A dataframe which is an expansion of ``df`` and includes extra columns
            with new value columns for each label in ``annotate_labels`` or all labels
            if ``annotate_labels`` is None. The new column for each label is NA if
            that label is not activated and equals ``value_col`` otherwise.
    """
    df = df.copy()
    if keep_cols is None:
        keep_cols = []

    y_col_style_dict = {}
    # The columns to be filled. They will be handled separately from ``keep_cols``.
    all_fill_cols = []
    if fill_cols is not None and len(fill_cols) > 0:
        if not isinstance(fill_cols[0], list):
            raise ValueError(f"fill_cols must be a list of lists of strings.")
        all_fill_cols = [c for c_list in fill_cols for c in c_list]
        # Removes a col in keep_cols if it is already in fill_cols.
        keep_cols = [col for col in keep_cols if col not in all_fill_cols]

    df = df[[x_col, value_col, label_col] + keep_cols + all_fill_cols]

    if annotate_labels is None:
        annotate_labels = df[label_col].unique()

    for col in [value_col] + keep_cols:
        y_col_style_dict[col] = {
            "mode": "lines"
        }

    for label in annotate_labels:
        new_col = f"{value_col}_label_{label}"
        df[new_col] = df[value_col]
        df.loc[~(df[label_col] == label), new_col] = None
        y_col_style_dict[new_col] = {
            "mode": "markers"}

    y_col_style_dict[value_col] = {
        "mode": "lines",
        "line": {
            "color": "rgba(0, 0, 255, 0.3)",
            "dash": "solid"}}

    del df[label_col]

    # Instead of calling plot_multivariate, we make the figure manually to avoid the column order to be
    # flipped due to the fact that dictionary is unordered. This may affect the fill between certain columns.
    data = []
    # Plots fill_cols first to make them in the bottom layer.
    if fill_cols is not None:
        first_fill_color = "rgba(142, 215, 246, 0.3)"  # blue

        for g, col_list in enumerate(fill_cols):
            for i, col in enumerate(col_list):
                line = go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    name=col,
                    legendgroup=f"{g}",
                    mode="lines",
                    line=dict(color=(first_fill_color if g == 0 else None)),
                    fill=("tonexty" if i != 0 else None),
                    fillcolor=(first_fill_color if g == 0 else None)
                )
                data.append(line)

    for column, style_dict in y_col_style_dict.items():
        style_dict = dict(
            name=column,
            **style_dict
        )
        line = go.Scatter(
            x=df[x_col],
            y=df[column],
            **style_dict)
        data.append(line)

    layout = go.Layout(
        xaxis=dict(title=x_col),
        yaxis=dict(title=value_col),
        title=title,
        title_x=0.5,
        showlegend=True,
    )
    fig = go.Figure(data=data, layout=layout)

    return {
        "fig": fig,
        "annotated_df": df}


def plt_compare_series_annotations(
        df,
        x_col,
        actual_col,
        actual_label_col,
        forecast_label_col,
        forecast_col=None,
        keep_cols=None,
        fill_cols=None,
        standardize_col=None,
        title=None,
        x_label=None,
        y_label=None):
    """A function which plots ``actual_col`` and ``forecast_col`` and well as
        ``forecast_lower_col``, ``forecast_upper_col`` with respect to x axis
        (often time) given in ``x_col``. Then it annotates:

    - the ``actual_col`` with markers when ``actual_label_col`` is non-zero.
    - the ``forecast_col`` with markers when ``forecast_label_col`` is non-zero.

    This will help user to identify where the two annotations are consistent.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data frame with all columns used in this function (see arguments passed
            ending with "_col").
    x_col: `str`
        Which column to plot on the x-axis.
    actual_col: `str`
        The column in the dataframe which includes the actual values (ground truth).
    actual_label_col: `str`
        The column which has labels including 0, and will be used to annotate
        the values in ``actual_col`` for non-zero values. The non-zero labels
        are often considerd anomalies.
    forecast_label_col: `str`
        The column which has labels including 0, and will be used to annotate
        the values in ``forecast_col`` for non-zero values. The non-zero labels
        are often considerd anomalies.
    forecast_col : `str` or None, default None
        The column in the dataframe which includes the forecast values
    keep_cols: `List` [`str`] or None, default None
        Extra columns to be plotted. The color for these extra columns will be
        opaque grey so they do not be the center of attention in the plot.
        Often these columns will be model upper and lower bound predictions.
    fill_cols: `list` [`list` [`str`] ] or None, default None
        Fill between each list of columns in ``fill_cols``.
    standardize_col : `str` or `None`
        In case we like to standardize all values with respect to a column, that
        column name would be passed here. Often ``actual_col`` will be a good
        candidate to be used here.
    title : `str` or None, default None
        Plot title. If None, default is based on axis labels.
    x_label : `str` or None, default None
        x axis label, if None ``x_col`` will be used.
    y_label : `str` or None, default None
        y axis label, if None, no label will be used for y-axis. Note that the
        legend will include the label for various curves / markers.


    Returns
    -------
    fig : `plotly.graph_objects.Figure`
        Interactive plotly graph.
    """

    df = df.copy()
    df[f"model_anomaly"] = df[actual_col]
    df.loc[df[forecast_label_col].map(float) == 0, f"model_anomaly"] = None
    df[f"true_anomaly"] = df[actual_col]
    df.loc[df[actual_label_col].map(float) == 0, f"true_anomaly"] = None

    cols = [
        actual_col,
        f"model_anomaly",
        f"true_anomaly"]

    all_fill_cols = []
    # The columns in fill_cols will be handled separately.
    if fill_cols is not None and len(fill_cols) > 0:
        if not isinstance(fill_cols[0], list):
            raise ValueError(f"fill_cols must be a list of lists of strings.")
        all_fill_cols = [c for c_list in fill_cols for c in c_list]
        if keep_cols is not None:
            # Removes a col in keep_cols if it is already in fill_cols.
            keep_cols = [col for col in keep_cols if col not in all_fill_cols]

    if keep_cols is not None:
        cols += keep_cols

    if forecast_col is not None:
        cols += [forecast_col]

    cols += all_fill_cols

    if standardize_col:
        x = df[standardize_col]
        for col in cols:
            df[col] = df[col] / x

    y_col_style_dict = {
        actual_col: {
            "mode": "lines",
            "line": {
                "color": "rgba(0, 255, 0, 0.3)",
                "dash": "solid"
            }
        },
        "true_anomaly": {
            "mode": "markers",
            "marker": dict(
                color="rgba(255, 0, 0, 0.5)",
                symbol="square")
        },
        "model_anomaly": {
            "mode": "markers",
            "marker": dict(
                color="rgba(0, 0, 255, 0.6)",
                symbol="star")
        }
    }

    if forecast_col is not None:
        y_col_style_dict[forecast_col] = {
            "mode": "lines",
            "line": {
                "color": "rgba(0, 0, 255, 0.4)",
                "dash": "solid"
            }
        }

    if keep_cols is not None:
        for i in range(len(keep_cols)):
            col = keep_cols[i]
            # this is to be able to distinguish between the ``keep_cols`` using opacity
            opacity = (i + 1) / (len(keep_cols) + 1)
            y_col_style_dict[col] = {
                "mode": "lines",
                "line": {
                    "color": f"rgba(191, 191, 191, {opacity})",
                    "dash": "solid"}}

    # Instead of calling plot_multivariate, we make the figure manually to avoid the column order to be
    # flipped due to the fact that dictionary is unordered. This may affect the fill between certain columns.
    data = []
    # Plots fill_cols first to make them in the bottom layer.
    if fill_cols is not None:
        first_fill_color = "rgba(142, 215, 246, 0.3)"  # blue

        for g, col_list in enumerate(fill_cols):
            for i, col in enumerate(col_list):
                line = go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    name=col,
                    legendgroup=f"{g}",
                    mode="lines",
                    line=dict(color=(first_fill_color if g == 0 else None)),
                    fill=("tonexty" if i != 0 else None),
                    fillcolor=(first_fill_color if g == 0 else None)
                )
                data.append(line)

    for column, style_dict in y_col_style_dict.items():
        style_dict = dict(
            name=column,
            **style_dict
        )
        line = go.Scatter(
            x=df[x_col],
            y=df[column],
            **style_dict)
        data.append(line)

    layout = go.Layout(
        xaxis=dict(title=(x_label if x_label is not None else x_col)),
        yaxis=dict(title=(y_label if y_label is not None else None)),
        title=title,
        title_x=0.5,
        showlegend=True,
    )
    fig = go.Figure(data=data, layout=layout)

    return fig


def plot_lines_markers(
        df,
        x_col,
        line_cols=None,
        marker_cols=None,
        band_cols=None,
        band_cols_dict=None,
        line_colors=None,
        marker_colors=None,
        band_colors=None,
        title=None):
    """A lightweight, easy-to-use function to create a plotly figure of given

    - lines (curves)
    - markers (points)
    - filled bands (e.g. error bands)

    from the columns of a dataframe with a legend which matches the column names.

    This can be used for example to annotate multiple curves,  markers and bands
    with an easy function call.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data frame with ``x_col`` and value columns specified in ``line_cols`` and ``marker_cols``.
    x_col : `str`
        The column used for the x-axis.
    line_cols : `list` [`str`] or None, default None
        The list of y-axis variables to be plotted as lines / curves.
    marker_cols : `list` [`str`] or None, default None
        The list of y-axis variables to be plotted as markers / points.
    band_cols : `list` [`str`] or None, default None
        The list of y-axis variables to be plotted as bands.
        Each column is expected to have tuples, each of which denote the upper
        and lower bounds.
    band_cols_dict : `dict` [`str`: [`str`]] or None, default None
        This is another way to specify bands.
        In this case:

            - each key will be the name for the band
            - the value contains the two bound colums of `df` for the band.

        For example `{
            "forecast": ["forecast_upper", "forecast_lower"],
            "w": ["w1", "w2"]}`
        Specifies two bands, one is based on the forecast prediction intervals
        and one is based on a variables denoted by "w" which has two corresponding
        columns in df: `"w1"` and `"w2"`.

    line_colors : `list` [`str`] or None, default None
        The list of colors to be used for each corresponding line column given in ``line_cols``.
    marker_colors : `list` [`str`] or None, default None
        The list of colors to be used for each corresponding marker column given in ``line_cols``.
    band_colors : `list` [`str`] or None, default None
        The list of colors to be used for each corresponding band column given in ``band_cols``.
        Each of these colors are used as filler for each band.
    title : `str` or None, default None
        Plot title. If None, no title will appear.

    Returns
    -------
    fig : `plotly.graph_objects.Figure`
        Interactive plotly graph of one or more columns in ``df`` against ``x_col``.
    """

    if line_colors is not None and line_cols is not None:
        if len(line_colors) < len(line_cols):
            raise ValueError(
                "If `line_colors` is passed, its length must be at least `len(line_cols)`")

    if marker_colors is not None and marker_cols is not None:
        if len(marker_colors) < len(marker_cols):
            raise ValueError(
                "If `marker_colors` is passed, its length must be at least `len(marker_cols)`")

    if band_colors is not None and band_cols is not None:
        if len(band_colors) < len(band_cols):
            raise ValueError(
                "If `band_colors` is passed, its length must be at least `len(band_cols)`")

    if band_colors is not None and band_cols_dict is not None:
        if len(band_colors) < len(band_cols_dict):
            raise ValueError(
                "If `band_colors` is passed, its length must be at least `len(band_cols_dict)`")

    if (
            line_cols is None and
            marker_cols is None and
            band_cols is None and
            band_cols_dict is None):
        raise ValueError(
                "At least one of `line_cols` or `marker_cols` or `band_cols`"
                " or `band_cols_dict` must be passed as a list (not None).")

    fig = go.Figure()
    # Below we count the number of figure components to assign proper labels to legends.
    count_fig_data = -1
    if line_cols is not None:
        for i, col in enumerate(line_cols):
            if line_colors is not None:
                line = go.scatter.Line(color=line_colors[i])
            else:
                line = go.scatter.Line()

            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="lines",
                line=line,
                showlegend=True))
            count_fig_data += 1
            fig["data"][count_fig_data]["name"] = col

    if marker_cols is not None:
        for i, col in enumerate(marker_cols):
            if marker_colors is not None:
                marker = go.scatter.Marker(color=marker_colors[i])
            else:
                marker = go.scatter.Marker()
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="markers",
                marker=marker,
                showlegend=True))
            count_fig_data += 1
            fig["data"][count_fig_data]["name"] = col

    if band_cols is not None:
        if band_colors is None:
            band_colors = get_distinct_colors(
                num_colors=len(band_cols),
                opacity=0.2)

        for i, col in enumerate(band_cols):
            fig.add_traces([
                go.Scatter(
                    x=df[x_col],
                    y=df[col].map(lambda b: b[1]),
                    mode="lines",
                    line=line,
                    line_color="rgba(0, 0, 0, 0)",
                    showlegend=True),
                go.Scatter(
                    x=df[x_col],
                    y=df[col].map(lambda b: b[0]),
                    mode="lines",
                    line_color="rgba(0, 0, 0, 0)",
                    line=line,
                    fill="tonexty",
                    fillcolor=band_colors[i],
                    showlegend=True)
            ])

            # The code below adds legend for each band.
            # We increment the count by two this time because each band comes with the
            # inner filling and lines around it.
            # In this case, we have made the lines around each band to be invisible.
            # However, they do appear in the figure data and we want to only include
            # one legend for each band.
            count_fig_data += 2
            # This adds the legend corresponding to the band filler color.
            fig["data"][len(fig["data"]) - 1]["name"] = col
            # The name for this added data is the empty string,
            # because we do not want to add a legend for the empty lines
            # around the bands.
            fig["data"][len(fig["data"]) - 2]["name"] = ""

    if band_cols_dict is not None:
        if band_colors is None:
            band_colors = get_distinct_colors(
                num_colors=len(band_cols_dict),
                opacity=0.2)

        for i, name in enumerate(band_cols_dict):
            col1 = band_cols_dict[name][0]
            col2 = band_cols_dict[name][1]
            fig.add_traces([
                go.Scatter(
                    x=df[x_col],
                    y=df[col2],
                    mode="lines",
                    line=line,
                    line_color="rgba(0, 0, 0, 0)",
                    showlegend=True),
                go.Scatter(
                    x=df[x_col],
                    y=df[col1],
                    mode="lines",
                    line_color="rgba(0, 0, 0, 0)",
                    line=line,
                    fill="tonexty",
                    fillcolor=band_colors[i],
                    showlegend=True)
            ])

            count_fig_data += 2
            fig["data"][len(fig["data"]) - 1]["name"] = name
            fig["data"][len(fig["data"]) - 2]["name"] = ""

    fig.update_layout(title=title)
    return fig


def plot_event_periods_multi(
        periods_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        freq=None,
        grouping_col=None,
        min_timestamp=None,
        max_timestamp=None,
        new_cols_tag="_is_anomaly",
        title="anomaly periods"):
    """For a dataframe (``periods_df``) with rows denoting start and end of the periods,
    it plots the periods. If there extra segmentation is given (``grouping_col``) then
    the periods in each segment/slice will be plotted separately on top of each other so that
    their overlap can be seen easily.

    Parameters
    ----------
    periods_df : `pandas.DataFrame`
        Data frame with ``start_time_col`` and ``end_time_col`` and optionally
        ``grouping_col`` if passed.
    start_time_col : `str`, default START_TIME_COL
        The column denoting the start of a period. The type can be any type
        admissable as `pandas.to_datetime`.
    end_time_col : `str`, default END_TIME_COL
        The column denoting the start of a period. The type can be any type
        admissable as `pandas.to_datetime`.
    freq : `str` or None, default None
        Frequency of the generated time grid which is used to plot the horizontal
        line segments (points).

        If None, we use hourly as default (freq = "H") which will be accurate
        for timestamps which are rounded up to an hour. For finer timestamps
        user should specify higher frequencies e.g. "min" for minutely. Also for
        daily, weekly, monthly data, user can use a lower frequency to make the plot
        size on disk smaller.
    grouping_col : `str` or None, default None
        A column which specifies the slicing.
        Each slice's event / anomaly period will be plotted with a specific color
        for that slice. Each segment will appear at a different height of the y-axis
        and its periods would be annotated at that height.
    min_timestamp : `str` or None, default None
        A string denoting the starting point (time) the x axis.
        If None, the minimum of ``start_time_col`` will be used.
    max_timestamp : `str` or None, default None
        A string denoting the end point (time) for the x axis.
        If None, the maximum of ``end_time_col`` will be used.
    title : `str` or None, default None
        Plot title. If None, default is based on axis labels.
    new_cols_tag : `str`, default "_is_anomaly"
        The tag used in the column names for each group.
        The column name has this format: f"{group}{new_cols_tag}".
        For example if a group is "impressions", with the default value
        of this argument the added column name is "impressions_is_anomaly"

    Returns
    -------
    result : `dict`

        - "fig" : `plotly.graph_objects.Figure`
            Interactive plotly graph of periods given for each group.
        - "labels_df" : `pandas.DataFrame`
            A dataframe which includes timestamps as one column (TIME_COL) and one
            dummy string column for each group.
            The values of the new columns are None, except for the time periods specified
            in each corresponding group.
        - "groups" : `list` [`str`]
            The distinct values seen in ``df[grouping_col]`` which are used for
            slicing of data.
        - "new_cols" : `list` [`str`]
            The new columns generated and added to ``labels_df``.
            Each column corresponds to one slice of the data as specified in
            ``grouping_col``.
        - "ts" : `list` [`pandas._libs.tslibs.timestamps.Timestamp`]
            A time-grid generated by ``pandas.date_range``
        - "min_timestamp" : `str` or None, default None
            A string denoting the starting point (time) the x axis.
        - "max_timestamp" : `str` or None, default None
            A string denoting the end point (time) for the x axis.
        - "marker_colors" : `list` [`str`]
            A list of strings denoting the colors used for various slices.

    """
    periods_df = periods_df.copy()
    if min_timestamp is None:
        min_timestamp = periods_df[start_time_col].min()
    if max_timestamp is None:
        max_timestamp = periods_df[end_time_col].max()

    periods_df = periods_df[
        (periods_df[start_time_col] >= min_timestamp) &
        (periods_df[end_time_col] <= max_timestamp)].reset_index(drop=True)

    if freq is None:
        freq = "H"

    ts = pd.date_range(start=min_timestamp, end=max_timestamp, freq=freq)
    labels_df = pd.DataFrame({TIME_COL: ts})
    # Converting the time to standard string format in order to use plotly safely
    labels_df[TIME_COL] = labels_df[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")

    def add_periods_dummy_column_for_one_group(
            labels_df,
            periods_df,
            new_col,
            label):
        """This function will add a dummy column for the time periods of each group to the ``labels_df``.
        Parameters
        ----------
        labels_df : `pandas.DataFrame`
            A data frame which at least inludes a timestamp column (TIME_COL).
        periods_df : `pandas.DataFrame`
            Data frame with ``start_time_col`` and ``end_time_col``.
        new_col : `str`
            The column name for the new dummy column to be added.
        label : `str`
            The label to be used in the new column when an event / anomaly is
            happening. Other values will be None.


        Returns
        -------
        labels_df : `pandas.DataFrame`
            A dataframe which has one extra column as compared with the input ``labels_df``.
            This extra column is a dummy string column for the input ``group``.
        """
        for i, row in periods_df.iterrows():
            t1 = row[start_time_col]
            t2 = row[end_time_col]
            if t2 < t1:
                raise ValueError(
                    f"End Time: {t2} cannot be before Start Time: {t1}, in ``periods_df``.")
            bool_index = (ts >= t1) & (ts <= t2)
            labels_df.loc[bool_index, new_col] = label

        return labels_df

    new_cols = []
    # If there is no grouping column we add a grouping column with only one value ("metric")
    if grouping_col is None:
        grouping_col = "metric"
        periods_df["metric"] = "metric"

    groups = set(periods_df[grouping_col].values)

    marker_colors = get_distinct_colors(
        len(groups),
        opacity=0.8)

    for group in groups:
        new_col = f"{group}{new_cols_tag}"
        new_cols.append(new_col)
        periods_df_group = periods_df.loc[
            periods_df[grouping_col] == group]
        labels_df = add_periods_dummy_column_for_one_group(
            labels_df=labels_df,
            periods_df=periods_df_group,
            new_col=new_col,
            label=group)

    # Plotting line segments for each period.
    # The line segments will have the same color for the same group.
    # Each group will be occupying one horizontal level in the plot.
    # The groups are stacked vertically in the plot so that the user can
    # scan and compare the periods across groups.
    fig = plot_lines_markers(
        df=labels_df,
        x_col=TIME_COL,
        line_cols=None,
        marker_cols=new_cols,
        line_colors=None,
        marker_colors=marker_colors)

    # Specify the y axis range
    fig.update_yaxes(range=[-1, len(groups)])

    # We add rectangles for each event period.
    # The rectangle colors for each group will be the same and consistent with
    # the line segments generated before for the same group.
    # The rectangles span all the way through the y-axis so that the user can
    # inspect the intersections between various groups better.
    shapes = []
    for i, group in enumerate(groups):
        ind = (periods_df[grouping_col] == group)
        periods_df_group = periods_df.loc[ind]

        fillcolor = marker_colors[i]
        for j, row in periods_df_group.iterrows():
            x0 = row[start_time_col]
            x1 = row[end_time_col]
            y0 = -1
            y1 = len(groups)

            # Specify the corners of the rectangles
            shape = dict(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                fillcolor=fillcolor,
                opacity=0.6,
                line_width=2,
                line_color=fillcolor,
                layer="below")

            shapes.append(shape)

    fig.update_layout(
        shapes=shapes,
        title=title,
        plot_bgcolor="rgba(233, 233, 233, 0.3)")  # light grey (lighter than default background)

    return {
        "fig": fig,
        "labels_df": labels_df,
        "groups": groups,
        "new_cols": new_cols,
        "ts": ts,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "marker_colors": marker_colors,
        "periods_df": periods_df
    }


def add_multi_vrects(
        fig,
        periods_df,
        grouping_col=None,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        y_min=None,
        y_max=None,
        annotation_text_col=None,
        annotation_position="top left",
        opacity=0.5,
        grouping_color_dict=None):
    """Adds vertical rectangle shadings to existing figure.
    Each vertical rectangle information is given in rows of data frame ``periods_df``.
    The information includes the beginning and end of vertical ranges of each rectangle as well as optional
    annotation.
    Rectangle colors can be grouped using a grouping given in ``grouping_col`` if available.

    Parameters
    ----------
    fig : `plotly.graph_objects.Figure`
        Existing plotly object which is going to be augmented with vertical
        rectangles and annotations.
    periods_df : `pandas.DataFrame`
        Data frame with at least ``start_time_col`` and ``end_time_col`` to denote the
        beginning and end of each vertical rectangle.
        This might also include ``grouping_col`` if the rectangle colors are to be grouped
        into same color within the group.
    grouping_col : `str` or None, default None
        The column which is used for grouping the vertical rectangle colors.
        For each group, the same color will be used for all periods in that group.
        If None, a dummy column will be added ("metric") with a single value (also "metric")
        which results in generating only one color for all rectangles.
    start_time_col : `str`, default ``START_TIME_COL``
        The column denoting the start of a period. The type can be any type
        consistent with the type of existing x axis in ``fig``.
    end_time_col : `str`, default ``END_TIME_COL``
        The column denoting the start of a period. The type can be any type
        consistent with the type of existing x axis in ``fig``.
    y_min : `float` or None, default None
        The lower limit of the rectangles.
    y_max : `float` or None, default None
        The upper limit of the rectangles.
    annotation_text_col : `str` or None, default None
        A column which includes annotation texts for each vertical rectangle.
    annotation_position : `str`, default "top left"
        The position of annotation texts with respect to the vertical rectangle.
    opacity : `float`, default 0.5
        The opacity of the colors. Note that the passed colors could have opacity
        as well, in which case this opacity will act as a relative opacity.
    grouping_color_dict : `dict` [`str`, `str`] or None, default None
        A dictionary to specify colors for each group given in ``grouping_col``.
        If there is no ``grouping_col`` passed, there will be only one color needed
        and in that case a dummy ``grouping_col`` will be created with the name "metric".
        Therefore user needs to specify ``grouping_color_dict = {"metric": desired_color}``.

    Returns
    -------
    result : `dict`
        - "fig": `plotly.graph_objects.Figure`
            Updated plotly object which is augmented with vertical
            rectangles and annotations.
        - "grouping_color_dict": `dict` [`str`, `str`]
            A dictionary with keys being the groups and the values being the colors.

    """
    # If there is no grouping column we add a grouping column with only one value ("metric")
    if grouping_col is None:
        grouping_col = "metric"
        periods_df["metric"] = "metric"

    if start_time_col not in periods_df.columns:
        raise ValueError(
            f"start_time_col: {start_time_col} is not found in ``periods_df`` columns: {periods_df.columns}")

    if end_time_col not in periods_df.columns:
        raise ValueError(
            f"end_time_col: {end_time_col} is not found in ``periods_df`` columns: {periods_df.columns}")

    if grouping_col is not None and grouping_col not in periods_df.columns:
        raise ValueError(
            f"grouping_col: {grouping_col} is passed but not found in ``periods_df`` columns: {periods_df.columns}")

    if annotation_text_col is not None and annotation_text_col not in periods_df.columns:
        raise ValueError(
            f"annotation_text_col: {annotation_text_col} is passed but not found in ``periods_df`` columns: {periods_df.columns}")

    groups = list(set(periods_df[grouping_col]))
    groups.sort()

    if grouping_color_dict is None:
        colors = get_distinct_colors(
            len(groups),
            opacity=1.0)
        grouping_color_dict = {groups[i]: colors[i] for i in range(len(groups))}

    for i, group in enumerate(groups):
        ind = (periods_df[grouping_col] == group)
        periods_df_group = periods_df.loc[ind]

        fillcolor = grouping_color_dict[group]
        for j, row in periods_df_group.iterrows():
            x0 = row[start_time_col]
            x1 = row[end_time_col]

            if annotation_text_col is not None:
                annotation_text = row[annotation_text_col]
            else:
                annotation_text = ""
            # Adds the vertical rectangles
            fig.add_vrect(
                x0=x0,
                y0=y_min,
                x1=x1,
                y1=y_max,
                fillcolor=fillcolor,
                opacity=opacity,
                line_width=2,
                line_color=fillcolor,
                layer="below",
                annotation_text=annotation_text,
                annotation_position=annotation_position)
    return {
        "fig": fig,
        "grouping_color_dict": grouping_color_dict}


def plot_overlay_anomalies_multi_metric(
        df,
        time_col,
        value_cols,
        anomaly_df,
        anomaly_df_grouping_col=None,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        annotation_text_col=None,
        annotation_position="top left",
        lines_opacity=0.6,
        markers_opacity=0.8,
        vrect_opacity=0.3):
    """This function operates on a given data frame (``df``) which includes time (given in ``time_col``) and
    metrics (given in ``value_cols``), as well as ``anomaly_df`` which includes the anomaly periods
    corresponding to those metrics. It generates a plot of the metrics annotated with anomaly
    values as markers on the curves and vertical rectangles for the same periods.
    Each metric, its anomaly values and vertical rectangles use the same color with
    varying opacity.

    Parameters
    ----------
    df : `pandas.DataFrame`
        A data frame which at least inludes a timestamp column (``TIME_COL``) and
        ``value_cols`` which represent the metrics.
    time_col : `str`
        The column name in ``df`` representing time for the time series data.
        The time column can be anything that can be parsed by `pandas.DatetimeIndex`.
    value_cols : `list` [`str`]
        The columns which include the metrics.
    anomaly_df : `pandas.DataFrame`
        Data frame with ``start_time_col`` and ``end_time_col`` and ``grouping_col``
        (if provided). This contains the anomaly periods for each metric
        (one of the ``value_cols``). Each row of this dataframe corresponds
        to an anomaly occurring between the times given in ``row[start_time_col]``
        and ``row[end_time_col]``.
        The ``grouping_col`` (if not None) determines which metric that
        anomaly corresponds too (otherwise we assume all anomalies apply to all metrics).
    anomaly_df_grouping_col : `str` or None, default None
        The column name for grouping the list of the anomalies which is to appear
        in ``anomaly_df``.
        This column should include some of the metric names
        specified in ``value_cols``. The ``grouping_col`` (if not None) determines which metric that
        anomaly corresponds too (otherwise we assume all anomalies apply to all metrics).
    start_time_col : `str`, default ``START_TIME_COL``
        The column name in ``anomaly_df`` representing the start timestamp of
        the anomalous period, inclusive.
        The format can be anything that can be parsed by pandas DatetimeIndex.
    end_time_col : `str`, default ``END_TIME_COL``
        The column name in ``anomaly_df`` representing the start timestamp of
        the anomalous period, inclusive.
        The format can be anything that can be parsed by pandas DatetimeIndex.
    annotation_text_col : `str` or None, default None
        A column which includes annotation texts for each vertical rectangle.
    annotation_position : `str`, default "top left"
        The position of annotation texts with respect to the vertical rectangle.
    lines_opacity : `float`, default 0.6
        The opacity of the colors used in the lines (curves) which represent the
        metrics given in ``value_cols``.
    markers_opacity: `float`, default 0.8
        The opacity of the colors used in the markersc which represent the
        value of the metrics given in ``value_cols`` during anomaly times as
        specified in ``anomaly_df``.
    vrect_opacity : `float`, default 0.3
        The opacity of the colors for the vertical rectangles.


    Returns
    -------
    result : `dict`
        A dictionary with following items:

        - "fig": `plotly.graph_objects.Figure`
            Plotly object which includes the metrics augmented with vertical
            rectangles and annotations.
        - "augmented_df": `pandas.DataFrame`
            This is a dataframe obtained by augmenting the input ``df`` with new
            columns determining if the metrics appearing in ``df`` are anomaly
            or not and the new columns denoting anomaly values and normal values
            (described below).
        - "is_anomaly_cols": `list` [`str`]
            The list of add boolean columns to determine if a value is an anomaly for
            a given metric. The format of the columns is ``f"{metric}_is_anomaly"``.
        - "anomaly_value_cols": `list` [`str`]
            The list of columns containing only anomaly values (`np.nan` otherwise) for each corresponding
            metric. The format of the columns is ``f"{metric}_anomaly_value"``.
        - "normal_value_cols": `list` [`str`]
            The list of columns containing only non-anomalous / normal values (`np.nan` otherwise)
            for each corresponding metric. The format of the columns is ``f"{metric}_normal_value"``.
        - "line_colors": `list` [`str`]
            The colors generated for the metric lines (curves).
        - "marker_colors": `list` [`str`]
            The colors generated for the anomaly values markers.
        - "vrect_colors": `list` [`str`]
            The colors generated for the vertical rectangles.

    """
    # Adds anomaly information columns to the data
    # For every column specified in ``value_cols``, there will be 3 new columns are added to ``df``:
    # ``f"{value_col}_is_anomaly"``
    # ``f"{value_col}_anomaly_value"``
    # ``f"{value_col}_normal_value"``
    augmenting_data_res = label_anomalies_multi_metric(
        df=df,
        time_col=time_col,
        value_cols=value_cols,
        anomaly_df=anomaly_df,
        anomaly_df_grouping_col=anomaly_df_grouping_col,
        start_time_col=start_time_col,
        end_time_col=end_time_col)

    augmented_df = augmenting_data_res["augmented_df"]
    is_anomaly_cols = augmenting_data_res["is_anomaly_cols"]
    anomaly_value_cols = augmenting_data_res["anomaly_value_cols"]
    normal_value_cols = augmenting_data_res["normal_value_cols"]

    line_colors = get_distinct_colors(
        len(value_cols),
        opacity=lines_opacity)

    marker_colors = get_distinct_colors(
        len(value_cols),
        opacity=markers_opacity)

    vrect_colors = get_distinct_colors(
        len(value_cols),
        opacity=vrect_opacity)

    fig = plot_lines_markers(
        df=augmented_df,
        x_col=time_col,
        line_cols=value_cols,
        marker_cols=anomaly_value_cols,
        line_colors=line_colors,
        marker_colors=marker_colors)

    grouping_color_dict = {value_cols[i]: vrect_colors[i] for i in range(len(value_cols))}

    y_min = df[value_cols].min(numeric_only=True).min()
    y_max = df[value_cols].max(numeric_only=True).max()

    augmenting_fig_res = add_multi_vrects(
        fig=fig,
        periods_df=anomaly_df,
        grouping_col=anomaly_df_grouping_col,
        start_time_col=start_time_col,
        end_time_col=end_time_col,
        y_min=y_min,
        y_max=y_max,
        annotation_text_col=annotation_text_col,
        annotation_position="top left",
        opacity=1.0,
        grouping_color_dict=grouping_color_dict)

    fig = augmenting_fig_res["fig"]

    return {
        "fig": fig,
        "augmented_df": augmented_df,
        "is_anomaly_cols": is_anomaly_cols,
        "anomaly_value_cols": anomaly_value_cols,
        "normal_value_cols": normal_value_cols,
        "line_colors": line_colors,
        "marker_colors": marker_colors,
        "vrect_colors": vrect_colors
    }


def plot_precision_recall_curve(
        df,
        grouping_col=None,
        recall_col="recall",
        precision_col="precision",
        axis_font_size=18,
        title_font_size=20,
        title="Precision - Recall Curve",
        opacity=0.95):
    """Plots a Precision - Recall curve, where the x axis is recall and the y axis is precision.
    If ``grouping_col`` is None, it creates one Precision - Recall curve given the data in ``df``.
    Otherwise, this function creates an overlay plot for multiple Precision - Recall curves, one for each level in the ``grouping_col``.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input dataframe. Must contain the columns:

            - ``recall_col``: `float`
            - ``precision_col``: `float`

        If ``grouping_col`` is not None, it must also contain the column ``grouping_col``.
    grouping_col : `str` or None, default None
        Column name for the grouping column.
    recall_col : `str`, default "recall"
        Column name for recall.
    precision_col : `str`, default "precision"
        Column name for precision.
    axis_font_size : `int`, default 18
        Axis font size.
    title_font_size : 20
        Title font size.
    title : `str`, default "Precision - Recall Curve"
        Plot title.
    opacity : `float`, default 0.95
        The opacity of the color. This has to be a number between 0 and 1.

    Returns
    -------
        fig : `plotly.graph_objs._figure.Figure`
            Plot figure.
    """
    if any([col not in df.columns for col in [recall_col, precision_col]]):
        raise ValueError(f"`df` must contain the `recall_col`: '{recall_col}' and the `precision_col`: '{precision_col}' specified!")
    # Stores the curves to be plotted.
    data = []
    # Creates the curve(s).
    if grouping_col is None:  # Creates one precision - recall curve.
        num_colors = 1
        df.sort_values(recall_col, inplace=True)
        line = go.Scatter(
            x=df[recall_col].tolist(),
            y=df[precision_col].tolist())
        data.append(line)
    else:  # Creates precision - recall curve for every level in `grouping_col`.
        if grouping_col not in df.columns:
            raise ValueError(f"`grouping_col` = '{grouping_col}' is not found in the columns of `df`!")
        num_colors = 0
        for level, indices in df.groupby(grouping_col).groups.items():
            df_subset = df.loc[indices].reset_index(drop=True).sort_values(recall_col)
            line = go.Scatter(
                name=f"{level}",
                x=df_subset[recall_col].tolist(),
                y=df_subset[precision_col].tolist())
            data.append(line)
            num_colors += 1
    # Creates a list of colors for the curve(s).
    color_list = get_distinct_colors(
        num_colors=num_colors,
        opacity=opacity)
    if color_list is not None:
        if len(color_list) < len(data):
            raise ValueError("`color_list` must not be shorter than the number of traces in this figure!")
        for i, v in enumerate(data):
            v.line.color = color_list[i]
    # Creates the layout.
    range_epsilon = 0.05  # Space at the beginning and end of the margins.
    layout = go.Layout(
        xaxis=dict(
            title=recall_col.title(),
            titlefont=dict(size=axis_font_size),
            range=[0 - range_epsilon, 1 + range_epsilon],  # Sets the range of xaxis.
            tickfont_size=axis_font_size,
            tickformat=".0%",
            hoverformat=",.1%"),  # Keeps 1 decimal place.
        yaxis=dict(
            title=precision_col.title(),
            titlefont=dict(size=axis_font_size),
            range=[0 - range_epsilon, 1 + range_epsilon],  # Sets the range of yaxis.
            tickfont_size=axis_font_size,
            tickformat=".0%",
            hoverformat=",.1%"),  # Keeps 1 decimal place.
        title=title.title(),
        title_x=0.5,
        titlefont=dict(size=title_font_size),
        autosize=False,
        width=1000,
        height=800)
    # Creates the figure.
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(
        constrain="domain",  # Compresses the yaxis by decreasing its "domain".
        automargin=True,
        rangemode="tozero")
    fig.update_xaxes(
        constrain="domain",  # Compresses the xaxis by decreasing its "domain".
        automargin=True,
        rangemode="tozero")
    fig.add_hline(y=0.0, line_width=1, line_color="gray")
    fig.add_vline(x=0.0, line_width=1, line_color="gray")
    return fig


def plot_anomalies_over_forecast_vs_actual(
        df,
        time_col=TIME_COL,
        actual_col=ACTUAL_COL,
        predicted_col=PREDICTED_COL,
        predicted_anomaly_col=PREDICTED_ANOMALY_COL,
        anomaly_col=ANOMALY_COL,
        marker_opacity=1,
        predicted_anomaly_marker_color="rgba(0, 90, 181, 0.9)",
        anomaly_marker_color="rgba(250, 43, 20, 0.7)",
        **kwargs):
    """Utility function which overlayes the predicted anomalies or anomalies on the forecast vs actual plot.
    The function calls the internal function `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`
    and then adds markers on top.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input dataframe.
    time_col : `str`, default `~greykite.common.constants.TIME_COL`
        Column in ``df`` with timestamp (x-axis).
    actual_col : `str`, default `~greykite.common.constants.ACTUAL_COL`
        Column in ``df`` with actual values.
    predicted_col : `str`, default `~greykite.common.constants.PREDICTED_COL`
        Column in ``df`` with predicted values.
    predicted_anomaly_col : `str` or None, default `~greykite.common.constants.PREDICTED_ANOMALY_COL`
        Column in ``df`` with predicted anomaly labels (boolean) in the time series.
        `True` denotes a predicted anomaly.
    anomaly_col : `str` or None, default `~greykite.common.constants.ANOMALY_COL`
        Column in ``df`` with anomaly labels (boolean) in the time series.
        `True` denotes an anomaly.
    marker_opacity : `float`, default 0.5
        The opacity of the marker colors.
    predicted_anomaly_marker_color : `str`, default "green"
        The color of the marker(s) for the predicted anomalies.
    anomaly_marker_color : `str`, default "red"
        The color of the marker(s) for the anomalies.
    **kwargs
        Additional arguments on how to decorate your plot.
        The keyword arguments are passed to `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`.

    Returns
    -------
    fig : `plotly.graph_objs._figure.Figure`
        Plot figure.
    """
    fig = plot_forecast_vs_actual(
        df=df,
        time_col=time_col,
        actual_col=actual_col,
        predicted_col=predicted_col,
        **kwargs)
    if anomaly_col is not None:
        fig.add_trace(go.Scatter(
            x=df.loc[df[anomaly_col].apply(lambda val: val is True), time_col],
            y=df.loc[df[anomaly_col].apply(lambda val: val is True), actual_col],
            mode="markers",
            marker_size=10,
            marker_symbol="square",
            marker=go.scatter.Marker(color=anomaly_marker_color),
            name=anomaly_col.title(),
            showlegend=True,
            opacity=marker_opacity))
    if predicted_anomaly_col is not None:
        fig.add_trace(go.Scatter(
            x=df.loc[df[predicted_anomaly_col].apply(lambda val: val is True), time_col],
            y=df.loc[df[predicted_anomaly_col].apply(lambda val: val is True), actual_col],
            mode="markers",
            marker_size=7,
            marker_symbol="diamond",
            marker=go.scatter.Marker(color=predicted_anomaly_marker_color),
            name=predicted_anomaly_col.title(),
            showlegend=True,
            opacity=marker_opacity))

    return fig
