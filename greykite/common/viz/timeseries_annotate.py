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

import plotly.graph_objects as go


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
