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
"""Plotting functions in matplotlib."""

import colorsys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters


matplotlib.use("agg")  # noqa: E402

register_matplotlib_converters()


def plt_compare_timeseries(
        df_dict,
        time_col,
        value_col,
        legends_dict=None,
        colors_dict=None,
        start_time=None,
        end_time=None,
        transform=lambda x: x,
        transform_name="",
        plt_title="",
        alpha=0.6,
        linewidth=4):
    """Compare a collection by of timeseries (given in `df_dict`)
       by overlaying them in the specified period between
       ``start_time`` and ``end_time``.

    :param df_dict: Dict[str, pd.DataFrame]
        The keys are the arbitrary labels for each dataframe provided by the user.
        The values are dataframes each containing a timeseries
        with `time_col` and `value_col` as columns.
    :param time_col: str
        The column denoting time in datetime format.
    :param value_col: str
        The value column of interest for the y axis.
    :param legends_dict: Optional[Dict[str, str]]
        Labels for plot legend.
        The keys are the df_dict labels (or a subset of them).
        The values are the labels appearing in the plot legend.
        If not provided or None, the `df_dict` keys will be used.
    :param colors_dict: Optional[Dict[str, str]]
        A dictionary determining the color for each series in `df_dict`.
        The keys are the df_dict labels (or a subset of them).
        The values are the colors appearing for each curve in the plot.
        If not provided or None, the colors will be generated.
    :param start_time: Optional[datetime.datetime]
        The start time of the series plot.
    :param end_time: Optional[datetime.datetime]
        The end time of the series plot.
    :param transform: Optional[func]
        A function to transform the y values before plotting.
    :param transform_name: Optional[str]
        The name of the transformation for using in the title.
    :param plt_title: str
        Plot title.
    :param alpha: Optional[float]
        Transparency of the curves.
    :param linewidth: Optional[float]
        The width of the curves.
    """
    if start_time is not None:
        for label, df in df_dict.items():
            df_dict[label] = df[df[time_col] >= start_time]

    if end_time is not None:
        for label, df in df_dict.items():
            df_dict[label] = df[df[time_col] <= end_time]

    n = len(df_dict)
    labels = list(df_dict.keys())
    if colors_dict is None:
        hsv_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
        rgb_tuples = [(lambda x: colorsys.hsv_to_rgb(*x))(x) for x in hsv_tuples]
        colors_dict = {labels[i]: rgb_tuples[i] for i in range(n)}

    if legends_dict is None:
        legends_dict = {label: label for label in labels}

    fig, ax = plt.subplots()
    legend_patches = []
    for label in labels:
        df = df_dict[label]
        color = colors_dict[label]
        legend = legends_dict.get(label)
        # Avoids the following issue:
        # "ValueError: view limit minimum -36864.55 is less than 1 and is an invalid
        # Matplotlib date value. This often happens if you pass a non-datetime value
        # to an axis that has datetime units".
        dates = df[time_col].astype("O")
        # we add a legend for the given series if legend is not None
        # for that series
        if legend is not None:
            ax.plot(
                dates,
                transform(df[value_col]),
                alpha=alpha,
                color=color,
                label=legend,
                linewidth=linewidth)
            patch = matplotlib.patches.Patch(color=color, label=legend)
            legend_patches.append(patch)
        else:
            ax.plot(
                dates,
                transform(df[value_col]),
                alpha=alpha,
                color=color,
                linewidth=linewidth)

    legends = list(legends_dict.values())
    ax.legend(
        labels=[legend for legend in legends if legend is not None],
        handles=legend_patches)
    fig.autofmt_xdate()  # rotates the dates
    ax.set_title(plt_title + " " + transform_name)
    plt.show()


def plt_overlay_long_df(
        df,
        x_col,
        y_col,
        split_col,
        agg_dict=None,
        agg_col_names=None,
        overlay_color="black",
        agg_col_colors=None,
        plt_title=None):
    """Overlay by splitting wrt values of a column (split_col).
        If some agg metrics (specified by agg_dict) are also desired,
        we overlay the aggregate metrics as well.
    :param df: pd.DataFrame
        data frame which includes the data
    :param x_col: str
        the column for the values for the x-axis
    :param y_col: str
        the column for the values for the y-axis
    :param split_col: str
        the column which is used to split the data to various
        parts to be overlayed
    :param agg_dict: Optional[dict]
        a dictionary to specify aggregations.
        we could calculate multiple metrics for example mean and median
    :param agg_col_names: optional[list[str]]
        names for the aggregated columns.
        if not provided it will be generated during aggregations
    :param overlay_color: Optional[str]
        the color of the overlayed curves.
        The color will be transparent so we can see the curves overlap
    :param agg_col_colors: Optional[str]
        the color of the aggregate curves
    :param plt_title: Optional[str]
        the title of the plot.
        It will default to y_col if not provided
    """
    g = df.groupby([split_col], as_index=False)
    df_num = len(g)
    alpha = 5.0 / df_num

    for name, df0 in g:
        plt.plot(
            df0[x_col],
            df0[y_col],
            color=overlay_color,
            alpha=alpha,
            label=None)

    agg_df = None
    if agg_dict is not None:
        g2 = df.groupby([x_col], as_index=False)
        agg_df = g2.agg(agg_dict)
        agg_df.columns = [" ".join(col).strip() for col in agg_df.columns.values]
        if agg_col_names is not None:
            agg_df.columns = [x_col] + agg_col_names

        if agg_col_colors is None:
            n = len(agg_col_names)
            hsv_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
            rgb_tuples = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
            agg_col_colors = rgb_tuples

        legend_patches = []

        for i in range(len(agg_col_names)):
            col = agg_col_names[i]
            color = agg_col_colors[i]
            plt.plot(agg_df[x_col], agg_df[col], label=col, color=color)
            patch = matplotlib.patches.Patch(color=color, label=col)
            legend_patches.append(patch)

        plt.legend(labels=agg_col_names, handles=legend_patches)

    if plt_title is None:
        plt_title = y_col

    plt.title(plt_title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    return agg_df


def plt_overlay_with_bands(
        df,
        x_col,
        y_col,
        split_col,
        perc=(25, 75),
        overlay_color="black",
        agg_col_colors=("blue", "black", "red"),
        plt_title=None):
    """Overlay by splitting wrt a column and plot wrt time.
        We also add quantile (percentile) bands
    :param df: pd.DataFrame
        data frame which includes the data
    :param x_col: str
        the column for the values for the x-axis
    :param y_col: str
        the column for the values for the y-axis
    :param split_col: str
        the column which is used to split the data
        to various partitions t be overlayed
    :param perc: tuple[float, float]
        the percentiles for the bands.
        The default is 25 and 75 percentiles
    :param overlay_color: Optional[str]
        the color of the overlayed curves.
        The color will be transparanet so we can see the curves overlap
    :param agg_col_colors: Optional[str]
        the color of the aggregate curves
    :param plt_title: Optional[str]
        the title of the plot.
        It will default to y_col if not provided
    """

    def quantile_fcn(q):
        return lambda x: np.nanpercentile(a=x, q=q)

    agg_dict = {
        y_col: [np.nanmean, quantile_fcn(perc[0]), quantile_fcn(perc[1])]
    }

    res = plt_overlay_long_df(
        df=df,
        x_col=x_col,
        y_col=y_col,
        split_col=split_col,
        agg_dict=agg_dict,
        agg_col_names=["mean", ("Q" + str(perc[0])), ("Q" + str(perc[1]))],
        overlay_color=overlay_color,
        agg_col_colors=agg_col_colors,
        plt_title=plt_title)

    return res


def plt_longterm_ts_agg(
        df,
        time_col,
        window_col,
        value_col,
        color_col=None,
        agg_func=np.nanmean,
        plt_title="",
        color="blue",
        choose_color_func=None):
    """Make a longterm avg plot by taking the average in a window
        and moving that across the time.
        That window can be a week, month, year etc

    :param df: pd.DataFrame
        The data frame with the data
    :param time_col: str
        The column with timestamps
    :param window_col: str
        This is a column which represents a coarse time granularity
        eg "2018-01" represents year-month
        In this case the avg across that month is calculated
    :param value_col: str
        The column denoting the values
    :param color_col: Optional[str]
        The column denoting the color for each point.
        We allow the curve color to change.
        When aggregating for each window the first color
        appearing in that window will be used by default.
    :param agg_func: Optional[func]
        the aggregation function to be used across the window
    :param plt_title: Optional[str]
        Plot title
    :param color: Optional[str]
        Color of the curve if it is not provided by `color_col`
        in original dataframe
    :param choose_color_func: func
        Function to determine how pointwise colors are translated to aggregated
        to be used for the aggregate curve.
        The default choosed the first color appearing in the data for
        the corresponding slice.
    :return pd.DataFrame
        The aggregated data frame constructed for the plot
    """
    df = df.copy()

    # this is for choosing the color for the aggregated curve
    # after averaging
    if choose_color_func is None:
        def choose_color_func(color_values):
            return color_values.iloc[0]

    if color_col is None:
        color_col = "curve_color"
        df["curve_color"] = color

    agg_dict = {
        time_col: np.nanmean,
        value_col: agg_func,
        color_col: choose_color_func}

    g = df.groupby([window_col], as_index=False)
    df_agg = g.agg(agg_dict)

    df_agg.columns = [window_col, time_col, value_col, color_col]
    df_agg.sort_values(by=time_col, inplace=True)
    # plotting the whole curve in grey
    plt.plot(
        df_agg[time_col],
        df_agg[value_col],
        alpha=0.5,
        color="grey")
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    colors_set = set(list(df[color_col].values))
    # adding scatter points in colors for each part of curve
    # based on `color_col`
    for c in colors_set:
        df_agg_subset = df_agg.loc[
            df_agg[color_col] == c].reset_index(drop=True)
        plt.scatter(
            df_agg_subset[time_col].values,
            df_agg_subset[value_col].values,
            alpha=0.5,
            color=c)
    plt.title(plt_title)

    return df_agg
