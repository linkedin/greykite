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
# author: Reza Hosseini

import pandas as pd

from greykite.common.viz.timeseries_plotting import plot_multivariate


def compare_two_dfs_on_index(
        dfs,
        df_labels,
        index_col,
        diff_cols=None,
        relative_diff=False):
    """Calculates difference between two dataframes macthed on a given index.
    One intended application is to compare breakdown dfs from two related ML models
    or same model at different times. However it can be used to compare generic
    dataframes as well.

    Parameters
    ----------
    dfs: `list` [`pandas.DataFrame`]
        A list of two dataframes which includes minimally
            - the index column (``index_col``)
            - columns which include the values to be compared.
            If ``diff_cols`` is passed, we expect them to appear in these dataframes.
    df_labels : `list` [`str`]
        A list of two strings denoting the label for each ``df`` respectively.
    index_col : `str`
        The column name of the index column which will be used for joining and
        as x axis of the difference plot.
    diff_cols :  `list` [`str`], default None
        A list of columns to be differenced.
        Each column represents a variable / component to be compared.
        If it is not passed or None, all the columns except for ``index_col`` will
        be used.
    relative_diff : `bool`, default True
        It determines if diff is to be simple diff or relative diff obtained by
        dividing the diff by the absolute values of first dataframe for each value column.

    Returns
    -------
    result : `dict`
        A dictionary with following items:

            - "diff_df": `pandas.DataFrame`
                A dataframe which includes the diff for each group / component.
            - "diff_fig": `plotly.graph_objs._figure.Figure`
                plotly plot overlaying various variable diffs

    """
    assert len(dfs) == 2, "We expect two dataframes."
    assert len(df_labels) == 2, "We expect two labels"
    assert list(dfs[0].columns) == list(dfs[1].columns), "We expect same column names in both dataframes"
    assert index_col in dfs[0].columns, "`index_col` must be found in both dataframes."
    assert index_col in dfs[1].columns, "`index_col` must be found in both dataframes."

    diff_df = pd.merge(
        dfs[0],
        dfs[1],
        on=index_col)

    if diff_cols is None:
        diff_cols = list(dfs[0].columns)
        diff_cols.remove(index_col)

    y_label = "diff"
    for col in diff_cols:
        diff_df[col] = diff_df[f"{col}_y"] - diff_df[f"{col}_x"]
        if relative_diff:
            diff_df[col] = diff_df[col] / abs(diff_df[f"{col}_x"])
            y_label = "relative diff"

    diff_df = diff_df[[index_col] + diff_cols]

    diff_fig = plot_multivariate(
        df=diff_df,
        x_col=index_col,
        title=f"change in variables from {df_labels[0]} to {df_labels[1]}",
        ylabel=y_label)

    return {
        "diff_df": diff_df,
        "diff_fig": diff_fig
    }
