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
"""Preprocessing function to normalize data before training."""

import warnings


def normalize_df(
        df,
        method,
        drop_degenerate_cols=True,
        replace_zero_denom=False):
    """For a given dataframe with columns including numerical values,
    it generates a function which can be applied to original data as well as
    any future data to normalize using two possible methods.
    The `"statistical"` method removes the "mean" and divides by "std".
    The `"zero_to_one"` method removes the "minimum" and divides by the
    "maximum - minimum".
    If desired, the function also drops the columns which have only one
    possible value and can cause issues not only during normalizaton
    (returning a column with all NAs) but also potentially during fitting
    as an example.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Input dataframe which (only) includes numerical values in all columns.
    method : `str`
        The method to be used for normalization.

        - "statistical" method removes the "mean" and divides by "std" for each column.
        - "zero_to_one" method removes the "min" and divides by the "max - min"
            for each column.
        - "minus_half_to_half" method will remove the "(min + max)/2" and divides by the "max - min"
            for each column.
        - "zero_at_origin" method will remove the first data point and divides by the "max - min"
            for each column.

    drop_degenerate_cols : `bool`, default True
        A boolean to determine if columns with only one possible value should be
        dropped in the normalized dataframe.
    replace_zero_denom : `bool`, default False
        A boolean to decide if zero denominator (e.g. standard deviation for
        ``method="statistical"``) for normalization should be replaced by 1.0.

    Returns
    -------
    normalize_info : `dict`

        A dictionary with with the main item being a normalization function.
        The items are as follows:

        ``"normalize_df_func"`` : callable (pd.DataFrame -> pd.DataFrame)
            A function which normalizes the input dataframe (``df``)
        ``"normalized_df"`` : normalized dataframe version of ``df``
        ``"keep_cols"`` : `list` [`str`]
            The list of kept columns after normalization.
        ``"drop_cols"`` : `list` [`str`]
            The list of dropped columns after normalization.
        ``"subtracted_series"`` : `pandas.Series`
            The series to be subtracted which has one value for each column of ``df``.
        ``"denominator_series"`` : `pandas.Series`
            The denominator series for normalization which has one value for each
            column of ``df``.
    """
    if method == "statistical":
        subtracted_series = df.mean()
        denominator_series = df.std()
    elif method == "zero_to_one":
        subtracted_series = df.min()
        denominator_series = (df.max() - df.min())
    elif method == "minus_half_to_half":
        subtracted_series = (df.min() + df.max()) / 2
        denominator_series = (df.max() - df.min())
    elif method == "zero_at_origin":
        subtracted_series = df.iloc[0]
        denominator_series = (df.max() - df.min())
    else:
        raise NotImplementedError(f"Method {method} is not implemented")

    # Replaces 0.0 in denominator series with 1.0 to avoid dividing by zero
    # when the variable has zero variance
    if replace_zero_denom:
        denominator_series.replace(to_replace=0.0, value=1.0, inplace=True)
    drop_cols = []
    keep_cols = list(df.columns)
    normalized_df = (df - subtracted_series) / denominator_series

    if drop_degenerate_cols:
        for col in df.columns:
            if normalized_df[col].isnull().any():
                drop_cols.append(col)
                warnings.warn(
                    f"{col} was dropped during normalization as it had only one "
                    "possible value (degenerate)")

        keep_cols = [col for col in list(df.columns) if col not in drop_cols]
        if len(keep_cols) == 0:
            raise ValueError(
                "All columns were degenerate (only one possible value per column).")

        subtracted_series = subtracted_series[keep_cols]
        denominator_series = denominator_series[keep_cols]

    def normalize_df_func(new_df):
        """A function which applies to a potentially new data frame (``new_df``)
        with the same columns as ``df`` (different values or row number is allowed)
        and returns a normalized dataframe with the same normalization parameters
        applied to ``df``.

        This function uses the series `subtracted_series` and
        ``denominator_series`` generated in its outer scope for normalization,
        and in this way ensures the same mapping for new data.

        Parameters
        ----------
        new_df : `pandas.DataFrame`
            Input dataframe which (only) includes numerical values in all columns.
            The columns of ``new_df`` must be the same as ``df`` which is passed to
            the outer function (``normalize_df``) to construct this function.

        Returns
        -------
        normalized_df : `pandas.dataframe`
            Normalized dataframe version of ``new_df``.
        """
        normalized_df = new_df.copy()
        if drop_degenerate_cols:
            normalized_df = normalized_df[keep_cols]
        return (normalized_df - subtracted_series) / denominator_series

    return {
        "normalize_df_func": normalize_df_func,
        "normalized_df": normalized_df,
        "keep_cols": keep_cols,
        "drop_cols": drop_cols,
        "subtracted_series": subtracted_series,
        "denominator_series": denominator_series
    }
