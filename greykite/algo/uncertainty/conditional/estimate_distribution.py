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
"""Estimates the empirical distribution of a variable."""

import numpy as np


def estimate_empirical_distribution(
        df,
        distribution_col,
        quantile_grid_size=0.05,
        quantiles=(0.025, 0.975),
        conditional_cols=None,
        remove_conditional_mean=True):
    """Estimates the empirical distribution for a given variable in the ``distribution_col``
    of ``df``.
    The distribution is an approximated quantile function defined on
    the quantiles specified in ``quantiles`` which are values in [0, 1].
    The function returns mean, and standard deviation (std) as well.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The dataframe with the following columns:

            - distribution_col,
            - conditional_cols (optional)

    distribution_col : `str`
        The column name for the value column of interest.
    quantile_grid_size : `float` or None
        The grid size for the quantiles.
        E.g. if grid_size = 0.25 then the quantiles for 0.25, 0.5, 0.75 are calculated.
        If grid_size = 0.05 then then the quantiles for 0.05, 0.1, ..., 0.90, 0.95 are calculated.
    quantiles : `list` [`float`] or None, default (0.025, 0.975)
        The probability grid for which quantiles are calculated.
        If None, ``quantile_grid_size`` is used to generate a regularly spaced quantile grid.
    conditional_cols : `list` [`str`] or None, default None
        These columns are used to slice the data first then calculate quantiles
        for each slice.
    remove_conditional_mean : `bool`, default True
        If True, for every slice (defined by ``conditional_cols``), the conditional mean
        is removed when calculating quantiles.

    Returns
    -------
    model_dict : `dict`
        A dictionary consisting two dataframes

            - "ecdf_df" : `pandas.DataFrame`
                The dataframe consists of a row for each combination of categories in
                 ``conditional_cols``. The columns are conditional_cols, {distribution_col}_mean,
                 {distribution_col}_std, {distribution_col}_ecdf_quantile_summary,
                 {distribution_col}_min, {distribution_col}_max and {distribution_col}_count (sample size).
            - "ecdf_df_overall" : `pandas.DataFrame`
                Has only one row with the overall mean, standard deviation,
                quantiles (no conditional_cols), min, max, overall sample size (count).
    """
    if quantiles is None:
        num = int(1.0 / quantile_grid_size - 1.0)
        quantiles = np.linspace(
            quantile_grid_size,
            1 - quantile_grid_size,
            num=num)

    def ecdf_quantile_summary(x):
        if remove_conditional_mean:
            y = x - np.mean(x)
        else:
            y = x
        return tuple(np.quantile(a=y, q=quantiles))

    # Aggregates w.r.t. given columns in conditional_cols
    agg_dict = {distribution_col: [ecdf_quantile_summary, "min", "mean", "max", "std", "count"]}
    # Creates an overall distribution by simply aggregating every value column globally.
    # This is useful for cases where no matching data is available for a given combination of values in conditional_cols
    ecdf_df_overall = df.groupby([True]*len(df), as_index=False).agg(agg_dict)
    # Flattens multi-index (result of aggregation) to have flat and descriptive column names.
    ecdf_df_overall.columns = [f"{metric}_{statistics}" if statistics else metric for (metric, statistics) in ecdf_df_overall.columns]
    if (conditional_cols is None) or (conditional_cols == []):
        ecdf_df = ecdf_df_overall
    else:
        ecdf_df = df.groupby(conditional_cols, as_index=False).agg(agg_dict)
        # Flattens multi-index (result of aggregation) to have flat and descriptive column names.
        ecdf_df.columns = [f"{metric}_{statistics}" if statistics else metric for (metric, statistics) in ecdf_df.columns]

    model_dict = {
        "ecdf_df": ecdf_df,
        "ecdf_df_overall": ecdf_df_overall}

    return model_dict
