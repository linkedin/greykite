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
        value_col,
        quantile_grid_size=0.05,
        quantiles=None,
        conditional_cols=None):
    """Estimates an empirical distribution for a given variable in the dataframe (df) "value_col" column
    The distribution is an approximated quantile function defined on
    the quantiles specified in "quantiles" which are values in [0, 1]
    The function returns, mean, and standard deviation (std) as well.

    :param df: the dataframe which includes the dataframe
    :param value_col: the column name for the value column of interest
    :param quantile_grid_size: the grid size for the quantiles.
        E.g. if grid_size = 0.25 then the quantiles for 0.25, 0.5, 0.75 are calculated
        If grid_size = 0.05 then then the quantiles for 0.05, 0.1, ..., 0.90, 0.95 are calculated
    :param quantiles: the probability grid for which quantiles are calculated.
        If None quantile_grid_size is used to generate a regularly spaced one.
    :param conditional_cols: the grouping variables used to condition on

    :return: The output consists of a dictionary with two dataframes
        (1) first dataframe ("ecdf_df") with columns being
            conditional_cols
            mean, sd
            quantiles (as many as specified in quantiles)
            min, max
            sample_size (count) for each combination
        (2) second dataframe ("ecdf_df_overall") has only one row and returns
            the overall mean, sd, quantiles (no conditional_cols), min, max, overall sample size (count)
    """
    if quantiles is None:
        num = int(1.0 / quantile_grid_size - 1.0)
        quantiles = np.linspace(
            quantile_grid_size,
            1 - quantile_grid_size,
            num=num)

    def quantile_summary(x):
        return tuple(np.quantile(a=x, q=quantiles))

    # Aggregates w.r.t. given columns in conditional_cols
    agg_dict = {value_col: [quantile_summary, "min", "mean", "max", "std", "count"]}
    # Creates an overall distribution by simply aggregating every value column globally.
    # This is useful for cases where no matching data is available for a given combination of values in conditional_cols
    ecdf_df_overall = df.groupby([True]*len(df), as_index=False).agg(agg_dict)
    # Flattens multi-index (result of aggregation) to have flat and descriptive column names.
    ecdf_df_overall.columns = [
        f"{value_col}_{x}" for x in ["quantile_summary", "min", "mean", "max", "std", "count"]]
    if conditional_cols is None:
        ecdf_df = ecdf_df_overall
    else:
        ecdf_df = df.groupby(conditional_cols, as_index=False).agg(agg_dict)
        # Flattens multi-index (result of aggregation) to have flat and descriptive column names.
        ecdf_df.columns = [f"{a}_{b}" if b else a for (a, b) in ecdf_df.columns]

    model = {
        "ecdf_df": ecdf_df,
        "ecdf_df_overall": ecdf_df_overall}

    return model
