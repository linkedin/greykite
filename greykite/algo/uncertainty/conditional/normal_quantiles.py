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
"""Calculates normal quantiles for each row of dataframe."""

from scipy import stats


def normal_quantiles_df(
        df,
        std_col,
        mean_col=None,
        fixed_mean=0.0,
        quantiles=(0.025, 0.975),
        quantile_summary_col="normal_quantiles"):
    """Calculates normal quantiles for each row of ``df``
    using available means (either given in ``mean_col`` or ``fixed_mean``) and
    standard deviations (given in ``std_col``).

    Parameters
    ----------
    df : `pandas.DataFrame`
        A DataFrame containing the standard deviations.
    fixed_mean : `float`, default 0
        Fixed sample mean to add to each row of ``df``.
        It is only used if ``mean_col`` is None.
    mean_col : `str` or None, default None
        The column with the sample means for each row of ``df``.
    std_col : `str`
        The column with the sample std for each row
    quantiles : `list` [`float`]
        The normal quantiles to be calculated for each row of ``df``.
    quantile_summary_col : `str`, default "normal_quantiles"
        The column name where the computed quantiles are stored.

    Returns
    -------
    df : `pandas.DataFrame`
        ``df`` augmented with normal quantiles for each row.
    """
    df = df.copy()
    if mean_col is None:
        df[quantile_summary_col] = df.apply(
            lambda row: stats.norm.ppf(
                loc=fixed_mean,
                scale=row[std_col],
                q=quantiles),
            axis=1)
    else:
        df[quantile_summary_col] = df.apply(
            lambda row: stats.norm.ppf(
                loc=row[mean_col],
                scale=row[std_col],
                q=quantiles),
            axis=1)

    return df
