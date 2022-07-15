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
"""Functions to impute missing data using past data."""
import numpy as np

from greykite.common.features.timeseries_lags import build_agg_lag_df


def impute_with_lags(
        df,
        value_col,
        orders,
        agg_func="mean",
        iter_num=1):
    """A function to impute timeseries values (given in ``df``) and in ``value_col``
    with chosen lagged values or an aggregated of those.
    For example for daily data one could use the 7th lag to impute using the
    value of the same day of past week as opposed to the closest value available
    which can be inferior for business related timeseries.

    The imputation can be done multiple times by specifying ``iter_num``
    to decrease the number of missing in some cases.
    Note that there are no guarantees to impute all missing values with this
    method by design.
    However the original number of missing values and the final number of missing
    values are returned by the function along with the imputed dataframe.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Input dataframe which must include `value_col` as a column.
    value_col : `str`
        The column name in ``df`` representing the values of the timeseries.
    orders : list of `int`
        The lag orders to be used for aggregation.
    agg_func : "mean" or callable, default: "mean"
        `pandas.Series` -> `float`
        An aggregation function to aggregate the chosen lags.
        If "mean", uses `pandas.DataFrame.mean`.
    iter_num : `int`, default `1`
        Maximum number of iterations to impute the series.
        Each iteration represent an imputation of the series using the provided
        lag orders (``orders``) and return an imputed dataframe.
        It might be the case that with one iterations some values
        are not imputed but with more iterations one can achieve more
        imputed values.

    Returns
    -------
    impute_info : `dict`
        A dictionary with following items:

        "df" : `pandas.DataFrame`
            A dataframe with the imputed values.
        "initial_missing_num" : `int`
            Initial number of missing values.
        "final_missing_num" : `int`
            Final number of missing values after imputations.
    """
    df = df.copy()
    nan_index = df[value_col].isnull()
    missing_num = sum(nan_index)
    initial_missing_num = missing_num

    def nan_agg_func(x):
        """To avoid warnings due to aggregation performed on sub-series
        with only NAs, we define an internal modified version of ``agg_func``.

        Note that `np.mean` passed to `pandas.DataFrame.apply` ignores NAs and
        already has the same effect as `np.nanmean`.
        """
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        else:
            return agg_func(x)

    safe_agg_func = agg_func if isinstance(agg_func, str) else nan_agg_func

    i = 0
    while i < iter_num and missing_num > 0:
        agg_lag_info = build_agg_lag_df(
            value_col=value_col,
            df=df,
            orders_list=[orders],
            interval_list=[],
            agg_func=safe_agg_func,
            agg_name="avglag")

        agg_lag_df = agg_lag_info["agg_lag_df"]
        # In this case the lagged dataframe will have only one column
        # which can be extracted from the list given in the ``"col_names"`` item
        agg_col_name = agg_lag_info["col_names"][0]
        df.loc[nan_index, value_col] = agg_lag_df.loc[nan_index, agg_col_name]
        nan_index = df[value_col].isnull()
        missing_num = sum(nan_index)
        i += 1

    return {
        "df": df,
        "initial_missing_num": initial_missing_num,
        "final_missing_num": missing_num}


def impute_with_lags_multi(
        df,
        orders,
        agg_func=np.mean,
        iter_num=1,
        cols=None):
    """Imputes every column of ``df`` using
    `~greykite.common.features.timeseries_impute.impute_with_lags`.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Input dataframe which must include `value_col` as a column.
    orders : list of `int`
        The lag orders to be used for aggregation.
    agg_func : callable, default `np.mean`
        `pandas.Series` -> `float`
        An aggregation function to aggregate the chosen lags.
    iter_num : `int`, default `1`
        Maximum number of iterations to impute the series.
        Each iteration represent an imputation of the series using the provided
        lag orders (``orders``) and return an imputed dataframe.
        It might be the case that with one iterations some values
        are not imputed but with more iterations one can achieve more
        imputed values.
    cols : `list` [`str`] or None, default None
        Which columns to impute. If None, imputes all columns.

    Returns
    -------
    impute_info : `dict`
        A dictionary with following items:

        "df" : `pandas.DataFrame`
            A dataframe with the imputed values.
        "missing_info" : `dict`
            Dictionary with information about the missing info.

            Key = name of a column in ``df``
            Value = dictionary containing:

                "initial_missing_num" : `int`
                    Initial number of missing values.
                "final_missing_num" : `int`
                    Final number of missing values after imputation.
    """
    if cols is None:
        cols = df.columns
    impute_info = {
        "df": df.copy(),
        "missing_info": {}
    }
    for col in cols:
        # updates each column with the imputed values
        col_impute_info = impute_with_lags(
            df=df,
            value_col=col,
            orders=orders,
            agg_func=agg_func,
            iter_num=iter_num,
        )
        impute_info["df"][col] = col_impute_info["df"][col]
        impute_info["missing_info"][col] = {
            "initial_missing_num": col_impute_info["initial_missing_num"],
            "final_missing_num": col_impute_info["final_missing_num"],
        }
    return impute_info
