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
"""Functions to construct lagged time series features."""
import numpy as np
import pandas as pd

from greykite.common import constants as cst


def build_lag_df(
        value_col,
        df=None,
        max_order=None,
        orders=None):
    """A function which builds a dataframe including time series lags
    (in the form of data frame columns)
    for a given value column (value_col) from an input dataframe (df)

    :param value_col: str
        the column name for the column which includes the values
    :param df: Optional[pd.DataFrame]
        data frame which include the value column of interest
        df could be passed as None if col_names are desired only
        in the output dictionary
    :param max_order: Optional[int]
        if lag orders are not specified then max_order is used.
        max_order is a natural number specifying
        the orders needed. e.g. if "max_order = 5"
        then we add these lag orders 1, 2, 3, 4, 5
    :param orders: List[int]
        a list of the lag orders needed.
        e.g. if orders = [1, 2, 7] for a given time series
        denoted in mathematical notation by Y(t), we calculate
        Y(t-1), Y(t-2), Y(t-7) and store them in the returned data frame
    :return: dict
        dictionary with these items:

            - "col_names": List[str]
                the generated column names
            - "lag_df": Optional[pd.DataFrame]
                a data frame consisting of the lagged values of the given orders.
                For example, if the value_col = "y" and the orders = [1, 2, 7],
                then the returned data frame
                include the following columns: "y_lag_1", "y_lag_2" and "y_lag_7".
                These correspond to the 1st lag, the 2nd lag and the 7th lag
                respectively.
    """
    # initializes the returned items
    lag_df = None
    col_names = []

    if max_order is None and orders is None:
        raise ValueError(
            "at least one of 'max_order' or 'orders' must be provided")

    if orders is None:
        orders = range(1, max_order + 1)

    if df is not None:
        lag_df = pd.DataFrame()

    for i in orders:
        col_name = f"{value_col}{cst.LAG_INFIX}{i}"
        col_names.append(col_name)
        if df is not None:
            lag_df[col_name] = df[value_col].shift(i)
    return {
        "lag_df": lag_df,
        "col_names": col_names}


def min_max_lag_order(
        lag_dict=None,
        agg_lag_dict=None):
    """Calculating min and max lag order needed given the prescribed lags
    in the model.

    :param lag_dict: dict
        dictionary with these fields:

            - "orders": Optional[List(int)]
             a list of positive integers or None
            - "max_order": Optional[int]

        see function "build_lag_df" arguments with same names
        to understand use case
    :param agg_lag_dict: dict
        dictionary with these items

            - "orders_list": List[List[int]], default: []
                a list of lists of integers
            - "interval_list": List[tuple[int]], default: []
                a list of int tuples of each with length 2

        see function "build_agg_lag_df" arguments with same names
        to understand use-case

    :return: dict[str, float]
        dictionary with two items with keys:

            - "max_order": the maximum lag used
            - "min_order": the mimimum lag used
    """
    max_order = 0
    min_order = np.inf

    if lag_dict is not None:
        if lag_dict.get("orders") is not None:
            max_order = np.nanmax(lag_dict["orders"])
            min_order = np.nanmin(lag_dict["orders"])
        elif lag_dict.get("max_order") is not None:
            max_order = np.nanmax([lag_dict["max_order"], max_order])
            # if max_order is not None in `lag_dict`
            # then lags 1, 2, ..., `max_order` are used
            # Therfore `min_order` is 1
            min_order = 1

    if agg_lag_dict is not None:
        orders_list = []
        if agg_lag_dict.get("orders_list") is not None:
            orders_list = orders_list + agg_lag_dict.get("orders_list")
        if agg_lag_dict.get("interval_list") is not None:
            orders_list = orders_list + agg_lag_dict.get("interval_list")
        flatten_orders = [order for sublist in orders_list for order in sublist]
        max_order = np.nanmax(flatten_orders + [max_order])
        min_order = np.nanmin(flatten_orders + [min_order])

    return {
        "max_order": max_order,
        "min_order": min_order}


def build_agg_lag_df(
        value_col,
        df=None,
        orders_list=[],
        interval_list=[],
        agg_func="mean",
        agg_name=cst.AGG_LAG_INFIX,
        max_order=None):
    """A function which returns a dataframe including aggregated
    (e.g. averaged) time series lags in the form of dataframe columns.
    By "aggregated lags", we mean an aggregate of several lags using an
    aggregation function given in "agg_func".
    The advantage of "aggregated lags" over regular lags is we can aggregate
    (e.g. average) many lags in the past instead of using a large number of lags.
    This is useful in many applications and avoids over-fitting.

    For a time series mathematically denoted by Y(t),
    one could consider the average lag processes as follows:
        the average of last 3 values:
            "avg(t) = (Y(t-1) + Y(t-2) + Y(t-3)) / 3"
        the average of 7th, 14th and 21st lags:
            "avg(t) = (Y(t-7) + Y(t-14) + Y(t-21)) / 3"

    See following references:
        Reza Hosseini et al. (2014)
        Non-linear time-varying stochastic models for agroclimate risk assessment,
        Environmental and Ecological Statistics
        https://link.springer.com/article/10.1007/s10651-014-0295-2

        Alireza Hosseini et al. (2017)
        Capturing the time-dependence in the precipitation process for weather risk assessment,
        Stochastic Environmental Research and Risk Assessment
        https://link.springer.com/article/10.1007/s00477-016-1285-8

    :param value_col: str
        the column name for the values of interest
    :param df: Optional[pd.DataFrame]
        the data frame which includes the time series of interest
    :param orders_list: List[int]
        a list including the order range for the average lags. For example if
        agg_func = np.mean and orders_list = [[1, 2, 3], [7, 14, 21]]
        then we construct two averaged lags:
            avg(t) = (Y(t-1) + Y(t-2) + Y(t-3)) / 3 and
            avg(t) = (Y(t-7) + Y(t-14) + Y(t-21)) / 3
    :param interval_list: List[tuple[int]]
        a list of (lag) intervals
        where interval is a tuple of length 2 with
            - first element denoting the lower bound and
            - second is the upper
        For example if interval_list = [(1, 3), (8, 11)]
        then we construct two "average lagged" variables:
            avg(t) = (Y(t-1) + Y(t-2) + Y(t-3)) / 3 and
            avg(t) = (Y(t-8) + Y(t-9) + Y(t-10) + Y(t-11)) / 4
    :param agg_func: "mean" or callable, default: "mean"
        the function used to aggregate the lag orders for each of
        orders specified in either of order_list or interval_list.
        Typically this function is an averaging function such as
        np.mean or np.median but more sophisticated functions are allowed.
        If "mean", uses `pandas.DataFrame.mean`.
    :param agg_name: str, default: "avglag"
        the aggregate function name used in constructing the column names for
        the output data frame.
        For example if
            - value_col = "y"
            - orders = [7 , 14, 21]
            - agg_name = "avglag"
        then the column name appearing in the output data frame
        will be "y_avglag_7_14_21".
    :param max_order: Optional[int]
        maximum order of lags needed in calculations of lag aggregates
        this is usually calculated/inferred from these arguments:
            orders_list, interval_list
        unless the max_order is already pre-calculated before calling
        this function. Hence this argument is optional and only included for
        computational efficiency gains.
    :return: dict
        dictionary with following items:
            - "col_names": List[str]
                the generated column names
            - "agg_lag_df": Optional[pd.DataFrame]
                a data frame with the average lag columns.
                The column names are constructed in a way
                that reflects what lags are averaged. For example if
                    - value_col = "y"
                    - agg_name = "avglag"
                    - orders_list = [[1, 2, 3], [7, 14, 21]]
                Then the column names are
                "y_avglag_1_2_3", "y_avglag_7_14_21"
                and if
                    - interval_list = [(1, 3), (8, 11)]
                Then the column names are
                "y_avglag_1_to_3", "y_avglag_8_to_11"
    """

    if orders_list is None and interval_list is None:
        raise ValueError(
            "at least one of 'orders_list' or 'interval_list' must be provided")
    # Finds out which lags we need by finding the maximum lag used.
    # Note that `max_order` should be usually passed as None (default)
    # unless it is pre-calculated before calling this function.
    if max_order is None:
        max_order = min_max_lag_order(
            lag_dict=None,
            agg_lag_dict={
                "orders_list": orders_list,
                "interval_list": interval_list})["max_order"]

    # intializes the returned items
    agg_lag_df = None
    col_names = []

    if df is not None:
        lag_info = build_lag_df(
            df=df.copy(),
            value_col=value_col,
            max_order=max_order,
            orders=None)
        lag_df = lag_info["lag_df"]
        agg_lag_df = pd.DataFrame()

    for orders in orders_list:
        if len(orders) > len(set(orders)):
            raise Exception(
                "a list of orders in orders_list contains a duplicate element")
        col_suffix = "_".join([str(x) for x in orders])
        orders_col_index = [x-1 for x in orders]
        col_name = f"{value_col}_{agg_name}_{col_suffix}"
        col_names.append(col_name)
        if df is not None:
            if agg_func == "mean":
                # uses vectorized mean for speed
                agg_lag_df[col_name] = (
                    lag_df.iloc[:, orders_col_index].mean(axis=1))
            else:
                # generic aggregation
                agg_lag_df[col_name] = (
                    lag_df.iloc[:, orders_col_index].apply(agg_func, axis=1))

    for interval in interval_list:
        if len(interval) != 2:
            raise Exception("interval must be a tuple of length 2")
        lower = interval[0]
        upper = interval[1]
        if lower > upper:
            raise Exception(
                "we must have interval[0] <= interval[1], "
                "for each interval in interval_list")
        orders = range(lower, upper + 1)
        col_suffix = f"{lower}_to_{upper}"
        orders_col_index = [x-1 for x in orders]
        col_name = f"{value_col}_{agg_name}_{col_suffix}"
        col_names.append(col_name)
        if df is not None:
            agg_lag_df[col_name] = (
                lag_df.iloc[:, orders_col_index].apply(agg_func, axis=1))

    return {
        "agg_lag_df": agg_lag_df,
        "col_names": col_names}


def build_autoreg_df(
        value_col,
        lag_dict=None,
        agg_lag_dict=None,
        series_na_fill_func=lambda s: s.bfill().ffill()):
    """This function generates a function ("build_lags_func" in the returned dict)
        which when called builds a lag data frame and an aggregated lag data frame using
        "build_lag_df" and "build_agg_lag_df" functions.
        Note: In case of training, validation and testing (e.g. cross-validation)
        for forecasting, this function needs to be applied after the data split is done.
        This is especially important if "series_na_fill_func" is using future values
        in interpolation - that is the case for the default which is
        lambda s: s.bfill().ffill()

   :param value_col: str
        the column name for the values of interest
   :param lag_dict: Optional[dict]
        A dictionary which encapsulates the needed params to be passed to the
        function "build_lag_df"
        Expected items are:

            - "max_order": Optional[int]
                the max_order for creating lags
            - "orders": Optional[List[int]]
                the orders for which lag is needed

    :param agg_lag_dict: Optional[dict]
        A dictionary encapsulating the needed params to be passed to the function
        "build_agg_lag_df"
        Expected items are:

            - "orders_list": List[List[int]]
                A list of list of integers.
                Each int list is to be used as order of lags to be aggregated
                See build_lag_df arguments for more details
            - "interval_list": List[tuple]
                A list of tuples each of length 2.
                Each tuple is used to construct an aggregated lag using all orders within that range
                See build_agg_lag_df arguments for more details
            - "agg_func": "mean" or func (pd.Dataframe -> pd.Dataframe)
                The function used for aggregation in "build_agg_lag_df"
                If this key is not passed, the default of "build_agg_lag_df" will be used.
                If "mean", uses `pandas.DataFrame.mean`.

    :param series_na_fill_func: (pd.Series -> pd.Series)
        default: lambda s: s.bfill.ffill()
        This function is used to fill in the missing data
        The default works by first back-filling and then forward-filling
        This function should not be applied to data before CV split is done.
    :return: dict
        a dictionary with following items

            - "build_lags_func": func
                pd.Daframe -> dict(lag_df=pd.DataFrame, agg_lag_df=pd.DataFrame)
                A function which takes a df (need to have value_col) as input
                calculates the lag_df and agg_lag_df and returns them
            - "lag_col_names": Optional[List[str]]
                The list of generated column names for the returned lag_df
                when "build_lags_func" is applied
            - "agg_lag_col_names": Optional[List[str]]
                The list of generated column names for returned agg_lag_df when
                "build_lags_func" is applied
            - "max_order": int
                the maximum lag order needed in the calculation of "build_lags_func"
            - "min_order": int
                the minimum lag order needed in the calculation of "build_lags_func"

    """
    # building arguments for passing to build_lag_df
    # when lag_dict is not None
    build_lag_df_args = None
    if lag_dict is not None:
        build_lag_df_args = {"value_col": value_col}
        build_lag_df_args.update(lag_dict)

    # building arguments for passing to build_agg_lag_df
    # when agg_lag_dict is not None
    build_agg_lag_df_args = None
    if agg_lag_dict is not None:
        build_agg_lag_df_args = {"value_col": value_col}
        build_agg_lag_df_args.update(agg_lag_dict)

    # we get the col_names for lag_df
    lag_col_names = None
    if lag_dict is not None:
        lag_info = build_lag_df(
            df=None,
            **build_lag_df_args)
        lag_col_names = lag_info["col_names"]

    # we get col_names for agg_lag_df
    agg_lag_col_names = None
    if agg_lag_dict is not None:
        agg_lag_info = build_agg_lag_df(
            df=None,
            **build_agg_lag_df_args)
        agg_lag_col_names = agg_lag_info["col_names"]

    # we find out the max_order needed
    # outside the internal function: build_lags_func
    min_max_order = min_max_lag_order(
        lag_dict=build_lag_df_args,
        agg_lag_dict=build_agg_lag_df_args)
    max_order = min_max_order["max_order"]
    min_order = min_max_order["min_order"]

    def build_lags_func(df, past_df=None):
        """A function which uses:
            df (pd.Dataframe), past_df (pd.DataFrame)
            and returns lag_df and agg_lag_df for df.
            This function infers some parameters
            e.g. value_col, max_order, series_na_fill_func from its environment.
        :param df: pd.DataFrame
             The input dataframe which is expected to have value_col as a column
             The returned lag_df and agg_lag_df are generated for df
             (past_df will be used in calculation if provided)
        :param past_df: Optional[pd.DataFrame]
            When provided it will be appended to df (from left)
            in order for the lags to be calculated.
            past_df is considered to include the past values for the time series
            leading up to df values. So if df values start at time t0:
                Y(t0), ..., Y(t0 + len(df))
            past_df will include these values
                Y(t-len(past_df)), ..., Y(t0-1)
            Note that the last value in past_df is the immediate value in time
            before t0 which is the first time in df.
            Also note that we do not require a timestamp column in df and past_df
            as that is not needed in the logic.
            If past_df is None, and series_na_fill_func is also None.
            Therefore lag_df and agg_lag_df will include NULLs at the beginning
            depending on how many lags are calculated
        :return: dict
            A dictionary including lag_df and agg_lag_df for the input df
            The dictionary includes following items:
                - "lag_df": Optional[pd.DataFrame]
                - "agg_lag_df": Optional[pd.DataFrame]
        """
        df = df[[value_col]].reset_index(drop=True)

        # if past_df is None, we create one with np.nan
        # also if it is shorter than max_order we expand it with np.nan
        if past_df is None:
            past_df = pd.DataFrame({value_col: [np.nan]*max_order})
        else:
            if value_col not in list(past_df.columns):
                raise ValueError(
                    f"{value_col} must appear in past_df if past_df is not None")
            past_df = past_df[[value_col]].reset_index(drop=True)
            # if past_df length (number of rows) is smaller than max_order
            # we expand it to avoid NULLs
            if past_df.shape[0] < max_order:
                past_df_addition = pd.DataFrame(
                    {value_col: [np.nan]*(max_order - past_df.shape[0])})
                past_df = past_df_addition.append(past_df)

        # df is expanded by adding past_df as the past data for df
        # this will help in avoiding NULLs to appear in lag_df and agg_lag_df
        # as long as past_df has data in it or expanded df is interpolated
        df_expanded = past_df.append(df)
        if series_na_fill_func is not None:
            df_expanded[value_col] = series_na_fill_func(df_expanded[value_col])

        # we get the col_names for lag_df
        lag_df = None
        if lag_dict is not None:
            lag_info = build_lag_df(
                df=df_expanded,
                **build_lag_df_args)
            # since the lag calculation is done on expanded dataframe (`df_expanded`)
            # we need to pick only the relevant rows which match original
            # dataframe (via `iloc`)
            lag_df = lag_info["lag_df"].iloc[-(df.shape[0]):].reset_index(
                drop=True)
            # cast dtype to 'bool' if original dtype is 'bool', need to make sure there is no NaN
            # there is an edge case where ``df[value_col]`` is NaN but ``past_df[value_col]`` is not
            # e.g., in predict phase with autoregression, ``df_fut[value_col]`` would be NaN
            if (df[value_col].dtype == 'bool' or past_df[value_col].dtype == 'bool') and lag_df.isna().sum().sum() == 0:
                lag_df = lag_df.astype('bool')

        # we get col_names for agg_lag_df
        agg_lag_df = None
        if agg_lag_dict is not None:
            agg_lag_info = build_agg_lag_df(
                df=df_expanded,
                **build_agg_lag_df_args)
            # since the lag calculation is done on expanded dataframe (`df_expanded`)
            # we need to pick only the relevant rows which match original
            # dataframe (via `iloc`)
            agg_lag_df = agg_lag_info["agg_lag_df"].iloc[
                -(df.shape[0]):].reset_index(drop=True)

        return {
            "lag_df": lag_df,
            "agg_lag_df": agg_lag_df
        }

    return {
        "build_lags_func": build_lags_func,
        "lag_col_names": lag_col_names,
        "agg_lag_col_names": agg_lag_col_names,
        "min_order": min_order,
        "max_order": max_order
    }


def build_autoreg_df_multi(
        value_lag_info_dict,
        series_na_fill_func=lambda s: s.bfill().ffill()):
    """A function which returns a function to build autoregression
    dataframe for multiple value columns.
    This function should not be applied to data before CV split is done.


    Parameters
    ----------
    value_lag_info_dict : `dict` [`str`, `dict`]
        A dictionary with keys being the target value columns: `value_col`
        For each of these value columns, a dictionary with following keys

            `lag_dict`,
            `agg_lag_dict`,
            `series_na_fill_func`

        The `value_col` and the above three variables are then passed to the
        following function:

            build_autoreg_df(
                value_col,
                lag_dict,
                agg_lag_dict,
                series_na_fill_func)

        Check the
        `greykite.common.features.timeseries_lags.build_autoreg_df`
        docstring for more details for each argument.
    series_na_fill_func : callable, (pd.Series -> pd.Series)
        default: `lambda s: s.bfill.ffill()`
        This function is used to fill in the missing data
        The default works by first back-filling and then forward-filling

    Returns
    -------
    A dictionary with following items

        "autoreg_func" : callable, (pd.DataFrame -> pd.DataFrame)
            A function which can be applied to a dataframe and return a dataframe
            which has the lagged values for all the relevant columns
        "autoreg_col_names" : List[str]
            A list of all the generated columns
        "autoreg_orig_col_names" : List[str]
            A list of all the original target value columns
        "max_order" : int
            Maximum lag order for all target value columns
        "min_order" : int
            Minimum lag order for all target value columns

    """

    multi_autoreg_info = {}
    autoreg_col_names = []
    autoreg_orig_col_names = list(value_lag_info_dict.keys())
    min_order = np.inf
    max_order = 0

    for value_col, lag_info in value_lag_info_dict.items():
        # we assign the interpolation function to be the default specified above
        # in this function
        # if a custom interpolation function is made available for that
        # `value_col`, we replace the default
        series_na_fill_func0 = lag_info.get(
            "series_na_fill_func",
            series_na_fill_func)

        autoreg_info = build_autoreg_df(
            value_col,
            lag_dict=lag_info.get("lag_dict"),
            agg_lag_dict=lag_info.get("agg_lag_dict"),
            series_na_fill_func=series_na_fill_func0)

        # store the result (`autoreg_info`) for each `value_col` in a dictionary
        # the result for each `value_col` will be
        # a dictionary with following keys:
        # "build_lags_func"
        # "lag_col_names"
        # "agg_lag_col_names"
        # "min_order"
        # "max_order"
        multi_autoreg_info[value_col] = autoreg_info

        # extract column names for lagged variables and add to the full list of
        # all lagged variable column names: `autoreg_col_names`
        lag_col_names = autoreg_info["lag_col_names"]
        agg_lag_col_names = autoreg_info["agg_lag_col_names"]
        if lag_col_names is not None:
            autoreg_col_names += lag_col_names
        if agg_lag_col_names is not None:
            autoreg_col_names += agg_lag_col_names

        # extract the min_order and max_order for each col and update the overall lagged_regressor_order
        min_order = np.nanmin([min_order, autoreg_info["min_order"]])
        max_order = np.nanmax([max_order, autoreg_info["max_order"]])

    def build_lags_func_multi(df, past_df=None):
        """A function which generates a lagged dataframe
        for a given dataframe with potentially multiple value columns
        to be lagged. Note `df` and `past_df` must have the same extract
        columns. However time column is not needed as the function assumes
        `past_df` is the data which precedes `df` (without gaps).

        Parameters
        ----------
        df : `pd.DataFrame`
            The dataframe which includes the value columns for which lagged
            data is needed
        past_df : `pandas.DataFrame` or None, default None
            The past data which is immediately before `df` with same
            value columns.


        Returns :
        -------
        autoreg_df : `pandas.DataFrame`
            A DataFrame which includes the lagged values as columns
        """

        autoreg_df = None
        lag_dfs = []
        for value_col, lag_info in multi_autoreg_info.items():
            build_lags_func = lag_info["build_lags_func"]
            res = build_lags_func(
                df=df[[value_col]],
                past_df=past_df)
            lag_df = res["lag_df"]
            agg_lag_df = res["agg_lag_df"]
            lag_dfs.append(lag_df)
            lag_dfs.append(agg_lag_df)

        try:
            # Concatenates the columns to the result
            # dataframe in column-wise fashion
            autoreg_df = pd.concat(lag_dfs, axis=1)
        except ValueError:  # All objects passed are None
            autoreg_df = None

        return autoreg_df

    return {
        "autoreg_func": build_lags_func_multi,
        "autoreg_col_names": autoreg_col_names,
        "autoreg_orig_col_names": autoreg_orig_col_names,
        "min_order": min_order,
        "max_order": max_order
    }
