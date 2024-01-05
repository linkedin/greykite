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
# original author: Sayan Patra, Reza Hosseini
from functools import reduce

import pandas as pd

from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.features.outlier import ZScoreOutlierDetector
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.detection.detector.constants import Z_SCORE_CUTOFF


def partial_return(func, k):
    """For a given function ``func`` which returns multiple outputs accessible by
    ``[]`` e.g. python list or dictionary, it construct a new function which only
    returns part of the output given in position `k` where key can be a key or
    other index.

    Parameters
    ----------
    func : callable
        A function which returns multiple output accessible by ``[]``
    k : `int` or `str`
        A key or index which is implemented by ``[]`` on output of the
        input function ``func``

    Returns
    -------
    result : callable
        A function which only returns part of what the input ``func`` returns given
        in position ``[k]``. In the case that the key does not exist or index is
        out of bound, it returns None.
    """
    def func_k(*args, **kwargs):
        res = func(*args, **kwargs)
        # If result is a dictionary we check if key `k` exist,
        # If so return the value for that key.
        if type(res) == dict:
            if k in res.keys():
                return res[k]
        # This is for the non-dict case eg `list`, `pandas.Series`, `np.array`.
        # In this case we check if `k` is an `integer` and `res` has sufficient length.
        # If so, we return the k-th element.
        elif type(k) == int and len(res) > k:
            return res[k]
        # Otherwise it returns None.
        return None

    return func_k


def vertical_concat_dfs(
        df_list,
        join_cols,
        common_value_cols=[],
        different_value_cols=[]):
    """For a given set of datarfames with same columns
    in ``df_dict``, it will concat them vertically by using
    ``join_cols`` as joining columns.

    For ``common_value_cols`` it only extract the columns from the first
    dataframe.

    For ``different_value_cols`` it will extract them for each
    df and concat them horizontally. The new column names will
    have an added index based on their order in ``df_list``.

    Parameters
    ----------
    df_list : `list` [`pandas.DataFrame`]
        A list of dataframes which are to be concatenated.
    join_cols : `list` [`str`]
        The list of columns which are to be used for joining.
    common_value_cols : `list` [`str`], default ``[]``
        The list of column names for which we assume the values are the
        same across dataframes in ``df_list``. For these columns only data
        is pulled from the first dataframe appearing in ``df_list`.`
    different_value_cols : `list` [`str`], default ``[]``
        The list of columns which are assumed to have potentially different
        values across dataframes.

    Returns
    -------
    result : `pd.DataFrame`
        The resulting concatenated dataframe.
    """
    new_df_list = []
    for i, df in enumerate(df_list):
        df = df.copy()
        # only keeps the `common_value_cols` from the first df
        if i != 0:
            for col in common_value_cols:
                del df[col]

        for col in different_value_cols:
            df[f"{col}{i}"] = df[col]
            del df[col]
        new_df_list.append(df)

    concat_df = reduce(
        lambda left, right: pd.merge(left, right, on=join_cols),
        new_df_list)

    return concat_df


def add_new_params_to_records(
        new_params,
        records=None):
    """For a list of records (each being a `dict`) and a set of parameters
    each having a list of potential values, it expands each record in all possible
    ways based on all possible values for each param in ``new_params``.
    Then it returns all possible augmented records in a list.

    Parameters
    ----------
    new_params : `dict` {`str`: `list`}
        A dictionary with keys representing (new) variables and values for each
        key being the possible values for that variable.
    records : `list` [`dict`] or None, default None
        List of existing records which are to be augmented with all possible
        combinations of the new variables. If None, it is assigned to ``[{}]``
        which means we start from an empty record.

    Returns
    -------
    expanded_records : `list` [`dict`]
        The resulting list of augmented records.
    """
    if records is None:
        records = [{}]

    def add_new_param_values(name, values, records):
        # `records` is a list and it is copied so that its not altered
        # Note that `deepcopy` is not possible for lists
        # Therefore inside the for loop we copy each param (`dict`)
        records = records.copy()
        expanded_records = []
        for param in records:
            for v in values:
                # Copies to avoid over-write
                expanded_param = param.copy()
                expanded_param.update({name: v})
                expanded_records.append(expanded_param)
        return expanded_records

    expanded_records = records.copy()
    for name, values in new_params.items():
        expanded_records = add_new_param_values(
            name=name,
            values=values,
            records=expanded_records)

    return expanded_records


def get_anomaly_df(
        df,
        time_col=TIME_COL,
        anomaly_col=ANOMALY_COL):
    """Computes anomaly dataframe from a labeled ``df``.

    Parameters
    ----------
    df : `pandas.DataFrame`
        A data frame which includes minimally
            - the timestamp column (``time_col``)
            - the anomaly column (``anomaly_col``).
    time_col : `str` or None
        The column name of timestamps in ``df``.
        If None, it is set to
        `~greykite.common.constants.TIME_COL`.
    anomaly_col : `str` or None
        The column name of anomaly labels in ``df``.
        ``True`` indicates anomalous data.
        ``False`` indicates non-anomalous data.
        If None, it is set to
        `~greykite.detection.detector.constants.ANOMALY_COL`.

    Returns
    -------
    anomaly_df : `pandas.DataFrame`
        The dataframe that contains anomaly info.
        It should have
            - the anomaly start column "start_time"
            `~greykite.detection.detector.constants.ANOMALY_START_TIME`.
            - the anomaly end column "end_time"
            `~greykite.detection.detector.constants.ANOMALY_END_TIME`.
        Both should be inclusive.
    """
    # Copies `df` by resetting the index to avoid alteration to input df
    df = df.reset_index(drop=True)
    # When all rows are True/ anomalies
    if df[anomaly_col].all():
        start_index = [0]
        end_index = [df.index[-1]]
    # When all rows are False/ not anomalies
    elif not df[anomaly_col].any():
        start_index = []
        end_index = []
    else:
        df[anomaly_col] = df[anomaly_col].astype(int)
        df[f"{anomaly_col}_diff"] = df[anomaly_col].diff()

        start_index = df.index[df[f"{anomaly_col}_diff"] == 1.0].tolist()
        end_index = df.index[df[f"{anomaly_col}_diff"] == -1.0].tolist()
        end_index = [index-1 for index in end_index]  # to make end points inclusive

        # The first entry of df is an anomaly
        if df.iloc[0][anomaly_col]:
            start_index.insert(0, 0)

        # The last entry of df is an anomaly
        if df.iloc[df.index[-1]][anomaly_col]:
            end_index.append(df.index[-1])

    anomaly_df = pd.DataFrame({
        START_TIME_COL: df.iloc[start_index][time_col].values,
        END_TIME_COL: df.iloc[end_index][time_col].values
    })

    return anomaly_df


def get_canonical_anomaly_df(
        anomaly_df,
        freq,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL):
    """Validates and merges overlapping anomaly periods in anomaly dataframe.
    Also standardizes column names.

    For example, consider the following input ``anomaly_df``:
     start_time       end_time
    "2020-01-01"    "2020-01-02"
    "2020-01-03"    "2020-01-05"

    For a daily dataset i.e. ``freq = "D"``, the end time "2020-01-02" and start time
    "2020-01-03" are consecutive. Hence, in the output the ``anomaly_df`` is converted to
    start_time       end_time
    "2020-01-01"    "2020-01-05"

    However, for an hourly dataset i.e. ``freq = "H"`` the end time "2020-01-02" and start time
    "2020-01-03" are not consecutive. Hence, the output is the same as the input ``anomaly_df``.

    Parameters
    ---------
    anomaly_df : `pandas.DataFrame`
        The dataframe that contains anomaly info.
        It should at least have
            - the anomaly start column ``start_time_col``
            - the anomaly end column ``end_time_col``
        Both are assumed to be inclusive of the start and end times.
    freq : `str`
        Frequency of the timeseries represented in ``anomaly_df``.
        This is used to determine if the timestamps in the ``anomaly_df`` are consecutive.
    start_time_col : `str` or None
        The column name containing anomaly start timestamps in ``anomaly_df``.
        If None, it is set to
        `~greykite.detection.detector.constants.START_TIME_COL`.
    end_time_col : `str` or None
        The column name containing anomaly end timestamps in ``anomaly_df``.
        If None, it is set to
        `~greykite.detection.detector.constants.END_TIME_COL`.
    Returns
    -------
    anomaly_df : `pandas.DataFrame`
        Standardized anomaly dataframe.
        It should have
            - the anomaly start column "start_time"
            `~greykite.detection.detector.constants.ANOMALY_START_TIME`.
            - the anomaly end column "end_time"
            `~greykite.detection.detector.constants.ANOMALY_END_TIME`.
        Both should be inclusive.
        The anomaly periods are non-overlapping and sorted from earliest to latest.
    """
    df = anomaly_df.copy()
    df[start_time_col] = pd.to_datetime(df[start_time_col])
    df[end_time_col] = pd.to_datetime(df[end_time_col])
    df = df.sort_values(by=[start_time_col]).reset_index(drop=True)
    row_num = df.shape[0]-1
    for row in range(row_num):
        start_time = df[start_time_col][row]
        end_time = df[end_time_col][row]
        if start_time > end_time:
            raise ValueError(f"Anomaly 'start_time' ({start_time}) is after the anomaly 'end_time' ({end_time}).")
        # Merges anomalies
        next_start_time = df[start_time_col][row+1]
        next_end_time = df[end_time_col][row+1]
        num_periods = (next_start_time.to_period(freq=freq) - end_time.to_period(freq=freq)).n
        # Start times and end times are inclusive for anomaly df. Hence, the anomaly periods should
        # be merged if the number of periods between the anomalies are less than 1.
        # e.g. The anomaly periods ["2020-01-01", "2020-01-02"] and ["2020-01-03", "2020-01-04"]
        # should be merged into a single anomaly period ["2020-01-01", "2020-01-04"].
        if num_periods <= 1:
            df[start_time_col][row+1] = start_time
        if next_end_time < end_time:
            df[end_time_col][row+1] = end_time
    df = df.drop_duplicates(
        subset=[start_time_col],
        keep="last"
    ).rename({
        start_time_col: START_TIME_COL,
        end_time_col: END_TIME_COL
    }, axis=1).reset_index(drop=True)
    return df


def optimize_df_with_constraints(
        df,
        objective_col,
        constraint_col,
        constraint_value):
    """Function that solves the following constrained optimization problem.

        maximize ``df``[``objective_col``]
        subject to ``df``[``constraint_col``] >= ``constraint_value``.

    However, unlike traditional constrained optimization, which returns None when no
    values satisfy the constraint, this function maximizes ``df``[``constraint_col``].
    Note that in this case, since the constraint is not satisfied, this will get
    as close as possible to the ``constraint_value``.

    To understand the reasoning behind this choice, it is helpful to think
    about ``objective_col`` as precision and ``constraint_col`` as recall. Thus,
    the optimization problem becomes:
    maximize precision subject to recall >= target_recall.

    The algorithm proceeds as follows:
    1. Find rows which satisfy ``df``[``constraint_col``] >= ``constraint_value``.
        1.1. If such rows exist, find rows that have highest ``df``[``objective_col``].
            1.1.1. Find rows that have highest ``df``[``constraint_col``].
            1.1.2. Among these, find row that maximize ``df``[``objective_col``].
                   This solves for multiple ties, if any.
        1.2. If no such rows exist,
            1.2.1. Find rows that maximizes ``df``[``constraint_col``].
            1.2.2. Among these, find row that maximize ``df``[``objective_col``].
                   This solves for multiple ties, if any.
   2. Return corresponding ``df`` row.

    Parameters
    ----------
    df : `pandas.DataFrame`
        A data frame which includes minimally
            - the objective column (``objective_col``)
            - the constraint column (``constraint_col``).
    objective_col : `str`
        The column name of the variable to be optimized.
    constraint_col : `str`
        The column name of the constraint variable.
    constraint_value : `float`
        The value of the constraint.

    Returns
    -------
    optimal_dict : `dict`
        The row of the ``df`` which is the optimal of the
        corresponding optimization problem..
    """
    df = df.copy()
    constraint_match_indices = df[constraint_col] >= constraint_value
    if constraint_match_indices.any():
        log_message(f"Values satisfying the constraint are found.\n"
                    f"Solving the following optimization problem:\n"
                    f"Maximize {objective_col} subject to {constraint_col} >= {constraint_value}.",
                    LoggingLevelEnum.INFO)
        df = df[constraint_match_indices]
        df = df[df[objective_col] == max(df[objective_col])]
        df = df[df[constraint_col] == max(df[constraint_col])]
    else:
        log_message(f"No values satisfy the constraint.\n"
                    f"Maximizing ``constraint_col`` ({constraint_col}) so that it is as "
                    f"close as possible to the ``constraint_value`` ({constraint_value}).",
                    LoggingLevelEnum.INFO)
        df = df[df[constraint_col] == max(df[constraint_col])]
        df = df[df[objective_col] == max(df[objective_col])]

    return df.iloc[-1].to_dict()


def validate_volatility_features(
        volatility_features_list,
        valid_features=None):
    """Removes duplicate values from ``volatility_features_list``
     and validates the features against ``valid_features``.

     Parameters
     ----------
     volatility_features_list: `list` [`list` [`str`]]
        Lists of volatility features used to optimize anomaly detection performance.
        Valid volatility feature column names are
        either columns of ``df`` or belong to
        `~greykite.common.constants.TimeFeaturesEnum`.

     valid_features: `list` [`str`] or None
        ``volatility_features_list`` is validated against this list.

    Returns
    -------
    validated_features_list: `list` [`list` [`str`]]
        List of validated volatility features.
     """
    # Removes duplicates
    validated_features_list = []
    for features in volatility_features_list:
        # Removes duplicates within a set of features
        features = list(dict.fromkeys(features))
        # Removes duplicates among the feature sets
        if features not in validated_features_list:
            validated_features_list.append(features)

    # Checks features against the provided features in ``valid_features``
    if valid_features is not None:
        all_features = sum(validated_features_list, [])
        unique_features = set(all_features)
        missing_features = unique_features - set(valid_features)
        if missing_features:
            raise ValueError(f"Unknown feature(s) ({missing_features}) in `volatility_features_list`. "
                             f"Valid features are: [{valid_features}].")

    return validated_features_list


def get_timestamp_ceil(ts, freq):
    """Returns the smallest timestamp that is greater than or equal to `ts`
    and is also a multiple of the `freq`.
    Assume hourly frequency i.e. `freq` = "H". Then
    If `ts` = 1:30, this function returns 2:00.
    If `ts` = 1:00, this function returns 1:00.

    Parameters
    ----------
    ts: `str`
        Timestamp in `str` format.
    freq: `str`
        Pandas timeseries frequency string.

    Returns
    -------
    dt_ceil: `pd.Timestamp`
        The smallest timestamp that is greater than or equal to `ts`
        and is also a multiple of the `freq`.
    """
    dt = pd.to_datetime(ts)
    try:
        return dt.ceil(freq=freq)
    # `pd.Timestamp.ceil` raises a ValueError when `freq` is a non-fixed frequency
    # e.g. weekly ("W-MON"), business day ("B) or monthly ("M")
    except ValueError:
        return dt.to_period(freq).to_timestamp(how="E").normalize()


def get_timestamp_floor(ts, freq):
    """Returns the largest timestamp that is smaller than or equal to `ts`
    and is also a multiple of the `freq`.
    Assume hourly frequency i.e. `freq` = "H". Then
    If `ts` = 1:30, this function returns 1:00.
    If `ts` = 1:00, this function returns 1:00.

    Parameters
    ----------
    ts: `str`
        Timestamp in `str` format.
    freq: `str`
        Pandas timeseries frequency string.

    Returns
    -------
    dt_floor: `pd.Timestamp`
        The largest timestamp that is smaller than or equal to `ts`
        and is also a multiple of the `freq`.
    """
    dt_ceil = get_timestamp_ceil(ts, freq)
    # If input `ts` is not on the `freq` offset, `dt_ceil` > `ts`.
    # e.g. Assume `freq` = "H". If `ts` = 1:30, then `dt_ceil` = 2:00.
    # Then `dt_ceil` is reduced one `freq` offset to get `dt_floor`.
    if dt_ceil > pd.to_datetime(ts):
        return dt_ceil - pd.tseries.frequencies.to_offset(freq)
    else:
        return dt_ceil


def get_anomaly_df_from_outliers(
        df,
        time_col,
        value_col,
        freq,
        z_score_cutoff=Z_SCORE_CUTOFF,
        trim_percent=1.0):
    """This function identifies extreme values as outliers based on z-scores.
    A normal distribution will be fit on ``value_col`` of input df,
    and the time points with corresponding values that satisfy abs(z-scores) >
    ``Z_SCORE_CUTOFF`` will be considered as outliers.
    The function will then construct ``anomaly_df`` based on identified outliers.
    If trimming is specified via ``trim_percent`` to be non-zero,
    data is trimmed in symmetric fashion (removes high and low values) before
    calculating mean and variance of the standard normal.
    This is done to deal with large outliers.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The dataframe with ``time_col`` and ``value_col``.
    time_col : `str`
        The column name of timestamps in ``df``.
    value_col : `str`
        The column name of values in ``df`` on which z-scores will be calculated to identify outliers.
    freq : `str`
        Pandas timeseries frequency string, e.g. "H", "D", etc.
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
    z_score_cutoff : `float`, default ``Z_SCORE_CUTOFF``
        Z score cutoff for outlier detection.
    trim_percent : `float`, default 1.0
        Trimming percent for calculating the variance.
        The function first removes this amount of data in symmetric fashion from
        both ends and then it calculates the mean and the variance.

    Returns
    -------
    anomaly_df : `pandas.DataFrame`
        The dataframe that contains anomaly info based on identified outliers.
        It should have

            - the anomaly start column "start_time"
                `~greykite.common.constants.START_TIME_COL`.
            - the anomaly end column "end_time"
                `~greykite.common.constants.END_TIME_COL.

        Both are inclusive.
    """
    if df.empty:
        return None

    if time_col not in df.columns:
        raise ValueError(f"`df` does not have `time_col` with name {time_col}.")

    if value_col not in df.columns:
        raise ValueError(f"`df` does not have `value_col` with name {value_col}.")

    df[time_col] = pd.to_datetime(df[time_col])

    # Calculates z-scores after trimming (if trimming is not 0)
    # and identifies points with abs(z-score) > `Z_SCORE_CUTOFF` as outliers.
    detect_outlier = ZScoreOutlierDetector(
        diff_method=None,
        trim_percent=trim_percent,
        z_score_cutoff=z_score_cutoff)

    detect_outlier.fit(df[value_col])
    fitted = detect_outlier.fitted
    cond_outlier = fitted.is_outlier

    # Extracts time points when outliers occur.
    outlier_points = df.loc[cond_outlier, time_col].reset_index(drop=True)

    anomaly_df = pd.DataFrame(columns=[START_TIME_COL, END_TIME_COL])
    if len(outlier_points) > 0:
        anomaly_df[START_TIME_COL] = outlier_points
        anomaly_df[END_TIME_COL] = anomaly_df[START_TIME_COL]
        anomaly_df = get_canonical_anomaly_df(
            anomaly_df=anomaly_df,
            freq=freq)
    return anomaly_df
