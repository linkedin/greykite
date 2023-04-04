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
"""Sliding window benchmarking."""

import warnings

import numpy as np
import pandas as pd


def gen_moving_timeseries_forecast(
        df,
        time_col,
        value_col,
        train_forecast_func,
        train_move_ahead,
        forecast_horizon,
        min_training_end_point=None,
        min_training_end_timestamp=None,
        max_forecast_end_point=None,
        max_forecast_end_timestamp=None,
        regressor_cols=None,
        keep_cols=None,  # extra cols in df which we want to keep from the raw data
        forecast_keep_cols=None,  # extra cols we want to keep from the forecast result
        **model_params):
    """Applies a forecast function (`train_forecast_func`) to many derived
    timeseries from `df` which are moving windows of `df`. For each derived
    series a model is trained and forecast is generated.
    It returns a `compare_df` to compare actuals and forecasts.

    Parameters
    ----------

    df : `pandas.DataFrame`
        A data frame which includes the timestamp column
        as well as the value column.
        The time column is assumed to be in increasing order (
        timestamps increase).
    time_col : `str`
        The column name in ``df`` representing time for the time series data.
        The time column can be anything that can be parsed by pandas DatetimeIndex.
    value_col: `str`
        The column name which has the value of interest to be forecasted.
    train_forecast_func : `func`
        A function with this signature::

        train_forecast_func(
            df,
            time_col,
            value_col,
            forecast_horizon,
            new_external_regressor_df=None)

        This function is required to return a dictionary which has at minimum this item
            "fut_df": pd.DataFrame which includes the forecasts in the column
            ``value_col``

    train_move_ahead : `int`
        The number of steps moving forward for each window
        This can be set to 1 often to get the maximum number of validations
        However other numbers can be used e.g. if computation is an issue
    forecast_horizon : `int`
        The number of forecasts needed
    min_training_end_point : `int` or None, default None
        The minimum number of training time points
    min_training_end_timestamp : `str` or None, default None
        The minimum timestamp to be used.
        If this is not None, ``min_training_end_point`` will be overwritten.
    max_forecast_end_point : `int` or None, default None
        The end point to be forecasted. The input ``df`` will be limited
        to this point.
    max_forecast_end_timestamp : `str` or None, default None
        The last timestamp allowed to be forecasted.
        If this is not None, ``max_forecast_end_point`` will be overwritten.
    regressor_cols : `list` [`str`] or None, default None
        If regressors are to be used, they are listed here.
    keep_cols : `list` [`str`] or None, default None
        Extra columns in ``df`` which we want to keep
    forecast_keep_cols : `list` [`str`] or None, default None
        Extra columns in the forecat result (dataframe) which we want to keep

    Return : `dict`
    ----------
    A dictionary with following items:

    - "compare_df": `pd.DataFrame`
        A dataframe which includes
        (a) actual true values (observed) given in "y_true" column;
        (b) forecasted values given in "y_hat";
        (c) horizon given in a column "horizon" which determines the number of
            points into the future for that forecast;
        (d) training end point given in ``training_end_point`` column
    - "max_possible_validation_num" : `int`
        Maximum possible number of validations
    - "validation_num" : `int`
        Number of validations used

    """
    if max_forecast_end_timestamp is not None:
        max_forecast_end_point = max(
            np.where(df[time_col] <= max_forecast_end_timestamp)[0])

    if min_training_end_timestamp is not None:
        min_training_end_point = min(
            np.where(df[time_col] >= min_training_end_timestamp)[0])

    if max_forecast_end_point is not None:
        df = df[:max_forecast_end_point]

    compare_df = None
    n = df.shape[0]
    if (n - forecast_horizon) <= min_training_end_point:
        raise ValueError("No reasonble train test period is found for validation")

    # Maximum possible validation number for this set
    max_possible_validation_num = n - forecast_horizon - min_training_end_point
    # Actual validation number
    validation_num = max_possible_validation_num / train_move_ahead

    training_end_times = np.arange(
        min_training_end_point,
        n - forecast_horizon,
        train_move_ahead)

    def get_compare_df_row(m):
        """Calculates comparison df with actuals and forecasted for the given
        horizon, using the training data up to time ``m``

        ----------
        Parameters
        m : `int`
            Last row of data to be used for training

        -------
        Returns
        compare_df0 : `pandas.DataFrame`
            A pandas dataframe with ``forecast_horizon`` rows containing
            observed values and forecasted values.
            It includes
                (a) actual true values (observed) given in "y_true" column;
                (b) forecasted values given in "y_hat";
                (c) horizon given in a column "horizon" which determines the number of
                    points into the future for that forecast;
                (d) training end point given in ``training_end_point`` column
        """
        train_df = df[:m]
        test_df = df.loc[range(m, m + forecast_horizon), :].reset_index(drop=True)

        if regressor_cols is not None:
            new_external_regressor_df = test_df[regressor_cols]
            obtained_forecast = train_forecast_func(
                df=train_df,
                value_col=value_col,
                time_col=time_col,
                forecast_horizon=forecast_horizon,
                new_external_regressor_df=new_external_regressor_df,
                **model_params)
        else:
            obtained_forecast = train_forecast_func(
                df=train_df,
                value_col=value_col,
                time_col=time_col,
                forecast_horizon=forecast_horizon,
                **model_params)

        fut_df = obtained_forecast["fut_df"]
        fut_df = fut_df.reset_index(drop=True)
        y_hat = fut_df[value_col].values
        y_true = test_df[value_col]

        timestamps = test_df[time_col]
        compare_df0 = pd.DataFrame({
            time_col: timestamps,
            "y_hat": y_hat,
            "y_true": y_true})

        if keep_cols is not None:
            compare_df0 = pd.concat(
                [compare_df0, test_df[keep_cols]],
                axis=1)
        if forecast_keep_cols is not None:
            compare_df0 = pd.concat(
                [compare_df0, fut_df[forecast_keep_cols]],
                axis=1)
        compare_df0["horizon"] = range(1, forecast_horizon + 1)
        compare_df0["training_end_point"] = m

        return compare_df0
    # Runs the function for all m and stores the results dataframes for each m
    compare_df_list = [get_compare_df_row(m) for m in training_end_times]
    # Concats all the dataframes
    compare_df = pd.concat(compare_df_list, axis=0)

    na_df = compare_df[compare_df.isnull().any(axis=1)]
    if na_df.shape[0] > 0:
        warnings.warn("NA was generated in compare_df.")

    return {
        "compare_df": compare_df,
        "max_possible_validation_num": max_possible_validation_num,
        "validation_num": validation_num}
