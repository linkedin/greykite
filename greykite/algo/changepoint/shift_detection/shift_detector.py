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
# original author: Katherine Li, Kaixu Yang
"""This module conducts level shift detection.
The level shifts are handled with regressors in the Silverkite forecasting model.
This module contains a class which can generate corresponding regressors to the input dataframe.
This dataframe can be fed into the Silverkite model.
The level shift algorithm takes the first order differencing of the data.
It will calculate the z score on the differenced values.
If the z score is larger than the predefined threshold,
the dates will be marked as level shift and corresponding regressors will be created.
For every level shift, it will create a regressor that has values 0 before it and 1 after it,
so the model is able to shift at those dates."""

import re
import warnings
from datetime import datetime
from enum import Enum
from typing import List
from typing import Optional

import pandas as pd

from greykite.common.constants import LEVELSHIFT_COL_PREFIX_SHORT
from greykite.common.viz.timeseries_plotting import plot_multivariate


# An enum of supported time series frequencies
# see details at
# https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
class TimeSeriesFrequency(Enum):
    T = "minutes"
    H = "hours"
    D = "days"
    W = "weeks"
    M = "months"
    Y = "years"


def find_min_max_of_block(indices: List[int]):
    """Given a list of indices, with some of them being consecutive,
    find the start and end of each block.
    Indices are considered to be in the same block if they are consecutive numbers.
    For example, [1, 4, 5, 6, 12, 14, 15] will give [[1, 1], [4, 6], [12, 12], [14, 15]].

    Parameters
    ----------
    indices: `list`
        List of indices in ascending order.
        Example: [1, 4, 5, 6, 12, 14, 15].

    Returns
    ----------
        index_blocks : `list`
        List of list with start and end index of each block.
        Example: [[1, 1], [4, 6], [12, 12], [14, 15]].
    """
    if not indices:
        return []
    block_start_i = 0
    index_blocks = []
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] != 1:
            index_blocks.append((indices[block_start_i], indices[i - 1]))
            block_start_i = i
    index_blocks.append((indices[block_start_i], indices[-1]))
    return index_blocks


class ShiftDetection:
    """The level shifts are handled with regressors in the Silverkite forecasting model.
    This class can generate corresponding regressors to the input dataframe that can be fed into the Silverkite model.
    For every level shift, we create a regressor that has values 0 before it and 1 after it,
    so the model is able to shift at those dates.

    The main method to run level shift detection is `detect`.
    The level shift algorithm takes the first order differencing of the data.
    It will calculate the z score on the differenced values.
    If the z score is larger than the predefined threshold,
    the dates will be marked as level shift and corresponding regressors are created.
    For example, if it detects a sudden increase during
    [2020-01-01, 2020-01-03], it will create a regressor column named "ctp_2020_01_02"
    by taking middle of the start and end date of the level shift period.
    for dates before 2020_01_02, values will be set to 0 while after this date, values are 1.

    Attributes
    ----------
    df : `pandas.DataFrame`
        The dataframe used for level shift detection.
        It must have two columns:
        ``time_col`` indicating the column of time
        (the format should be able to be parsed by pd.to_datetime),
        and ``value_col`` indicating the column of observed time series values.
    time_col : `str`
        The column name for time column.
    value_col : `str`
        The column name for value column.
    forecast_horizon: `int`
        The number of datapoints to forecast.
        This is used to generate dataframe after adding regressors for level shift.
    freq : `str`
        Frequency of the dataframe such as "D" for daily and "W" for weekly.
        The allowed values can be found in the TimeSeriesFrequency Enum object.
    z_score_cutoff : `int`
        The z score cutoff value to define the level shift.
        By default is 3.
    df_shift : `pandas.DataFrame`
        Dataframe with additional columns calculated by levelshift algorithm:
        column "actual_diff" represents the abs of the first order differencing,
        column "zscore" of actual_diff of the two adjacent dates.
        This dataframe will be used for function `plot_level_shift`.
    shift_dates : `list`
        The shift start and end date. below is an example:
        [(Timestamp("2020-01-11 00:00:00"), Timestamp("2020-01-12 00:00:00")),
            (Timestamp("2020-01-21 00:00:00"), Timestamp("2020-01-21 00:00:00"))].
    regressor_col : `list`
        List of names of the regressor columns which will start with `LEVELSHIFT_COL_PREFIX_SHORT`
        E.g. ["ctp_2020_01_11"]
    final_df : `pandas.DataFrame`
        Dataframe with regressor columns which is expanded to future dates based on ``forecast_horizon``.
        For example, if the input has 2 rows and forecast_horizon = 1,
        it will create a new row with ``value_col`` = NaN and a few regressor columns of the level shift detected.

    Methods
    -------
    detect : callable
        Runs shift detection algorithm and create dataframe with levelshift
        regressor columns for a given time series df.
    plot_level_shift : callable
        Plots the results after running fucntion `find_shifts` or `detect`.
    find_shifts : callable
        Finds the start and end dates of the level shift for a given time series df.
    create_df_with_regressor : callable
        Appends level shift regressor column to a given time series df.
    create_regressor_for_future_dates : callable
        Creates future dates with level shift regressor columns
        to a given time series df.
    """

    def __init__(self):
        self.original_df: Optional[pd.DataFrame] = None
        self.time_col: Optional[str] = None
        self.value_col: Optional[str] = None
        self.forecast_horizon: Optional[int] = None
        self.freq: Optional[str] = None
        self.z_score_cutoff: Optional[int] = None
        self.df_shift: Optional[pd.DataFrame] = None
        self.shift_dates: Optional[list] = None
        self.regressor_col: Optional[list] = None
        self.final_df: Optional[pd.DataFrame] = None

    def detect(
            self,
            original_df: pd.DataFrame,
            time_col: str,
            value_col: str,
            forecast_horizon: int = 0,
            freq: str = "D",
            z_score_cutoff: float = 3):
        """This is the main function to create dataframe
        with level shift regressor columns.
        It will detect the level shifts, return a dataframe with regressor columns
        representing the dates of sudden value shift, and return a list of regressor
        column names that will be passed into the config of the Forecaster.run_forecast_config.

        Attributes
        ----------
        df : `pandas.DataFrame`
            The dataframe used for level shift detection.
            It must have at least two columns:
            ``time_col`` in the format should be able to be parsed by pd.to_datetime
            and ``value_col`` in the `int` or `float` format for observed time series values.
        time_col : `str`
            The column name for time column.
        value_col : `str`
            The column name for value column.
        forecast_horizon: `int`, default is 0 (don't need to expand to future dates)
            The number of datapoints to forecast.
            This is used to generate dataframe after adding regressors for level shift.
        freq : `str`, default is "D"
            Frequency of the dataframe such as "D" for daily and "W" for weekly.
            The allowed values can be found in the TimeSeriesFrequency Enum object.
        z_score_cutoff : `float`
            The z score cutoff value to define the level shift.
            By default is 3.

        Returns
        ------
        regressor_col: `list`
            List of names of the regressor_col which will start with `LEVELSHIFT_COL_PREFIX_SHORT`
            E.g. ["ct0", "ctp_2020_01_11"]
        final_df: `pandas.DataFrame`
            Dataframe with time_col, value_col and additional regressor columns
            of the level shift detected.
            It is also expanded to future dates based on ``forecast_horizon``.
            For example, if the input has 2 rows and forecast_horizon = 1,
            it will create a new row with value_col = NaN and a few regressor columns
            of the level shift detected.
        """
        # Initialize the variables.
        self.original_df = original_df
        self.time_col = time_col
        self.value_col = value_col
        self.forecast_horizon = forecast_horizon
        self.freq = freq
        self.z_score_cutoff = z_score_cutoff

        # Check frequency type matching the enum TimeSeriesFrequency.
        if freq not in TimeSeriesFrequency.__members__.keys():
            raise ValueError("freq should be one of the values of "
                             + str(list(TimeSeriesFrequency.__members__.keys())))

        # Filter columns and ensure the datetime type of the time_col.
        df = original_df[[time_col, value_col]].copy()
        df[time_col] = pd.to_datetime(df[time_col])

        # Find shifts.
        self.df_shift, self.shift_dates = self.find_shifts(df, time_col, value_col, z_score_cutoff)

        # Create regressor columns.
        df_w_regressor = self.create_df_with_regressor(self.df_shift, time_col, self.shift_dates)
        self.regressor_col, self.final_df = self.create_regressor_for_future_dates(
            df_w_regressor,
            time_col,
            value_col,
            forecast_horizon,
            freq)
        return self.regressor_col, self.final_df

    def plot_level_shift(self):
        """Makes a plot to show the observations with level shifts.

        Attributes
        ----------
        None

        Returns
        ----------
        fig : `plotly.graph_objects.Figure` The plot object.
        """
        if self.df_shift is not None and self.time_col is not None:
            fig = plot_multivariate(self.df_shift, self.time_col)
            for pair in self.shift_dates:
                fig.add_vrect(x0=pair[0], x1=pair[1], fillcolor="red", opacity=0.2)
            fig.show()
        else:
            warnings.warn("please run either detect() or find_shifts() first.")

    def find_shifts(
            self,
            df: pd.DataFrame,
            time_col: str,
            value_col: str,
            z_score_cutoff: int = 3):
        """This is the main function to detect level shifts based on the z score threshold.

        Attributes
        ----------
        df : `pandas.DataFrame`
            The dataframe used for level shift detection.
            It must have at least two columns:
            ``time_col`` in the format should be able to be parsed by pd.to_datetime
            and ``value_col`` in the `int` or `float` format for observed time series values.
        time_col : `str`
            The column name for time column.
        value_col : `str`
            The column name for value column.
        z_score_cutoff : `int`, default is 3.
            The z score cutoff value to define the level shift.

        Returns
        ----------
        df_find_shifts : `pandas.DataFrame`
            Dataframe with two additional columns:
            actual_diff: abs of the first order differencing.
            zscore: standard deviation between two datapoints.
        shift_dates : `list`
            The shift start and end date. below is an example:
            [(Timestamp("2020-01-11 00:00:00"), Timestamp("2020-01-12 00:00:00")),
                (Timestamp("2020-01-21 00:00:00"), Timestamp("2020-01-21 00:00:00"))].
        """
        df_find_shifts = df.copy()
        df_find_shifts["actual_diff"] = df_find_shifts[value_col].diff(1).abs()
        df_find_shifts["zscore"] = ((df_find_shifts["actual_diff"] - df_find_shifts["actual_diff"].mean())
                                    / df_find_shifts["actual_diff"].std())

        shifts = df_find_shifts[df_find_shifts["zscore"] > z_score_cutoff].index.tolist()

        shift_dates = find_min_max_of_block(shifts)
        shift_dates = [(df_find_shifts[time_col].loc[pair[0]], df_find_shifts[time_col].loc[pair[1]]) for pair in shift_dates]
        return df_find_shifts, shift_dates

    def create_df_with_regressor(
            self,
            df: pd.DataFrame,
            time_col: str,
            shift_dates: List[datetime]):
        """Create dataframe with additional regressor columns of the level shift detected.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The dataframe used for level shift detection.
            It must have at least two columns:
            ``time_col`` in the format should be able to be parsed by pd.to_datetime
            and ``value_col`` in the `int` or `float` format for observed time series values.
        time_col : `str`
            The column name for time column.
        shift_dates : `list`
            The shift start and end date. this is generated by find_shifts().
            below is an example:
            [(Timestamp("2020-01-11 00:00:00"), Timestamp("2020-01-12 00:00:00")),
                (Timestamp("2020-01-21 00:00:00"), Timestamp("2020-01-21 00:00:00"))]

        Returns
        ----------
        df_regressor : `pandas.DataFrame`
            Dataframe with additional regressor columns of the level shift detected.
            The regressor columns will start with `LEVELSHIFT_COL_PREFIX_SHORT`.
            It is in minutely format.
            regressor is calculated based on the middle point of each shift time block.
            This middle point might not exist in the raw data. E.g, for a shift block of
            [(Timestamp("2020-09-01 00:00:00"), Timestamp("2020-10-01 00:00:00")],
            regressor is created as `ctp_2020_09_16_00_00`.
        """
        df_regressor = df.copy()
        for i, pair in enumerate(shift_dates):
            # If we have a consecutive block, we use the date in the middle to create the regressor column.
            ctp = pair[0] + (pair[1] - pair[0]) / 2
            df_regressor[f"{LEVELSHIFT_COL_PREFIX_SHORT}_{ctp.strftime('%Y_%m_%d_%H_%M')}"] = (
                (df_regressor[time_col] >= ctp).astype(int))
        return df_regressor

    def create_regressor_for_future_dates(
            self,
            df: pd.DataFrame,
            time_col: str,
            value_col: str,
            forecast_horizon: int,
            freq: TimeSeriesFrequency):
        """Expand dataframe to future dates based on the forecast_horizon.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The dataframe generated by calling create_df_with_regressor().
            It must have at least the columns listed below:
            time_col in the format should be able to be parsed by pd.to_datetime,
            value_col in the `int` or `float` format for observed time series values,
            a few regressor columns identified by level shift algorithm.
        time_col : `str`
            The column name for time column.
        value_col : `str`
            The column name for value column.
        forecast_horizon : `int`
            The number of datapoints to forecast.
            This is used to generate dataframe after adding regressors for level shift.
        freq : `str`
            Frequency of the dataframe such as "D" for daily and "W" for weekly.
            The allowed values can be found in the TimeSeriesFrequency Enum object.

        Returns
        ----------
        ctp_col : `list`
            List of names of the regressor columns which will start with `LEVELSHIFT_COL_PREFIX_SHORT`
            E.g. ["ctp_2020_01_11_00_00"]
        df_result : `pandas.DataFrame`
             Dataframe with regressor columns which is expanded to future dates based on ``forecast_horizon``.
            For example, if the input has 2 rows and forecast_horizon = 1,
            it will create a new row with ``value_col`` = NaN and a few regressor columns of the level shift detected.
        """
        # Check frequency type matching the enum TimeSeriesFrequency.
        if forecast_horizon != 0 and freq not in TimeSeriesFrequency.__members__.keys():
            raise ValueError("freq should be one of the values of "
                             + str(list(TimeSeriesFrequency.__members__.keys())))

        df_regressor = df.copy()
        re_pattern = re.compile(fr"^{LEVELSHIFT_COL_PREFIX_SHORT}_\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}$")
        ctp_col = [column for column in df_regressor.columns if re_pattern.match(column)]
        df_regressor = df_regressor[[time_col, value_col] + ctp_col]
        if forecast_horizon == 0:
            return ctp_col, df_regressor
        expand_df = pd.DataFrame({
            time_col: pd.date_range(df_regressor[time_col].max(), freq=freq, periods=forecast_horizon+1)})
        expand_df[ctp_col] = 1
        df_result = pd.concat([df_regressor, expand_df.loc[1:]]).reset_index(drop=True)
        return ctp_col, df_result
