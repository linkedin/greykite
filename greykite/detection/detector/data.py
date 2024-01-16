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

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Data:
    """This class is useful in constructing the data consumed in
        `~greykite.detection.detector.optimizer.Optimizer`
    class.

    Attributes
    ----------
    df : `pandas.DataFrame` or None, default None
        If not None, it's a dataframe.
    """
    df: Optional[pd.DataFrame] = None


@dataclass
class DetectorData(Data):
    """This class is useful in constructing the data consumed in
        `~greykite.detection.detector.Detector`
    class.

    Attributes
    ----------
    pred_df : `pandas.DataFrame` or None, default None
        If not None, it's a dataframe which typically includes predicted data.
    y_true : `pandas.Series` or None, default None
        If not None, a pandas series of typically boolean values denoting anomaly occurrences
        in observed data.
    y_pred : `pandas.Series` or None, default None
        If not None, a pandas series of typically boolean values denoting anomaly occurrences
        in predicted data.
    anomaly_df : `pandas.DataFrame` or None, default None
        A dataframe which includes the start and end times of observed anomalies.
    """
    pred_df: Optional[pd.DataFrame] = None
    y_true: Optional[list] = None
    y_pred: Optional[list] = None
    anomaly_df: Optional[pd.DataFrame] = None


@dataclass
class ForecastDetectorData(DetectorData):
    """This class is useful in constructing the data consumed in
    `~greykite.detection.detector.forecast_based.ForecastBasedDetector`

    Attributes
    ----------
    forecast_dfs : `list` [`pandas.DataFrame`] or None, default None
        A list of dataframes, which are typically expected to include forecasts.
        Each one is typically joined with ``df`` to construct a ``joined_df``,
        results of which will be stored in ``joined_dfs`` (see below).
    joined_dfs : `list` [`pandas.DataFrame`] or None, default None
        A list of dataframes, which are typically the result of joining ``df``
        with each dataframe in ``forecast_dfs`` (see above).
    """
    forecast_dfs: Optional[list] = None
    joined_dfs: Optional[list] = None
