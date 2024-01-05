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

import numpy as np

from greykite.common.constants import PREDICTED_ANOMALY_COL
from greykite.detection.detector.best_forecast import BestForecastDetector
from greykite.detection.detector.optimizer import CalcResult


# The parameter space to iterate over for APE threshold (absolute percent error)
APE_PARAM_ITERABLE = [{"ape_thresh": x} for x in np.arange(0, 4, 0.05)]


class APEDetector(BestForecastDetector):
    """This class implements APE (absolute percent error) based detector.
    The class finds the

    - best forecast among multiple forecasts which can be
        passed as baselines
    - as well as optimal APE threshold to use to denote an anomaly


    This class inherits its parameters and attributes from

    `~greykite.detection.detector.forecast_based.BestForecast`

    and all methods apply here as well. The only difference is: ``param_iterable``
    is not passed but constructed during the ``__init__``.


    Parameters
    ----------
    Solely inherited from
    `~greykite.detection.detector.forecast_based.BestForecast`
    except for the input parameter `param_iterable` which is constructed in the
    ``__init__``.

    Attributes
    ----------
    Solely inherited from
    `~greykite.detection.detector.forecast_based.BestForecast`

    """

    def __init__(
            self,
            value_cols,
            pred_cols,
            is_anomaly_col=None,
            join_cols=None,
            reward=None,
            anomaly_percent_dict=None):
        super().__init__(
            value_cols=value_cols,
            pred_cols=pred_cols,
            is_anomaly_col=is_anomaly_col,
            join_cols=join_cols,
            reward=reward,
            anomaly_percent_dict=anomaly_percent_dict)

        self.param_iterable = APE_PARAM_ITERABLE

    def add_features_one_df(
            self,
            joined_df):
        """Adds features to one joined dataframe. This will be used to add
        features to all joined dataframes.

        Parameters
        ----------
        joined_df : `pandas.DataFrame`
            An input dataframe.

        Returns
        -------
        joined_df : `pandas.DataFrame`
            An output dataframe, with an extra column of APE values.
        """

        for col in (self.value_cols + self.pred_cols):
            if col not in joined_df.columns:
                raise ValueError(
                    "f{col} was not found in joined data with columns:",
                    "{joined_df.columns}")

        # Multivariate APE using Euclidean distance
        joined_df["ape"] = (
            np.linalg.norm(
                joined_df[self.value_cols].values - joined_df[self.pred_cols].values,
                axis=1) /
            np.linalg.norm(
                joined_df[self.value_cols].values,
                axis=1))

    def calc_with_param(self, param, data):
        """It assigns anomaly label to any points which has a larger mape
        than the threshold.

        Parameters
        ----------
        data : See class docstring

        Returns
        -------
        result : `pandas.DataFrame`
            Data frame with predictions added in a new column: ``PREDICTED_ANOMALY_COL``
        """
        pred_df = data.joined_dfs[param["forecast_id"]]
        y_pred = (pred_df["ape"] > param["ape_thresh"])
        pred_df[PREDICTED_ANOMALY_COL] = y_pred
        data.y_pred = y_pred
        data.pred_df = pred_df
        return CalcResult(data=data)
