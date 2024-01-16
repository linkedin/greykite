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

from greykite.detection.detector.detector import Detector


class ForecastBasedDetector(Detector):
    """This class enables anomaly detection algorithms which use baseline
    forecasts in their logic.

    The class assumes that for a given dataset (`df`) which can for example include
    timestamps and observed values for a timeseries as well as anomaly labels (optional),
    a number of (say k) forecasts are given.
    The goal is to use those k forecasts as baselines to detect anomalies.

    To that end this class implements joining of the k forecasts with the observed
    data (`df`), thus producing k joined dataframes.

    This class facilitates the join by implementing it by using the input `join_cols`
    using its `join_with_forecasts` method within the class.
    In order to use this class user only required to implement these three methods:

        - ``add_features_one_df``: Note that the class uses this method to fully implement
            ``add_features`` because the class can assume that the same function can
            be used for all k forecasts.
        - ``calc_with_param``: the prediction logic, assuming the optimal param
            is determined
        - ``fit``: The fit method

    This class also already implements ``prep_df_for_predict`` which is basically
    a combination of

    - Joining with baselines: ``join_with_forecasts``
    - Adding features: ``add_features``

    In this way, with minimal implementation, one can implement a large variety of
    "forecast based" anomaly detectors.


    Parameters
    ----------
    value_cols : `list` [`str`] or None
        The columns for the response metric (which can be multivariate).
        If not None, and also ``pred_cols`` (below) also not None, we expect them
        to be ordered consistently.
    pred_cols : `list` [`str`] or None
        The columns for the response metric (which can be multivariate).
        If not None, and also ``value_cols`` (above) also not None, we expect them
        to be ordered consistently.
    is_anomaly_col : `str` or None
    join_cols : `list` [`str`] or None
    reward : See docstring for
        `~/greykite.detection.detector.detector.Detector`
    anomaly_percent_dict : See docstring for
        `~/greykite.detection.detector.detector.Detector`
    param_iterable : See docstring for
        `~/greykite.detection.detector.detector.Detector`

    Attributes
    ----------
    Solely inherited from `~/greykite.detection.detector.detector.Detector`
    """
    def __init__(
            self,
            value_cols=None,
            pred_cols=None,
            is_anomaly_col=None,
            join_cols=None,
            reward=None,
            anomaly_percent_dict=None,
            param_iterable=None):
        super().__init__(
            reward=reward,
            anomaly_percent_dict=anomaly_percent_dict,
            param_iterable=param_iterable)
        self.value_cols = value_cols
        self.pred_cols = pred_cols
        self.is_anomaly_col = is_anomaly_col
        self.join_cols = join_cols

    def join_with_forecasts(
            self,
            forecast_dfs,
            df=None):
        """Joins data with forecasts.
        This will be used both in training and prediction phases.
        Parameters
        ----------
        forecast_dfs : `dict` [`str`: `pandas.DataFrame`] or None, default None
            Dict of baselines (forecasts) to be joined with observed data given
            in ``df``.
            If ``df`` is None, no join is needed and ``forecast_dfs`` are
            returned.
        df : `pandas.DataFrame` or None, default None
            A dataframe which includes the observed data and potentially the
            observed labels.
            If None, it is assumed that the ``forecast_dfs``
            list has already been joined or has enough information to fit.

        Returns
        -------
        result : `list` [`pandas.DataFrame`]
            The list of baselines (forecasts) after being joined with ``df`` (if needed).
        """

        # If either `df` or `self.join_cols` is None, we assume data is joined
        # already or has enough info already
        if df is None or self.join_cols is None:
            return forecast_dfs

        joined_dfs = {}
        for k, forecast_df in forecast_dfs.items():
            joined_dfs[k] = df.merge(
                forecast_df,
                how="inner",
                on=self.join_cols)

        return joined_dfs

    def add_features_one_df(
            self,
            joined_df):
        """Adds features to one joined dataframe.
        This will be used to add features to all joined dataframes.
        Classes inherting from this class can implement this to get new detectors.

        Parameters
        ----------
        joined_df : `pandas.DataFrame`
            An input dataframe.

        Returns
        -------
        joined_df : `pandas.DataFrame`
            An output dataframe, potentially with new columns or altered columns.

        """
        return joined_df

    def add_features(
            self,
            joined_dfs=None):
        """Adds features to `joined_dfs` passed.
        Note that if nothing is passed, this will update ``self.joined_dfs``

        Parameters
        ----------
        joined_dfs : `list` [`pandas.DataFrame`] or None, default None
            A list of dataframes. If None, ``self.joined_dfs`` will be
            used as input.

        Returns
        -------
        joined_dfs : `list` [`pandas.DataFrame`]
            The resulting list of dataframes.

        """
        if joined_dfs is None:
            joined_dfs = self.joined_dfs

        if joined_dfs is None:
            raise ValueError(
                "`joined_dfs` cannot be None."
                "`join_with_forecasts` method is to be called before")

        for k, joined_df in joined_dfs.items():
            self.add_features_one_df(joined_df)
        return joined_dfs

    def prep_df_for_predict(
            self,
            data):
        """This will prepares the detection data (``data``) by applying
        the joins and adding features.

        Parameters
        ----------
        data : `~greykite.detection.detector.ForecastDetectorData`
            Object including the data.

        Returns
        -------
        None
            The input ``data`` will be altered.
        """
        data.joined_dfs = self.join_with_forecasts(
            df=data.df,
            forecast_dfs=data.forecast_dfs)
        assert len(data.joined_dfs) == len(data.forecast_dfs)
        data.joined_dfs = self.add_features(
            data.joined_dfs)

        return None
