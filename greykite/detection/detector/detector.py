# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

import numpy as np
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
import pandas as pd

from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import PREDICTED_ANOMALY_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.viz.timeseries_annotate import plot_anomalies_over_forecast_vs_actual
from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.common.ad_evaluation import precision_score
from greykite.detection.common.ad_evaluation import recall_score
from greykite.detection.detector.ad_utils import get_anomaly_df
from greykite.detection.detector.ad_utils import partial_return
from greykite.detection.detector.constants import PHASE_PREDICT
from greykite.detection.detector.constants import PHASE_TRAIN
from greykite.detection.detector.constants import PenalizeMethod
from greykite.detection.detector.optimizer import Optimizer
from greykite.detection.detector.reward import Reward


# Default for `anomaly_percent_dict` which is needed in
# `Detector` class if no `reward` is passed.
DEFAULT_ANOMALY_PERCENT_DICT = {"range": (4, 6), "penalty": -1}


def build_anomaly_percent_reward(anomaly_percent_dict):
    """This builds an reward function given an expected anomaly percent range.
    This is the expected percent of anomalies in the data by user.
    This is useful when the user does now know which points are anomalies,
    but has some idea about what percent of data are anomalies.
    The reward function constructed here will penalize for being far from the
    center of the range specified by the user (``anomaly_percent_dict["range"]``)
    and it will add an extra penalty for being outside that interval. The penalty
    is specified in``anomaly_percent_dict["penalty"]``.

    Parameters
    ----------
    anomaly_percent_dict : `dict` or None, default None
        If not None, a dictionary with items:

        - ``"range"`` : `tuple`
            We expect a tuple with two elements.
        - ``"penalty"`` : `float`
            A real number to specify the penalty of being outside the range
            specified by user. It should be typically a negative value and it
            could be set to ``float("-inf")`` to make this reward a restriction
            while using it along with other rewards.

        The dictionary is used to construct an reward which will be the reward in
        the optimization is no reward is passed to the ``Detector`` class below.

        If another reward is passed then this will be added to the passed reward.

        The constructed reward based on ``anomaly_percent_dict`` penalizes
        by the distance between the center of the ``"range"`` and predicted anomaly
        percent as long as the predicted is within range.

        For values outside the range an extra penalty given in`"penalty"`
        will be applied.
        If `"penalty"` is None, it will be set to -1.

        If None, the default ``DEFAULT_ANOMALY_PERCENT_DICT`` will be used.

    Returns
    -------
    result : `~greykite.detection.detector.reward.Reward`
        The reward which reflects the information given in the input.
    """
    min_percent = anomaly_percent_dict["range"][0]
    max_percent = anomaly_percent_dict["range"][1]
    target_percent = (min_percent + max_percent) / 2.0

    def reward_func(data):
        percent_anomaly = 100 * np.mean(data.y_pred)
        diff = abs(percent_anomaly - target_percent) / 100.0
        # `-diff` will be returned because we assume higher is better
        return -diff

    # Below intends to calculate which diffs using the above reward
    # function transalte to the `percent_anomaly` being outside the
    # specified range in `anomaly_percent_dict`
    # Considering the above `reward_func` measures the diff from the mid-point
    # of the range: we can deduce that maximum diff which still remains in
    # the range is the length of the range.
    # since the`reward_func` also divides the diff by 100, we also divide by
    # 100.
    max_acceptable_diff = (max_percent - min_percent) / (2 * 100.0)

    penalty = anomaly_percent_dict["penalty"]
    if penalty is None:
        penalty = -1

    # Below `min_unpenalized=-max_acceptable_diff` will ensure that `percent_anomaly`
    # which is outside the specified range in `anomaly_percent_reward` will be
    # penalized by an extra penalty
    # with default -1, if not provided in `anomaly_percent_dict`
    anomaly_percent_reward = Reward(
        reward_func=reward_func,
        min_unpenalized=-max_acceptable_diff,
        max_unpenalized=float("inf"),
        penalize_method=PenalizeMethod.ADDITIVE.value,
        penalty=penalty)

    return anomaly_percent_reward


class Detector(Optimizer):
    """Base detector class for Anomaly Detection.
    The class initializes by passing an arbitrary ``reward`` for optimization
    and a potentially multivariate parameter (given in ``param_iterable``) to optimize.

    The ``reward`` object is required to implement the ``apply`` method which is the case for this class:
    `~greykite.detection.detector.reward.Reward`

    The class behaves similar to typical machine-learning algorithms as it includes

        - ``fit``
        - ``predict``

    methods.

    The optimization method (``optimize_param``) is inherited from:

    `~greykite.detection.detector.optimizer.Optimizer`

    It works simply by iterating over ``param_iterable`` and calculating
    the reward to choose the optimal parameter via ``calc_with_param`` method
    which is an abstract method already appearing in the ``Optimizer`` class.

    The class assumes that larger is better for the reward function,
    during optimization.

    The classes inheriting this class, need to implement ``calc_with_param``
    method to be able to use the optimizer and given that implementation.

    The ``predict`` method is already implemented here and should work for most cases.
    This is because in most cases, ``predict`` is simply ``calc_with_param``
    applied to the best param found during the optimization step.

    Parameters
    ----------
    reward : `~greykite.detection.detector.reward.Reward` or None, default None
        The reward to be used in the optimization.
        If None, an reward will be built using the other input
        ``anomaly_percent_dict``.

    anomaly_percent_dict : `dict` or None, default None
        If not None, a dictionary with items:

        - ``"range"`` : `tuple`
            We expect a tuple with two elements denoting the min and max
            of the interval range.
        - ``"penalty"`` : `float`
            A real number to specify the penalty of being outside the range
            specified by user. It should be typically a negative value and it
            could be set to ``float("-inf")`` to make this reward a restriction
            while using it along with other rewards.

        which is used to construct an reward which will be the reward in
        the optimization is no reward is passed.
        If another reward is passed then this will be added to the passed reward.
        The constructed reward based on ``anomaly_percent_dict`` penalizes
        by the distance between the center of the ``"range"`` and predicted anomaly
        percent as long as the predicted is within range.
        For values outside the range an extra penalty given in``"penalty"`` will be applied.
        If `"penalty"` is None, it will be set to -1.

    param_iterable : iterable or None, default None
        An iterable with every element being a parameter passed to the method
        ``calc_with_param`` which takes ``param`` as one of its arguments.
        Each `param` can be a dictionary including values for a set of variables,
        but that is not a requirement.
        The optimizer method (``optimize_param``) will iterate over all the
        parameters to find the best parameter in terms of the specified reward.

    Attributes
    ----------
    data : `dataclasses.dataclass` or None, default None
        A data class object which includes the data for fitting or
        prediction. Depending on the model, this data class might
        include various fields. The prominent used class which can support
        forecast based approaches is given in
        `~greykite.detection.detector.DetectorData`.
    fitted_df : `pandas.DataFrame` or None, default None
        The fitted data after applying the detector.
    fit_info : `dict`
        A dictionary which includes information about the fitted model.
        It is expected that this includes ``"full_param"`` after the fitting
        so that the ``predict`` function can use that param during the prediction
        and simply call ``calc_with_param``.
        In that case the ``predict`` function does not need further implementation
        in child classes as it's already implemented in this class.
    """
    def __init__(
            self,
            reward=None,
            anomaly_percent_dict=None,
            param_iterable=None):
        # If both `reward` and `anomaly_percent_dict` are None,
        # the detector will use a default `DEFAULT_ANOMALY_PERCENT_DICT`
        # the default reward will be smaller for values away from the center of
        # `anomaly_percent_dict["range"]`
        # and it adds a penalty (which is -1 if not passed)
        # if the anomaly percent is outside the range
        if reward is None and anomaly_percent_dict is None:
            anomaly_percent_dict = DEFAULT_ANOMALY_PERCENT_DICT

        anomaly_percent_reward = None
        if anomaly_percent_dict is not None:
            anomaly_percent_reward = build_anomaly_percent_reward(
                anomaly_percent_dict)

        if reward is None:
            self.reward = anomaly_percent_reward
        elif anomaly_percent_dict is None:
            self.reward = reward
        else:
            self.reward = reward + anomaly_percent_reward

        self.anomaly_percent_dict = anomaly_percent_dict
        self.param_iterable = param_iterable
        # Initialize attributes
        self.data = None
        self.fitted_df = None
        self.fit_info = {"param_full": None}

        # Set by the predict method
        self.pred_df = None

    def fit(
            self,
            data=None):
        pass

    def prep_df_for_predict(
            self,
            data=None):
        """A method to prepares the data for ``fit``, ``calc_with_param``.

        Parameters
        ----------
        data : See class attributes.

        Returns
        -------
        result : None
            It updates ``data``
        """
        if data is not None and data.df is not None:
            if TIME_COL in data.df.columns:
                data.df[TIME_COL] = pd.to_datetime(data.df[TIME_COL])
        return data

    def predict(
            self,
            data,
            **kwargs):
        """``predict`` method is already implemented here and should work for most cases.
            This is because in most cases, ``predict`` is simply
            ``calc_with_param`` applied to the best param found during optimization.

        Parameters
        ----------
        data : See class attributes.

        Returns
        -------
        result : return type of ``calc_with_param``
            Usually a ``pandas.DataFrame`` which includes the predicted anomalies.
        """
        self.prep_df_for_predict(data)
        calc_result = self.calc_with_param(
            data=data,
            param=self.fit_info["param_full"],
            **kwargs)
        if data is not None:
            self.pred_df = calc_result.data.pred_df
        return calc_result.data

    def plot(self, phase=PHASE_PREDICT, title=None):
        """Plots the predicted anomalies over the actual anomalies.

        Parameters
        ----------
        phase : str, default ``PHASE_PREDICT``
            The phase of the detector to plot.
            Must be one of ``PHASE_PREDICT`` or ``PHASE_TRAIN``.
        title : str, default None
            The title of the plot.
            If None, a default title will be used.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            The plotly figure object.
        """
        if phase == PHASE_PREDICT:
            if self.pred_df is None:
                raise ValueError("No data to plot. Please run `predict` first.")
            else:
                train_end_date = self.fitted_df[TIME_COL].max()
                if train_end_date < self.pred_df[TIME_COL].min():
                    train_end_date = None
                if title is None:
                    title = "Detected vs actual anomalies - Prediction phase"
                fig = plot_anomalies_over_forecast_vs_actual(
                    df=self.pred_df,
                    time_col=TIME_COL,
                    actual_col=ACTUAL_COL,
                    predicted_col=PREDICTED_COL,
                    predicted_anomaly_col=PREDICTED_ANOMALY_COL,
                    anomaly_col=ANOMALY_COL,
                    marker_opacity=1,
                    predicted_anomaly_marker_color="rgba(0, 90, 181, 0.9)",
                    anomaly_marker_color="rgba(250, 43, 20, 0.7)",
                    predicted_lower_col=PREDICTED_LOWER_COL,
                    predicted_upper_col=PREDICTED_UPPER_COL,
                    train_end_date=train_end_date,
                    title=title)
                return fig
        elif phase == PHASE_TRAIN:
            if self.fitted_df is None:
                raise ValueError("No data to plot. Please run `fit` first.")
            else:
                if title is None:
                    title = "Predicted vs actual anomalies - Training phase"
                fig = plot_anomalies_over_forecast_vs_actual(
                    df=self.fitted_df,
                    time_col=TIME_COL,
                    actual_col=ACTUAL_COL,
                    predicted_col=PREDICTED_COL,
                    predicted_anomaly_col=PREDICTED_ANOMALY_COL,
                    anomaly_col=ANOMALY_COL,
                    marker_opacity=1,
                    predicted_anomaly_marker_color="rgba(0, 90, 181, 0.9)",
                    anomaly_marker_color="rgba(250, 43, 20, 0.7)",
                    predicted_lower_col=PREDICTED_LOWER_COL,
                    predicted_upper_col=PREDICTED_UPPER_COL,
                    train_end_date=self.fitted_df[TIME_COL].max(),
                    title=title)
                return fig
        else:
            raise ValueError(f"phase {phase} is not supported. Must be one of {PHASE_PREDICT}, {PHASE_TRAIN}.")

    def summary(self):
        """Returns a summary of the fitted model."""
        if self.fitted_df is None:
            raise ValueError("No data to summarize. Please run `fit` first.")
        else:
            # `fitted_df` can be a pandas dataframe or a DetectorData object.
            if isinstance(self.fitted_df, pd.DataFrame):
                df = self.fitted_df.copy()
            else:
                df = self.fitted_df.df.copy()

        # Adds the model name and number of observations to the summary.
        content = " Anomaly Detection Model Summary ".center(80, "=") + "\n\n"
        content += f"Number of observations: {len(df)}\n"
        content += f"Model: {self.__class__.__name__}\n"
        content += f"Number of detected anomalies: {np.sum(df[PREDICTED_ANOMALY_COL])}\n\n"

        # Calculates the duration of each anomaly block.
        if TIME_COL in df:
            pred_anomaly_df = get_anomaly_df(
                df=df,
                time_col=TIME_COL,
                anomaly_col=PREDICTED_ANOMALY_COL)
            # To calculate the duration of each anomaly block, we add 1
            # to the difference between the end and start times. This is
            # because both the start time and the end time is inclusive.
            freq = pd.infer_freq(df[TIME_COL])
            pred_anomaly_df["anomaly_interval"] = (
                    pred_anomaly_df[END_TIME_COL] + pd.Timedelta(value=1, unit=freq) -
                    pred_anomaly_df[START_TIME_COL])

            # Adds anomaly duration info to the summary.
            duration_mean = np.mean(pred_anomaly_df["anomaly_interval"])
            content += f"Average Anomaly Duration: {duration_mean}\n"
            duration_min = np.min(pred_anomaly_df["anomaly_interval"])
            content += f"Minimum Anomaly Duration: {duration_min}\n"
            duration_max = np.max(pred_anomaly_df["anomaly_interval"])
            content += f"Maximum Anomaly Duration: {duration_max}\n\n"

        content += f"Alert Rate(%): {np.mean(df[PREDICTED_ANOMALY_COL] * 100)}"
        # Calculates metrics e,g, precision and recall.
        if ANOMALY_COL in df and not df[ANOMALY_COL].isnull().all():
            content += ",   "
            content += f"Anomaly Rate(%): {np.mean(df[ANOMALY_COL]) * 100}\n"
            # Calculates Precision score for the True label.
            calc_precision = partial_return(precision_score, True)
            precision = calc_precision(
                y_true=df[ANOMALY_COL],
                y_pred=df[PREDICTED_ANOMALY_COL])
            content += f"Precision: {round(precision, 3)},"
            # Calculates Recall score for the True label.
            calc_recall = partial_return(recall_score, True)
            recall = calc_recall(
                y_true=df[ANOMALY_COL],
                y_pred=df[PREDICTED_ANOMALY_COL])
            content += f"   Recall: {round(recall, 3)},"
            # Calculates F1 score for the True label.
            calc_f1 = partial_return(f1_score, True)
            f1 = calc_f1(
                y_true=df[ANOMALY_COL],
                y_pred=df[PREDICTED_ANOMALY_COL])
            content += f"   F1 Score: {round(f1, 3)}\n"

        # Adds the optimal objective value and parameters to the summary.
        content += "\n"
        obj_value = round(self.fit_info["obj_value"], 3)
        content += f"Optimal Objective Value: {obj_value}\n"
        optimal_params = self.fit_info["param"]
        content += f"Optimal Parameters: {optimal_params}\n"
        content += "\n"

        return content
