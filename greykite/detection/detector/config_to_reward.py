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

from greykite.detection.common.ad_evaluation import soft_f1_score
from greykite.detection.common.ad_evaluation import soft_precision_score
from greykite.detection.common.ad_evaluation import soft_recall_score
from greykite.detection.detector.ad_utils import partial_return
from greykite.detection.detector.config import F1
from greykite.detection.detector.config import PRECISION
from greykite.detection.detector.config import RECALL
from greykite.detection.detector.detector import build_anomaly_percent_reward
from greykite.detection.detector.reward import Reward


# Evaluation metrics needed.
# Soft F1 score for the True label:
calc_soft_f1 = partial_return(soft_f1_score, True)
# Soft Precision score, for the True label:
calc_soft_precision = partial_return(soft_precision_score, True)
# Soft Recall score for the True label:
calc_soft_recall = partial_return(soft_recall_score, True)


OBJECTIVE_FUNC_MAP = {
    F1: calc_soft_f1,
    PRECISION: calc_soft_precision,
    RECALL: calc_soft_recall
}
"""This is a mapping from objective (string) to a function."""


def config_to_reward(ad_config):
    """Uses information in `ADConfig` to construct a reward function.
    The constructed reward function will be the sum of various rewards
    related to the objective and other information given in `ADConfig`.

    The relevant fields in `ADConfig` are:

        - target_anomaly_percent:
            An `anomaly_percent_range` will be created here with penalty of -1
            for not hitting that range. The range will be the given `target_anomaly_percent`
            plus / minus 10 percent.
        - soft_window_size:
            This is used as a parameter in calculating objective and target_precision / target_recall (below).
        - objective:
            It is either of `F1`, `RECALL` and `PRECISION`.
            Soft versions will be used if `soft_window_size` is not None.
        - target_precision:
            This is the minimal precision we aim for.
            Any precision below this will be penalized by -1.
        - target_recall
            This is the minimal recall we aim for.
            Any recall below this will be penalized by -1.

    Parameters
    ----------
    ad_config : `~greykite.detection.detector.config.ADConfig`
    See the linked dataclass (`ADConfig`) for details.

    Returns
    -------
    result : `~greykite.detection.detector.reward.Reward`
        See the linked class (`Reward`) for details.
    """
    # We initialize the reward function to return 0 regradless of input.
    # This will be then added to other rewards based on `ADConfig`.
    def reward_func(data):
        return 0

    reward = Reward(reward_func=reward_func)

    if ad_config.target_anomaly_percent is not None:
        anomaly_percent_upper = min(1.1 * ad_config.target_anomaly_percent, 100)
        anomaly_percent_lower = max(0.9 * ad_config.target_anomaly_percent, 0)
        anomaly_percent_range = (anomaly_percent_lower, anomaly_percent_upper)
        anomaly_percent_dict = {
            "range": anomaly_percent_range, "penalty": -1}
        anomaly_percent_reward = build_anomaly_percent_reward(
            anomaly_percent_dict)

        reward = reward + anomaly_percent_reward

    # Handles objective.
    # Determine soft evaluation window.
    window = 0
    if ad_config.soft_window_size is not None:
        window = ad_config.soft_window_size
    if ad_config.objective in [F1, PRECISION, RECALL]:
        def reward_func(data):
            obj = OBJECTIVE_FUNC_MAP[ad_config.objective](
                y_true=data.y_true,
                y_pred=data.y_pred,
                window=window)
            if obj is not None:
                return obj
            return 0
        reward = reward + Reward(reward_func=reward_func)

    if ad_config.target_precision is not None:
        def reward_func(data):
            precision = OBJECTIVE_FUNC_MAP[PRECISION](
                y_true=data.y_true,
                y_pred=data.y_pred,
                window=window)
            if precision is not None:
                return precision
            return 0
        reward = reward + Reward(
            reward_func=reward_func,
            min_unpenalized=ad_config.target_precision,
            penalty=-1)

    if ad_config.target_recall is not None:
        def reward_func(data):
            recall = OBJECTIVE_FUNC_MAP[RECALL](
                y_true=data.y_true,
                y_pred=data.y_pred,
                window=window)
            if recall is not None:
                return recall
            return 0

        reward = reward + Reward(
            reward_func=reward_func,
            min_unpenalized=ad_config.target_recall,
            penalty=-1)

    return reward
