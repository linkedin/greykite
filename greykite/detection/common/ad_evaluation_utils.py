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
# original author: Saad Eddin Al Orjany

"""util functions"""

from typing import List
from typing import Optional

import numpy as np


def shift(arr, num: int, fill_value: int = np.nan):
    """Rolls a 1d-array in either direction, and applies a fill value on a portion of the array

    Parameters
    ----------
    arr : array-like, 1-D
        Array to roll and apply partial mask on
    num : `int`
        The patterns to be kept.
    fill_value: `int`
        After shifting elements, a value to fill in place of NAs in the resulint shifted array

    Returns
    -------
    arr : array-like, 1-D
        Rolled and partially-masked array
    """
    arr = np.roll(arr, num)
    if num < 0:
        arr[num:] = fill_value
    elif num > 0:
        arr[:num] = fill_value
    return arr


def prepare_anomaly_ranges(pointwise_anomalies, range_based: bool = True):
    """Convert a list of pointwise anomalies into a list of anomaly ranges

    Parameters
    ----------
    pointwise_anmalies : array-like, 1-D
        List of pointwise anomalies
    range_based : `bool`
        If False, each pointwise anomaly is treated as an anomaly range
        If True, adjacent pointwise anomalies will be merged into a single anomaly interval

    Returns
    -------
    anomaly_ranges: array-like, 2D
        2D-array representing anomaly ranges, with each element indicating the
        start and end indexes for an anomaly period.
    """

    if range_based:
        pointwise_anomalies = np.argwhere(pointwise_anomalies == 1).ravel()
        anomaly_ranges_shift_forward = shift(
            pointwise_anomalies,
            1,
            fill_value=pointwise_anomalies[0])
        anomaly_ranges_shift_backward = shift(
            pointwise_anomalies,
            -1,
            fill_value=pointwise_anomalies[-1])
        anomaly_ranges_start = np.argwhere((
            anomaly_ranges_shift_forward - pointwise_anomalies) != -1).ravel()
        anomaly_ranges_end = np.argwhere((
            pointwise_anomalies - anomaly_ranges_shift_backward) != -1).ravel()
        anomaly_ranges = np.hstack([
            pointwise_anomalies[anomaly_ranges_start].reshape(-1, 1),
            pointwise_anomalies[anomaly_ranges_end].reshape(-1, 1)])
    else:
        anomaly_ranges = np.argwhere(pointwise_anomalies == 1).repeat(2, axis=1)

    return anomaly_ranges


def get_cardinality_factor(
        overlap_count,
        cardinality_bias: Optional[str] = None):
    """Cardinalty factor used to penalize the overlap size & positional reward.

    Parameters
    ----------
    overlap_count : array-like, 1-D
        Accumulator used to keep track of how of overlaps between an anomaly range and a set of anomaly ranges.
    cardinality_bias: `str` or None, default None
        In the overlap reward, this is a penalization factor. If None, no cardinality penalty will be applied. If "reciprocal", the
        overlap reward will be penalized as it gets multiplied by the reciprocal of the number of detected anomaly ranges overlapping
        with the predicted anomaly range.

    Returns
    -------
    cardinality_factor: `float`
        A multiplying factor used to penalize higher cardinality: the number of overlaps of
        predicted anomaly ranges with a real anomaly range, and vice versa.
    """

    if cardinality_bias is not None:
        assert cardinality_bias == "reciprocal"

    overlap = overlap_count[0]
    assert overlap >= 0

    cardinality_factor = 1.0

    if cardinality_bias == "reciprocal" and overlap > 1:
        cardinality_factor /= overlap

    return cardinality_factor


def get_positional_reward(
        loc: int,
        anomaly_length: int,
        positional_bias: str = "flat"):
    """Positional reward for a single pointwise anomaly with an anomaly range

    Parameters
    ----------
    loc : `int`
        Location of the pointwise anomaly, within the anomaly range. Takes value in the range[1, anomaly_length]
    anomaly_length: `int`
        Length of the anomaly range used to score a pointwise anomaly within it.
    positional_bias : `str`, default "flat"
        The accepted options are:
        * "flat": Each index position of an anomaly range is equally important. Return the same value of 1.0 as the positional
          reward regardless of the location of the pointwise anomaly within the anomaly range.
        * "front": Positional reward is biased towards early detection, as earlier overlap locations of pointwise
          anomalies with an anomaly range are assigned higher rewards.
        * "middle": Positional reward is biased towards the detection of anomaly closer to its middle point, as overlap locations
          closer to the middle of an anomaly range are assigned higher rewards.
        * "back":  Positional reward is biased towards later detection, as later overlap locations of pointwise anomalies with an
          anomaly range are assigned higher rewards.

    Returns
    -------
    positional_reward: `float`
        Positional reward for the pointwise anomaly within an anomaly range.
    """

    assert 1 <= loc <= anomaly_length
    positional_reward = 1.0

    if positional_bias == "flat":
        return positional_reward
    elif positional_bias == "front":
        positional_reward = float(anomaly_length - loc + 1.0)
    elif positional_bias == "middle":
        if loc <= anomaly_length / 2.0:
            positional_reward = float(loc)
        else:
            positional_reward = float(anomaly_length - loc + 1.0)
    elif positional_bias == "back":
        positional_reward = float(loc)
    else:
        raise Exception("Invalid positional bias value")
    return positional_reward


def get_overlap_size_and_position_reward(
        anomaly_range_1: List[int],
        anomaly_range_2: List[int],
        overlap_count,
        positional_bias: str = "flat"):
    """Calculates overlap reward for both size and position of two anomaly ranges

    Parameters
    ----------
    anomaly_range_1 : list [int]
        A list of two integers, representing the start and end indexes of an anomaly range
    anomaly_range_2 : list [int]
        A list of two integers, representing the start and end indexes of an anomaly range
    overlap_count : array-like, 1-D
        Accumulator used to keep track of how of overlaps between an anomaly range and a set of anomaly ranges.
    positional_bias : `str`, default "flat"
        If "flat", each index position of an anomaly range is equally important. Return the same
          number, 1.0, as the positional reward regardless of the location of the pointwise anomaly
          within the anomaly range.
        If "front", reward is biased towards early detection, as earlier overlap locations of pointwise
          anomalies with an anomaly range are assigned higher rewards.
        If "middle", reward is biased towards the detection of anomaly closer to its middle point, as
          overlap locations closer to the middle of an anomaly range are assigned higher rewards.
        If "back", reward is biased towards later detection, as later overlap locations of pointwise
          anomalies with an anomaly range are assigned higher rewards.

    Returns
    -------
    overlap_size_and_position_reward: `float`
        Overlap reward for both size and position of two anomaly ranges.
    """
    overlap_size_and_position_reward = 0

    if anomaly_range_1[1] < anomaly_range_2[0] or anomaly_range_1[0] > anomaly_range_2[1]:
        return overlap_size_and_position_reward
    else:
        overlap_count[0] += 1
        overlap = np.zeros(anomaly_range_1.shape)
        overlap[0] = max(anomaly_range_1[0], anomaly_range_2[0])
        overlap[1] = min(anomaly_range_1[1], anomaly_range_2[1])

        anomaly_length = anomaly_range_1[1] - anomaly_range_1[0] + 1
        overlap_positional_reward = 0
        max_positional_reward = 0
        for local_idx in range(1, anomaly_length + 1):
            temp_reward = get_positional_reward(local_idx, anomaly_length, positional_bias)
            max_positional_reward += temp_reward

            idx = anomaly_range_1[0] + local_idx - 1
            if overlap[0] <= idx <= overlap[1]:
                overlap_positional_reward += temp_reward

        if max_positional_reward > 0:
            overlap_size_and_position_reward = overlap_positional_reward / max_positional_reward

        return overlap_size_and_position_reward


def compute_range_based_score(
        anomaly_ranges_1,
        anomaly_ranges_2,
        alpha: float = 0.5,
        positional_bias: str = "flat",
        cardinality_bias: Optional[str] = None):
    """Given two lists of anomaly ranges, calculate a range-based score over the time series represented by the first list
    of anomaly ranges. If ``anomaly_ranges_1`` is the predicted anomaly ranges, the result is range-based precision score. If
    ``anomaly_range_1`` is the real anomaly ranges, the result is range-based recall score.

    Parameters
    ----------
    anomaly_ranges_1 : array-like, 2D
        2D-array representing anomaly ranges, with each element indicating the
        start and end indexes for an anomaly period.
    anomaly_ranges_2 : array-like, 2D
        2D-array representing anomaly ranges, with each element indicating the
        start and end indexes for an anomaly period.
    alpha : `float`
        Reward weighting term for the two main reward terms for the real anomaly range recall score: existence
        and overlap rewards.
    positional_bias : `str`, default "flat"
        If "flat", each index position of an anomaly range is equally important. Return the same
          number, 1.0, as the positional reward regardless of the location of the pointwise anomaly
          within the anomaly range.
        If "front", reward is biased towards early detection, as earlier overlap locations of pointwise
          anomalies with an anomaly range are assigned higher rewards.
        If "middle", reward is biased towards the detection of anomaly closer to its middle point, as
          overlap locations closer to the middle of an anomaly range are assigned higher rewards.
        If "back", reward is biased towards later detection, as later overlap locations of pointwise
          anomalies with an anomaly range are assigned higher rewards.
    cardinality_bias: `str` or None, default None
        In the overlap reward, this is a penalization factor. If None, no cardinality penalty will be applied. If "reciprocal", the
        overlap reward will be penalized as it gets multiplied by the reciprocal of the number of detected anomaly ranges overlapping
        with the predicted anomaly range.

    Returns
    -------
    overlap_size_and_position_reward: `float`
        Overlap reward for both size and position of two anomaly ranges.
    """

    assert 0 <= alpha <= 1
    assert positional_bias in ["flat", "front", "middle", "back"]
    if cardinality_bias is not None:
        assert cardinality_bias == "reciprocal"

    score = 0.0

    for range_idx1 in range(len(anomaly_ranges_1)):
        overlap_count = [0]
        overlap_size_and_position_reward = 0
        real_range = anomaly_ranges_1[range_idx1, :]
        for range_idx2 in range(len(anomaly_ranges_2)):
            predicted_range = anomaly_ranges_2[range_idx2, :]
            overlap_size_and_position_reward += get_overlap_size_and_position_reward(
                real_range,
                predicted_range,
                overlap_count,
                positional_bias)

        cardinality_factor = get_cardinality_factor(overlap_count, cardinality_bias)
        overlap_reward = cardinality_factor * overlap_size_and_position_reward

        existence_reward = 1 if overlap_count[0] > 0 else 0
        score += alpha * existence_reward + (1 - alpha) * overlap_reward

    score /= len(anomaly_ranges_1)
    return score
