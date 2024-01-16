import numpy as np

from greykite.detection.common.ad_evaluation_utils import compute_range_based_score
from greykite.detection.common.ad_evaluation_utils import get_cardinality_factor
from greykite.detection.common.ad_evaluation_utils import get_overlap_size_and_position_reward
from greykite.detection.common.ad_evaluation_utils import get_positional_reward
from greykite.detection.common.ad_evaluation_utils import prepare_anomaly_ranges


def test_prepare_anomaly_ranges():
    """Tests for prepare_anomaly_ranges function"""

    y_true = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    y_pred = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0]

    real_anomaly_ranges = prepare_anomaly_ranges(np.array(y_true))
    expected_real_anomaly_ranges = np.array([[3, 7], [11, 18]])
    predicted_anomaly_ranges = prepare_anomaly_ranges(np.array(y_pred))
    expected_predicted_anomaly_ranges = np.array([[2, 5], [11, 12], [15, 17]])

    assert np.array_equal(real_anomaly_ranges, expected_real_anomaly_ranges)
    assert np.array_equal(predicted_anomaly_ranges, expected_predicted_anomaly_ranges)

    real_anomaly_ranges = prepare_anomaly_ranges(np.array(y_true), range_based=False)
    expected_real_anomaly_ranges = [[3, 3], [4, 4], [5, 5], [6, 6], [7, 7],
                                    [11, 11], [12, 12], [13, 13], [14, 14],
                                    [15, 15], [16, 16], [17, 17], [18, 18]]
    predicted_anomaly_ranges = prepare_anomaly_ranges(np.array(y_pred), range_based=False)
    expected_predicted_anomaly_ranges = [[2, 2], [3, 3], [4, 4], [5, 5], [11, 11],
                                         [12, 12], [15, 15], [16, 16], [17, 17]]

    assert all(anomaly_range in real_anomaly_ranges for anomaly_range in expected_real_anomaly_ranges)
    assert all(anomaly_range in predicted_anomaly_ranges for anomaly_range in expected_predicted_anomaly_ranges)


def test_get_cardinality_factor():
    """Tests for get_cardinality_factor function"""

    cardinality_factor = get_cardinality_factor(overlap_count=[1])
    assert cardinality_factor == 1.0

    # When cardinality bias is set to "reciprocal", return the reciprocal of x where x is the number of
    # overlapping anomaly ranges with a certain anomaly range.
    # An example of this is when there is a real anomaly range, with two predicted anomaly ranges overlapping
    # with it. In this case, the cardinality factor should be 1/2
    cardinality_factor = get_cardinality_factor(overlap_count=[2], cardinality_bias="reciprocal")
    assert cardinality_factor == 0.5


def test_get_positional_reward():
    """Tests for get_positional_reward function"""

    # With "flat" positional bias, no matter where a pointwise anomaly is within an anomaly
    # range of length 5, return a fixed value (1.0)
    positional_reward = get_positional_reward(loc=1, anomaly_length=5, positional_bias="flat")
    assert positional_reward == 1.0
    positional_reward = get_positional_reward(loc=3, anomaly_length=5, positional_bias="flat")
    assert positional_reward == 1.0
    positional_reward = get_positional_reward(loc=5, anomaly_length=5, positional_bias="flat")
    assert positional_reward == 1.0

    # With "front" positional bias, higher positional reward is allocated to a pointwise
    # anomaly the earlier it is in the anomaly range of length 5
    positional_reward = get_positional_reward(loc=1, anomaly_length=5, positional_bias="front")
    assert positional_reward == 5.0
    positional_reward = get_positional_reward(loc=5, anomaly_length=5, positional_bias="front")
    assert positional_reward == 1.0

    # With "middle" positional bias, higher positional reward is allocated to a pointwise anomaly
    # the closer to the middle it is of the anomaly range of length 5
    positional_reward = get_positional_reward(loc=1, anomaly_length=5, positional_bias="middle")
    assert positional_reward == 1.0
    positional_reward = get_positional_reward(loc=3, anomaly_length=5, positional_bias="middle")
    assert positional_reward == 3.0
    positional_reward = get_positional_reward(loc=5, anomaly_length=5, positional_bias="middle")
    assert positional_reward == 1.0

    # With "back" positional bias, higher positional reward is allocated to a pointwise anomaly
    # the later it is in the anomaly range of length 5
    positional_reward = get_positional_reward(loc=1, anomaly_length=5, positional_bias="back")
    assert positional_reward == 1.0
    positional_reward = get_positional_reward(loc=5, anomaly_length=5, positional_bias="back")
    assert positional_reward == 5.0


def test_get_overlap_size_and_position_reward():
    """Tests for get_overlap_size_and_position_reward function"""

    # No overlap produces an overlap_size_and_position_reward of zero
    anomaly_range_1 = np.array([3, 8])
    anomaly_range_2 = np.array([9, 10])
    overlap_size_and_position_reward = get_overlap_size_and_position_reward(
        anomaly_range_1=anomaly_range_1,
        anomaly_range_2=anomaly_range_2,
        overlap_count=[0],
        positional_bias="flat"
    )
    assert overlap_size_and_position_reward == 0

    # Overlap of anomaly_range_2 with anomaly_range_1 happens at the beginning of anomaly_range_1
    # This overlap yields higher size and positional reward when positional bias is set to "front" than when
    # set to "middle" or "back".
    anomaly_range_1 = np.array([3, 8])
    anomaly_range_2 = np.array([3, 5])
    overlap_size_and_position_reward = get_overlap_size_and_position_reward(
        anomaly_range_1=anomaly_range_1,
        anomaly_range_2=anomaly_range_2,
        overlap_count=[0],
        positional_bias="front")
    assert round(overlap_size_and_position_reward, 3) == 0.714

    overlap_size_and_position_reward = get_overlap_size_and_position_reward(
        anomaly_range_1=anomaly_range_1,
        anomaly_range_2=anomaly_range_2,
        overlap_count=[0],
        positional_bias="middle")
    assert round(overlap_size_and_position_reward, 3) == 0.50

    overlap_size_and_position_reward = get_overlap_size_and_position_reward(
        anomaly_range_1=anomaly_range_1,
        anomaly_range_2=anomaly_range_2,
        overlap_count=[0],
        positional_bias="back")
    assert round(overlap_size_and_position_reward, 3) == 0.286

    # Overlap of anomaly_range_2 with anomaly_range_1 happens at the end of anomaly_range_1
    # This overlap yields higher size and positional reward when positional bias is set to "back" than when
    # set to "middle" or "front".
    anomaly_range_1 = np.array([3, 8])
    anomaly_range_2 = np.array([5, 9])
    overlap_size_and_position_reward = get_overlap_size_and_position_reward(
        anomaly_range_1=anomaly_range_1,
        anomaly_range_2=anomaly_range_2,
        overlap_count=[0],
        positional_bias="front")
    assert round(overlap_size_and_position_reward, 3) == 0.476

    overlap_size_and_position_reward = get_overlap_size_and_position_reward(
        anomaly_range_1=anomaly_range_1,
        anomaly_range_2=anomaly_range_2,
        overlap_count=[0],
        positional_bias="middle")
    assert round(overlap_size_and_position_reward, 3) == 0.75

    overlap_size_and_position_reward = get_overlap_size_and_position_reward(
        anomaly_range_1=anomaly_range_1,
        anomaly_range_2=anomaly_range_2,
        overlap_count=[0],
        positional_bias="back")
    assert round(overlap_size_and_position_reward, 3) == 0.857


def test_compute_range_based_score():
    """Tests for compute_range_based_score function"""

    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0])

    real_anomaly_ranges = prepare_anomaly_ranges(y_true)
    predicted_anomaly_ranges = prepare_anomaly_ranges(y_pred)

    recall = compute_range_based_score(
        real_anomaly_ranges,
        predicted_anomaly_ranges,
        alpha=0.5,
        positional_bias="flat")
    precision = compute_range_based_score(
        predicted_anomaly_ranges,
        real_anomaly_ranges,
        alpha=0.5, positional_bias="flat")
    assert round(precision, 3) == 0.958
    assert round(recall, 3) == 0.806

    recall = compute_range_based_score(
        real_anomaly_ranges,
        predicted_anomaly_ranges,
        alpha=0.5, positional_bias="front")
    precision = compute_range_based_score(
        predicted_anomaly_ranges,
        real_anomaly_ranges,
        alpha=0.5,
        positional_bias="front")
    assert round(precision, 3) == 0.933
    assert round(recall, 3) == 0.867

    recall = compute_range_based_score(
        real_anomaly_ranges,
        predicted_anomaly_ranges,
        alpha=0.5,
        positional_bias="middle")
    precision = compute_range_based_score(
        predicted_anomaly_ranges,
        real_anomaly_ranges,
        alpha=0.5,
        positional_bias="middle")
    assert round(precision, 3) == 0.972
    assert round(recall, 3) == 0.817

    recall = compute_range_based_score(
        real_anomaly_ranges,
        predicted_anomaly_ranges,
        alpha=0.5, positional_bias="back")
    precision = compute_range_based_score(
        predicted_anomaly_ranges,
        real_anomaly_ranges,
        alpha=0.5, positional_bias="back")
    assert round(precision, 3) == 0.983
    assert round(recall, 3) == 0.746

    recall = compute_range_based_score(
        real_anomaly_ranges,
        predicted_anomaly_ranges,
        alpha=0.5,
        positional_bias="front",
        cardinality_bias="reciprocal")
    precision = compute_range_based_score(
        predicted_anomaly_ranges,
        real_anomaly_ranges,
        alpha=0.5,
        positional_bias="front",
        cardinality_bias="reciprocal")
    assert round(precision, 3) == 0.933
    assert round(recall, 3) == 0.783
