from greykite.detection.common.ad_evaluation import soft_f1_score
from greykite.detection.common.ad_evaluation import soft_precision_score
from greykite.detection.common.ad_evaluation import soft_recall_score
from greykite.detection.detector.ad_utils import partial_return
from greykite.detection.detector.config import F1
from greykite.detection.detector.config import PRECISION
from greykite.detection.detector.config import RECALL
from greykite.detection.detector.config import ADConfig
from greykite.detection.detector.config_to_reward import config_to_reward
from greykite.detection.detector.data import DetectorData


# Soft F1 score for the True label:
calc_soft_f1 = partial_return(soft_f1_score, True)
# Soft Precision score, for the True label:
calc_soft_precision = partial_return(soft_precision_score, True)
# Soft Recall score for the True label:
calc_soft_recall = partial_return(soft_recall_score, True)


def test_config_to_reward():
    """Tests `Reward` class."""
    # Defines Test Data.
    y_true = [True, True, False, True, True, False, False, False, True, True]
    y_pred = [False, True, False, True, True, False, True, False, True, False]
    data = DetectorData(y_true=y_true, y_pred=y_pred)

    # This calculates the metrics in simple way.
    # We will use these values during testing.
    raw_f1_value = calc_soft_f1(y_true=y_true, y_pred=y_pred, window=0)
    raw_recall_value = calc_soft_recall(y_true=y_true, y_pred=y_pred, window=0)
    raw_precision_value = calc_soft_precision(y_true=y_true, y_pred=y_pred, window=0)

    # Soft F1 with window of 2.
    soft_f1_value = calc_soft_f1(y_true=y_true, y_pred=y_pred, window=2)

    # Tests anomaly percent config.
    # Case 1:
    ad_config = ADConfig(target_anomaly_percent=50)
    reward = config_to_reward(ad_config)

    # Since the actual percent in `y_pred` is 50%, we expect a zero reward (best case).
    assert reward.apply(data) == 0

    # Case 2:
    ad_config = ADConfig(target_anomaly_percent=20)
    reward = config_to_reward(ad_config)

    # Due to mismatch between 20% and 30% anomaly percent (diff = -0.3)
    # and the penaly being -1, we expect -1.3
    assert reward.apply(data) == -1.3

    # Tests F1 as objective.
    ad_config = ADConfig(objective=F1)
    assert ad_config.objective == F1
    reward = config_to_reward(ad_config)
    assert reward.apply(data) == raw_f1_value

    # Tests recall as objective.
    ad_config = ADConfig(objective=RECALL)
    assert ad_config.objective == RECALL
    reward = config_to_reward(ad_config)
    assert reward.apply(data) == raw_recall_value

    # Tests precision as objective.
    ad_config = ADConfig(objective=PRECISION)
    assert ad_config.objective == PRECISION
    reward = config_to_reward(ad_config)
    assert reward.apply(data) == raw_precision_value

    # Tests F1 with window of 2 as objective.
    ad_config = ADConfig(
        objective=F1,
        soft_window_size=2)
    assert ad_config.objective == F1
    reward = config_to_reward(ad_config)
    assert reward.apply(data) == soft_f1_value

    # Tests recall target.
    # Case 1: We set the recall as the actual recall.
    # In this case, we should not get any penalty.
    ad_config = ADConfig(target_recall=raw_recall_value)
    assert ad_config.target_recall == raw_recall_value
    reward = config_to_reward(ad_config)
    assert reward.apply(data) == raw_recall_value

    # Case 2: We set the recall as the actual recall plus a very small value (0.01).
    # In this case we should get penalized by -1.
    ad_config = ADConfig(target_recall=raw_recall_value + 0.01)
    reward = config_to_reward(ad_config)
    assert reward.apply(data) == raw_recall_value - 1.0

    # Test objective being RECALL and having a target precision.
    # Case 1: We let the precision to be the actual precision.
    # We expect no penalty in this case.
    ad_config = ADConfig(
        objective=RECALL,
        target_precision=raw_precision_value)
    reward = config_to_reward(ad_config)
    assert reward.apply(data) == raw_recall_value + raw_precision_value

    # Case 2: We let the precision to be the actual precision plus a small value (0.01).
    # We expect to be penalized by -1 this time.
    ad_config = ADConfig(
        objective=RECALL,
        target_precision=raw_precision_value + 0.01)
    assert ad_config.target_precision == raw_precision_value + 0.01
    reward = config_to_reward(ad_config)
    assert round(reward.apply(data), 2) == round(raw_recall_value + raw_precision_value - 1, 2)
