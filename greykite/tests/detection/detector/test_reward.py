from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.common.ad_evaluation import precision_score
from greykite.detection.common.ad_evaluation import recall_score
from greykite.detection.detector.ad_utils import partial_return
from greykite.detection.detector.reward import Reward


def test_reward():
    """Tests `Reward` class."""
    y_true = [True, True, False, True, True, False, False, False, True, True]
    y_pred = [False, True, False, True, True, False, True, False, True, False]

    f1 = partial_return(f1_score, True)
    prec = partial_return(precision_score, True)
    rec = partial_return(recall_score, True)

    raw_f1_value = f1(y_true=y_true, y_pred=y_pred)
    raw_recall_value = rec(y_true=y_true, y_pred=y_pred)

    assert round(raw_f1_value, 2) == 0.73
    assert round(raw_recall_value, 2) == 0.67

    obj_value = Reward(f1).apply(y_true=y_true, y_pred=y_pred)

    assert obj_value == raw_f1_value

    # This penalizes f1 values under 0.75 by `penalty == -1`
    obj_value = Reward(
        f1,
        min_unpenalized=0.75,
        penalty=-1).apply(y_true=y_true, y_pred=y_pred)

    assert obj_value == raw_f1_value - 1.0

    # The penalty won't take effect since `min_unpenalized < raw_f1_value`
    obj_value = Reward(
        f1,
        min_unpenalized=0.70,
        penalty=-1).apply(y_true=y_true, y_pred=y_pred)

    assert obj_value == raw_f1_value

    # Multiplicative penalty
    obj_value = Reward(
        f1,
        min_unpenalized=0.75,
        penalize_method="multiplicative",
        penalty=0.1).apply(y_true=y_true, y_pred=y_pred)

    assert obj_value == raw_f1_value * 0.1

    # Recall, penalty won't take effect since `0.66 < raw_recall_value`
    obj_value = Reward(
        rec,
        min_unpenalized=0.66,
        penalize_method="additive",
        penalty=-1).apply(y_true=y_true, y_pred=y_pred)

    assert obj_value == raw_recall_value

    # Recall, penalty will take effect since `0.66 > raw_recall_value`
    obj_value = Reward(
        rec,
        max_unpenalized=0.66,
        penalize_method="additive",
        penalty=+3.0).apply(y_true=y_true, y_pred=y_pred)

    assert obj_value == raw_recall_value + 3.0

    # Combining rewards (adding them)
    # In this scenario, we penalize all recalls less than 0.8 by -1
    # While we add it to raw F1
    combined_reward = (
        Reward(
            rec,
            min_unpenalized=0.8,
            penalize_method="additive",
            penalty=-1) +
        Reward(f1))

    obj_value = combined_reward.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == raw_f1_value + raw_recall_value - 1.0

    # This adds a numeric value to f1
    combined_reward = Reward(f1) + 13
    obj_value = combined_reward.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == raw_f1_value + 13

    # This multiplies a numeric value to f1
    combined_reward = (Reward(f1) * 0)
    obj_value = combined_reward.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == 0

    # This divides f1 by a numeric value
    combined_reward = (Reward(f1) / 0)
    obj_value = combined_reward.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == float("inf")

    # This divides 0 by f1
    combined_reward = (0 / Reward(f1))
    obj_value = combined_reward.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == 0

    # This divides 17 by f1
    combined_reward = (17 / Reward(f1))
    obj_value = combined_reward.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == 17.0 / raw_f1_value

    # This adds from right
    combined_reward = 13 + Reward(f1)
    obj_value = combined_reward.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == raw_f1_value + 13

    # Punishes recalls less than 0.8 harshly by assigning -inf
    # This is useful in constrained optimization
    combined_reward = (
        Reward(
            rec,
            min_unpenalized=0.8,
            penalize_method="additive",
            penalty=float("-inf")) +
        Reward(f1))

    obj_value = combined_reward.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == float("-inf")

    # Multiplication of recall and F1 with penalty on the recall part
    combined_reward = (
        Reward(
            rec,
            min_unpenalized=0.8,
            penalize_method="multiplicative",
            penalty=0.1) *
        Reward(f1))

    obj_value = combined_reward.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == raw_f1_value * raw_recall_value * 0.1

    # Apply the class operations to construct f1
    rec_obj = Reward(rec)
    prec_obj = Reward(prec)
    half_f1_obj = (2 * rec_obj * prec_obj) / (rec_obj + prec_obj)
    obj_value = half_f1_obj.apply(y_true=y_true, y_pred=y_pred)
    assert obj_value == raw_f1_value
