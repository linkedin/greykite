import numpy as np
import pandas as pd
import pytest

from greykite.detection.common.ad_evaluation import confusion_matrix
from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.common.ad_evaluation import informedness_statistic
from greykite.detection.common.ad_evaluation import matthews_corrcoef
from greykite.detection.common.ad_evaluation import precision_score
from greykite.detection.common.ad_evaluation import range_based_precision_score
from greykite.detection.common.ad_evaluation import range_based_recall_score
from greykite.detection.common.ad_evaluation import recall_score
from greykite.detection.common.ad_evaluation import soft_f1_score
from greykite.detection.common.ad_evaluation import soft_precision_score
from greykite.detection.common.ad_evaluation import soft_recall_score


@pytest.fixture
def input_values():
    values = {
        "y_true": ["0", "0", "0", "0", "0", "1", "1", "1", "a", "a", "a", "a"],
        "y_pred": ["0", "0", "0", "1", "1", "1", "1", "1", "a", "a", "a", "a"],
        "expected_precision": {
            "0": 1.0,
            "1": 0.6,
            "a": 1.0
        },
        "expected_recall": {
            "0": 0.6,
            "1": 1.0,
            "a": 1.0
        },
        "expected_f1_score": {
            "0": 0.75,
            "1": 0.75,
            "a": 1.0
        },
        "expected_confusion_matrix": np.array([[3, 2, 0], [0, 3, 0], [0, 0, 4]]),
        "sample_weight": [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "expected_precision_with_weight": {
            "0": 1.0,
            "1": 1.0,
            "a": 0.0
        },
        "expected_matthews_corrcoef": 0.7872340425531915,
        "expected_matthews_corrcoef_with_weight": 1.0,
        "expected_informedness_statistic": 0.8,
        "expected_informedness_statistic_with_weight": 1.0

    }
    return values


@pytest.fixture
def soft_input_values():
    values = {
        "y_true": [0, 1, 1, 1, 0, 0, np.nan, np.nan, 0],
        "y_pred": [0, 0, 0, 1, 0, 1, np.nan, 1, np.nan],
        "expected_soft_precision": [
            {0.0: 0.5, 1.0: 0.5},
            {1.0: 0.5, 0.0: 0.0},
            {1.0: 1.0, 0.0: 0.0}],
        "expected_soft_recall": [
            {0.0: 2/3, 1.0: 1/3},
            {0.0: 1/3, 1.0: 2/3},
            {0.0: 1/3, 1.0: 1.0}],
        "expected_soft_f1": [
            {0.0: 0.5714285714285715, 1.0: 0.4},
            {0.0: 0.0, 1.0: 0.5714285714285715},
            {0.0: 0.0, 1.0: 1.0}]
    }

    return values


@pytest.fixture
def range_based_input_values():
    values = {
        "y_true": [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        "y_pred": [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        # Expected precision values with different positional biases
        "expected_range_based_precision_flat_positional_bias": 0.938,
        "expected_range_based_precision_front_positional_bias": 0.9,
        "expected_range_based_precision_middle_positional_bias": 0.958,
        "expected_range_based_precision_back_positional_bias": 0.975,
        # Expected recall values with different positional biases
        "expected_range_based_recall_flat_positional_bias": 0.817,
        "expected_range_based_recall_front_positional_bias": 0.908,
        "expected_range_based_recall_middle_positional_bias": 0.854,
        "expected_range_based_recall_back_positional_bias": 0.725,

    }

    return values


def test_precision_score(input_values):
    y_true = input_values["y_true"]
    y_pred = input_values["y_pred"]
    expected_precision = input_values["expected_precision"]

    # Tests list input.
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred
    )
    assert precision == expected_precision

    # Tests numpy array input.
    precision = precision_score(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred)
    )
    assert precision == expected_precision

    # Tests pandas Series input.
    precision = precision_score(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred)
    )
    assert precision == expected_precision

    # Tests pandas DataFrame input.
    precision = precision_score(
        y_true=pd.DataFrame(y_true),
        y_pred=pd.DataFrame(y_pred)
    )
    assert precision == expected_precision


def test_recall_score(input_values):
    y_true = input_values["y_true"]
    y_pred = input_values["y_pred"]
    expected_recall = input_values["expected_recall"]

    # Tests list input.
    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred
    )
    assert recall == expected_recall

    # Tests numpy array input.
    recall = recall_score(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred)
    )
    assert recall == expected_recall

    # Tests pandas Series input.
    recall = recall_score(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred)
    )
    assert recall == expected_recall

    # Tests pandas DataFrame input.
    recall = recall_score(
        y_true=pd.DataFrame(y_true),
        y_pred=pd.DataFrame(y_pred)
    )
    assert recall == expected_recall


def test_f1_score(input_values):
    y_true = input_values["y_true"]
    y_pred = input_values["y_pred"]
    expected_f1_score = input_values["expected_f1_score"]

    # Tests list input.
    f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred
    )
    assert {key: round(value, 2) for key, value in f1.items()} == expected_f1_score

    # Tests numpy array input.
    f1 = f1_score(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred)
    )
    assert {key: round(value, 2) for key, value in f1.items()} == expected_f1_score

    # Tests pandas Series input.
    f1 = f1_score(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred)
    )
    assert {key: round(value, 2) for key, value in f1.items()} == expected_f1_score

    # Tests pandas DataFrame input.
    f1 = f1_score(
        y_true=pd.DataFrame(y_true),
        y_pred=pd.DataFrame(y_pred)
    )
    assert {key: round(value, 2) for key, value in f1.items()} == expected_f1_score


def test_confusion_matrix(input_values):
    y_true = input_values["y_true"]
    pred = input_values["y_pred"]
    expected_confusion_matrix = input_values["expected_confusion_matrix"]
    confusion_mat = confusion_matrix(
        y_true=y_true,
        y_pred=pred
    )
    assert (confusion_mat.values == expected_confusion_matrix).all().all()


def test_error_and_warnings():
    # Not 1-D array.
    with pytest.raises(
            ValueError,
            match="The input for scoring must be 1"):
        precision_score(
            y_true=[[1, 2, 3], [4, 5, 6]],
            y_pred=[[1, 2, 3], [4, 5, 6]]
        )

    # Not equal length.
    with pytest.raises(
            ValueError,
            match="The input lengths must be the same, found"):
        precision_score(
            y_true=[1, 2, 3],
            y_pred=[1, 2]
        )

    # Warnings 1.
    with pytest.warns(
            UserWarning,
            match="The following categories do not appear in y_true column,"):
        precision_score(
            y_true=[1, 2, 2],
            y_pred=[1, 2, 4]
        )

    # Warnings 2.
    with pytest.warns(
            UserWarning,
            match="The following categories do not appear in y_pred column,"):
        precision_score(
            y_true=[1, 2, 3],
            y_pred=[1, 2, 2]
        )


def test_sample_weight(input_values):
    y_true = input_values["y_true"]
    y_pred = input_values["y_pred"]
    sample_weight = input_values["sample_weight"]
    expected_precision = input_values["expected_precision_with_weight"]

    # Tests list input.
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight
    )
    assert precision == expected_precision


def test_soft_precision_score(soft_input_values):
    y_true = soft_input_values["y_true"]
    y_pred = soft_input_values["y_pred"]
    expected_soft_precision = soft_input_values["expected_soft_precision"]

    # Tests list input.
    soft_precision = [soft_precision_score(
        y_true=y_true,
        y_pred=y_pred,
        window=window) for window in [0, 1, 2]]
    assert soft_precision == expected_soft_precision

    # Tests numpy array input.
    soft_precision = [soft_precision_score(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        window=window) for window in [0, 1, 2]]
    assert soft_precision == expected_soft_precision

    # Tests pandas Series input.
    soft_precision = [soft_precision_score(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred),
        window=window) for window in [0, 1, 2]]
    assert soft_precision == expected_soft_precision

    # Tests pandas DataFrame input.
    soft_precision = [soft_precision_score(
        y_true=pd.DataFrame(y_true),
        y_pred=pd.DataFrame(y_pred),
        window=window) for window in [0, 1, 2]]
    assert soft_precision == expected_soft_precision


def test_soft_recall_score(soft_input_values):
    y_true = soft_input_values["y_true"]
    y_pred = soft_input_values["y_pred"]
    expected_soft_recall = soft_input_values["expected_soft_recall"]

    # Tests list input.
    soft_recall = [soft_recall_score(
        y_true=y_true,
        y_pred=y_pred,
        window=window) for window in [0, 1, 2]]
    assert soft_recall == expected_soft_recall

    # Tests numpy array input.
    soft_recall = [soft_recall_score(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        window=window) for window in [0, 1, 2]]
    assert soft_recall == expected_soft_recall

    # Tests pandas Series input.
    soft_recall = [soft_recall_score(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred),
        window=window) for window in [0, 1, 2]]
    assert soft_recall == expected_soft_recall

    # Tests pandas DataFrame input.
    soft_recall = [soft_recall_score(
        y_true=pd.DataFrame(y_true),
        y_pred=pd.DataFrame(y_pred),
        window=window) for window in [0, 1, 2]]
    assert soft_recall == expected_soft_recall


def test_soft_f1(soft_input_values):
    y_true = soft_input_values["y_true"]
    y_pred = soft_input_values["y_pred"]
    expected_soft_f1 = soft_input_values["expected_soft_f1"]

    # Tests list input.
    soft_f1 = [soft_f1_score(
        y_true=y_true,
        y_pred=y_pred,
        window=window) for window in [0, 1, 2]]
    assert soft_f1 == expected_soft_f1

    # Tests numpy array input.
    soft_f1 = [soft_f1_score(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        window=window) for window in [0, 1, 2]]
    assert soft_f1 == expected_soft_f1

    # Tests pandas Series input.
    soft_f1 = [soft_f1_score(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred),
        window=window) for window in [0, 1, 2]]
    assert soft_f1 == expected_soft_f1

    # Tests pandas DataFrame input.
    soft_f1 = [soft_f1_score(
        y_true=pd.DataFrame(y_true),
        y_pred=pd.DataFrame(y_pred),
        window=window) for window in [0, 1, 2]]
    assert soft_f1 == expected_soft_f1


def test_range_based_precision_score(range_based_input_values):
    """Tests for range_based_precision_score function"""

    y_true = range_based_input_values["y_true"]
    y_pred = range_based_input_values["y_pred"]

    # Tests range-based precision with flat positional bias
    expected_precision = range_based_input_values["expected_range_based_precision_flat_positional_bias"]
    range_based_precision = range_based_precision_score(
        y_true=y_true,
        y_pred=y_pred,
        positional_bias="flat")
    assert round(range_based_precision, 3) == expected_precision

    # Tests range-based precision with front positional bias
    expected_precision = range_based_input_values["expected_range_based_precision_front_positional_bias"]
    range_based_precision = range_based_precision_score(
        y_true=y_true,
        y_pred=y_pred,
        positional_bias="front")
    assert round(range_based_precision, 3) == expected_precision

    # Tests range-based precision with middle positional bias
    expected_precision = range_based_input_values["expected_range_based_precision_middle_positional_bias"]
    range_based_precision = range_based_precision_score(
        y_true=y_true,
        y_pred=y_pred,
        positional_bias="middle")
    assert round(range_based_precision, 3) == expected_precision

    # Tests range-based precision with back positional bias
    expected_precision = range_based_input_values["expected_range_based_precision_back_positional_bias"]
    range_based_precision = range_based_precision_score(
        y_true=y_true,
        y_pred=y_pred,
        positional_bias="back")
    assert round(range_based_precision, 3) == expected_precision

    # Tests if the range_based implementation subsumes the classical recall implementation
    classical_precision = precision_score(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred)
    )
    precision = range_based_precision_score(
        y_true=y_true,
        y_pred=y_pred,
        range_based=False)
    assert round(precision, 3) == round(classical_precision[1], 3)


def test_range_based_recall_score(range_based_input_values):
    """Tests for range_based_recall_score function"""

    y_true = range_based_input_values["y_true"]
    y_pred = range_based_input_values["y_pred"]

    # Tests range-based recall with flat positional bias
    expected_recall = range_based_input_values["expected_range_based_recall_flat_positional_bias"]
    range_based_recall = range_based_recall_score(
        y_true=y_true,
        y_pred=y_pred,
        positional_bias="flat")
    assert round(range_based_recall, 3) == expected_recall

    # Tests range-based recall with front positional bias
    expected_recall = range_based_input_values["expected_range_based_recall_front_positional_bias"]
    range_based_recall = range_based_recall_score(
        y_true=y_true,
        y_pred=y_pred,
        positional_bias="front")
    assert round(range_based_recall, 3) == expected_recall

    # Tests range-based recall with middle positional bias
    expected_recall = range_based_input_values["expected_range_based_recall_middle_positional_bias"]
    range_based_recall = range_based_recall_score(
        y_true=y_true,
        y_pred=y_pred,
        positional_bias="middle")
    assert round(range_based_recall, 3) == expected_recall

    # Tests range-based recall with back positional bias
    expected_recall = range_based_input_values["expected_range_based_recall_back_positional_bias"]
    range_based_recall = range_based_recall_score(
        y_true=y_true,
        y_pred=y_pred,
        positional_bias="back")
    assert round(range_based_recall, 3) == expected_recall

    # Tests if the range_based implementation subsumes the classical recall implementation
    classical_recall = recall_score(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred)
    )
    recall = range_based_recall_score(
        y_true=y_true,
        y_pred=y_pred,
        range_based=False)
    assert round(recall, 3) == round(classical_recall[1], 3)


def test_matthews_corrcoef(input_values):
    y_true = input_values["y_true"]
    y_pred = input_values["y_pred"]
    sample_weight = input_values["sample_weight"]
    expected_mcc = input_values["expected_matthews_corrcoef"]
    expected_mcc_with_weight = input_values["expected_matthews_corrcoef_with_weight"]

    # Tests list input.
    mcc = matthews_corrcoef(
        y_true=y_true,
        y_pred=y_pred
    )
    assert mcc == pytest.approx(expected_mcc)

    # Tests numpy array input.
    mcc = matthews_corrcoef(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred)
    )
    assert mcc == pytest.approx(expected_mcc)

    # Tests pandas Series input.
    mcc = matthews_corrcoef(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred)
    )
    assert mcc == pytest.approx(expected_mcc)

    # Tests pandas DataFrame input.
    mcc = matthews_corrcoef(
        y_true=pd.DataFrame(y_true),
        y_pred=pd.DataFrame(y_pred)
    )
    assert mcc == pytest.approx(expected_mcc)

    # Tests expected result when using sample weights.
    mcc = matthews_corrcoef(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight
    )
    assert mcc == pytest.approx(expected_mcc_with_weight)


def test_informedness_statistic(input_values):
    y_true = input_values["y_true"]
    y_pred = input_values["y_pred"]
    sample_weight = input_values["sample_weight"]
    expected_informedness = input_values["expected_informedness_statistic"]
    expected_informedness_with_weight = input_values["expected_informedness_statistic_with_weight"]

    # Tests list input.
    informedness = informedness_statistic(
        y_true=y_true,
        y_pred=y_pred
    )
    assert informedness == pytest.approx(expected_informedness)

    # Tests numpy array input.
    informedness = informedness_statistic(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred)
    )
    assert informedness == pytest.approx(expected_informedness)

    # Tests pandas Series input.
    informedness = informedness_statistic(
        y_true=pd.Series(y_true),
        y_pred=pd.Series(y_pred)
    )
    assert informedness == pytest.approx(expected_informedness)

    # Tests pandas DataFrame input.
    informedness = informedness_statistic(
        y_true=pd.DataFrame(y_true),
        y_pred=pd.DataFrame(y_pred)
    )
    assert informedness == pytest.approx(expected_informedness)

    # Tests expected result when using sample weights.
    informedness = informedness_statistic(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight
    )
    assert informedness == pytest.approx(expected_informedness_with_weight)

    # Tests that informedness_statistic returns the same results as sensitivity + specificity - 1 for binary output.
    y_true = ["0", "0", "0", "0", "0", "1", "1", "1"]
    y_pred = ["0", "0", "0", "1", "1", "1", "1", "1"]
    informedness = informedness_statistic(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=None
    )
    recalls = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=None)
    assert informedness == sum(recalls.values()) - 1.0
