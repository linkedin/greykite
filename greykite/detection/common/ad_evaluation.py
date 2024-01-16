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
# original author: Saad Eddin Al Orjany, Sayan Patra, Reza Hosseini, Kaixu Yang

"""Evaluation functions."""

import functools
import warnings
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from sklearn import metrics

from greykite.detection.common.ad_evaluation_utils import compute_range_based_score
from greykite.detection.common.ad_evaluation_utils import prepare_anomaly_ranges


INPUT_COL_NAME = "input_col"


def validate_categorical_input(score_func):
    """Decorator function to validate categorical scoring function input,
    and unifies the input type to pandas.Series.
    """

    @functools.wraps(score_func)
    def score_func_wrapper(
            y_true: Union[list, np.array, pd.Series, pd.DataFrame],
            y_pred: Union[list, np.array, pd.Series, pd.DataFrame],
            *args,
            **kwargs) -> np.array:
        actual = pd.DataFrame(y_true).reset_index(drop=True)
        pred = pd.DataFrame(y_pred).reset_index(drop=True)
        if actual.shape[-1] != 1 or pred.shape[-1] != 1:
            raise ValueError(f"The input for scoring must be 1-D array, found {actual.shape} and {pred.shape}")
        if actual.shape != pred.shape:
            raise ValueError(f"The input lengths must be the same, found {actual.shape} and {pred.shape}")
        actual.columns = [INPUT_COL_NAME]
        pred.columns = [INPUT_COL_NAME]
        # Drop rows with NA values in either actual or pred
        merged_df = pd.concat([actual, pred], axis=1).dropna()
        actual = merged_df.iloc[:, [0]]
        pred = merged_df.iloc[:, [1]]
        category_in_actual_set = set(actual[INPUT_COL_NAME])
        category_in_pred_set = set(pred[INPUT_COL_NAME])
        pred_minus_actual = category_in_pred_set.difference(category_in_actual_set)
        if pred_minus_actual:
            warnings.warn(f"The following categories do not appear in y_true column, "
                          f"the recall may be undefined.\n{pred_minus_actual}")
        actual_minus_pred = category_in_actual_set.difference(category_in_pred_set)
        if actual_minus_pred:
            warnings.warn(f"The following categories do not appear in y_pred column, "
                          f"the precision may be undefined.\n{actual_minus_pred}")
        # Adds a list wrapper below since `sklearn >= 1.1` restricts the input types and shapes.
        return score_func(
            y_true=list(actual[INPUT_COL_NAME].reset_index(drop=True)),
            y_pred=list(pred[INPUT_COL_NAME].reset_index(drop=True)),
            *args,
            **kwargs
        )

    return score_func_wrapper


@validate_categorical_input
def precision_score(
        y_true,
        y_pred,
        sample_weight=None):
    """Computes the precision scores for two arrays.

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual categories.
    y_pred : array-like, 1-D
        The predicted categories.
    sample_weight : array-like, 1-D
        The sample weight.

    Returns
    -------
    precision : `dict`
        The precision score for different categories.
        The keys are the categories, and the values are the precisions.
    """
    actual_category = pd.unique(y_true)
    pred_category = pd.unique(y_pred)
    labels = pd.unique(np.concatenate([actual_category, pred_category]))
    precisions_array = metrics.precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels=labels,
        sample_weight=sample_weight,
        zero_division=0
    )
    precisions = {}
    for label, precision in zip(labels, precisions_array):
        precisions[label] = precision
    return precisions


@validate_categorical_input
def recall_score(
        y_true,
        y_pred,
        sample_weight=None):
    """Computes the recall scores for two arrays.

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual categories.
    y_pred : array-like, 1-D
        The predicted categories.
    sample_weight : array-like, 1-D
        The sample weight.

    Returns
    -------
    recall : `dict`
        The recall score for different categories.
        The keys are the categories, and the values are the recalls.
    """
    actual_category = pd.unique(y_true)
    pred_category = pd.unique(y_pred)
    labels = pd.unique(np.concatenate([actual_category, pred_category]))
    recalls_array = metrics.recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels=labels,
        sample_weight=sample_weight,
        zero_division=0
    )
    recalls = {}
    for label, recall in zip(labels, recalls_array):
        recalls[label] = recall
    return recalls


@validate_categorical_input
def f1_score(
        y_true,
        y_pred,
        sample_weight=None):
    """Computes the F1 scores for two arrays.

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual categories.
    y_pred : array-like, 1-D
        The predicted categories.
    sample_weight : array-like, 1-D
        The sample weight.

    Returns
    -------
    recall : `dict`
        The recall score for different categories.
        The keys are the categories, and the values are the recalls.
    """
    actual_category = pd.unique(y_true)
    pred_category = pd.unique(y_pred)
    labels = pd.unique(np.concatenate([actual_category, pred_category]))
    f1s_array = metrics.f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels=labels,
        sample_weight=sample_weight,
        zero_division=0
    )
    f1_scores = {}
    for label, f1 in zip(labels, f1s_array):
        f1_scores[label] = f1
    return f1_scores


@validate_categorical_input
def matthews_corrcoef(
        y_true,
        y_pred,
        sample_weight=None):
    """Computes the Matthews correlation coefficient for two arrays.
    The statistic is also known as the phi coefficient.
    The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications.
    It takes into account true and false positives and negatives and is generally regarded as a balanced measure
    which can be used even if the classes are of very different sizes.
    The MCC is in essence a correlation coefficient value between -1 and +1 (inclusive).
    One can interpret this coefficient as follows:

        - +1 represents a perfect prediction.
        - 0 represents an average random prediction.
        - -1 represents an inverse prediction.

    For more information, please consult the `wiki page <https://en.wikipedia.org/wiki/Phi_coefficient>`_.

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual categories.
    y_pred : array-like, 1-D
        The predicted categories.
    sample_weight : array-like, 1-D or None, default None
        The sample weight.

    Returns
    -------
    result : `float`
        The Matthews correlation coefficient.
    """
    return metrics.matthews_corrcoef(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight)


@validate_categorical_input
def informedness_statistic(
        y_true,
        y_pred,
        sample_weight=None):
    """Computes the Informedness also known as the Youden's J statistic for two arrays.
    Youden's J statistic is defined as J = sensitivity + specificity - 1 for a binary output.
    Informedness is its generalization to the multiclass case and estimates the probability of an informed decision.
    Note that in binary classification, we have:

        - sensitivity: recall of the positive class.
        - specificity: recall of the negative class.

    The index gives equal weight to false positive and false negative values.
    In other words, all algorithms with the same value of the index give the same proportion of total misclassified results.
    Its value ranges from -1 through +1 (inclusive).
    One can interpret this statistic as follows:

        - +1 represents that there are no false positives or false negatives, i.e. the algorithm is perfect.
        - 0  respresents when an algorithm gives the same proportion of positive results with and without an anomaly, i.e the test is useless.
        - -1 represents that the classification yields only false positives and false negatives. It's an inverse prediction.

    For more information, please consult the `wiki page <https://en.wikipedia.org/wiki/Youden%27s_J_statistic>`_.

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual categories.
    y_pred : array-like, 1-D
        The predicted categories.
    sample_weight : array-like, 1-D or None, default None
        The sample weight.

    Returns
    -------
    result : `float`
        The informedness statistic.
    """
    return metrics.balanced_accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        adjusted=True)


@validate_categorical_input
def confusion_matrix(
        y_true,
        y_pred,
        sample_weight=None):
    """Computes the confusion matrix for two arrays.

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual categories.
    y_pred : array-like, 1-D
        The predicted categories.
    sample_weight : array-like, 1-D
        The sample weight.

    Returns
    -------
    confusion_matrix : `pandas.DataFrame`
        The confusion matrix.
    """
    actual_category = pd.unique(y_true)
    pred_category = pd.unique(y_pred)
    all_category = pd.unique(np.concatenate([actual_category, pred_category]))
    matrix = metrics.confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=all_category,
        sample_weight=sample_weight
    )
    matrix = pd.DataFrame(matrix)
    matrix.index = pd.MultiIndex.from_arrays([["Actual"] * len(all_category), matrix.index])
    matrix.columns = pd.MultiIndex.from_arrays([["Pred"] * len(all_category), matrix.columns])
    return matrix


@validate_categorical_input
def soft_recall_score(
        y_true,
        y_pred,
        window):
    """Computes the soft recall score for two classes, usually labeled 1 and 0 to denote
    an anomaly/ alert and not an anomaly/ alert, for `y_true` and `y_pred` respectively.

    soft_recall(window) is defined as the proportion of anomalies that were correctly
    alerted within the window size. Mathematically,

    soft_precision(window) = TruePositive(window)/ sum_i (y_true_i == 1),
    where
    TruePositive(window) = sum_i(y_true_i == 1, max(y_pred_{i-window}, ..., y_pred_{i+window}) == 1).

    For example, let window = 2.
    If the ith value in `y_true` is an anomaly (labeled 1), then we say the anomaly
    is predicted if any of i-2, i-1, i, i+1, i+2 value in `y_pred` is an alert (labeled 1).
    True Positive (window) is the sum of all such predicted anomalies.

    As far as we know these soft metrics do not appear in related work at least in this simple form.
    These metric were introduced by Reza Hosseini and Sayan Patra as a part of this work.
    They are found to be a very simple yet powerful extension of Precision/Recall in our work.

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual categories.
    y_pred : array-like, 1-D
        The predicted categories.
    window : `int`
        The window size to determine True Positives.

    Returns
    -------
    recall : `dict`
        The recall scores for various categories.

    Examples
    --------
    >>> y_true = [0, 1, 1, 1, 0, 0]
    >>> y_pred = [0, 0, 0, 1, 0, 1]
    >>> print([soft_recall_score(y_true, y_pred, window) for window in [0, 1, 2]])
    [0.3333333333333333, 0.6666666666666666, 1.0]
    """
    if not isinstance(window, int) or window < 0:
        raise ValueError(f"Input value of the parameter window ({window}) is not a non-negative integer.")

    # If `window` is 0, we revert to the standard definition directly (to save computation time).
    if window == 0:
        return recall_score(y_true=y_true, y_pred=y_pred)

    lag_y_pred = pd.DataFrame({
        "y_pred_0": y_pred
    })

    for i in np.arange(-window, window + 1):
        # Due to shifting there will be missing data at the beginning and the end of
        # the series. ffill and bfill interpolates these edge cases.
        lag_y_pred[f"y_pred_{i}"] = lag_y_pred["y_pred_0"].shift(i).ffill().bfill()

    y_pred_soft = lag_y_pred.any(axis="columns")

    return recall_score(y_true, y_pred_soft)


@validate_categorical_input
def soft_precision_score(
        y_true,
        y_pred,
        window):
    """Computes the soft precision score for two classes, usually labeled 1 and 0 to denote
    an anomaly and not an anomaly.

    soft_precision(window) is defined as the proportion of alerts that corresponds to an
    anomaly within the window size. Mathematically,

    soft_precision(window) = TruePositive(window)/ sum_i (y_pred_i == 1),
    where
    TruePositive(window) = sum_i(y_pred_i == 1, max(actual_{i-window}, ..., actual_{i+window}) == 1)

    For eg. let window = 2.
    If the ith value in `y_pred` is an alert (labeled 1), then we say the anomaly
    is predicted if any of i-2, i-1, i, i+1, i+2 value in `y_pred` is an anomaly (labeled 1).

    True Positive (window) is the sum of all such captured anomalies.

    As far as we know these soft metrics do not appear in related work at least in this simple form.
    These metric were introduced by Reza Hosseini and Sayan Patra as a part of this work.
    They are found to be a very simple yet powerful extension of Precision/Recall in our work.

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual categories.
    y_pred : array-like, 1-D
        The predicted categories.
    window : `int`
        The window size to determine True Positives.


    Returns
    -------
    recall : `dict`
        The precision scores for various categories.

    Examples
    --------
    >>> y_true = [0, 1, 1, 1, 0, 0]
    >>> y_pred = [0, 0, 0, 1, 0, 1]
    >>> print([soft_precision_score(y_true, y_pred, window) for window in [0, 1, 2]])
    [0.5, 0.5, 1.0]
    """
    if not isinstance(window, int) or window < 0:
        raise ValueError(f"Input value of the parameter window ({window}) is not a non-negative integer.")

    # If `window` is 0, we revert to the standard definition directly (to save computation time).
    if window == 0:
        return precision_score(y_true=y_true, y_pred=y_pred)

    lag_y_true = pd.DataFrame({
        "y_true_0": y_true
    })

    for i in np.arange(-window, window + 1):
        # Due to shifting there will be missing data at the beginning and the end of
        # the series. ffill and bfill interpolates these edge cases.
        lag_y_true[f"y_true_{i}"] = lag_y_true["y_true_0"].shift(i).ffill().bfill()

    y_true_soft = lag_y_true.any(axis="columns")

    return precision_score(y_true_soft, y_pred)


@validate_categorical_input
def soft_f1_score(
        y_true,
        y_pred,
        window):
    """Computes the soft F1 score for two classes, usually labeled 1 and 0 to denote
    an anomaly and not an anomaly.
    Soft F1 is simply calculated from
    - Soft Precision: `~greykite.detection.common.evaluation.soft_precision_score` and
    - Soft Recall: `~greykite.detection.common.evaluation.soft_recall_score`
    using the standard formula for F1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual categories.
    y_pred : array-like, 1-D
        The predicted categories.
    window : `int`
        The window size to determine True Positives.

    Returns
    -------
    recall : `dict`
        The precision scores for various categories.
    """
    soft_precision = soft_precision_score(
        y_true=y_true,
        y_pred=y_pred,
        window=window)

    soft_recall = soft_recall_score(
        y_true=y_true,
        y_pred=y_pred,
        window=window)

    soft_f1 = {}
    # Soft f1 can only be defined for categories (e.g. True, False) that
    # appear in results of both precision and recall.
    # Therefore we first find the intersection.
    admissable_categories = set(soft_precision.keys()).intersection(soft_recall.keys())
    for categ in admissable_categories:
        soft_f1[categ] = (
            2 * (soft_precision[categ] * soft_recall[categ]) /
            (soft_precision[categ] + soft_recall[categ]))

    return soft_f1


@validate_categorical_input
def range_based_precision_score(
        y_true,
        y_pred,
        alpha: float = 0.5,
        positional_bias: str = "flat",
        cardinality_bias: Optional[str] = None,
        range_based: bool = True):
    """Compute a precision score for two classes, usually labeled 1 and 0 to denote an anomaly and not an anomaly.
    Both ``y_true`` and ``y_pred`` need to be sorted by timestamp.

    This precision implementation is from the paper:
    Precision and Recall for Time Series
    <https://arxiv.org/abs/1803.03639>;

    Point-wise real and predicted anomalies are first transformed into anomaly ranges. Then, given the set of real
    anomaly ranges, R = {R_1, ..., R_Nr}, and predicted anomaly ranges, P = {P_1, ..., P_Np}, a precision score Precision_T(R, P_j) is
    calculated for each predicted anomaly range, P_j. Those precision scores are then added into a total precision score and divided by the
    total number of predicted anomaly ranges, Np, to obtain an average precision score for the whole timeseries.

    Multiple considerations are taken into account when computing the individual precision scores for each real anomaly range, such as:
        Existence: Catching the existence of an anomaly (even by predicting only a single point in R_i), by itself, might be valuable
        for the application.
        Size: The larger the size of the correctly predicted portion of R_i, the higher the precision score.
        Position: In some cases, not only size, but also the relative position of the correctly predicted portion of R_i might matter
        to the application.
        Cardinality: Detecting R_i with a single prediction range P_j ∈ P may be more valuable than doing so with multiple different
        ranges in P in a fragmented manner.

    All of those considerations are captured using two main reward terms: Existence Reward and Overlap Reward, weighted by a weighting
    constant ``alpha``. The precision score for each predicted anomaly range will be calculated as:
        Precision_T = alpha * Existence Reward + (1 - alpha) * Overlap Reward

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual point-wise anomalies
    y_pred : array-like, 1-D
        The predicted point-wise anomalies
    alpha : `float`
        Reward weighting term for the two main reward terms for the predicted anomaly range precision score: existence and overlap rewards.
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
    cardinality_bias: `str` or None, default None
        In the overlap reward, this is a penalization factor. If None, no cardinality penalty will be applied. If "reciprocal", the
        overlap reward will be penalized as it gets multiplied by the reciprocal of the number of detected anomaly ranges overlapping
        with the predicted anomaly range.
    range_based: `bool`, default True
        This implementation of range-based precision subsumes the classic precision. If True range based precision will be calculated, otherwise
        classic precision will be calculated.

    Returns
    -------
    precision : `float`
        The overall precision score for the time series.
    """
    assert len(y_true) == len(y_pred)
    assert 0 <= alpha <= 1
    assert positional_bias in ["flat", "front", "middle", "back"]
    if cardinality_bias is not None:
        assert cardinality_bias == "reciprocal"

    real_anomaly_ranges = prepare_anomaly_ranges(np.array(y_true), range_based)
    predicted_anomaly_ranges = prepare_anomaly_ranges(np.array(y_pred), range_based)

    precision = compute_range_based_score(
        predicted_anomaly_ranges,
        real_anomaly_ranges,
        alpha, positional_bias, cardinality_bias)
    return precision


@validate_categorical_input
def range_based_recall_score(
        y_true,
        y_pred,
        alpha: float = 0.5,
        positional_bias: str = "flat",
        cardinality_bias: Optional[str] = None,
        range_based: bool = True):
    """Compute a recall score for two classes, usually labeled 1 and 0 to denote an anomaly and not an anomaly.
    Both ``y_true`` and ``y_pred`` need to be in sorted by timestamp.

    This recall implementation is from the paper:
    Precision and Recall for Time Series
    <https://arxiv.org/abs/1803.03639>;

    Point-wise real and predicted anomalies are first transformed into anomaly ranges. Then, given the set of real
    anomaly ranges, R = {R_1, ..., R_Nr}, and predicted anomaly ranges, P = {P_1, ..., P_Np}, a recall score Recall_T(R_i, P) is
    calculated for each real anomaly range, R_i. Those recall scores are then added into a total recall score and divided by the
    total number of real anomaly ranges, Nr, to obtain an average recall score for the whole timeseries.

    Multiple considerations are taken into account when computing the individual recall scores for each real anomaly range, such as:
        Existence: Catching the existence of an anomaly (even by predicting only a single point in R_i), by itself, might be valuable
        for the application.
        Size: The larger the size of the correctly predicted portion of R_i, the higher the recall score.
        Position: In some cases, not only size, but also the relative position of the correctly predicted portion of R_i might matter
        to the application.
        Cardinality: Detecting R_i with a single prediction range P_j ∈ P may be more valuable than doing so with multiple different
        ranges in P in a fragmented manner.

    All of those considerations are captured using two main reward terms: Existence Reward and Overlap Reward, weighted by a weighting
    constant ``alpha``. The recall score for each real anomaly range will be calculated as:
        Recall_T = alpha * Existence Reward + (1 - alpha) * Overlap Reward

    Parameters
    ----------
    y_true : array-like, 1-D
        The actual point-wise anomalies
    y_pred : array-like, 1-D
        The predicted point-wise anomalies
    alpha : `float`
        Reward weighting term for the two main reward terms for the real anomaly range recall score: existence and overlap rewards.
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
    cardinality_bias: `str` or None, default None
        In the overlap reward, this is a penalization factor. If None, no cardinality penalty will be applied. If "reciprocal", the
        overlap reward will be penalized as it gets multiplied by the reciprocal of the number of detected anomaly ranges overlapping
        with the real anomaly range.
    range_based: `bool`, default True
        This implementation of range-based recall subsumes the classic recall. If True range based recall will be calculated, otherwise
        classic recall will be calculated.

    Returns
    -------
    recall : `float`
        The overall recall score for the time series.
    """
    assert len(y_true) == len(y_pred)
    assert 0 <= alpha <= 1
    assert positional_bias in ["flat", "front", "middle", "back"]
    if cardinality_bias is not None:
        assert cardinality_bias == "reciprocal"

    real_anomaly_ranges = prepare_anomaly_ranges(np.array(y_true), range_based)
    predicted_anomaly_ranges = prepare_anomaly_ranges(np.array(y_pred), range_based)
    recall = compute_range_based_score(
        real_anomaly_ranges,
        predicted_anomaly_ranges,
        alpha, positional_bias, cardinality_bias)
    return recall
