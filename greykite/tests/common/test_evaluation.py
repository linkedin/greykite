import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import COVERAGE_VS_INTENDED_DIFF
from greykite.common.constants import LOWER_BAND_COVERAGE
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTION_BAND_COVERAGE
from greykite.common.constants import PREDICTION_BAND_WIDTH
from greykite.common.constants import UPPER_BAND_COVERAGE
from greykite.common.evaluation import ElementwiseEvaluationMetricEnum
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import ValidationMetricEnum
from greykite.common.evaluation import add_finite_filter_to_scorer
from greykite.common.evaluation import add_preaggregation_to_scorer
from greykite.common.evaluation import aggregate_array
from greykite.common.evaluation import all_equal_length
from greykite.common.evaluation import calc_pred_coverage
from greykite.common.evaluation import calc_pred_err
from greykite.common.evaluation import correlation
from greykite.common.evaluation import elementwise_absolute_error
from greykite.common.evaluation import elementwise_absolute_percent_error
from greykite.common.evaluation import elementwise_outside_tolerance
from greykite.common.evaluation import elementwise_quantile
from greykite.common.evaluation import elementwise_residual
from greykite.common.evaluation import elementwise_squared_error
from greykite.common.evaluation import elementwise_within_bands
from greykite.common.evaluation import fraction_outside_tolerance
from greykite.common.evaluation import fraction_within_bands
from greykite.common.evaluation import mean_absolute_percent_error
from greykite.common.evaluation import median_absolute_percent_error
from greykite.common.evaluation import prediction_band_width
from greykite.common.evaluation import quantile_loss
from greykite.common.evaluation import quantile_loss_q
from greykite.common.evaluation import r2_null_model_score
from greykite.common.evaluation import root_mean_squared_error
from greykite.common.evaluation import symmetric_mean_absolute_percent_error
from greykite.common.evaluation import valid_elements_for_evaluation


def test_all_equal_length():
    """Tests all_equal_length function"""
    assert all_equal_length() is True

    assert all_equal_length(
        np.array([1, 2, 3])
    ) is True

    assert all_equal_length(
        np.array([1, 2, 3]),
        [4, 5, 6],
        pd.Series([7, 8, 9])
    ) is True

    assert all_equal_length(
        np.array([1, 2, 3]),
        [4, 6],
        pd.Series([7, 8, 9])
    ) is False

    # constants and None are ignored
    assert all_equal_length(
        np.array([1, 2, 3]),
        4,
        None,
        pd.Series([7, 8, 9])
    ) is True


def test_valid_elements_for_evaluation():
    """Tests valid_elements_for_evaluation function"""

    with pytest.warns(Warning) as record:
        y_true = [1.0, np.nan, 2.0, np.Inf]
        y_pred = [np.nan, 2.0, 1.0, 2.0]
        y_another = [2.0, 1.0, np.nan, np.Inf]
        y_true, y_pred, y_another = valid_elements_for_evaluation(
            reference_arrays=[y_true],
            arrays=[y_pred, y_another],
            reference_array_names="y_true",
            drop_leading_only=False,
            keep_inf=False)
        assert "2 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]
        assert_array_equal(y_true, np.array([1.0, 2.0]))
        assert_array_equal(y_pred, np.array([np.nan, 1.0]))
        assert_array_equal(y_another, np.array([2.0, np.nan]))

    # Leading NAs and keep inf
    with pytest.warns(Warning) as record:
        y_true = [np.nan, 2.0, np.nan, np.Inf]
        y_pred = [np.nan, 2.0, 1.0, 2.0]
        y_another = [2.0, 1.0, np.nan, np.Inf]
        y_true, y_pred, y_another = valid_elements_for_evaluation(
            reference_arrays=[y_true],
            arrays=[y_pred, y_another],
            reference_array_names="y_true",
            drop_leading_only=True,
            keep_inf=True)
        assert "1 value(s) in y_true were NA and are omitted in error calc." in record[0].message.args[0]
        assert_array_equal(y_true, np.array([2.0, np.nan, np.Inf]))
        assert_array_equal(y_pred, np.array([2.0, 1.0, 2.0]))
        assert_array_equal(y_another, np.array([1.0, np.nan, np.Inf]))

    # All NAs and drop inf
    with pytest.warns(Warning) as record:
        y_true = [np.nan, np.nan, 2.0, np.Inf]
        y_pred = [np.nan, 2.0, 1.0, 2.0]
        y_another = [2.0, 1.0, np.nan, np.Inf]
        y_true, y_pred, y_another = valid_elements_for_evaluation(
            reference_arrays=[y_true],
            arrays=[y_pred, y_another],
            reference_array_names="y_true",
            drop_leading_only=False,
            keep_inf=False)
        assert "3 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]
        assert_array_equal(y_true, np.array([2.0]))
        assert_array_equal(y_pred, np.array([1.0]))
        assert_array_equal(y_another, np.array([np.nan]))

    # All NAs and keep inf
    with pytest.warns(Warning) as record:
        y_true = [np.nan, 2.0, np.nan, np.Inf]
        y_pred = [np.nan, 2.0, 1.0, 2.0]
        y_another = [2.0, 1.0, np.nan, np.Inf]
        y_true, y_pred, y_another = valid_elements_for_evaluation(
            reference_arrays=[y_true],
            arrays=[y_pred, y_another],
            reference_array_names="y_true",
            drop_leading_only=False,
            keep_inf=True)
        assert "2 value(s) in y_true were NA and are omitted in error calc." in record[0].message.args[
            0]
        assert_array_equal(y_true, np.array([2.0, np.Inf]))
        assert_array_equal(y_pred, np.array([2.0, 2.0]))
        assert_array_equal(y_another, np.array([1.0, np.Inf]))

    with pytest.warns(Warning) as record:
        y_true = [1.0, np.nan, 2.0, np.Inf]
        y_pred = [np.nan, 2.0, 1.0, 2.0]
        y_another = 2.0
        y_last = None
        y_true, y_pred, y_another, y_last = valid_elements_for_evaluation(
            reference_arrays=[y_true],
            arrays=[y_pred, y_another, y_last],
            reference_array_names="y_true",
            drop_leading_only=False,
            keep_inf=False)
        assert_array_equal(y_true, np.array([1.0, 2.0]))
        assert_array_equal(y_pred, np.array([np.nan, 1.0]))
        assert y_another == 2.0
        assert y_last is None
        assert "2 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]

    with pytest.warns(Warning) as record:
        y_true = [np.nan, np.Inf]
        y_pred = [np.nan, 2.0]
        y_another = 2.0
        y_last = None
        y_true, y_pred, y_another, y_last = valid_elements_for_evaluation(
            reference_arrays=[y_true],
            arrays=[y_pred, y_another, y_last],
            reference_array_names="y_true",
            drop_leading_only=False,
            keep_inf=False)
        assert y_another == 2.0
        assert y_last is None
        assert "There are 0 non-null elements for evaluation." in record[0].message.args[0]


def test_aggregate_array():
    """Tests aggregate_array function"""
    # tests defaults
    assert_array_equal(
        aggregate_array(pd.Series(np.arange(15))),
        np.array([28.0, 77.0])
    )

    # tests warning
    with pytest.warns(Warning) as record:
        assert_array_equal(
            aggregate_array([1.0, 2.0], agg_periods=3, agg_func=np.sum),
            np.array([3.0])
        )
        assert "Using all for aggregation" in record[0].message.args[0]

    # tests aggregation with dropping incomplete bin from the left
    assert_array_equal(
        aggregate_array([1.0, 2.0, 3.0], agg_periods=3, agg_func=np.sum),
        np.array([6.0])
    )
    assert_array_equal(
        aggregate_array([1.0, 2.0, 3.0, 4.0], agg_periods=3, agg_func=np.sum),
        np.array([9.0])  # drops 1.0, adds the rest
    )
    assert_array_equal(
        aggregate_array([1.0, 2.0, 3.0, 4.0, 5.0], agg_periods=3, agg_func=np.sum),
        np.array([12.0])  # drops 1.0 and 2.0, adds the rest
    )
    assert_array_equal(
        aggregate_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], agg_periods=3, agg_func=np.sum),
        np.array([6.0, 15.0])
    )
    assert_array_equal(
        aggregate_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], agg_periods=3, agg_func=np.sum),
        np.array([9.0, 18.0])
    )
    # tests np.array input
    assert_array_equal(
        aggregate_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), agg_periods=3, agg_func=np.sum),
        np.array([9.0, 18.0])
    )
    # tests pd.Series input
    assert_array_equal(
        aggregate_array(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), agg_periods=3, agg_func=np.sum),
        np.array([9.0, 18.0])
    )
    # tests custom agg_func
    assert_array_equal(
        aggregate_array(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), agg_periods=3, agg_func=np.max),
        np.array([4.0, 7.0])
    )


def test_add_preaggregation_to_scorer():
    y_true = pd.Series([3, 1, 3])
    y_pred = pd.Series([1, 4, 2])

    # tests various types of aggregation with mean_absolute_error scorer
    agg_mean_absolute_error = add_preaggregation_to_scorer(mean_absolute_error, agg_periods=3, agg_func=np.sum)
    assert agg_mean_absolute_error(y_true, y_pred) == 0.0  # 7 vs 7

    agg_mean_absolute_error = add_preaggregation_to_scorer(mean_absolute_error, agg_periods=3, agg_func=np.max)
    assert agg_mean_absolute_error(y_true, y_pred) == 1.0  # 4 vs 3

    agg_mean_absolute_error = add_preaggregation_to_scorer(mean_absolute_error, agg_periods=2, agg_func=np.sum)
    assert agg_mean_absolute_error(y_true, y_pred) == 2.0  # 6 vs 4

    agg_mean_absolute_error = add_preaggregation_to_scorer(mean_absolute_error, agg_periods=2, agg_func=np.max)
    assert agg_mean_absolute_error(y_true, y_pred) == 1.0  # 4 vs 3

    agg_mean_absolute_error = add_preaggregation_to_scorer(mean_absolute_error, agg_periods=1, agg_func=np.max)
    assert agg_mean_absolute_error(y_true, y_pred) == mean_absolute_error(y_true, y_pred)  # agg_periods=1 does nothing

    # tests aggregation on scorer with arguments
    y_true = pd.Series([2.0, 1.0, 9.0, 4.0, 5.0, 9.0, 2.0])
    y_pred = pd.Series([3.0, 0.0, 0.0, -4.0, 2.0, 1.0, 3.0])
    agg_mean_absolute_error = add_preaggregation_to_scorer(mean_absolute_error, agg_periods=3, agg_func=np.std)
    agg_quantile_loss = add_preaggregation_to_scorer(quantile_loss, agg_periods=3, agg_func=np.std)
    # quantile loss with q=0.5 is equivalent to half the MAE
    assert 2.0 * agg_quantile_loss(y_true, y_pred, q=0.5) == agg_mean_absolute_error(y_true, y_pred)


def test_add_finite_filter_to_scorer():
    with pytest.warns(UserWarning) as record:
        y_true = pd.Series([3, 1, 3, np.nan])
        y_pred = pd.Series([1, 4, 2, 2])
        score_func = add_finite_filter_to_scorer(mean_absolute_error)
        assert score_func(y_true, y_pred) == 2.0
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        y_true = pd.Series([np.Inf, np.nan])
        y_pred = pd.Series([2, 2])
        score_func = add_finite_filter_to_scorer(mean_absolute_error)
        assert score_func(y_true, y_pred) is None
        assert "There are 0 non-null elements for evaluation." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record, \
            pytest.raises(ValueError, match="Input contains NaN"):
        y_true = pd.Series([3, 1, 3, np.nan])
        y_pred = pd.Series([1, 4, np.nan, 2])  # this causes an error
        # ``add_finite_filter_to_scorer`` does not drop NAN which are not heading NANs in ``y_pred``.
        score_func = add_finite_filter_to_scorer(mean_absolute_error)
        assert score_func(y_true, y_pred) is None
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]


def test_model_r2_score():
    """Checks r2_null_model_score function"""
    y_true = [1.0, 2.0, 3.0, np.nan]  # the last value is ignored when computing model scores
    y_pred = [1.5, 2.5, 2.5, 2.0]
    y_pred_null = [1.0, 3.0, 4.0, 3.0]
    y_train = [0.0, 1.0, 2.0, 100.0]

    r2 = add_finite_filter_to_scorer(r2_score)  # to check equivalence under certain conditions
    with pytest.warns(UserWarning) as record:
        # null model from test data
        assert r2_null_model_score(y_true, y_pred) == r2(y_true, y_pred) == 0.625
        # constant null model
        assert r2_null_model_score(y_true, y_pred, y_pred_null=1.0) == 0.85
        # array null model
        assert r2_null_model_score(y_true, y_pred, y_pred_null=y_pred_null) == 0.625
        # null model from training data
        assert r2_null_model_score(y_true, y_pred, y_train=y_train) == 0.85
        # null model takes precedence over training null model
        r2_score_value = (r2(y_true, y_pred) - r2(y_true, y_pred_null)) / (1.0 - r2(y_true, y_pred_null))
        assert r2_null_model_score(y_true, y_pred, y_pred_null=y_pred_null, y_train=y_train) == r2_score_value == 0.625
        # specify custom loss function
        assert r2_null_model_score(y_true, y_pred, loss_func=median_absolute_error) == 0.5
        # no data to evaluate
        assert r2_null_model_score([np.Inf], [1.0]) is None
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]


def test_calc_pred_err1():
    """Checks calc_pred_err function"""
    y_true = [1, 3, 5, 9, 10]
    y_pred = [1, 4, 5, 9, 10]
    res = calc_pred_err(y_true, y_pred)

    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert res[enum.get_metric_name()] == 0.2
    enum = EvaluationMetricEnum.Correlation
    assert res[enum.get_metric_name()].round(1) == 1.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert np.round(res[enum.get_metric_name()], 1) == 0.4
    enum = EvaluationMetricEnum.MedianAbsolutePercentError
    assert res[enum.get_metric_name()] == 0.0

    with pytest.warns(UserWarning) as record:
        y_true = [1, 3, 5, 9, np.nan, 10]
        y_pred = [1, 4, 5, 9, 1, 10]
        res = calc_pred_err(y_true, y_pred)
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]
        enum = EvaluationMetricEnum.MeanAbsoluteError
        assert res[enum.get_metric_name()] == 0.2
        enum = EvaluationMetricEnum.Correlation
        assert res[enum.get_metric_name()].round(1) == 1.0
        enum = EvaluationMetricEnum.RootMeanSquaredError
        assert np.round(res[enum.get_metric_name()], 1) == 0.4
        enum = EvaluationMetricEnum.MeanAbsolutePercentError
        assert np.round(res[enum.get_metric_name()], 1) == 6.7
        enum = EvaluationMetricEnum.MedianAbsolutePercentError
        assert res[enum.get_metric_name()] == 0.0

    with pytest.warns(UserWarning) as record:
        res = calc_pred_err([np.Inf], [1.0])
        assert "There are 0 non-null elements for evaluation." in record[0].message.args[0]
        for key, value in res.items():
            assert value is None


def test_calc_pred_err2():
    """Checks calc_pred_err function"""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 4.0, 1.0]
    result = calc_pred_err(y_true, y_pred)

    enum = EvaluationMetricEnum.Correlation
    assert pytest.approx(result[enum.get_metric_name()]) == 0.0
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert result[enum.get_metric_name()] == 4.0 / 3.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert result[enum.get_metric_name()] == math.sqrt(8.0 / 3.0)
    enum = EvaluationMetricEnum.MedianAbsoluteError
    assert result[enum.get_metric_name()] == 2.0
    enum = EvaluationMetricEnum.MeanAbsolutePercentError
    assert result[enum.get_metric_name()] == pytest.approx(100.0 * 5.0 / 9.0)
    enum = EvaluationMetricEnum.MedianAbsolutePercentError
    assert result[enum.get_metric_name()] == pytest.approx(2.0 / 3.0 * 100)


def test_mean_absolute_percent_error():
    """Checks mean_absolute_percent_error function"""
    y_true = [1.0, 2.0, 4.0]
    y_pred = [0.5, 1.5, 3.5]
    assert mean_absolute_percent_error(y_true, y_pred) == pytest.approx((50 + 25 + 12.5) / 3.0)

    with pytest.warns(UserWarning) as record:
        y_true = [np.Inf, 1.0, 2.0, 4.0]
        y_pred = [34.0, 0.5, 1.5, 3.5]
        assert mean_absolute_percent_error(y_true, y_pred) == pytest.approx((50 + 25 + 12.5) / 3.0)
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        y_true = [1.0, 2.0, 0.0]
        y_pred = [0.5, 1.5, 3.5]
        assert mean_absolute_percent_error(y_true, y_pred) is None
        assert "y_true contains 0. MAPE is undefined." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        y_true = [1.0, 2.0, 1e-9]
        y_pred = [0.5, 1.5, 3.5]
        assert mean_absolute_percent_error(y_true, y_pred) is not None
        assert "y_true contains very small values. MAPE is likely highly volatile." in record[0].message.args[0]


def test_median_absolute_percent_error():
    """Checks mean_absolute_percent_error function"""
    y_true = [1.0, 2.0, 4.0]
    y_pred = [0.5, 1.5, 3.5]
    assert median_absolute_percent_error(y_true, y_pred) == 25

    with pytest.warns(UserWarning) as record:
        y_true = [np.Inf, 1.0, 2.0, 4.0]
        y_pred = [34.0, 0.5, 1.5, 3.5]
        assert median_absolute_percent_error(y_true, y_pred) == 25
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        y_true = [1.0, 2.0, 0.0]
        y_pred = [0.5, 1.5, 3.5]
        assert median_absolute_percent_error(y_true, y_pred) is None
        assert "y_true contains 0. MedAPE is undefined." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        y_true = [1.0, 1e-9, 1e-9]
        y_pred = [0.5, 1.5, 3.5]
        assert median_absolute_percent_error(y_true, y_pred) is not None
        assert "y_true contains very small values. MedAPE is likely highly volatile." in record[0].message.args[0]


def test_symmetric_mean_absolute_percent_error():
    """Checks symmetric_mean_absolute_percent_error function"""
    y_true = [1.0, 2.0, 4.0]
    y_pred = [0.5, 1.5, 3.5]
    assert symmetric_mean_absolute_percent_error(y_true, y_pred) == pytest.approx(100 * (.5 / 1.5 + .5 / 3.5 + .5 / 7.5)
                                                                                  / 3.0)
    with pytest.warns(UserWarning) as record:
        y_true = [np.Inf, 1.0, 2.0, 4.0]
        y_pred = [34.0, 0.5, 1.5, 3.5]
        assert symmetric_mean_absolute_percent_error(y_true, y_pred) == pytest.approx(
            100 * (.5 / 1.5 + .5 / 3.5 + .5 / 7.5)
            / 3.0)
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        y_true = [1.0, 2.0, 0.0]
        y_pred = [0.5, 1.5, 0.0]
        assert symmetric_mean_absolute_percent_error(y_true, y_pred) is None
        assert "denominator contains 0. sMAPE is undefined." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        y_true = [1.0, 2.0, 1e-9]
        y_pred = [0.5, -2.0, 1e-10]
        assert symmetric_mean_absolute_percent_error(y_true, y_pred) is not None
        assert "denominator contains very small values. sMAPE is likely highly volatile." in record[0].message.args[0]


def test_root_mean_squared_error():
    """Checks root mean_absolute_percent_error function"""
    y_true = [1.0, 2.0, 2.0]
    y_pred = [0.5, 1.5, 3.5]
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(math.sqrt((0.5 ** 2 + 0.5 ** 2 + 1.5 ** 2) / 3.0))

    with pytest.warns(UserWarning) as record:
        y_true = [np.Inf, 1.0, 2.0, 2.0]
        y_pred = [34.0, 0.5, 1.5, 3.5]
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
            math.sqrt((0.5 ** 2 + 0.5 ** 2 + 1.5 ** 2) / 3.0))
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]


def test_correlation():
    """Checks correlation function"""
    y_true = [1.0, 3.0, 7.0]
    y_pred = [0.5, 1.5, 3.5]
    assert correlation(y_true, y_pred) == pytest.approx(1.0)

    y_true = [1.0, 0.0, -1.0]
    y_pred = [-1.0, 0.0, 1.0]
    assert correlation(y_true, y_pred) == pytest.approx(-1.0)

    y_true = [1.0, 0.0, 1.0]
    y_pred = [-1.0, 0.0, 1.0]
    assert correlation(y_true, y_pred) == pytest.approx(0.0)

    with pytest.warns(UserWarning) as record:
        y_true = [np.Inf, 1.0, 3.0, 7.0]
        y_pred = [34.0, 0.5, 1.5, 3.5]
        assert correlation(y_true, y_pred) == pytest.approx(1.0)
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        y_true = [1.0, 1.0, 1.0]
        y_pred = [0.5, 1.5, 3.5]
        assert correlation(y_true, y_pred) is None
        assert "y_true is constant. Correlation is not defined." in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        y_true = [1.0, 3.0, 7.0]
        y_pred = [1.5, 1.5, 1.5]
        assert correlation(y_true, y_pred) is None
        assert "y_pred is constant. Correlation is not defined." in record[0].message.args[0]


def test_quantile_loss():
    """Checks quantile_loss function"""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [0.5, 1.5, 3.5]
    assert quantile_loss(y_true, y_pred) == pytest.approx(0.975 / 3.0)
    assert quantile_loss(y_true, y_pred, q=0.9) == pytest.approx(0.95 / 3.0)

    with pytest.warns(UserWarning) as record:
        y_true = [np.Inf, 1.0, 2.0, 3.0]
        y_pred = [34.0, 0.5, 1.5, 3.5]
        assert quantile_loss(y_true, y_pred) == pytest.approx(0.975 / 3.0)
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]
        assert quantile_loss(y_true, y_pred, q=0.9) == pytest.approx(0.95 / 3.0)

    with pytest.warns(UserWarning) as record:
        assert quantile_loss([np.Inf], [1.0]) is None
        assert "There are 0 non-null elements for evaluation." in record[0].message.args[0]


def test_quantile_loss_q():
    """Tests quantile loss function with fixed quantile"""
    q95 = quantile_loss_q(0.95)
    y_true = [1.0, 2.0, 3.0]
    y_pred = [0.5, 1.5, 3.5]
    assert q95(y_true=y_true, y_pred=y_pred) == pytest.approx(0.975 / 3.0)

    q90 = quantile_loss_q(0.90)
    y_true = [1.0, 2.0, 3.0]
    y_pred = [0.5, 1.5, 3.5]
    assert q90(y_true=y_true, y_pred=y_pred) == pytest.approx(0.95 / 3.0)


def test_fraction_within_bands():
    """Tests fraction_within_bands function"""
    assert fraction_within_bands(
        observed=[2, 2, 2, 2],
        lower=[0, 1, 2, 3],
        upper=[2, 3, 4, 5]
    ) == 0.25

    with pytest.warns(UserWarning) as record:
        assert fraction_within_bands([np.Inf], [1.0], [2.0]) is None
        assert "There are 0 non-null elements for evaluation." in record[0].message.args[0]

    with pytest.raises(Exception, match="length of arrays do not match"):
        fraction_within_bands(
            observed=[2, 2, 2],
            lower=[0, 1, 2, 3],
            upper=[2, 3, 4, 5]
        )

    with pytest.warns(UserWarning) as record:
        fraction_within_bands(
            observed=[2, 2, 2, 2],
            lower=[3, 1, 2, 3],
            upper=[2, 3, 4, 5]
        )
        assert "1 of 4 upper bound values are smaller than the lower bound" in record[0].message.args[0]


def test_prediction_band_width():
    """Tests prediction_band_width function"""
    assert prediction_band_width(
        observed=[2, 2, 2, 2],
        lower=[0, 1, 2, 3],
        upper=[2, 3, 4, 5]
    ) == 100.0

    assert prediction_band_width(
        observed=[1, 1, 2, 2],
        lower=[0, 1, 2, 3],
        upper=[1, 3, 5, 8]
    ) == 175.0

    assert prediction_band_width(
        observed=[0, 2, 2, 2],  # can't divide by 0
        lower=[0, 1, 2, 3],
        upper=[2, 3, 4, 5]
    ) is None

    with pytest.warns(UserWarning) as record:
        assert prediction_band_width([np.Inf], [1.0], [2.0]) is None
        assert "There are 0 non-null elements for evaluation." in record[0].message.args[0]

    with pytest.warns(Warning) as record:
        prediction_band_width(
            observed=[2, 2, 2, 2],
            lower=[3, 1, 2, 3],
            upper=[2, 3, 4, 5]
        )
        assert "1 of 4 upper bound values are smaller than the lower bound" in record[0].message.args[0]


def test_fraction_outside_tolerance():
    """Tests fraction_outside_tolerance function"""
    rtol = 1.05 - 1.00  # floating point, not exactly 0.05
    assert fraction_outside_tolerance([2.0], [2.101], rtol=rtol) == 1.0
    assert fraction_outside_tolerance([2.0], [2.100], rtol=rtol) == 0.0
    assert fraction_outside_tolerance([2.0], [2.009], rtol=rtol) == 0.0
    assert fraction_outside_tolerance([10.0], [8.0], rtol=rtol) == 1.0
    assert fraction_outside_tolerance([0.0], [1.0], rtol=rtol) == 1.0
    assert fraction_outside_tolerance([1.0], [0.0], rtol=rtol) == 1.0
    assert fraction_outside_tolerance([0.0], [0.0], rtol=rtol) == 0.0

    assert fraction_outside_tolerance(
        [0.0, 0.0],
        [0.0, 1.0],
        rtol=rtol) == 0.5
    assert fraction_outside_tolerance(
        [1.0, 0.0, 0.0],
        [1.02, 0.0, 1.0],
        rtol=rtol) == 1/3
    assert fraction_outside_tolerance(
        [1.0, 0.0, 0.0],
        [1.02, 0.0, 1.0],
        rtol=0.01) == 2/3

    # np.inf in `y_true` is ignored when computing the fraction
    with pytest.warns(UserWarning) as record:
        assert fraction_outside_tolerance(
            [1.0, 0.0, 0.0, np.inf],
            [1.02, 0.0, 1.0, 2.0],
            rtol=rtol) == 1/3
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc" in record[0].message.args[0]

    # np.inf in `y_pred` is outside tolerance
    assert fraction_outside_tolerance(
        [1.0, 0.0, 0.0, 2.0],
        [1.02, 0.0, 1.0, -np.inf],
        rtol=rtol) == 2/4


def test_calc_pred_coverage():
    """Checks calc_pred_coverage function"""
    observed = [2, 2, 2, 2]
    predicted = [1, 2, 3, 4]
    lower = [0, 1, 2, 3]
    upper = [2, 3, 4, 5]
    coverage = 0.95
    res = calc_pred_coverage(observed, predicted, lower, upper, coverage)

    assert res[PREDICTION_BAND_WIDTH] == 100.0
    assert res[PREDICTION_BAND_COVERAGE] == 0.25
    assert res[LOWER_BAND_COVERAGE] == 0.0
    assert res[UPPER_BAND_COVERAGE] == 0.0
    assert res[COVERAGE_VS_INTENDED_DIFF] == -0.70

    with pytest.warns(UserWarning) as record:
        observed = [0, 2, 2, np.nan]
        predicted = [1, 2.5, 3, 4]
        lower = [0, 1, 2, 3]
        upper = [2, 3, 4, 5]
        coverage = 0.7
        res = calc_pred_coverage(observed, predicted, lower, upper, coverage)
        assert "1 value(s) in y_true were NA or infinite and are omitted in error calc." in record[0].message.args[0]
        assert res[PREDICTION_BAND_WIDTH] is None  # Can't divide by 0
        assert res[PREDICTION_BAND_COVERAGE] == 1.0 / 3.0
        assert res[LOWER_BAND_COVERAGE] == 1.0 / 3.0
        assert res[UPPER_BAND_COVERAGE] == 0.0 / 3.0
        assert res[COVERAGE_VS_INTENDED_DIFF] == 1.0 / 3.0 - 0.7

    with pytest.warns(UserWarning) as record:
        res = calc_pred_coverage([np.Inf], [1.5], [1.0], [2.0], 0.9)
        assert "There are 0 non-null elements for evaluation" in record[0].message.args[0]
        for key, value in res.items():
            assert value is None


def test_elementwise_residual():
    """Tests elementwise_residual function"""
    assert elementwise_residual(1.0, 3.0) == -2.0
    assert elementwise_residual(3.0, 1.0) == 2.0


def test_elementwise_absolute_error():
    """Tests elementwise_absolute_error function"""
    assert elementwise_absolute_error(1.0, 3.0) == 2.0
    assert elementwise_absolute_error(3.0, 1.0) == 2.0


def test_elementwise_squared_error():
    """Tests elementwise_squared_error function"""
    assert elementwise_squared_error(1.0, 3.0) == 4.0
    assert elementwise_squared_error(3.0, 1.0) == 4.0


def test_elementwise_absolute_percent_error():
    """Tests elementwise_absolute_percent_error function"""
    assert elementwise_absolute_percent_error(1.0, 3.0) == pytest.approx(200.0)
    assert elementwise_absolute_percent_error(3.0, 1.0) == pytest.approx(100 * 2/3, rel=1e-5)

    assert elementwise_absolute_percent_error(1.0, 0.0) == 100.0
    with pytest.warns(UserWarning, match="Percent error is undefined"):
        assert np.isnan(elementwise_absolute_percent_error(0.0, 1.0))

    with pytest.warns(Warning) as record:
        elementwise_absolute_percent_error(1e-9, 1.0)
        assert "true_val is less than 1e-8. Percent error is very likely highly volatile." in record[0].message.args[0]


def test_elementwise_quantile():
    """Tests elementwise_quantile function"""
    assert elementwise_quantile(1.0, 3.0, q=0.8) == pytest.approx(2.0 * 0.2, rel=1e-5)
    assert elementwise_quantile(3.0, 1.0, q=0.8) == pytest.approx(2.0 * 0.8, rel=1e-5)


def test_elementwise_outside_tolerance():
    """Tests elementwise_outside_tolerance function"""
    rtol = 1.05 - 1.00  # floating point, not exactly 0.05
    assert elementwise_outside_tolerance(2.0, 2.101, rtol=rtol) == 1.0
    assert elementwise_outside_tolerance(2.0, 2.100, rtol=rtol) == 0.0
    assert elementwise_outside_tolerance(2.0, 2.009, rtol=rtol) == 0.0
    assert elementwise_outside_tolerance(10.0, 8.0, rtol=rtol*2) == 1.0
    assert elementwise_outside_tolerance(0.0, 1.0, rtol=rtol) == 1.0
    assert elementwise_outside_tolerance(1.0, 0.0, rtol=rtol) == 1.0
    assert elementwise_outside_tolerance(0.0, 0.0, rtol=rtol) == 0.0


def test_elementwise_within_bands():
    """Tests elementwise_within_bands function"""
    assert elementwise_within_bands(5.0, 3.0, 8.0) == 1.0
    assert elementwise_within_bands(5.0, 6.0, 8.0) == 0.0
    assert elementwise_within_bands(5.0, 3.0, 4.0) == 0.0


def test_elementwise_evaluation_metric_enum():
    """Tests ElementwiseEvaluationMetricEnum accessors"""
    assert ElementwiseEvaluationMetricEnum.Residual.get_metric_func() == elementwise_residual
    assert ElementwiseEvaluationMetricEnum.Residual.get_metric_name() == "residual"
    assert ElementwiseEvaluationMetricEnum.Residual.get_metric_args() == [ACTUAL_COL, PREDICTED_COL]

    assert ElementwiseEvaluationMetricEnum.AbsoluteError.get_metric_func() == elementwise_absolute_error
    assert ElementwiseEvaluationMetricEnum.SquaredError.get_metric_func() == elementwise_squared_error
    assert ElementwiseEvaluationMetricEnum.AbsolutePercentError.get_metric_func() == elementwise_absolute_percent_error

    # spot check a few values
    assert ElementwiseEvaluationMetricEnum.Quantile90.get_metric_func()(3.0, 4.0) == pytest.approx(0.1 * (4.0 - 3.0), rel=1e-5)
    assert ElementwiseEvaluationMetricEnum.OutsideTolerance5.get_metric_func()(1.00, 1.06) == 1.0
    assert ElementwiseEvaluationMetricEnum.OutsideTolerance5.get_metric_func()(1.00, 1.04) == 0.0
    assert ElementwiseEvaluationMetricEnum.Coverage.get_metric_func() == elementwise_within_bands

    assert ElementwiseEvaluationMetricEnum.Coverage.name in ElementwiseEvaluationMetricEnum.__members__


def test_evaluation_metric_enum():
    """Tests EvaluationMetricEnum accessors"""
    assert EvaluationMetricEnum.RootMeanSquaredError == EvaluationMetricEnum["RootMeanSquaredError"]
    assert EvaluationMetricEnum.RootMeanSquaredError.name == "RootMeanSquaredError"
    assert EvaluationMetricEnum.RootMeanSquaredError.value == (root_mean_squared_error, False, "RMSE")
    # tuple access works as usual
    assert EvaluationMetricEnum["RootMeanSquaredError"].value[0] == root_mean_squared_error
    assert EvaluationMetricEnum.RootMeanSquaredError.get_metric_func() == root_mean_squared_error
    assert not EvaluationMetricEnum["RootMeanSquaredError"].value[1]
    assert not EvaluationMetricEnum.RootMeanSquaredError.get_metric_greater_is_better()
    assert EvaluationMetricEnum["RootMeanSquaredError"].value[2] == "RMSE"
    assert EvaluationMetricEnum.RootMeanSquaredError.get_metric_name() == "RMSE"

    assert EvaluationMetricEnum["MeanAbsolutePercentError"].value[0] == mean_absolute_percent_error
    assert len(", ".join(EvaluationMetricEnum.__dict__["_member_names_"])) > 0

    assert EvaluationMetricEnum.Correlation.name == "Correlation"
    assert EvaluationMetricEnum.RootMeanSquaredError.name == "RootMeanSquaredError"
    assert EvaluationMetricEnum.MeanAbsoluteError.name == "MeanAbsoluteError"
    assert EvaluationMetricEnum.MedianAbsoluteError.name == "MedianAbsoluteError"
    assert EvaluationMetricEnum.MeanAbsolutePercentError.name == "MeanAbsolutePercentError"
    assert EvaluationMetricEnum.SymmetricMeanAbsolutePercentError.name == "SymmetricMeanAbsolutePercentError"
    assert EvaluationMetricEnum.Quantile80.name == "Quantile80"
    assert EvaluationMetricEnum.Quantile95.name == "Quantile95"
    assert EvaluationMetricEnum.Quantile99.name == "Quantile99"

    y_true = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y_pred = np.array([2.0*1.005, 3.0*1.015, 4.0*1.025, 5.0*1.035, 6.0*1.045, 7.0*1.055])
    r2 = EvaluationMetricEnum.CoefficientOfDetermination.get_metric_func()
    assert r2(y_true, y_pred) == r2_score(y_true, y_pred)
    outside_1 = EvaluationMetricEnum.FractionOutsideTolerance1.get_metric_func()
    assert outside_1(y_true, y_pred) == 5/6
    outside_2 = EvaluationMetricEnum.FractionOutsideTolerance2.get_metric_func()
    assert outside_2(y_true, y_pred) == 4/6
    outside_3 = EvaluationMetricEnum.FractionOutsideTolerance3.get_metric_func()
    assert outside_3(y_true, y_pred) == 3/6
    outside_4 = EvaluationMetricEnum.FractionOutsideTolerance4.get_metric_func()
    assert outside_4(y_true, y_pred) == 2/6
    outside_5 = EvaluationMetricEnum.FractionOutsideTolerance5.get_metric_func()
    assert outside_5(y_true, y_pred) == 1/6


def test_validation_metric_enum():
    """Tests ValidationMetricEnum accessors"""
    assert ValidationMetricEnum.BAND_WIDTH == ValidationMetricEnum["BAND_WIDTH"]
    assert ValidationMetricEnum.BAND_WIDTH.name == "BAND_WIDTH"
    assert ValidationMetricEnum.BAND_WIDTH.value == (prediction_band_width, False)
    # tuple access works as usual
    assert ValidationMetricEnum["BAND_WIDTH"].value[0] == prediction_band_width
    assert ValidationMetricEnum.BAND_WIDTH.get_metric_func() == prediction_band_width
    assert not ValidationMetricEnum["BAND_WIDTH"].value[1]
    assert not ValidationMetricEnum.BAND_WIDTH.get_metric_greater_is_better()

    assert len(", ".join(ValidationMetricEnum.__dict__["_member_names_"])) > 0

    assert ValidationMetricEnum.BAND_COVERAGE.name == "BAND_COVERAGE"
