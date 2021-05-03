import numpy as np
import pytest

from greykite.sklearn.cross_validation import RollingTimeSeriesSplit


def assert_splits_equal(actual, expected):
    """Asserts whether CV splits match expectation
    :param actual: np.array of np.arrays
    :param expected: np.array of np.arrays
    """
    actual = np.array(list(actual))
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            np.testing.assert_array_equal(actual[i][j], expected[i][j])


def test_rolling_time_series_split():
    """Tests rolling_time_series_split with default values."""
    tscv = RollingTimeSeriesSplit(forecast_horizon=3, max_splits=None)
    assert tscv.forecast_horizon == 3
    assert tscv.min_train_periods == 6
    assert not tscv.expanding_window
    assert not tscv.use_most_recent_splits
    assert tscv.periods_between_splits == 3
    assert tscv.periods_between_train_test == 0

    X = np.random.rand(20, 2)
    expected = np.array([  # offset applied, first two observations are not used in CV
        (np.array([2, 3, 4, 5, 6, 7]), np.array([8, 9, 10])),
        (np.array([5, 6, 7, 8, 9, 10]), np.array([11, 12, 13])),
        (np.array([8, 9, 10, 11, 12, 13]), np.array([14, 15, 16])),
        (np.array([11, 12, 13, 14, 15, 16]), np.array([17, 18, 19]))
    ])
    assert tscv.get_n_splits(X=X) == 4
    assert_splits_equal(tscv.split(X=X), expected)


def test_rolling_time_series_split2():
    """Tests rolling_time_series_split with custom values"""
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=2,
        min_train_periods=4,
        expanding_window=True,
        use_most_recent_splits=False,
        periods_between_splits=4,
        periods_between_train_test=2,
        max_splits=None)

    assert tscv.forecast_horizon == 2
    assert tscv.min_train_periods == 4
    assert tscv.expanding_window
    assert not tscv.use_most_recent_splits
    assert tscv.periods_between_splits == 4
    assert tscv.periods_between_train_test == 2

    X = np.random.rand(20, 4)
    expected = np.array([  # no offset
        (np.array([0, 1, 2, 3]), np.array([6, 7])),
        (np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([10, 11])),
        (np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), np.array([14, 15])),
        (np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), np.array([18, 19]))
    ])
    assert tscv.get_n_splits(X=X) == 4
    assert_splits_equal(tscv.split(X=X), expected)

    X = np.random.rand(25, 4)
    expected = np.array([
        # offset with expanding window, first training is set larger than min_train_periods to use all data
        (np.arange(5), np.array([7, 8])),
        (np.arange(9), np.array([11, 12])),
        (np.arange(13), np.array([15, 16])),
        (np.arange(17), np.array([19, 20])),
        (np.arange(21), np.array([23, 24]))
    ])
    assert tscv.get_n_splits(X=X) == 5
    assert_splits_equal(tscv.split(X=X), expected)


def test_rolling_time_series_split3():
    """Tests rolling_time_series_split with max_splits"""
    # only the last split is kept
    with pytest.warns(Warning) as record:
        max_splits = 1
        tscv = RollingTimeSeriesSplit(
            forecast_horizon=2,
            min_train_periods=4,
            expanding_window=True,
            periods_between_splits=4,
            periods_between_train_test=2,
            max_splits=max_splits)

        X = np.random.rand(20, 4)
        expected = np.array([
            (np.array([0, 1, 2, 3]), np.array([6, 7])),
            (np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([10, 11])),
            (np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), np.array([14, 15])),
            (np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), np.array([18, 19]))
        ])
        assert tscv.get_n_splits_without_capping(X=X) == 4
        assert tscv.get_n_splits(X=X) == max_splits
        assert_splits_equal(tscv.split(X=X), expected[-max_splits:])

        obtained_messages = "--".join([r.message.args[0] for r in record])
        assert "There is only one CV split" in obtained_messages

    # only the last two splits are kept
    max_splits = 2
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=2,
        min_train_periods=4,
        expanding_window=True,
        use_most_recent_splits=False,
        periods_between_splits=4,
        periods_between_train_test=2,
        max_splits=max_splits)
    assert tscv.get_n_splits_without_capping(X=X) == 4
    assert tscv.get_n_splits(X=X) == max_splits
    assert_splits_equal(tscv.split(X=X), expected[-max_splits:])

    # two splits and a random split are kept
    max_splits = 3
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=2,
        min_train_periods=4,
        expanding_window=True,
        use_most_recent_splits=False,
        periods_between_splits=4,
        periods_between_train_test=2,
        max_splits=max_splits)
    assert tscv.get_n_splits_without_capping(X=X) == 4
    assert tscv.get_n_splits(X=X) == max_splits
    assert_splits_equal(tscv.split(X=X), expected[[0, 2, 3]])  # picked at random (selection fixed by random seed)

    # all splits are kept (max_splits == get_n_splits)
    max_splits = 4
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=2,
        min_train_periods=4,
        expanding_window=True,
        use_most_recent_splits=False,
        periods_between_splits=4,
        periods_between_train_test=2,
        max_splits=max_splits)
    assert tscv.get_n_splits_without_capping(X=X) == 4
    assert tscv.get_n_splits(X=X) == 4
    assert_splits_equal(tscv.split(X=X), expected)

    # all splits are kept (max_splits > get_n_splits)
    max_splits = 5
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=2,
        min_train_periods=4,
        expanding_window=True,
        use_most_recent_splits=False,
        periods_between_splits=4,
        periods_between_train_test=2,
        max_splits=max_splits)
    assert tscv.get_n_splits_without_capping(X=X) == 4
    assert tscv.get_n_splits(X=X) == 4
    assert_splits_equal(tscv.split(X=X), expected)

    # rolling window evaluation
    # splits from end up to max_splits are kept
    max_splits = 3
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=2,
        min_train_periods=4,
        expanding_window=True,
        use_most_recent_splits=True,
        periods_between_splits=4,
        periods_between_train_test=2,
        max_splits=max_splits)
    assert tscv.use_most_recent_splits
    assert tscv.get_n_splits_without_capping(X=X) == 4
    assert tscv.get_n_splits(X=X) == max_splits
    assert_splits_equal(tscv.split(X=X), expected[[1, 2, 3]])


def test_rolling_time_series_split_empty():
    """Tests rolling_time_series_split when there is not enough data to create user splits"""
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=50,
        min_train_periods=160,
        expanding_window=True,
        periods_between_splits=4,
        periods_between_train_test=0,
        max_splits=None)

    with pytest.warns(Warning) as record:
        X = np.random.rand(200, 4)
        expected = np.array([
            (np.arange(180), np.arange(start=180, stop=200)),  # 90/10 split
        ])
        assert tscv.get_n_splits(X=X) == 1
        assert_splits_equal(tscv.split(X=X), expected)
        obtained_messages = "--".join([r.message.args[0] for r in record])
        assert "There are no CV splits under the requested settings" in obtained_messages

    with pytest.warns(Warning) as record:
        X = np.random.rand(150, 4)
        expected = np.array([
            (np.arange(135), np.arange(start=135, stop=150)),  # 90/10 split
        ])
        assert tscv.get_n_splits(X=X) == 1
        assert_splits_equal(tscv.split(X=X), expected)
        obtained_messages = "--".join([r.message.args[0] for r in record])
        assert "There are no CV splits under the requested settings" in obtained_messages


def test_rolling_time_series_split_error():
    """Tests rolling_time_series_split exceptions and warnings"""
    with pytest.warns(Warning) as record:
        RollingTimeSeriesSplit(forecast_horizon=3, min_train_periods=4)
        assert "`min_train_periods` is too small for your `forecast_horizon`" in record[0].message.args[0]

    with pytest.warns(Warning) as record:
        X = np.random.rand(12, 1)
        tscv = RollingTimeSeriesSplit(forecast_horizon=4, min_train_periods=8)
        list(tscv.split(X=X))
        assert "There is only one CV split" in record[0].message.args[0]

    with pytest.warns(Warning) as record:
        X = np.random.rand(48, 1)
        tscv = RollingTimeSeriesSplit(forecast_horizon=4, min_train_periods=8, max_splits=None)
        list(tscv.split(X=X))
        assert "There is a high number of CV splits" in record[0].message.args[0]


def test_get_n_splits():
    """Tests get_n_splits and get_n_splits_without_capping"""
    X = np.random.rand(100, 3)

    # normal case, no capping
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=10,
        min_train_periods=20,
        expanding_window=True,
        periods_between_splits=16,
        periods_between_train_test=3,
        max_splits=None)
    assert tscv.get_n_splits(X=X) == 5
    assert tscv.get_n_splits_without_capping(X=X) == 5

    # normal case, no capping
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=7,
        min_train_periods=30,
        expanding_window=False,
        periods_between_splits=30,
        periods_between_train_test=10,
        max_splits=None)
    assert tscv.get_n_splits(X=X) == 2
    assert tscv.get_n_splits_without_capping(X=X) == 2

    # max_splits reached
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=10,
        min_train_periods=20,
        expanding_window=True,
        periods_between_splits=16,
        periods_between_train_test=3,
        max_splits=4)
    assert tscv.get_n_splits(X=X) == 4
    assert tscv.get_n_splits_without_capping(X=X) == 5

    # not enough data, use default_split
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=10,
        min_train_periods=90,
        expanding_window=True,
        periods_between_splits=16,
        periods_between_train_test=3,
        max_splits=None)
    assert tscv.get_n_splits(X=X) == 1
    assert tscv.get_n_splits_without_capping(X=X) == 0


def test_get_offset():
    """Tests _get_offset"""
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=5,
        min_train_periods=10,
        periods_between_train_test=5,
        periods_between_splits=5)
    assert tscv._get_offset(X=np.random.rand(19, 1)) == 0  # no CV splits
    assert tscv._get_offset(X=np.random.rand(20, 1)) == 0
    assert tscv._get_offset(X=np.random.rand(21, 1)) == 1
    assert tscv._get_offset(X=np.random.rand(22, 1)) == 2
    assert tscv._get_offset(X=np.random.rand(23, 1)) == 3
    assert tscv._get_offset(X=np.random.rand(24, 1)) == 4
    assert tscv._get_offset(X=np.random.rand(25, 1)) == 0  # perfect alignment again, last data point is used

    tscv = RollingTimeSeriesSplit(
        forecast_horizon=3,
        min_train_periods=6,
        periods_between_train_test=11,
        periods_between_splits=3)
    assert tscv._get_offset(X=np.random.rand(20, 1)) == 0
    assert tscv._get_offset(X=np.random.rand(21, 1)) == 1
    assert tscv._get_offset(X=np.random.rand(22, 1)) == 2
    assert tscv._get_offset(X=np.random.rand(23, 1)) == 0  # perfect alignment again, last data point is used


def test_sample_splits():
    """Tests _sample_splits"""
    # final split
    max_splits = 1
    tscv = RollingTimeSeriesSplit(forecast_horizon=10, max_splits=max_splits)
    assert tscv._sample_splits(num_splits=10) == [9]

    # last two splits
    max_splits = 2
    tscv = RollingTimeSeriesSplit(forecast_horizon=10, max_splits=max_splits)
    assert tscv._sample_splits(num_splits=10) == [9, 8]

    # last two splits, plus randomly selected splits
    max_splits = 8
    tscv = RollingTimeSeriesSplit(forecast_horizon=10, max_splits=max_splits)
    assert tscv._sample_splits(num_splits=10) == [9, 8, 0, 7, 3, 4, 6, 2]

    # all splits
    max_splits = None
    tscv = RollingTimeSeriesSplit(forecast_horizon=10, max_splits=max_splits)
    assert tscv._sample_splits(num_splits=10) == list(range(10))

    max_splits = 10
    tscv = RollingTimeSeriesSplit(forecast_horizon=10, max_splits=max_splits)
    assert tscv._sample_splits(num_splits=10) == list(range(10))

    max_splits = 15
    tscv = RollingTimeSeriesSplit(forecast_horizon=10, max_splits=max_splits)
    assert tscv._sample_splits(num_splits=10) == list(range(10))

    # rolling window evaluation
    # splits from end upto max_splits are kept
    max_splits = 5
    tscv = RollingTimeSeriesSplit(forecast_horizon=10, use_most_recent_splits=True, max_splits=max_splits)
    assert tscv._sample_splits(num_splits=10) == [5, 6, 7, 8, 9]
