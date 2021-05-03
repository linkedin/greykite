import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from greykite.common.python_utils import assert_equal
from greykite.sklearn.transform.normalize_transformer import NORMALIZE_ALGORITHMS
from greykite.sklearn.transform.normalize_transformer import NormalizeTransformer


def test_normalize_transformer():
    X = pd.DataFrame({
        "a": [1.0, 4.0, -9.0, 3.0],
        "b": [11.0, 2.0, 8.0, 9.0],
        "c": [7.0, 7.0, 7.0, 7.0],
        "d": [3.0, 0.0, 2.0, -3.0],
    })
    # all algorithms work
    for algorithm, scaler_class in NORMALIZE_ALGORITHMS.items():
        nt = NormalizeTransformer(normalize_algorithm=algorithm)
        # init does not modify parameters
        assert nt.normalize_algorithm == algorithm
        assert nt.normalize_params is None
        assert nt.scaler is None
        nt.fit(X=X)
        assert isinstance(nt.scaler, scaler_class)
        result = nt.transform(X=X)
        assert not result.isnull().any().any()

    with pytest.raises(ValueError, match="`normalize_algorithm` 'unknown_algorithm' is not recognized"):
        nt = NormalizeTransformer(normalize_algorithm="unknown_algorithm")
        nt.fit(X=X)

    # parameters can be customized
    params = {"quantile_range": (10.0, 90.0)}
    nt = NormalizeTransformer(
        normalize_algorithm="RobustScaler",
        normalize_params=params,
    )
    with pytest.raises(NotFittedError, match="This instance is not fitted yet"):
        nt.transform(X)
    actual = nt.fit_transform(X=X)
    scaler = RobustScaler(**params)
    expected = scaler.fit_transform(X=X)
    assert_equal(
        np.array(actual),
        expected)
    assert_equal(actual.index, X.index)
    assert_equal(actual.columns, X.columns)
    assert (actual["c"] == 0.0).all()  # constant column

    # parameters can be customized
    params = {"feature_range": (0.0, 2.0)}
    nt = NormalizeTransformer(
        normalize_algorithm="MinMaxScaler",
        normalize_params=params,
    )
    actual = nt.fit_transform(X=X)
    scaler = MinMaxScaler(**params)
    expected = scaler.fit_transform(X=X)
    assert_equal(
        np.array(actual),
        expected)
    assert_equal(actual.index, X.index)
    assert_equal(actual.columns, X.columns)
    assert (actual["c"] == 0.0).all()  # constant column
    assert (actual.min() == 0.0).all()  # range is as specified
    assert (actual[["a", "b", "d"]].max() - 2.0 < 1e-10).all()  # max is 2.0 for non-constant column

    # stateful transform, uses params stored from `fit`
    X_new = X*2 + 1
    actual = nt.transform(X=X_new)
    expected = scaler.transform(X=X_new)
    assert_equal(
        np.array(actual),
        expected)
    assert (actual["c"] != 0.0).all()
