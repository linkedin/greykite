import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from greykite.common.python_utils import assert_equal
from greykite.sklearn.transform.drop_degenerate_transformer import DropDegenerateTransformer


def test_drop_degenerate_transformer():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [-1.0, -1.0, -1.0],
        "c": ["x", "x", "x"],
        "d": ["one", "two", "three"],
    })

    ddt = DropDegenerateTransformer(drop_degenerate=True)
    # init does not modify parameters
    assert ddt.drop_degenerate is True
    assert ddt.drop_cols is None
    assert ddt.keep_cols is None
    with pytest.raises(NotFittedError, match="This instance is not fitted yet"):
        ddt.transform(df)
    with pytest.warns(
            RuntimeWarning,
            match=r"Columns \['b', 'c'\] are degenerate \(constant value\), "
                  r"and will not be used in the forecast."):
        ddt.fit(df)
    assert ddt.keep_cols == ["a", "d"]
    assert ddt.drop_cols == ["b", "c"]
    assert_equal(ddt.transform(df), df[["a", "d"]])

    # stateful transform, uses params stored from `fit`
    df_test = pd.DataFrame({
        "a": [-1.0, -1.0, -1.0],
        "b": [1, 2, 3],
        "c": ["one", "two", "three"],
        "d": ["x", "x", "x"],
    })
    assert_equal(ddt.transform(df_test), df_test[["a", "d"]])

    ddt = DropDegenerateTransformer(drop_degenerate=False)
    assert ddt.drop_degenerate is False
    assert ddt.drop_cols is None
    assert ddt.keep_cols is None
    ddt.fit(df)
    assert ddt.drop_cols == []
    assert ddt.keep_cols == ["a", "b", "c", "d"]
    assert_equal(ddt.transform(df), df)
