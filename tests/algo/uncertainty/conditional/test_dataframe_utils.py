import pandas as pd
from pandas.util.testing import assert_frame_equal

from greykite.algo.uncertainty.conditional.dataframe_utils import limit_tuple_col
from greykite.algo.uncertainty.conditional.dataframe_utils import offset_tuple_col


def test_limit_tuple_col():
    """Testing limit_tuple_col"""
    df = pd.DataFrame({
        "a": [1, 4, 7],
        "b": [(1, 2), (3, 5), (4, 8)]})

    obtained_df = limit_tuple_col(
        df=df,
        tuple_col="b",
        lower=1.5,
        upper=6.5)

    expected_df = pd.DataFrame({
        "a": [1, 4, 7],
        "b": [(1.5, 2), (3, 5), (4, 6.5)]})

    assert_frame_equal(obtained_df, expected_df), (
        "The values of the tuple column are not damped correctly.")


def test_offset_tuple_col():
    """Testing offset_tuple_col."""
    df = pd.DataFrame({
        "a": [1, 4, 7],
        "b": [(1, 2), (3, 5), (4, 8)]})

    res = offset_tuple_col(
        df=df,
        offset_col="a",
        tuple_col="b")

    expected = pd.Series([(1+1, 2+1), (3+4, 5+4), (4+7, 8+7)])

    assert (res == expected).min(), "Tuple values are not offset correctly."
