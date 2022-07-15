import pandas as pd
import pytest

from greykite.common.features.normalize import normalize_df


def test_normalize_df():
    """Testing `normalize_df`"""
    df = pd.DataFrame({"x": [2, 1, 3], "y": [1, 1, 1]})

    expected_warning = "was dropped during normalization"

    # `"statistical"` method
    with pytest.warns(Warning) as record:
        normalization_info = normalize_df(
            df=df,
            method="statistical",
            drop_degenerate_cols=True)
        assert expected_warning in record[0].message.args[0]

    normalize_df_func = normalization_info["normalize_df_func"]
    normalized_df = normalize_df_func(df)
    assert list(normalized_df.columns) == ["x"]
    assert list(normalized_df["x"].values) == [0, -1, 1]

    # `"zero_at_origin"` method
    with pytest.warns(Warning) as record:
        normalization_info = normalize_df(
            df=df,
            method="zero_at_origin",
            drop_degenerate_cols=True)
        assert expected_warning in record[0].message.args[0]

    normalize_df_func = normalization_info["normalize_df_func"]
    normalized_df = normalize_df_func(df)
    assert list(normalized_df.columns) == ["x"]
    assert list(normalized_df["x"].values) == [0, -0.5, 0.5]

    # `"statistical"` method with replacement of zero denominator
    normalization_info = normalize_df(
        df=df,
        method="statistical",
        drop_degenerate_cols=True,
        replace_zero_denom=True)

    normalize_df_func = normalization_info["normalize_df_func"]
    normalized_df = normalize_df_func(df)
    assert list(normalized_df.columns) == ["x", "y"]
    assert list(normalized_df["x"].values) == [0, -1, 1]
    assert list(normalized_df["y"].values) == [0, 0, 0]

    # `"zero_to_one"` method
    with pytest.warns(Warning) as record:
        normalization_info = normalize_df(
            df=df,
            method="zero_to_one",
            drop_degenerate_cols=True)
        assert expected_warning in record[0].message.args[0]

    normalize_df_func = normalization_info["normalize_df_func"]
    normalized_df = normalize_df_func(df)
    assert list(normalized_df.columns) == ["x"]
    assert list(normalized_df["x"].values) == [0.5, 0, 1]

    # `"minus_half_to_half"` method
    with pytest.warns(Warning) as record:
        normalization_info = normalize_df(
            df=df,
            method="minus_half_to_half",
            drop_degenerate_cols=True)
        assert expected_warning in record[0].message.args[0]

    normalize_df_func = normalization_info["normalize_df_func"]
    normalized_df = normalize_df_func(df)
    assert list(normalized_df.columns) == ["x"]
    assert list(normalized_df["x"].values) == [0, -0.5, 0.5]

    # apply to a new dataframe with new elements
    new_df = pd.DataFrame({"x": [1, 2, 3, 4, -1], "y": [1, 1, 1, 5, 6]})
    with pytest.warns(Warning) as record:
        normalization_info = normalize_df(
            df=df,
            method="statistical",
            drop_degenerate_cols=True)
        assert expected_warning in record[0].message.args[0]

    normalize_df_func = normalization_info["normalize_df_func"]
    normalized_df = normalize_df_func(new_df)
    assert list(normalized_df.columns) == ["x"]
    assert list(normalized_df["x"].values) == [-1, 0, 1, 2, -3]

    # We do not drop degenerate columns: `drop_degenerate_cols=False`
    # no warning is expected
    normalization_info = normalize_df(
        df=df,
        method="statistical",
        drop_degenerate_cols=False)
    normalize_df_func = normalization_info["normalize_df_func"]
    normalized_df = normalize_df_func(df)
    assert list(normalized_df.columns) == ["x", "y"]
    assert list(normalized_df["x"].values) == [0, -1, 1]
    assert sum(normalized_df["y"].isnull()) == 3

    # Testing for raising exception if the `method` is not available
    expected_match = "is not implemented"
    with pytest.raises(NotImplementedError, match=expected_match):
        normalize_df(
            df=df,
            method="quantile_based",
            drop_degenerate_cols=True)

    # testing for exception for the case when all columns are degenerate
    df = pd.DataFrame({"x": [2, 2, 2], "y": [1, 1, 1]})
    expected_match = "All columns were degenerate"
    with pytest.raises(ValueError, match=expected_match):
        normalize_df(
            df=df,
            method="statistical",
            drop_degenerate_cols=True)
