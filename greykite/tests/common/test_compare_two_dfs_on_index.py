import pandas as pd

from greykite.common.compare_two_dfs_on_index import compare_two_dfs_on_index


def test_compare_two_dfs_on_index():

    n = 3
    df1 = pd.DataFrame({
        "x": range(n),
        "y": [1.0, 2.0, 3.0],
        "z": [-1.0, -2.0, -3.0]
        })

    df2 = pd.DataFrame({
        "x": range(n),
        "y": [1.0, 2.0, 6.0],
        "z": [-1.0, -2.0, -3.3]
        })

    res = compare_two_dfs_on_index(
        dfs=[df1, df2],
        df_labels=["one", "two"],
        index_col="x",
        diff_cols=None,
        relative_diff=False)

    diff_df = res["diff_df"]
    diff_fig = res["diff_fig"]

    expected_diff_df = pd.DataFrame({
        "x": range(n),
        "y": [0.0, 0.0, 3.0],
        "z": [0.0, 0.0, -0.30]})

    assert diff_fig.layout.title.text == "change in variables from one to two"
    assert len(diff_fig.data) == 2
    assert diff_df.round(2).equals(expected_diff_df.round(2))

    # Selects columns to be differenced
    res = compare_two_dfs_on_index(
        dfs=[df1, df2],
        df_labels=["one", "two"],
        index_col="x",
        diff_cols=["y"],
        relative_diff=False)

    diff_df = res["diff_df"]
    diff_fig = res["diff_fig"]

    expected_diff_df = pd.DataFrame({
        "x": range(n),
        "y": [0.0, 0.0, 3.0]})

    assert diff_fig.layout.title.text == "change in variables from one to two"
    assert len(diff_fig.data) == 1
    assert diff_df.round(2).equals(expected_diff_df.round(2))

    # Uses relative difference
    res = compare_two_dfs_on_index(
        dfs=[df1, df2],
        df_labels=["one", "two"],
        index_col="x",
        diff_cols=None,
        relative_diff=True)

    diff_df = res["diff_df"]
    diff_fig = res["diff_fig"]

    expected_diff_df = pd.DataFrame({
        "x": range(n),
        "y": [0.0, 0.0, 1.0],
        "z": [0.0, 0.0, -0.1]})

    assert diff_fig.layout.title.text == "change in variables from one to two"
    assert len(diff_fig.data) == 2
    assert diff_df.round(2).equals(expected_diff_df.round(2))
