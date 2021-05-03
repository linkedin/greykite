import pandas as pd

from greykite.algo.uncertainty.conditional.estimate_distribution import estimate_empirical_distribution


def test_estimate_empirical_distribution():
    """Testing estimate_empirical_distribution function"""
    df = pd.DataFrame({
        "x": ["a"]*100 + ["b"]*100,
        "y": list(range(100, 200)) + list(range(200, 300))})

    # test for the case with specified quantile_grid_size
    model = estimate_empirical_distribution(
        df=df,
        value_col="y",
        quantile_grid_size=0.25,
        quantiles=None,
        conditional_cols=["x"])

    # checking if the output dataframes have the expected column names
    expected_cols_overall = ["y_quantile_summary", "y_min", "y_mean", "y_max", "y_std", "y_count"]
    expected_cols = ["x"] + expected_cols_overall
    assert list(model["ecdf_df_overall"].columns) == expected_cols_overall
    assert list(model["ecdf_df"].columns) == expected_cols

    assert model["ecdf_df_overall"]["y_min"].values.round(1) == 100.0, (
        "minimum of response is not calculated correctly")
    assert model["ecdf_df_overall"]["y_quantile_summary"].values[0][0].round(2) == 149.75, (
        "quantile summary is not correct")
    assert model["ecdf_df_overall"]["y_std"].values.round(2) == 57.88, (
        "standard deviation of response is not calculated correctly")

    assert model["ecdf_df"].iloc[0]["y_quantile_summary"] == (124.75, 149.5, 174.25), (
        "quantile summary is not correct")
    assert model["ecdf_df"].iloc[1]["y_quantile_summary"] == (224.75, 249.5, 274.25), (
        "quantile summary is not correct")

    assert model["ecdf_df"].iloc[0]["y_count"] == 100, "sample size (count) is not correct"
    assert model["ecdf_df_overall"].iloc[0]["y_count"] == 200, "overall sample size (count) is not correct"

    # test for the case with specified quantiles argument
    model = estimate_empirical_distribution(
        df=df,
        value_col="y",
        quantile_grid_size=None,
        quantiles=[0.25, 0.50, 0.75],
        conditional_cols=["x"])

    assert model["ecdf_df_overall"]["y_min"].values.round(1) == 100.0, (
        "minimum of response is not calculated correctly")
    assert model["ecdf_df_overall"]["y_quantile_summary"].values[0][0].round(2) == 149.75, (
        "quantile summary is not correct")
    assert model["ecdf_df_overall"]["y_std"].values.round(2) == 57.88, (
        "standard deviation of response is not calculated correctly")

    assert model["ecdf_df"].iloc[0]["y_quantile_summary"] == (124.75, 149.5, 174.25), (
        "quantile summary is not correct")
    assert model["ecdf_df"].iloc[1]["y_quantile_summary"] == (224.75, 249.5, 274.25), (
        "quantile summary is not correct")

    # test for the case with conditional_cols = None
    model = estimate_empirical_distribution(
        df=df,
        value_col="y",
        quantile_grid_size=None,
        quantiles=[0.25, 0.50, 0.75],
        conditional_cols=None)

    assert model["ecdf_df_overall"]["y_min"].values.round(1) == 100.0, (
        "minimum of response is not calculated correctly")
    assert model["ecdf_df_overall"]["y_quantile_summary"].values[0][0].round(2) == 149.75, (
        "quantile summary is not correct")
    assert model["ecdf_df_overall"]["y_std"].values.round(2) == 57.88, (
        "standard deviation of response is not calculated correctly")

    # we expect same values for ecdf_df because conditional_cols = None
    assert model["ecdf_df"]["y_min"].values.round(1) == 100.0, (
        "minimum of response is not calculated correctly")
    assert model["ecdf_df"]["y_quantile_summary"].values[0][0].round(2) == 149.75, (
        "quantile summary is not correct")
    assert model["ecdf_df"]["y_std"].values.round(2) == 57.88, (
        "standard deviation of response is not calculated correctly")
