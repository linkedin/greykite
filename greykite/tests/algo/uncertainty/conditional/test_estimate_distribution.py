import pandas as pd

from greykite.algo.uncertainty.conditional.estimate_distribution import estimate_empirical_distribution
from greykite.common.python_utils import assert_equal


def test_estimate_empirical_distribution():
    """Testing estimate_empirical_distribution function"""
    df = pd.DataFrame({
        "x": ["a"]*100 + ["b"]*100,
        "y": list(range(100, 200)) + list(range(200, 300))
    })
    expected_ecdf_df = pd.DataFrame({
        "x": ["a", "b"],
        "y_ecdf_quantile_summary": [(-24.75, 0.0, 24.75), (-24.75, 0.0, 24.75)],
        "y_min": [100, 200],
        "y_mean": [149.5, 249.5],
        "y_max": [199, 299],
        "y_std": [29.011492, 29.011492],
        "y_count": [100, 100]
    })
    expected_ecdf_df_overall = pd.DataFrame({
        "y_ecdf_quantile_summary": [(-49.75, 0.0, 49.75)],
        "y_min": [100],
        "y_mean": [199.5],
        "y_max": [299],
        "y_std": [57.87915],
        "y_count": [200]
    })

    # Tests for the case with specified quantile_grid_size
    model = estimate_empirical_distribution(
        df=df,
        distribution_col="y",
        quantile_grid_size=0.25,
        quantiles=None,
        conditional_cols=["x"]
    )
    assert_equal(model["ecdf_df"], expected_ecdf_df)
    assert_equal(model["ecdf_df_overall"], expected_ecdf_df_overall)

    # Tests for the case with specified quantiles argument
    model = estimate_empirical_distribution(
        df=df,
        distribution_col="y",
        quantile_grid_size=None,
        quantiles=[0.25, 0.50, 0.75],
        conditional_cols=["x"],
        remove_conditional_mean=True
    )
    assert_equal(model["ecdf_df"], expected_ecdf_df)
    assert_equal(model["ecdf_df_overall"], expected_ecdf_df_overall)

    # Tests for the case when ``conditional_col`` is not given and
    # ``remove_conditional_col`` is False
    expected_ecdf_df_overall["y_ecdf_quantile_summary"] = expected_ecdf_df_overall["y_ecdf_quantile_summary"].apply(
        lambda x: tuple(e + expected_ecdf_df_overall["y_mean"].iloc[0] for e in x))
    for conditional_cols in [None, []]:
        model = estimate_empirical_distribution(
            df=df,
            distribution_col="y",
            quantile_grid_size=None,
            quantiles=[0.25, 0.50, 0.75],
            conditional_cols=conditional_cols,
            remove_conditional_mean=False
        )
        # No conditional col is given, hence ecdf_df is same as ecdf_df_overall
        assert_equal(model["ecdf_df"], expected_ecdf_df_overall)
        assert_equal(model["ecdf_df_overall"], expected_ecdf_df_overall)
