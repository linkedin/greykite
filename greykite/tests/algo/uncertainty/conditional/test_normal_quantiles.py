import pandas as pd

from greykite.algo.uncertainty.conditional.estimate_distribution import estimate_empirical_distribution
from greykite.algo.uncertainty.conditional.normal_quantiles import normal_quantiles_df
from greykite.common.testing_utils import gen_sliced_df


def test_normal_quantiles_df():
    """Testing calculating normal quantiles for a sliced dataframe."""
    df = gen_sliced_df()

    model_dict = estimate_empirical_distribution(
        df=df,
        distribution_col="y",
        quantile_grid_size=None,
        quantiles=[0.025, 0.975],
        conditional_cols=["x"])

    ecdf_df = model_dict["ecdf_df"]

    # check the case with regular mean
    quantiles_df = normal_quantiles_df(
        df=ecdf_df,
        mean_col="y_mean",
        std_col="y_std",
        quantiles=(0.025, 0.975))

    quantiles_df = quantiles_df[["x", "normal_quantiles"]].copy()
    quantiles_df["normal_quantiles"] = quantiles_df["normal_quantiles"].apply(lambda x: tuple(e.round(2) for e in x))
    expected_df = pd.DataFrame()
    expected_df["x"] = ["a", "b", "c", "d", "e"]
    expected_df["normal_quantiles"] = [
        (89.94, 102.01),
        (190.2, 201.9),
        (290.25, 301.91),
        (479.37, 561.93),
        (-14.15, 6.43)]

    assert expected_df.equals(quantiles_df), "quantiles are not calculated correctly"

    # check the case with fixed_mean
    quantile_summary_col = "custom_quantiles"
    quantiles_df = normal_quantiles_df(
        df=ecdf_df,
        mean_col=None,
        std_col="y_std",
        fixed_mean=0,
        quantiles=(0.025, 0.975),
        quantile_summary_col=quantile_summary_col
    )

    quantiles_df = quantiles_df[["x", quantile_summary_col]].copy()
    quantiles_df[quantile_summary_col] = quantiles_df[quantile_summary_col].apply(lambda x: tuple(e.round(2) for e in x))
    expected_df = pd.DataFrame()
    expected_df["x"] = ["a", "b", "c", "d", "e"]
    expected_df[quantile_summary_col] = [
        (-6.04, 6.04),
        (-5.85, 5.85),
        (-5.83, 5.83),
        (-41.28, 41.28),
        (-10.29, 10.29)]

    assert expected_df.equals(quantiles_df), "quantiles are not calculated correctly"
