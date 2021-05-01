import matplotlib  # isort:skip

matplotlib.use("agg")  # noqa: E402
import matplotlib.pyplot as plt  # isort:skip

from greykite.algo.forecast.similarity.forecast_similarity_based import forecast_similarity_based
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import calc_pred_err
from greykite.common.testing_utils import generate_df_for_tests


def test_forecast_similarity_based():
    """ Testing the function: forecast_similarity_based in various examples
    """
    data = generate_df_for_tests(freq="D", periods=30*8)  # 8 months
    df = data["df"]
    train_df = data["train_df"]
    test_df = data["test_df"]

    df["z"] = df["y"] + 1
    train_df["z"] = train_df["y"] + 1
    test_df["z"] = test_df["y"] + 1

    train_df = train_df[["ts", "y", "z"]]
    test_df = test_df[["ts", "y", "z"]]

    res = forecast_similarity_based(
        df=train_df,
        time_col="ts",
        value_cols=["y", "z"],
        agg_method="median",
        agg_func=None,
        match_cols=["dow"])

    # forecast using predict
    fdf_median = res["predict"](test_df)
    assert (fdf_median["z"] - fdf_median["y"] - 1.0).abs().max().round(2) == 0.0, \
        "forecast for z must be forecast for y + 1 at each timestamp"
    err = calc_pred_err(test_df["y"], fdf_median["y"])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.3

    # forecast using predict_n
    fdf_median = res["predict_n"](test_df.shape[0])
    err = calc_pred_err(test_df["y"], fdf_median["y"])
    assert err[enum.get_metric_name()] > 0.3

    res = forecast_similarity_based(
        df=train_df,
        time_col="ts",
        value_cols=["y", "z"],
        agg_method="mean",
        agg_func=None,
        match_cols=["dow"])

    # forecast using the mean of all similar times
    fdf_mean = res["predict"](test_df)
    err = calc_pred_err(test_df["y"], fdf_mean["y"])
    assert err[enum.get_metric_name()] > 0.3

    res = forecast_similarity_based(
        df=train_df,
        time_col="ts",
        value_cols=["y", "z"],
        agg_method="most_recent",
        agg_func=None,
        match_cols=["dow"],
        recent_k=3)

    # forecast using the mean of 3 recent times similar to the given time
    fdf_recent3_mean = res["predict"](test_df)
    err = calc_pred_err(test_df["y"], fdf_recent3_mean["y"])
    assert err[enum.get_metric_name()] > 0.3

    plt.plot(df["ts"].dt.strftime('%Y-%m-%d'), df["y"], label="true", alpha=0.5)
    plt.plot(fdf_median["ts"].dt.strftime('%Y-%m-%d'), fdf_median["y"], alpha=0.5, label="median pred wrt dow")
    plt.plot(fdf_mean["ts"].dt.strftime('%Y-%m-%d'), fdf_mean["y"], alpha=0.5, label="mean pred wrt dow")
    plt.plot(fdf_recent3_mean["ts"].dt.strftime('%Y-%m-%d'), fdf_recent3_mean["y"], alpha=0.5, label="mean recent 3 wrt dow")
    plt.xticks(rotation=15)
    plt.legend()
