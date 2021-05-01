import datetime
import math
from functools import partial

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats

from greykite.common.viz.timeseries_plotting_mpl import plt_compare_timeseries
from greykite.common.viz.timeseries_plotting_mpl import plt_longterm_ts_agg
from greykite.common.viz.timeseries_plotting_mpl import plt_overlay_long_df
from greykite.common.viz.timeseries_plotting_mpl import plt_overlay_with_bands


matplotlib.use("agg")  # noqa: E402


def test_plt_compare_timeseries():
    date_list = pd.date_range(
        start=datetime.datetime(2018, 6, 1),
        periods=24 * 600,
        freq="H").tolist()

    df = pd.DataFrame({"ts": date_list})
    df1 = df.copy()
    df2 = df.copy()

    value_col = "y"
    df1[value_col] = np.random.normal(size=df1.shape[0])
    df2[value_col] = np.random.normal(size=df2.shape[0])

    plt_compare_timeseries(
        df_dict={"obs": df1, "forecast": df2},
        time_col="ts",
        value_col="y",
        start_time=datetime.datetime(2019, 9, 1),
        end_time=datetime.datetime(2019, 9, 10))

    # custom legends
    plt_compare_timeseries(
        df_dict={"obs": df1, "forecast": df2},
        time_col="ts",
        value_col="y",
        legends_dict={"obs": "observed", "forecast": "silverkite forecast"},
        start_time=datetime.datetime(2019, 9, 1),
        end_time=datetime.datetime(2019, 9, 10))

    # custom colors
    plt_compare_timeseries(
        df_dict={"obs": df1, "forecast": df2},
        time_col="ts",
        value_col="y",
        colors_dict={"obs": "red", "forecast": "green"},
        start_time=datetime.datetime(2019, 9, 1),
        end_time=datetime.datetime(2019, 9, 10))


def test_plt_overlay_long_df():
    m = 200
    x = np.array(list(range(m)) * m) / (1.0 * m)
    x = x * 2 * math.pi
    z = []
    for u in range(m):
        z = z + [u] * m
    y = np.sin(x) + np.random.normal(0, 1, len(x))
    w = np.cos(x) + np.random.normal(0, 1, len(x))
    df = pd.DataFrame({"x": x, "y": y, "z": z, "w": w})

    agg_dict = {
        "y": [np.nanmean, partial(np.nanpercentile, q=25), partial(np.nanpercentile, q=75)],
        "w": np.nanmean
    }
    agg_col_names = ["mean", "lower", "upper", "w"]
    x_col = "x"
    y_col = "y"
    split_col = "z"
    df_plt = plt_overlay_long_df(
        df=df,
        x_col=x_col,
        y_col=y_col,
        split_col=split_col,
        plt_title="",
        agg_dict=agg_dict,
        agg_col_names=agg_col_names)
    assert list(df_plt.columns) == ["x"] + agg_col_names


def test_plt_overlay_with_bands():
    m = 200
    x = np.array(list(range(m)) * m) / (1.0 * m)
    x = x * 2 * math.pi
    z = []
    for u in range(m):
        z = z + [u] * m
    y = np.sin(x) + np.random.normal(0, 1, len(x))
    df = pd.DataFrame({"x": x, "y": y, "z": z})

    x_col = "x"
    y_col = "y"
    split_col = "z"

    plt_overlay_with_bands(
        df=df,
        x_col=x_col,
        y_col=y_col,
        split_col=split_col,
        perc=[25, 75],
        overlay_color="black",
        agg_col_colors=None,
        plt_title=None)


def test_plt_longterm_ts_agg_fixed_color():
    """Testing `plt_longterm_ts_agg` with fixed color across"""
    r = 10.0
    x = np.linspace(2.0, 2.0 + r, num=100)
    y = np.linspace(3.0, 3.0 + r, num=100) ** 2 + np.random.normal(0, 20, 100)

    df = pd.DataFrame({"x": x, "y": y})
    df["window"] = df["x"].map(round)

    plt_longterm_ts_agg(
        df=df,
        time_col="x",
        window_col="window",
        value_col="y",
        agg_func=np.nanmean,
        plt_title="fixed color",
        color="blue")


def test_plt_longterm_ts_agg_changing_color():
    """Testing `plt_longterm_ts_agg` with changing color across the curve"""
    r = 10.0
    x = np.linspace(2.0, 2.0 + r, num=100)
    y = np.linspace(3.0, 3.0 + r, num=100) ** 2 + np.random.normal(0, 20, 100)
    z = ["red"]*20 + ["green"]*20 + ["blue"]*60

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    df["window"] = df["x"].map(round)

    plt_longterm_ts_agg(
        df=df,
        time_col="x",
        window_col="window",
        value_col="y",
        color_col="z",
        agg_func=np.nanmean,
        plt_title="changing_color",
        color=None)


def test_plt_longterm_ts_agg_custom_choose_color():
    """Testing `plt_longterm_ts_agg` with custom choice of color"""
    r = 10.0
    x = np.linspace(2.0, 2.0 + r, num=100)
    y = np.linspace(3.0, 3.0 + r, num=100) ** 2 + np.random.normal(0, 20, 100)
    z = ["red"]*20 + ["green"]*20 + ["blue"]*60

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    df["window"] = df["x"].map(round)

    # we define a custom `choose_color_func`
    # this function returns the most common color seen for the give slice
    def choose_color_func(x):
        return scipy.stats.mode(x)[0][0]

    plt_longterm_ts_agg(
        df=df,
        time_col="x",
        window_col="window",
        value_col="y",
        color_col="z",
        agg_func=np.nanmean,
        plt_title="changing_color",
        color=None,
        choose_color_func=choose_color_func)
