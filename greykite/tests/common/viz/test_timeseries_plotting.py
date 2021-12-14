import datetime
from datetime import datetime as dt
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from plotly.colors import DEFAULT_PLOTLY_COLORS
from testfixtures import LogCapture

import greykite.common.constants as cst
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.viz.timeseries_plotting import add_groupby_column
from greykite.common.viz.timeseries_plotting import flexible_grouping_evaluation
from greykite.common.viz.timeseries_plotting import grouping_evaluation
from greykite.common.viz.timeseries_plotting import plot_forecast_vs_actual
from greykite.common.viz.timeseries_plotting import plot_multivariate
from greykite.common.viz.timeseries_plotting import plot_multivariate_grouped
from greykite.common.viz.timeseries_plotting import plot_univariate
from greykite.common.viz.timeseries_plotting import split_range_into_groups


def test_plot_multivariate():
    """Tests plot_multivariate function"""
    x_col = "time"
    df = pd.DataFrame({
        x_col: [dt(2018, 1, 1),
                dt(2018, 1, 2),
                dt(2018, 1, 3)],
        "oranges": [8.5, 2.0, 3.0],
        "apples": [1.4, 2.1, 3.4],
        "bananas": [4.2, 3.1, 3.0],
    })

    # plot with default values
    fig = plot_multivariate(
        df=df,
        x_col=x_col)
    assert fig.layout.showlegend
    assert fig.layout.xaxis.title.text == x_col
    assert fig.layout.yaxis.title.text == cst.VALUE_COL
    assert len(fig.data) == 3
    assert fig.data[0].mode == "lines"
    assert fig.data[0].name == "oranges"
    assert fig.data[1].name == "apples"
    assert fig.data[2].name == "bananas"

    # plot with style override
    fig = plot_multivariate(
        df=df,
        x_col=x_col,
        y_col_style_dict={
            "oranges": {
                "legendgroup": "one",
                "line": {
                    "color": "red",
                    "dash": "dot"
                }
            },
            "apples": None,
            "bananas": {
                "name": "plantain",
                "legendgroup": "one",
                "mode": "markers",
                "line": None  # Remove line params since we use mode="markers"
            }
        },
        xlabel="xlab",
        ylabel="ylab",
        title="New Title",
        showlegend=False
    )
    assert not fig.layout.showlegend
    assert fig.layout.xaxis.title.text == "xlab"
    assert fig.layout.yaxis.title.text == "ylab"
    assert len(fig.data) == 3
    assert fig.data[0].mode == "lines"
    assert fig.data[0].legendgroup == "one"
    assert fig.data[0].line.color == "red"
    assert fig.data[0].name == "oranges"
    assert fig.data[1].name == "apples"
    assert fig.data[2].name == "plantain"
    assert fig.data[2].mode == "markers"

    # ylabel is used for default title
    fig = plot_multivariate(
        df=df,
        x_col=x_col,
        xlabel="xlab",
        ylabel="ylab")
    assert fig.layout.title.text == f"ylab vs xlab"

    # plotly style
    fig = plot_multivariate(
        df=df,
        x_col=x_col,
        y_col_style_dict="plotly")
    assert [fig.data[i].name for i in range(len(fig.data))] ==\
           ["oranges", "apples", "bananas"]
    assert fig.data[0].line.color is None
    assert fig.data[1].fill is None

    # auto style
    fig = plot_multivariate(
        df=df,
        x_col=x_col,
        y_col_style_dict="auto")
    assert [fig.data[i].name for i in range(len(fig.data))] ==\
           ["apples", "bananas", "oranges"]  # sorted ascending
    assert fig.data[0].line.color == "rgba(0, 145, 202, 1.0)"
    assert fig.data[1].fill is None

    # auto-fill style
    fig = plot_multivariate(
        df=df,
        x_col=x_col,
        y_col_style_dict="auto-fill",
        default_color="blue")
    assert [fig.data[i].name for i in range(len(fig.data))] ==\
           ["apples", "bananas", "oranges"]  # sorted ascending
    assert fig.data[0].line.color == "blue"
    assert fig.data[1].fill == "tonexty"
    assert fig.data[2].fill == "tonexty"


def test_plot_multivariate_grouped():
    """Tests plot_multivariate_grouped function"""
    x_col = "time"
    df = pd.DataFrame({
        x_col: [dt(2018, 1, 1),
                dt(2018, 1, 2),
                dt(2018, 1, 3)],
        "y1": [8.5, 2.0, 3.0],
        "y2": [1.4, 2.1, 3.4],
        "y3": [4.2, 3.1, 3.0],
        "y4": [0, 1, 2],
        "y5": [10, 9, 8],
        "group": [1, 2, 1],
    })

    # Default values
    fig = plot_multivariate_grouped(
        df=df,
        x_col=x_col,
        y_col_style_dict={
            "y1": {
                "name": "apple",
                "legendgroup": "one",
                "mode": "markers",
                "line": None  # Remove line params since we use mode="markers"
            },
            "y2": None,
        },
        grouping_x_col="group",
        grouping_y_col_style_dict={
            "y3": {
                "line": {
                    "color": "blue"
                }
            },
            "y4": {
                "name": "banana",
                "line": {
                    "width": 2,
                    "dash": "dot"
                }
            },
            "y5": None,
        },
        grouping_x_col_values=None
    )

    assert fig.layout.showlegend
    assert fig.layout.xaxis.title.text == x_col
    assert fig.layout.yaxis.title.text == cst.VALUE_COL
    assert fig.layout.title.text == f"{cst.VALUE_COL} vs {x_col}"
    assert len(fig.data) == 8

    # In y_col_style dict, names and line colors are kept if given
    assert fig.data[0].name == "apple"
    assert fig.data[0].mode == "markers"
    assert fig.data[0].line.color == DEFAULT_PLOTLY_COLORS[0]

    assert fig.data[1].name == "y2"
    assert fig.data[1].mode == "lines"
    assert fig.data[1].line.color == DEFAULT_PLOTLY_COLORS[1]

    # In grouped_y_col_style_dict
    # names are augmented with grouping_x_col_value if given, else default name is provided
    # line colors and properties are kept if given, else line color is added
    # group == 1
    assert fig.data[2].name == "1_group_y3"
    assert fig.data[2].mode == "lines"
    assert fig.data[2].line.color == "blue"  # color was provided

    assert fig.data[3].name == "1_banana"
    assert fig.data[3].mode == "lines"
    assert fig.data[3].line.color == DEFAULT_PLOTLY_COLORS[2]
    assert fig.data[3].line.width == 2

    assert fig.data[4].name == "1_group_y5"
    assert fig.data[4].mode == "lines"
    assert fig.data[4].line.color == DEFAULT_PLOTLY_COLORS[2]  # same color as before

    # group == 2
    assert fig.data[5].name == "2_group_y3"
    assert fig.data[5].mode == "lines"
    assert fig.data[5].line.color == "blue"  # color was provided

    assert fig.data[6].name == "2_banana"
    assert fig.data[6].mode == "lines"
    assert fig.data[6].line.color == DEFAULT_PLOTLY_COLORS[3]
    assert fig.data[6].line.width == 2

    assert fig.data[7].name == "2_group_y5"
    assert fig.data[7].mode == "lines"
    assert fig.data[7].line.color == DEFAULT_PLOTLY_COLORS[3]  # same color as before

    # Custom values
    # len(colors) < len(fig.data), hence colors are interpolated
    colors = ["rgb(99, 114, 218)", "rgb(0, 145, 202)"]
    fig = plot_multivariate_grouped(
        df=df,
        x_col=x_col,
        y_col_style_dict={
            "y2": None,
        },
        grouping_x_col="group",
        grouping_y_col_style_dict={
            "y3": {
                "line": {
                    "color": "blue"
                }
            },
            "y5": None,
        },
        grouping_x_col_values=[1],
        colors=colors,
        xlabel="xlab",
        ylabel="ylab",
        title="title",
        showlegend=False)

    assert not fig.layout.showlegend
    assert fig.layout.xaxis.title.text == "xlab"
    assert fig.layout.yaxis.title.text == "ylab"
    assert fig.layout.title.text == "title"

    assert len(fig.data) == 3

    assert fig.data[0].name == "y2"
    assert fig.data[0].line.color == colors[0]

    assert fig.data[1].name == "1_group_y3"
    assert fig.data[1].line.color == "blue"  # color was provided

    assert fig.data[2].name == "1_group_y5"
    assert fig.data[2].line.color == colors[1]

    # Error when grouping_x_col_value is missing
    missing_values = {0, 3}
    with pytest.raises(ValueError, match=f"Following 'grouping_x_col_values' are missing in "
                                         f"'group' column: {missing_values}"):
        plot_multivariate_grouped(
            df=df,
            x_col=x_col,
            y_col_style_dict={
                "y2": None,
            },
            grouping_x_col="group",
            grouping_y_col_style_dict={
                "y3": None,
            },
            grouping_x_col_values=[0, 1, 3]
        )


def test_plot_univariate():
    """Tests plot_univariate function"""
    time_col = "time"
    value_col = "val"
    df = pd.DataFrame({
        time_col: [dt(2018, 1, 1, 0, 0, 1),
                   dt(2018, 1, 1, 0, 0, 2),
                   dt(2018, 1, 1, 0, 0, 3)],
        value_col: [1, 2, 3]
    })
    fig = plot_univariate(df, time_col, value_col)
    assert fig.data[0].name == value_col
    assert fig.layout.xaxis.title.text == time_col
    assert fig.layout.yaxis.title.text == value_col
    assert fig.layout.title.text == f"{value_col} vs {time_col}"
    assert fig.data[0].x.shape[0] == df.shape[0]
    assert fig.data[0].line.color == "rgb(32, 149, 212)"
    assert fig.data[0].mode == "lines"
    assert fig.layout.showlegend

    fig = plot_univariate(
        df,
        time_col,
        value_col,
        xlabel="x-axis",
        ylabel="y-axis",
        title="new title",
        color="blue",
        showlegend=False)
    assert fig.layout.xaxis.title.text == "x-axis"
    assert fig.layout.yaxis.title.text == "y-axis"
    assert fig.data[0].line.color == "blue"
    assert fig.layout.title.text == "new title"
    assert not fig.layout.showlegend


def test_plot_forecast_vs_actual():
    """Tests plot_forecast_vs_actual function"""
    size = 200
    df = pd.DataFrame({
        cst.TIME_COL: pd.date_range(start="2018-01-01", periods=size, freq="H"),
        cst.ACTUAL_COL: np.random.normal(scale=10, size=size)
    })
    df[cst.PREDICTED_COL] = df[cst.ACTUAL_COL] + np.random.normal(size=size)
    df[cst.PREDICTED_LOWER_COL] = df[cst.PREDICTED_COL] - 10
    df[cst.PREDICTED_UPPER_COL] = df[cst.PREDICTED_COL] + 10
    fig = plot_forecast_vs_actual(df)
    assert len(fig.data) == 4
    assert fig.layout.title.text == "Forecast vs Actual"
    assert fig.layout.xaxis.title.text == cst.TIME_COL
    assert fig.layout.yaxis.title.text == cst.VALUE_COL

    # checks if figure can be manipulated
    update_layout = dict(
        yaxis=dict(title="new ylabel"),
        title_text="new title",
        title_x=0.5,
        title_font_size=30)
    fig.update_layout(update_layout)
    assert len(fig.to_html(include_plotlyjs=False, full_html=True)) > 0

    # checks if lower and upper bound are optional
    fig = plot_forecast_vs_actual(df, predicted_lower_col=None)
    assert len(fig.data) == 3
    fig = plot_forecast_vs_actual(df, predicted_upper_col=None)
    assert len(fig.data) == 3
    fig = plot_forecast_vs_actual(
        df,
        predicted_lower_col=None,
        predicted_upper_col=None,
        title="new title")
    assert len(fig.data) == 2
    assert fig.layout.title.text == "new title"

    # checks if train end date can be plotted
    fig = plot_forecast_vs_actual(
        df,
        train_end_date=df[cst.TIME_COL][150],
        actual_mode="markers")
    assert len(fig.layout.annotations) > 0

    # sets a bunch of options
    fig = plot_forecast_vs_actual(
        df,
        train_end_date=df[cst.TIME_COL][150],
        showlegend=False,
        actual_mode="lines",
        actual_points_color="blue",
        actual_points_size=1.0,
        actual_color_opacity=0.3,
        forecast_curve_color="red",
        forecast_curve_dash="solid",
        ci_band_color="green",
        ci_boundary_curve_color="orange",
        ci_boundary_curve_width=2.0,
        vertical_line_color="black",
        vertical_line_width=5.0)
    assert len(fig.data) == 4
    assert not fig.layout.showlegend


def test_split_range_into_groups():
    """Tests split_range_into_groups function"""
    assert np.array_equal(
        split_range_into_groups(10, 1, "last"),
        np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]))
    assert np.array_equal(
        split_range_into_groups(10, 2, "last"),
        np.array([0., 0., 1., 1., 2., 2., 3., 3., 4., 4.]))
    assert np.array_equal(
        split_range_into_groups(10, 3, "last"),
        np.array([0., 1., 1., 1., 2., 2., 2., 3., 3., 3.]))
    assert np.array_equal(
        split_range_into_groups(10, 4, "last"),
        np.array([0., 0., 1., 1., 1., 1., 2., 2., 2., 2.]))
    assert np.array_equal(
        split_range_into_groups(10, 4, "first"),
        np.array([0., 0., 0., 0., 1., 1., 1., 1., 2., 2.]))
    assert np.array_equal(
        split_range_into_groups(10, 5, "last"),
        np.array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]))
    assert np.array_equal(
        split_range_into_groups(10, 6, "last"),
        np.array([0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]))
    assert np.array_equal(
        split_range_into_groups(10, 10, "last"),
        np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    assert np.array_equal(
        split_range_into_groups(10, 12, "last"),
        np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))


def test_add_groupby_column():
    """Tests add_groupby_column function"""
    # ``groupby_time_feature``
    df = pd.DataFrame({
        cst.TIME_COL: [
            datetime.datetime(2018, 1, 1),
            datetime.datetime(2018, 1, 2),
            datetime.datetime(2018, 1, 3),
            datetime.datetime(2018, 1, 4),
            datetime.datetime(2018, 1, 5)],
        cst.VALUE_COL: [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    result = add_groupby_column(
        df=df,
        time_col=cst.TIME_COL,
        groupby_time_feature="dow",
        groupby_sliding_window_size=None,
        groupby_custom_column=None)
    expected_col = "dow"  # ``groupby_time_feature`` is used as the column name
    expected = df.copy()
    expected[expected_col] = pd.Series([1, 2, 3, 4, 5])  # Monday, Tuesday, etc.
    assert_frame_equal(result["df"], expected)
    assert result["groupby_col"] == expected_col

    # ``groupby_sliding_window_size``
    result = add_groupby_column(
        df=df,
        time_col=cst.TIME_COL,
        groupby_time_feature=None,
        groupby_sliding_window_size=2,
        groupby_custom_column=None)
    expected_col = f"{cst.TIME_COL}_downsample"
    expected = df.copy()
    expected[expected_col] = pd.Series([
        datetime.datetime(2018, 1, 1),
        datetime.datetime(2018, 1, 3),
        datetime.datetime(2018, 1, 3),
        datetime.datetime(2018, 1, 5),
        datetime.datetime(2018, 1, 5),
    ])
    assert_frame_equal(result["df"], expected)
    assert result["groupby_col"] == expected_col

    # ``groupby_custom_column`` without a name
    df.index.name = "index_name"
    custom_groups = pd.Series(["g1", "g2", "g1", "g3", "g2"])
    result = add_groupby_column(
        df=df,
        time_col=cst.TIME_COL,
        groupby_time_feature=None,
        groupby_sliding_window_size=None,
        groupby_custom_column=custom_groups)
    expected_col = "groups"  # default name
    expected = df.copy()
    expected[expected_col] = custom_groups.values
    assert_frame_equal(result["df"], expected)
    assert result["groupby_col"] == expected_col

    # ``groupby_custom_column`` with a name
    custom_groups = pd.Series(
        ["g1", "g2", "g1", "g3", "g2"],
        name="custom_groups")
    result = add_groupby_column(
        df=df,
        time_col=cst.TIME_COL,
        groupby_time_feature=None,
        groupby_sliding_window_size=None,
        groupby_custom_column=custom_groups)
    expected_col = custom_groups.name
    expected = df.copy()
    expected[expected_col] = custom_groups.values
    assert_frame_equal(result["df"], expected)
    assert result["groupby_col"] == expected_col

    # If a column has the same name as the index, the
    # index name is set to None.
    custom_groups = pd.Series(["g1", "g2", "g1", "g3", "g2"], name="index_name")
    result = add_groupby_column(
        df=df,
        time_col=cst.TIME_COL,
        groupby_time_feature=None,
        groupby_sliding_window_size=None,
        groupby_custom_column=custom_groups)
    expected_col = custom_groups.name
    expected = df.copy()
    expected[expected_col] = custom_groups.values
    expected.index.name = None
    assert_frame_equal(result["df"], expected)
    assert result["groupby_col"] == expected_col

    # Throws exception if multiple grouping dimensions are provided
    with pytest.raises(ValueError, match="Exactly one of.*must be specified"):
        add_groupby_column(
            df=df,
            time_col=cst.TIME_COL,
            groupby_time_feature=None,
            groupby_sliding_window_size=2,
            groupby_custom_column=custom_groups)

    with pytest.raises(ValueError, match="Exactly one of.*must be specified"):
        add_groupby_column(
            df=df,
            time_col=cst.TIME_COL,
            groupby_time_feature="dow",
            groupby_sliding_window_size=None,
            groupby_custom_column=custom_groups)

    with pytest.raises(ValueError, match="Exactly one of.*must be specified"):
        add_groupby_column(
            df=df,
            time_col=cst.TIME_COL,
            groupby_time_feature="dow",
            groupby_sliding_window_size=2,
            groupby_custom_column=None)

    with pytest.raises(ValueError, match="Exactly one of.*must be specified"):
        add_groupby_column(
            df=df,
            time_col=cst.TIME_COL,
            groupby_time_feature="dow",
            groupby_sliding_window_size=2,
            groupby_custom_column=custom_groups)


def test_grouping_evaluation():
    """Tests grouping_evaluation function"""
    groupby_col = "custom_groups"
    df = pd.DataFrame({
        cst.TIME_COL: [
            datetime.datetime(2018, 1, 1),
            datetime.datetime(2018, 1, 2),
            datetime.datetime(2018, 1, 3),
            datetime.datetime(2018, 1, 4),
            datetime.datetime(2018, 1, 5)],
        cst.VALUE_COL: [1.0, 2.0, 3.0, 4.0, 5.0],
        groupby_col: ["g1", "g2", "g1", "g3", "g2"]
    })

    # grouping function on one column
    def grouping_func_min(grp):
        return np.min(grp[cst.VALUE_COL])
    grouping_func_name = "min"
    grouped_df = grouping_evaluation(
        df=df,
        groupby_col=groupby_col,
        grouping_func=grouping_func_min,
        grouping_func_name=grouping_func_name)
    expected = pd.DataFrame({
        groupby_col: ["g1", "g2", "g3"],
        grouping_func_name: [1.0, 2.0, 4.0]
    })
    assert_frame_equal(grouped_df, expected)

    # grouping function on two columns
    df[cst.ACTUAL_COL] = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    df[cst.PREDICTED_COL] = pd.Series([1.0, 4.0, 3.0, 2.0, 3.0])

    metric = EvaluationMetricEnum.MeanAbsolutePercentError
    score_func = metric.get_metric_func()

    def grouping_func_mape(grp):
        return score_func(grp[cst.ACTUAL_COL], grp[cst.PREDICTED_COL])
    grouped_df = grouping_evaluation(
        df=df,
        groupby_col=groupby_col,
        grouping_func=grouping_func_mape,
        grouping_func_name=metric.get_metric_name())
    expected = pd.DataFrame({
        groupby_col: ["g1", "g2", "g3"],
        metric.get_metric_name(): [0.0, 70.0, 50.0]
    })
    assert_frame_equal(grouped_df, expected)


def test_flexible_grouping_evaluation():
    """Tests flexible_grouping_evaluation function"""
    nrows = 365
    df = pd.DataFrame({
        cst.TIME_COL: pd.date_range(start='2018-01-01', periods=nrows, freq="D"),
        cst.ACTUAL_COL: np.random.random(nrows),
        cst.PREDICTED_COL: np.random.random(nrows),
        "groups": np.random.choice(list("ABCDE"), size=nrows)
    })

    # no changes
    mapped_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=None)
    assert_frame_equal(mapped_df, df)

    # `map_func_dict`
    map_func_dict = {
        "abs_error": lambda row: np.abs(row[cst.PREDICTED_COL] - row[cst.ACTUAL_COL]),
        "squared_error": lambda row: (row[cst.PREDICTED_COL] - row[cst.ACTUAL_COL])**2
    }
    mapped_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=map_func_dict)

    expected = df.copy()
    expected["abs_error"] = (df[cst.PREDICTED_COL] - df[cst.ACTUAL_COL]).abs()
    expected["squared_error"] = (df[cst.PREDICTED_COL] - df[cst.ACTUAL_COL])**2
    assert_frame_equal(mapped_df, expected)

    with pytest.raises(ValueError,
                       match="Must specify `agg_kwargs` if grouping is requested via `groupby_col`."):
        flexible_grouping_evaluation(
            df=df,
            map_func_dict=map_func_dict,
            groupby_col="groups"
        )

    # `agg_kwargs` with tuples for named aggregation
    agg_kwargs = {
        f"{cst.ACTUAL_COL}_mean": pd.NamedAgg(column=cst.ACTUAL_COL, aggfunc="mean"),
        f"{cst.PREDICTED_COL}_min": pd.NamedAgg(column=cst.ACTUAL_COL, aggfunc="min"),
        f"{cst.PREDICTED_COL}_max": pd.NamedAgg(column=cst.ACTUAL_COL, aggfunc="max"),
        "squared_error_MedianSquaredError": pd.NamedAgg(column="squared_error", aggfunc=np.nanmedian),
        "squared_error_MSE": pd.NamedAgg(column="squared_error", aggfunc=np.nanmean),
        "abs_error_MAE": pd.NamedAgg(column="abs_error", aggfunc=np.nanmean),
        "abs_error_q95_abs_error": pd.NamedAgg(column="abs_error", aggfunc=partial(np.nanquantile, q=0.95)),
        "abs_error_q05_abs_error": pd.NamedAgg(column="abs_error", aggfunc=partial(np.nanquantile, q=0.05)),
    }
    eval_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=map_func_dict,
        groupby_col="groups",
        agg_kwargs=agg_kwargs)
    assert eval_df.shape[0] == df["groups"].nunique()
    assert eval_df.index.name == "groups"
    assert list(eval_df.columns) == [
        f"{cst.ACTUAL_COL}_mean",
        f"{cst.PREDICTED_COL}_min",
        f"{cst.PREDICTED_COL}_max",
        "squared_error_MedianSquaredError",
        "squared_error_MSE",
        "abs_error_MAE",
        "abs_error_q95_abs_error",
        "abs_error_q05_abs_error"]
    assert all(eval_df["squared_error_MedianSquaredError"].values == expected.groupby("groups")["squared_error"].agg("median").values)
    assert all(eval_df["abs_error_MAE"].values == expected.groupby("groups")["abs_error"].agg("mean").values)

    with LogCapture(cst.LOGGER_NAME) as log_capture:
        flexible_grouping_evaluation(
            df=df,
            map_func_dict=map_func_dict,
            agg_kwargs=agg_kwargs)
        expected_message = "`agg_kwargs` is ignored because `groupby_col` is None. " \
                           "Specify `groupby_col` to allow aggregation."
        log_capture.check(
            (cst.LOGGER_NAME, "WARNING", expected_message))

    # `agg_kwargs` with `func`
    agg_kwargs = {
        "func": {
            cst.ACTUAL_COL: "mean",
            cst.PREDICTED_COL: ["min", "max"],
            "squared_error": [np.nanmedian, np.nanmean],
            "abs_error": [np.nanmean, partial(np.nanquantile, q=0.95), partial(np.nanquantile, q=0.05)],
        }
    }
    eval_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=map_func_dict,
        groupby_col="groups",
        agg_kwargs=agg_kwargs)
    assert eval_df.shape[0] == df["groups"].nunique()
    assert eval_df.index.name == "groups"
    assert list(eval_df.columns) == [
        f"{cst.ACTUAL_COL}_mean",
        f"{cst.PREDICTED_COL}_min",
        f"{cst.PREDICTED_COL}_max",
        "squared_error_nanmedian",
        "squared_error_nanmean",
        "abs_error_nanmean",
        "abs_error_nanquantile",
        "abs_error_nanquantile"]  # duplicate name

    # same result without extended column names
    with pytest.warns(UserWarning) as record:
        eval_df2 = flexible_grouping_evaluation(
            df=df,
            map_func_dict=map_func_dict,
            groupby_col="groups",
            agg_kwargs=agg_kwargs,
            extend_col_names=False)
        assert list(eval_df2.columns) == [
            "mean",
            "min",
            "max",
            "nanmedian",
            "nanmean",
            "nanmean",
            "nanquantile",
            "nanquantile"]
        assert "Column names are not unique. Use `extend_col_names=True` " \
               "to uniquely identify every column." in record[0].message.args[0]

    # same result with multilevel index
    eval_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=map_func_dict,
        groupby_col="groups",
        agg_kwargs=agg_kwargs,
        extend_col_names=None)
    assert eval_df.columns.nlevels == 2

    # `agg_kwargs` with aggregation dictionary and list value
    agg_list = [np.nanmedian, np.nanmean]
    eval_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=None,
        groupby_col="groups",
        agg_kwargs={"func": agg_list}
    )
    agg_dict = {  # equivalent dictionary specification
        "actual": [np.nanmedian, np.nanmean],
        "forecast": [np.nanmedian, np.nanmean],
    }
    eval_df_expected = flexible_grouping_evaluation(
        df=df,
        map_func_dict=None,
        groupby_col="groups",
        agg_kwargs={"func": agg_dict}
    )
    assert_frame_equal(eval_df, eval_df_expected)

    # `unpack_list` without `list_names_dict`
    actual_quantiles = [0.1, 0.2, 0.8, 0.9]
    forecast_quantiles = [0.25, 0.75]
    agg_kwargs = dict(
        actual_Q=pd.NamedAgg(
            column="actual",
            aggfunc=lambda grp_values: partial(np.nanquantile, q=actual_quantiles)(grp_values).tolist()),
        forecast_Q=pd.NamedAgg(
            column="forecast",
            aggfunc=lambda grp_values: partial(np.nanquantile, q=forecast_quantiles)(grp_values).tolist()),
    )
    eval_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=None,
        groupby_col="groups",
        agg_kwargs=agg_kwargs,
        extend_col_names=True,
        unpack_list=True)
    assert list(eval_df.columns) == (
            [f"actual_Q{i}" for i in range(len(actual_quantiles))] +
            [f"forecast_Q{i}" for i in range(len(forecast_quantiles))])
    assert np.all(eval_df["actual_Q0"] <= eval_df["actual_Q1"])
    assert np.all(eval_df["actual_Q1"] <= eval_df["actual_Q2"])
    assert np.all(eval_df["actual_Q2"] <= eval_df["actual_Q3"])
    assert np.all(eval_df["forecast_Q0"] <= eval_df["forecast_Q1"])

    # `unpack_list` with `list_names_dict`
    list_names_dict = {
        # keys match the keys in `agg_kwargs` above
        "actual_Q": [f"actual_Q{q}" for q in actual_quantiles],
        "forecast_Q": [f"forecast_Q{q}" for q in forecast_quantiles]
    }
    eval_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=None,
        groupby_col="groups",
        agg_kwargs=agg_kwargs,
        extend_col_names=True,
        unpack_list=True,
        list_names_dict=list_names_dict)
    assert list(eval_df.columns) == list_names_dict["actual_Q"] + list_names_dict["forecast_Q"]

    with pytest.raises(ValueError, match="list_names_dict\\['actual_Q'\\] has length 2, but there are 4 columns to name."):
        list_names_dict = {
            "actual_Q": [f"actual_Q{q}" for q in forecast_quantiles],  # reversed, wrong length
            "forecast_Q": [f"forecast_Q{q}" for q in actual_quantiles]
        }
        flexible_grouping_evaluation(
            df=df,
            map_func_dict=None,
            groupby_col="groups",
            agg_kwargs=agg_kwargs,
            extend_col_names=True,
            unpack_list=True,
            list_names_dict=list_names_dict)

    with pytest.warns(
            UserWarning,
            match=r"These names from `list_names_dict` are not used, because the "
                  r"column \(key\) is not found in the dataframe after aggregation:\n"
                  r"\['actual', 'forecast'\].\nAvailable columns are:\n"
                  r"\['actual_Q', 'forecast_Q'\]."):
        list_names_dict = {
            "actual": [f"actual_Q{q}" for q in actual_quantiles],
            "forecast": [f"forecast_Q{q}" for q in forecast_quantiles]
        }
        flexible_grouping_evaluation(
            df=df,
            map_func_dict=None,
            groupby_col="groups",
            agg_kwargs=agg_kwargs,
            extend_col_names=True,
            unpack_list=True,
            list_names_dict=list_names_dict)

    # `agg_kwargs` with string format
    eval_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=None,
        groupby_col="groups",
        agg_kwargs={"func": "mean"})
    assert list(eval_df.columns) == ["actual", "forecast"]
    # only one level, no extension
    eval_df = flexible_grouping_evaluation(
        df=df,
        map_func_dict=None,
        groupby_col="groups",
        agg_kwargs={"func": "mean"},
        extend_col_names=True)
    assert list(eval_df.columns) == ["actual", "forecast"]
