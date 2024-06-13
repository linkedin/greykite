import numpy as np
import pandas as pd
import pytest

from greykite.common.constants import END_TIME_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.python_utils import assert_equal
from greykite.common.viz.colors_utils import get_distinct_colors
from greykite.common.viz.timeseries_annotate import add_multi_vrects
from greykite.common.viz.timeseries_annotate import plot_anomalies_over_forecast_vs_actual
from greykite.common.viz.timeseries_annotate import plot_event_periods_multi
from greykite.common.viz.timeseries_annotate import plot_lines_markers
from greykite.common.viz.timeseries_annotate import plot_overlay_anomalies_multi_metric
from greykite.common.viz.timeseries_annotate import plot_precision_recall_curve
from greykite.common.viz.timeseries_annotate import plt_annotate_series
from greykite.common.viz.timeseries_annotate import plt_compare_series_annotations


def test_plt_annotate_series():
    """Tests ``plt_annotate_series`` function."""
    df = pd.DataFrame({
        "x": list(range(10)),
        "y": [1, 2, 3, 4, 5, 4, 3, 3, 2, 1],
        "label": ["cat", "cat", "cat", "dog", "dog", "dog", "horse", "cat", "cat", "cat"]
        })
    fig = plt_annotate_series(
        df=df,
        x_col="x",
        value_col="y",
        label_col="label",
        annotate_labels=["cat", "dog"],
        keep_cols=None,
        title=None)["fig"]

    assert fig.layout.showlegend is True
    assert fig.layout.xaxis.title.text == "x"
    assert fig.layout.yaxis.title.text == "y"
    assert len(fig.data) == 3
    assert fig.data[0].mode == "lines"
    assert fig.data[0].name == "y"
    assert fig.data[1].name == "y_label_cat"
    assert fig.data[2].name == "y_label_dog"


def test_plt_compare_series_annotations():
    """Tests ``plt_compare_series_annotations`` function."""
    df = pd.DataFrame({
        "x": list(range(10)),
        "actual": [1, 2, 3, 4, 5, 4, 3, 3, 2, 1],
        "forecast": [1.5, 2.5, 3.5, 4.5, 5.5, 5, 4, 4, 1, 0.5],
        "actual_label": [0]*5 + [1]*5,
        "forecast_label": [0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        "forecast_upper": [2.5, 3.5, 4.5, 5.5, 6.5, 6, 5, 5, 2, 2],
        "forecast_lower": [.5, .5, 1.5, 3.5, 4.5, 3, 3, 2, -0.5, -1]
        })

    fig = plt_compare_series_annotations(
        df=df,
        x_col="x",
        actual_col="actual",
        actual_label_col="actual_label",
        forecast_label_col="forecast_label",
        keep_cols=["forecast_upper", "forecast_lower"],
        forecast_col="forecast",
        standardize_col=None)

    assert fig.layout.showlegend is True
    assert fig.layout.xaxis.title.text == "x"
    assert fig.layout.yaxis.title.text is None
    assert len(fig.data) == 6
    assert fig.data[0].mode == "lines"
    assert fig.data[0].name == "actual"
    assert fig.data[1].name == "true_anomaly"
    assert fig.data[2].name == "model_anomaly"
    assert fig.data[3].name == "forecast"

    # test with standardize
    fig = plt_compare_series_annotations(
        df=df,
        x_col="x",
        actual_col="actual",
        actual_label_col="actual_label",
        forecast_label_col="forecast_label",
        forecast_col="forecast",
        keep_cols=["forecast_upper", "forecast_lower"],
        standardize_col="actual")

    assert fig.layout.showlegend is True
    assert fig.layout.xaxis.title.text == "x"
    assert fig.layout.yaxis.title.text is None
    assert len(fig.data) == 6
    assert fig.data[0].mode == "lines"
    assert fig.data[0].name == "actual"
    assert fig.data[1].name == "true_anomaly"
    assert fig.data[2].name == "model_anomaly"
    assert fig.data[3].name == "forecast"

    # test with no ``forecast_col``
    fig = plt_compare_series_annotations(
        df=df,
        x_col="x",
        actual_col="actual",
        actual_label_col="actual_label",
        forecast_label_col="forecast_label",
        forecast_col=None,
        keep_cols=["forecast_upper", "forecast_lower"],
        standardize_col="actual")

    assert fig.layout.showlegend is True
    assert fig.layout.xaxis.title.text == "x"
    assert fig.layout.yaxis.title.text is None
    assert len(fig.data) == 5
    assert fig.data[0].mode == "lines"
    assert fig.data[0].name == "actual"
    assert fig.data[1].name == "true_anomaly"
    assert fig.data[2].name == "model_anomaly"


def test_fill():
    # tests plot_annotate_series
    df = pd.DataFrame({
        "x": list(range(10)),
        "actual": [1, 2, 3, 4, 5, 4, 3, 3, 2, 1],
        "forecast": [1.5, 2.5, 3.5, 4.5, 5.5, 5, 4, 4, 1, 0.5],
        "actual_label": [0] * 5 + [1] * 5,
        "forecast_label": [0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        "forecast_upper": [2.5, 3.5, 4.5, 5.5, 6.5, 6, 5, 5, 2, 2],
        "forecast_lower": [.5, .5, 1.5, 3.5, 4.5, 3, 3, 2, -0.5, -1]
    })
    fig = plt_annotate_series(
        df=df,
        x_col="x",
        value_col="actual",
        label_col="actual_label",
        annotate_labels=[0, 1],
        keep_cols=None,
        fill_cols=[["forecast_lower", "forecast_upper"]],
        title=None)["fig"]

    assert fig.data[0]["name"] == "forecast_lower"
    assert fig.data[1]["name"] == "forecast_upper"
    assert fig.data[1]["fill"] == "tonexty"

    # tests plot_compare_series_annotations
    fig = plt_compare_series_annotations(
        df=df,
        x_col="x",
        actual_col="actual",
        actual_label_col="actual_label",
        forecast_label_col="forecast_label",
        forecast_col=None,
        fill_cols=[["forecast_lower", "forecast_upper"]],
        standardize_col="actual")
    assert fig.data[0]["name"] == "forecast_lower"
    assert fig.data[1]["name"] == "forecast_upper"
    assert fig.data[1]["fill"] == "tonexty"

    with pytest.raises(
            ValueError,
            match="fill_cols must be a list of lists of strings."):
        plt_annotate_series(
            df=df,
            x_col="x",
            value_col="actual",
            label_col="actual_label",
            annotate_labels=[0, 1],
            keep_cols=None,
            fill_cols=["forecast_lower", "forecast_upper"],
            title=None)

    with pytest.raises(
            ValueError,
            match="fill_cols must be a list of lists of strings."):
        plt_compare_series_annotations(
            df=df,
            x_col="x",
            actual_col="actual",
            actual_label_col="actual_label",
            forecast_label_col="forecast_label",
            forecast_col=None,
            fill_cols=["forecast_lower", "forecast_upper"],
            standardize_col="actual")


def test_plot_lines_markers():
    """Tests ``plot_lines_markers``."""
    df = pd.DataFrame({
        "ts": [1, 2, 3, 4, 5, 6],
        "line1": [3, 4, 5, 6, 7, 7],
        "line2": [4, 5, 6, 7, 8, 8],
        "marker1": [np.nan, np.nan, 5, 6, np.nan, np.nan],
        "marker2": [np.nan, 5, 6, np.nan, np.nan, 8],
        })

    fig = plot_lines_markers(
        df=df,
        x_col="ts",
        line_cols=["line1", "line2"],
        marker_cols=["marker1", "marker2"],
        line_colors=None,
        marker_colors=None)

    assert len(fig.data) == 4
    assert fig.data[0].line.color is None
    assert fig.data[1].line.color is None
    assert fig.data[2].marker.color is None
    assert fig.data[3].marker.color is None

    # Next we make the marker and line colors consistent
    marker_colors = get_distinct_colors(
        num_colors=2,
        opacity=1.0)

    line_colors = get_distinct_colors(
        num_colors=2,
        opacity=0.5)

    fig = plot_lines_markers(
        df=df,
        x_col="ts",
        line_cols=["line1", "line2"],
        marker_cols=["marker1", "marker2"],
        line_colors=line_colors,
        marker_colors=marker_colors)

    assert len(fig.data) == 4
    assert fig.data[0].line.color == "rgba(31, 119, 180, 0.5)"
    assert fig.data[1].line.color == "rgba(255, 127, 14, 0.5)"
    assert fig.data[2].marker.color == "rgba(31, 119, 180, 1.0)"
    assert fig.data[3].marker.color == "rgba(255, 127, 14, 1.0)"

    # Length of `line_colors` must be larger than or equal to length of `line_cols` if passed.
    with pytest.raises(
            ValueError,
            match="If `line_colors` is passed"):
        plot_lines_markers(
            df=df,
            x_col="ts",
            line_cols=["line1", "line2"],
            marker_cols=["marker1", "marker2"],
            line_colors=line_colors[:1],
            marker_colors=marker_colors)

    # At least one of `line_cols` or `marker_cols` or `band_cols`
    # must be provided (not None).
    with pytest.raises(
            ValueError,
            match="At least one of"):
        plot_lines_markers(
            df=df,
            x_col="ts",
            line_cols=None,
            marker_cols=None,
            band_cols=None,
            line_colors=None,
            marker_colors=None)


def test_plot_lines_markers_with_bands():
    """Tests ``plot_lines_markers`` with bands."""
    df = pd.DataFrame({
        "x": range(4),
        "y": range(4),
        "z1": range(1, 5),
        "z2": range(-1, 3),
        "w": [(0, 1), (1, 3), (1, 5), (3, 5)],
        "u": [(2, 3), (3, 3), (4, 4), (6, 8)]})

    fig = plot_lines_markers(
        df=df,
        x_col="x",
        line_cols=["y", "z1"],
        band_cols=["u", "w"])

    assert len(fig.data) == 6
    assert fig.data[0].line.color is None
    assert fig.data[1].line.color is None
    assert fig.data[2].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[3].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[4].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[5].line.color == "rgba(0, 0, 0, 0)"

    assert fig.data[3].name == "u"
    assert fig.data[5].name == "w"
    assert fig.data[3].fillcolor == "rgba(31, 119, 180, 0.2)"
    assert fig.data[5].fillcolor == "rgba(255, 127, 14, 0.2)"
    assert fig.layout.title.text is None

    # Bands with custom colors and a title for the plot.
    fig = plot_lines_markers(
        df=df,
        x_col="x",
        line_cols=["y", "z1"],
        band_cols=["u", "w"],
        band_colors=["rgba(0, 255, 0, 0.2)", "rgba(255, 0, 0, 0.2)"],
        title="custom band colors")

    assert len(fig.data) == 6
    assert fig.data[0].line.color is None
    assert fig.data[1].line.color is None
    assert fig.data[2].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[3].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[4].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[5].line.color == "rgba(0, 0, 0, 0)"

    assert fig.data[3].name == "u"
    assert fig.data[5].name == "w"

    assert fig.data[3].fillcolor == "rgba(0, 255, 0, 0.2)"
    assert fig.data[5].fillcolor == "rgba(255, 0, 0, 0.2)"
    assert fig.layout.title.text == "custom band colors"

    # Bands specified by dictionary.
    df = pd.DataFrame({
        "x": range(4),
        "y": [2, 3, 4, 5],
        "z1": [4, 5, 6, 8],
        "z2": range(-1, 3),
        "w1": [5, 6, 6, 8],
        "w2": [7, 8, 9, 9],
        "u1": [2, 3, 5, 7],
        "u3": [4, 5, 8, 8]})

    fig = plot_lines_markers(
        df=df,
        x_col="x",
        line_cols=["y", "z1"],
        band_cols_dict={"u": ["u1", "u3"], "w": ["w1", "w2"]},
        band_colors=["rgba(0, 255, 0, 0.2)", "rgba(255, 0, 0, 0.2)"],
        title="bands via dict")

    assert len(fig.data) == 6
    assert fig.data[0].line.color is None
    assert fig.data[1].line.color is None
    assert fig.data[2].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[3].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[4].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[5].line.color == "rgba(0, 0, 0, 0)"

    assert fig.data[3].name == "u"
    assert fig.data[5].name == "w"

    assert fig.data[3].fillcolor == "rgba(0, 255, 0, 0.2)"
    assert fig.data[5].fillcolor == "rgba(255, 0, 0, 0.2)"
    assert fig.layout.title.text == "bands via dict"


def test_plot_event_periods_multi():
    """Tests ``plot_event_periods_multi`` function."""
    df = pd.DataFrame({
        "start_time": ["2020-01-01", "2020-02-01", "2020-01-02", "2020-02-02", "2020-02-05"],
        "end_time": ["2020-01-03", "2020-02-04", "2020-01-05", "2020-02-06", "2020-02-08"],
        "metric": ["impressions", "impressions", "clicks", "clicks", "bookings"]
        })

    # ``grouping_col`` is not passed.
    res = plot_event_periods_multi(
        periods_df=df,
        start_time_col="start_time",
        end_time_col="end_time",
        freq=None,
        grouping_col=None,
        min_timestamp=None,
        max_timestamp=None)

    fig = res["fig"]
    labels_df = res["labels_df"]
    ts = res["ts"]
    min_timestamp = res["min_timestamp"]
    max_timestamp = res["max_timestamp"]
    new_cols = res["new_cols"]

    assert labels_df.shape == (913, 2)
    assert list(labels_df.columns) == ["ts", "metric_is_anomaly"]
    assert len(fig.data) == 1
    assert len(fig.data[0]["x"]) == 913
    assert len(ts) == 913
    assert min_timestamp == "2020-01-01"
    assert max_timestamp == "2020-02-08"
    assert new_cols == ["metric_is_anomaly"]

    # ``grouping_col`` is passed.
    res = plot_event_periods_multi(
        periods_df=df,
        start_time_col="start_time",
        end_time_col="end_time",
        freq=None,
        grouping_col="metric",
        min_timestamp=None,
        max_timestamp=None)

    fig = res["fig"]
    labels_df = res["labels_df"]
    ts = res["ts"]
    min_timestamp = res["min_timestamp"]
    max_timestamp = res["max_timestamp"]
    new_cols = res["new_cols"]

    assert labels_df.shape == (913, 4)
    assert set(labels_df.columns) == {
        "ts", "bookings_is_anomaly", "impressions_is_anomaly", "clicks_is_anomaly"}
    assert len(fig.data) == 3
    assert len(fig.data[0]["x"]) == 913
    assert len(ts) == 913
    assert min_timestamp == "2020-01-01"
    assert max_timestamp == "2020-02-08"
    assert set(new_cols) == {"bookings_is_anomaly", "impressions_is_anomaly", "clicks_is_anomaly"}

    # Specifies ``freq``
    res = plot_event_periods_multi(
        periods_df=df,
        start_time_col="start_time",
        end_time_col="end_time",
        freq="min",
        grouping_col="metric",
        min_timestamp=None,
        max_timestamp=None)

    fig = res["fig"]
    labels_df = res["labels_df"]
    ts = res["ts"]
    min_timestamp = res["min_timestamp"]
    max_timestamp = res["max_timestamp"]
    new_cols = res["new_cols"]

    assert labels_df.shape == (54721, 4)
    assert set(labels_df.columns) == {
        "ts", "bookings_is_anomaly", "impressions_is_anomaly", "clicks_is_anomaly"}
    assert len(fig.data) == 3
    assert len(fig.data[0]["x"]) == 54721
    assert len(ts) == 54721
    assert min_timestamp == "2020-01-01"
    assert max_timestamp == "2020-02-08"
    assert set(new_cols) == {"bookings_is_anomaly", "impressions_is_anomaly", "clicks_is_anomaly"}

    # ``min_timestamp``, ``max_timestamp`` specified.
    res = plot_event_periods_multi(
        periods_df=df,
        start_time_col="start_time",
        end_time_col="end_time",
        freq=None,
        grouping_col="metric",
        min_timestamp="2019-12-15",
        max_timestamp="2020-02-15")

    fig = res["fig"]
    labels_df = res["labels_df"]
    ts = res["ts"]
    min_timestamp = res["min_timestamp"]
    max_timestamp = res["max_timestamp"]
    new_cols = res["new_cols"]

    assert labels_df.shape == (1489, 4)
    assert set(labels_df.columns) == {
        "ts", "bookings_is_anomaly", "impressions_is_anomaly", "clicks_is_anomaly"}
    assert len(fig.data) == 3
    assert len(fig.data[0]["x"]) == 1489
    assert len(ts) == 1489
    assert min_timestamp == "2019-12-15"
    assert max_timestamp == "2020-02-15"
    assert set(new_cols) == {"bookings_is_anomaly", "impressions_is_anomaly", "clicks_is_anomaly"}

    # Tests for raising ``ValueError`` due to start time being larger than end time.
    df = pd.DataFrame({
        "start_time": ["2020-01-03"],
        "end_time": ["2020-01-02"],
        "metric": ["impressions"]
        })

    expected_match = "End Time:"
    with pytest.raises(ValueError, match=expected_match):
        plot_event_periods_multi(
            periods_df=df,
            start_time_col="start_time",
            end_time_col="end_time",
            freq=None,
            grouping_col=None,
            min_timestamp=None,
            max_timestamp=None)


def test_add_multi_vrects():
    """Tests ``add_multi_vrects``."""
    periods_df = pd.DataFrame({
        "start_time": ["2019-12-28", "2020-01-25", "2020-01-02", "2020-02-02", "2020-02-05"],
        "end_time": ["2020-01-03", "2020-02-04", "2020-01-05", "2020-02-06", "2020-02-08"],
        "metric": ["impressions", "impressions", "clicks", "clicks", "bookings"],
        "reason": ["C1", "C2", "GCN1", "GCN2", "Lockdown"]
        })

    ts = pd.date_range(start="2019-12-15", end="2020-02-15", freq="D")
    df = pd.DataFrame({
        "ts": ts,
        "y1": range(len(ts))})

    df["y2"] = (df["y1"] - 10)**(1.5)

    fig = plot_lines_markers(
        df=df,
        x_col="ts",
        line_cols=["y1", "y2"],
        marker_cols=None,
        line_colors=None,
        marker_colors=None)

    # With multiple groups
    res = add_multi_vrects(
        periods_df=periods_df,
        fig=fig,
        start_time_col="start_time",
        end_time_col="end_time",
        grouping_col="metric",
        y_min=-15,
        y_max=df["y1"].max(),
        annotation_text_col="reason",
        grouping_color_dict=None)

    fig = res["fig"]
    grouping_color_dict = res["grouping_color_dict"]
    assert grouping_color_dict == {
        "bookings": "rgba(31, 119, 180, 1.0)",
        "clicks": "rgba(255, 127, 14, 1.0)",
        "impressions": "rgba(44, 160, 44, 1.0)"}
    assert len(fig.data) == 2
    assert len(fig.layout.shapes) == 5

    # No groups
    fig = plot_lines_markers(
        df=df,
        x_col="ts",
        line_cols=["y1", "y2"],
        marker_cols=None,
        line_colors=None,
        marker_colors=None)

    res = add_multi_vrects(
        periods_df=periods_df,
        fig=fig,
        start_time_col="start_time",
        end_time_col="end_time",
        grouping_col=None,
        y_min=-15,
        y_max=df["y1"].max(),
        annotation_text_col="reason",
        opacity=0.4,
        grouping_color_dict=None)

    fig = res["fig"]
    grouping_color_dict = res["grouping_color_dict"]
    assert grouping_color_dict == {"metric": "rgba(31, 119, 180, 1.0)"}
    assert len(fig.data) == 2
    assert len(fig.layout.shapes) == 5

    # No groups, no text annotations
    fig = plot_lines_markers(
        df=df,
        x_col="ts",
        line_cols=["y1", "y2"],
        marker_cols=None,
        line_colors=None,
        marker_colors=None)

    res = add_multi_vrects(
        periods_df=periods_df,
        fig=fig,
        start_time_col="start_time",
        end_time_col="end_time",
        grouping_col=None,
        y_min=-15,
        y_max=df["y1"].max(),
        annotation_text_col=None,
        opacity=0.4,
        grouping_color_dict=None)

    fig = res["fig"]
    grouping_color_dict = res["grouping_color_dict"]
    assert grouping_color_dict == {"metric": "rgba(31, 119, 180, 1.0)"}
    assert len(fig.data) == 2
    assert len(fig.layout.shapes) == 5

    # Tests for raising ``ValueError`` due to non-existing column name
    expected_match = "start_time_col"
    with pytest.raises(ValueError, match=expected_match):
        add_multi_vrects(
            periods_df=periods_df,
            fig=fig,
            start_time_col="start_timestamp",  # This column does not exist
            end_time_col="end_time",
            grouping_col=None,
            y_min=-15,
            y_max=df["y1"].max(),
            annotation_text_col=None,
            opacity=0.4,
            grouping_color_dict=None)


def test_plot_overlay_anomalies_multi_metric():
    """Tests ``plot_overlay_anomalies_multi_metric``."""
    anomaly_df = pd.DataFrame({
        "start_time": ["2020-01-01", "2020-02-01", "2020-01-02", "2020-02-02", "2020-02-05"],
        "end_time": ["2020-01-03", "2020-02-04", "2020-01-05", "2020-02-06", "2020-02-08"],
        "metric": ["impressions", "impressions", "clicks", "clicks", "bookings"],
        "reason": ["C1", "C2", "GCN1", "GCN2", "Lockdown"]
        })

    ts = pd.date_range(start="2019-12-01", end="2020-03-01", freq="D")

    np.random.seed(1317)
    df = pd.DataFrame({"ts": ts})
    size = len(df)
    value_cols = ["impressions", "clicks", "bookings"]
    df["impressions"] = np.random.normal(loc=0.0, scale=1.0, size=size)
    df["clicks"] = np.random.normal(loc=1.0, scale=1.0, size=size)
    df["bookings"] = np.random.normal(loc=2.0, scale=1.0, size=size)

    # Without annotation texts
    res = plot_overlay_anomalies_multi_metric(
        df=df,
        time_col="ts",
        value_cols=value_cols,
        anomaly_df=anomaly_df,
        anomaly_df_grouping_col="metric",
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL)

    fig = res["fig"]
    augmented_df = res["augmented_df"]
    is_anomaly_cols = res["is_anomaly_cols"]
    line_colors = res["line_colors"]

    assert len(fig.data) == 6
    assert augmented_df.shape == (92, 13)

    assert set(is_anomaly_cols) == {
        "impressions_is_anomaly",
        "clicks_is_anomaly",
        "bookings_is_anomaly"}

    assert line_colors == [
        "rgba(31, 119, 180, 0.6)",
        "rgba(255, 127, 14, 0.6)",
        "rgba(44, 160, 44, 0.6)"]

    # With annotation texts
    ts = pd.date_range(start="2019-12-25", end="2020-02-15", freq="D")

    np.random.seed(1317)
    df = pd.DataFrame({"ts": ts})
    size = len(df)
    value_cols = ["impressions", "clicks", "bookings"]
    df["impressions"] = np.random.normal(loc=0.0, scale=1.0, size=size)
    df["clicks"] = np.random.normal(loc=1.0, scale=1.0, size=size)
    df["bookings"] = np.random.normal(loc=2.0, scale=1.0, size=size)

    res = plot_overlay_anomalies_multi_metric(
        df=df,
        time_col="ts",
        value_cols=value_cols,
        anomaly_df=anomaly_df,
        anomaly_df_grouping_col="metric",
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        annotation_text_col="reason")

    fig = res["fig"]
    augmented_df = res["augmented_df"]
    is_anomaly_cols = res["is_anomaly_cols"]
    line_colors = res["line_colors"]

    assert len(fig.data) == 6
    assert augmented_df.shape == (53, 13)

    assert set(is_anomaly_cols) == {
        "impressions_is_anomaly",
        "clicks_is_anomaly",
        "bookings_is_anomaly"}

    assert line_colors == [
        "rgba(31, 119, 180, 0.6)",
        "rgba(255, 127, 14, 0.6)",
        "rgba(44, 160, 44, 0.6)"]


def test_plot_precision_recall_curve():
    """Tests ``plot_precision_recall_curve``."""
    # Creates fake data.
    precision = np.linspace(0, 1, num=25)
    recall = np.linspace(0, 0.5, num=25)
    pr_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "key": 1})
    groups_df = pd.DataFrame({"groups": ["A", "B", "C"], "key": 1})
    df = pr_df.merge(groups_df, how="inner", on="key").reset_index(drop=True).drop(labels="key", axis=1)
    # Generates the precision recall curve.
    fig = plot_precision_recall_curve(
        df=df,
        grouping_col="groups",
        precision_col="precision",
        recall_col="recall")
    # Asserts one curve per group.
    assert len(fig.data) == groups_df.shape[0]
    # Checks that the data is correct.
    for index, row in groups_df.iterrows():
        assert_equal(np.array(fig.data[index]["x"]), recall)
        assert_equal(np.array(fig.data[index]["y"]), precision)
        assert_equal(np.array(fig.data[index]["name"]), row["groups"])

    # Generates the precision recall curve when `grouping_col` is None.
    fig = plot_precision_recall_curve(
        df=pr_df,
        grouping_col=None,
        precision_col="precision",
        recall_col="recall")
    # Asserts only one curve.
    assert len(fig.data) == 1
    # Checks that the data is correct.
    assert_equal(np.array(fig.data[0]["x"]), recall)
    assert_equal(np.array(fig.data[0]["y"]), precision)

    # Tests expected errors.
    expected_match = "must contain"
    with pytest.raises(ValueError, match=expected_match):
        plot_precision_recall_curve(
                df=df,
                grouping_col=None,
                precision_col="wrong_column",
                recall_col="recall")

    expected_match = "must contain"
    with pytest.raises(ValueError, match=expected_match):
        plot_precision_recall_curve(
                df=df,
                grouping_col=None,
                precision_col="precision",
                recall_col="wrong_column")

    expected_match = "is not found"
    with pytest.raises(ValueError, match=expected_match):
        plot_precision_recall_curve(
                df=df,
                grouping_col="wrong_column",
                precision_col="precision",
                recall_col="recall")


def test_plot_anomalies_over_forecast_vs_actual():
    """Tests ``plot_anomalies_over_forecast_vs_actual`` function."""
    size = 200
    num_anomalies = 10
    num_predicted_anomalies = 15
    df = pd.DataFrame({
        "ts": pd.date_range(start="2018-01-01", periods=size, freq="H"),
        "actual": np.random.normal(scale=10, size=size)
    })
    df["forecast"] = df["actual"] + np.random.normal(size=size)
    df["forecast_lower"] = df["forecast"] - 10
    df["forecast_upper"] = df["forecast"] + 10
    df["is_anomaly"] = False
    df["is_anomaly_predicted"] = False
    df.loc[df.sample(num_anomalies).index, "is_anomaly"] = True
    df.loc[df.sample(num_predicted_anomalies).index, "is_anomaly_predicted"] = True

    # Tests plots when both `predicted_anomaly_col` and `anomaly_col` are not None.
    fig = plot_anomalies_over_forecast_vs_actual(
        df=df,
        time_col="ts",
        actual_col="actual",
        predicted_col="forecast",
        predicted_lower_col="forecast_lower",
        predicted_upper_col="forecast_upper",
        predicted_anomaly_col="is_anomaly_predicted",
        anomaly_col="is_anomaly",
        predicted_anomaly_marker_color="lightblue",
        anomaly_marker_color="orange",
        marker_opacity=0.4)
    assert len(fig.data) == 6
    # Checks the predicted anomaly data is correct.
    fig_predicted_anomaly_data = [data for data in fig.data if data["name"] == "is_anomaly_predicted".title()][0]
    assert len(fig_predicted_anomaly_data["x"]) == num_predicted_anomalies
    assert fig_predicted_anomaly_data["marker"]["color"] == "lightblue"
    assert fig_predicted_anomaly_data["opacity"] == 0.4
    # Checks the anomaly data is correct.
    fig_anomaly_data = [data for data in fig.data if data["name"] == "is_anomaly".title()][0]
    assert len(fig_anomaly_data["x"]) == num_anomalies
    assert fig_anomaly_data["marker"]["color"] == "orange"
    assert fig_anomaly_data["opacity"] == 0.4

    # Tests plots when `predicted_anomaly_col` is None.
    fig = plot_anomalies_over_forecast_vs_actual(
        df=df,
        time_col="ts",
        actual_col="actual",
        predicted_col="forecast",
        predicted_lower_col="forecast_lower",
        predicted_upper_col="forecast_upper",
        predicted_anomaly_col=None,
        anomaly_col="is_anomaly")
    assert len(fig.data) == 5

    # Tests plots when `anomaly_col` is None.
    fig = plot_anomalies_over_forecast_vs_actual(
        df=df,
        time_col="ts",
        actual_col="actual",
        predicted_col="forecast",
        predicted_lower_col="forecast_lower",
        predicted_upper_col="forecast_upper",
        predicted_anomaly_col="is_anomaly_predicted",
        anomaly_col=None)
    assert len(fig.data) == 5

    # Tests plots when both `predicted_anomaly_col` and `anomaly_col` are None.
    fig = plot_anomalies_over_forecast_vs_actual(
        df=df,
        time_col="ts",
        actual_col="actual",
        predicted_col="forecast",
        predicted_lower_col="forecast_lower",
        predicted_upper_col="forecast_upper",
        predicted_anomaly_col=None,
        anomaly_col=None)
    assert len(fig.data) == 4
