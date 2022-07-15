import pandas as pd
import pytest

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
