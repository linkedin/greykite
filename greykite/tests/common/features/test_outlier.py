import numpy as np
import pandas as pd
import pytest

from greykite.common.features.outlier import EXPONENTIAL_SMOOTHING
from greykite.common.features.outlier import MOVING_MEDIAN
from greykite.common.features.outlier import BaseOutlierDetector
from greykite.common.features.outlier import DetectionResult
from greykite.common.features.outlier import DiffMethod
from greykite.common.features.outlier import TukeyOutlierDetector
from greykite.common.features.outlier import ZScoreOutlierDetector
from greykite.common.viz.timeseries_annotate import plot_lines_markers


# Boolean to decide if figures are to be shown or not when this test file is run.
# Turn this on when changes are made and include in code reviews.
# Compare before and after the change to confirm everything is as expected.
FIG_SHOW = False


@pytest.fixture(scope="module")
def data():
    """Generates data for testing."""
    sampler = np.random.default_rng(1317)
    # Defines two clean vectors, one for `fit` and one for `detect`.
    y_clean = np.arange(0, 100)
    # Add small noise
    y_clean = y_clean + sampler.normal(loc=0.0, scale=3, size=len(y_clean))
    y_clean_test = np.arange(30, 40)
    y_clean_test = y_clean_test + sampler.normal(loc=0.0, scale=3, size=len(y_clean_test))

    # Constructs two scenarios:
    # Easy to detect (`y_easy_outlier`): there is an outlier in position two which is 10X the max
    # Hard to detect (`y_hard_outlier`): there is an outlier in position 3 which is equal to max,
    # however locally it is much larger than neighboring points.
    y_easy_outlier = y_clean.copy()
    y_hard_outlier = y_clean.copy()
    y_easy_outlier_test = y_clean_test.copy()
    y_hard_outlier_test = y_clean_test.copy()

    # Add outlier to `fit` data.
    y_easy_outlier[2] = 1000
    y_hard_outlier[2] = 100

    # Add outlier to test / `detect` data.
    y_easy_outlier_test[5] = 2000
    y_hard_outlier_test[5] = 150

    return {
        "y_clean": y_clean,
        "y_clean_test": y_clean_test,
        "y_easy_outlier": y_easy_outlier,
        "y_easy_outlier_test": y_easy_outlier_test,
        "y_hard_outlier": y_hard_outlier,
        "y_hard_outlier_test": y_hard_outlier_test}


def helper_plot_outliers(
        y,
        detection_result,
        title):
    """This is just a helper function to generate plots for outlier detection during tests.
    This plots the raw input and marks the discovered anomalies.

    Parameters
    ----------
    y: `pandas.Series`
        Input data.
    detection_result: `~greykite.common.features.outlier.DetectionResult`
        Outlier detection results.
    title: `str`
        Title of plot.
    Returns
    -------
    "fig" : `plotly.graph_objects.Figure`

    """
    df = pd.DataFrame({
        "ind": range(len(y)),
        "y": y,
        "scores": detection_result.scores,
        "is_outlier": detection_result.is_outlier})

    df["y_normal"] = None
    df["y_outlier"] = None

    df.loc[~df["is_outlier"], "y_normal"] = df.loc[~df["is_outlier"], "y"]
    df.loc[df["is_outlier"], "y_outlier"] = df.loc[df["is_outlier"], "y"]

    fig = plot_lines_markers(
        df=df,
        x_col="ind",
        line_cols=["y", "scores"],
        marker_cols=["y_outlier"],
        title=title)

    return fig


def test_detection_result():
    """Tests the dataclass `DetectionResult`."""
    detection_result = DetectionResult()

    assert detection_result.scores is None
    assert detection_result.is_outlier is None


def test_diff_methods_init():
    """Tests the dataclass DiffMethod."""
    baseline_method = DiffMethod()

    assert baseline_method.name is None
    assert baseline_method.param is not None


def test_base_outlier_detector_init():
    """Tests the basics of `BaseOutlierDetector` class."""
    # Tests default `__init__`.
    detect_outlier = BaseOutlierDetector()

    assert detect_outlier.trim_percent == 5.0
    assert detect_outlier.diff_method is not None
    assert detect_outlier.lower_bound is None
    assert detect_outlier.upper_bound is None
    assert detect_outlier.fitted_param == {}
    assert detect_outlier.y is None
    assert detect_outlier.y_diffed is None
    assert detect_outlier.y_na_removed is None
    assert detect_outlier.y_trimmed is None
    assert detect_outlier.y_ready_to_fit is None
    assert detect_outlier.fitted == DetectionResult(scores=None, is_outlier=None)
    assert detect_outlier.y_new is None
    assert detect_outlier.y_new_ready_to_predict is None
    assert detect_outlier.predicted == DetectionResult(scores=None, is_outlier=None)

    # Tests `__init__` with parameters.
    detect_outlier = BaseOutlierDetector(
        trim_percent=1,
        diff_method=DiffMethod(name="es"))

    assert detect_outlier.trim_percent == 1.0
    assert detect_outlier.diff_method.name == "es"
    assert detect_outlier.lower_bound is None
    assert detect_outlier.upper_bound is None
    assert detect_outlier.fitted_param == {}
    assert detect_outlier.y is None
    assert detect_outlier.y_diffed is None
    assert detect_outlier.y_na_removed is None
    assert detect_outlier.y_trimmed is None
    assert detect_outlier.y_ready_to_fit is None
    assert detect_outlier.fitted == DetectionResult(scores=None, is_outlier=None)
    assert detect_outlier.y_new is None
    assert detect_outlier.y_new_ready_to_predict is None
    assert detect_outlier.predicted == DetectionResult(scores=None, is_outlier=None)


def test_base_outlier_detector_trim():
    """Tests `trim` method."""
    detect_outlier = BaseOutlierDetector()
    y = np.arange(100)
    y_trimmed_1pcnt = detect_outlier.trim(y, 1)
    # This is the default (5%)
    y_trimmed_5pcnt = detect_outlier.trim(y)

    # Original range.
    assert max(y) == 99
    assert min(y) == 0

    # 1 percent case.
    assert max(y_trimmed_1pcnt) == 98
    assert min(y_trimmed_1pcnt) == 1

    # 5 percent case.
    assert max(y_trimmed_5pcnt) == 96
    assert min(y_trimmed_5pcnt) == 3

    with pytest.raises(
            ValueError,
            match="Trim percent:"):
        detect_outlier.trim(y, -1)

    with pytest.raises(
            ValueError,
            match="Trim percent:"):
        detect_outlier.trim(y, 150)


def test_base_outlier_detector_remove_na(data):
    """Tests `remove_na` method."""
    detect_outlier = BaseOutlierDetector()
    y = data["y_clean"].copy()
    y = pd.Series(y)
    y[5] = None
    y[70] = None

    y_na_removed = detect_outlier.remove_na(y)

    # Original length and new length.
    assert len(y) == 100
    assert len(y_na_removed) == 98

    y = pd.Series([None, None, None, 5, None])
    with pytest.raises(
            ValueError,
            match="Length of y after removing NAs is less than 2"):
        detect_outlier.remove_na(y)


def test_base_detect_diff_from_baseline_es():
    """Tests `diff_from_baseline` method with exponential smoothing."""
    detect_outlier = BaseOutlierDetector()
    y = np.arange(100)

    # Example with exponential smoothing.
    baseline_result = detect_outlier.diff_from_baseline(
        y=y,
        diff_method=EXPONENTIAL_SMOOTHING)

    residuals = baseline_result["residuals"]
    baseline_y = baseline_result["baseline_y"]

    # Original range.
    assert min(y) == 0
    assert max(y) == 99

    # Residuals range.
    assert min(residuals) == 0
    assert abs(max(residuals) - 1.0) < 0.1

    # Example with exponential smoothing.
    y = pd.Series([0, 0, 0, 10, 10, 10, 10, 0, 0, 0])
    baseline_result = detect_outlier.diff_from_baseline(
        y=y,
        diff_method=EXPONENTIAL_SMOOTHING)
    residuals = baseline_result["residuals"]
    baseline_y = baseline_result["baseline_y"]

    assert (round(baseline_y) == [0, 0, 0, 5, 8, 9, 9, 5, 2, 1]).all()
    assert (round(residuals) == [0, 0, 0, 5, 2, 1, 1, -5, -2, -1]).all()

    # Another example with custom `alpha = 1`.
    # This will imply that `y` is unchanged and residuals are all zero.
    y = pd.Series([0, 0, 0, 10, 10, 10, 10, 0, 0, 0])
    diff_method = DiffMethod(name="es", param={"alpha": 1})
    baseline_result = detect_outlier.diff_from_baseline(
        y=y,
        diff_method=diff_method)
    residuals = baseline_result["residuals"]
    baseline_y = baseline_result["baseline_y"]

    assert (baseline_y == y).all()
    assert (residuals == [0]*10).all()


def test_base_detect_diff_from_baseline_moving_med():
    """Tests `diff_from_baseline` method with exponential smoothing."""
    detect_outlier = BaseOutlierDetector()
    y = np.arange(100)

    # Example with moving median.
    baseline_result = detect_outlier.diff_from_baseline(
        y=y,
        diff_method=MOVING_MEDIAN)

    residuals = baseline_result["residuals"]
    baseline_y = baseline_result["baseline_y"]

    # Original range.
    assert min(y) == 0
    assert max(y) == 99

    # Residuals range.
    assert min(residuals) == -1
    assert abs(max(residuals) - 1.0) < 0.1

    # Example with moving median on a short vector.
    y = pd.Series([0, 0, 0, 10, 10, 10, 10, 0, 0, 0])
    baseline_result = detect_outlier.diff_from_baseline(
        y=y,
        diff_method=MOVING_MEDIAN)
    residuals = baseline_result["residuals"]
    baseline_y = baseline_result["baseline_y"]

    assert (round(baseline_y) == [0, 0, 0, 10, 10, 10, 10, 0, 0, 0]).all()
    assert (round(residuals) == [0]*10).all()

    # Another example with custom `window = 2`.
    # This will imply that `y` is unchanged and residuals are all zero.
    y = pd.Series([0, 0, 0, 10, 10, 10, 10, 0, 0, 0])
    diff_method = DiffMethod(
        name="moving_med",
        param={
            "window": 2,
            "min_periods": 1,
            "center": True})
    baseline_result = detect_outlier.diff_from_baseline(
        y=y,
        diff_method=diff_method)
    residuals = baseline_result["residuals"]
    baseline_y = baseline_result["baseline_y"]

    assert (baseline_y == [0, 0, 0, 5, 10, 10, 10, 5, 0, 0]).all()
    assert (residuals == [0, 0, 0, 5, 0, 0, 0, -5, 0, 0]).all()


def test_z_score_outlier_detector_init():
    """Tests `ZScoreOutlierDetector` init."""
    # Tests default `__init__`.
    detect_outlier = ZScoreOutlierDetector()
    assert detect_outlier.trim_percent == 5.0
    assert detect_outlier.diff_method is not None
    assert detect_outlier.lower_bound is None
    assert detect_outlier.upper_bound is None
    assert detect_outlier.fitted_param == {}
    assert detect_outlier.y is None
    assert detect_outlier.y_diffed is None
    assert detect_outlier.y_na_removed is None
    assert detect_outlier.y_trimmed is None
    assert detect_outlier.y_ready_to_fit is None
    assert detect_outlier.fitted == DetectionResult(scores=None, is_outlier=None)
    assert detect_outlier.y_new is None
    assert detect_outlier.y_new_ready_to_predict is None
    assert detect_outlier.predicted == DetectionResult(scores=None, is_outlier=None)
    # Specific to this class.
    assert detect_outlier.z_score_cutoff == 5.0

    # Tests `__init__` with parameters.
    detect_outlier = ZScoreOutlierDetector(z_score_cutoff=10)
    assert detect_outlier.z_score_cutoff == 10.0


def test_tukey_outlier_detector_init():
    """Tests `TukeyOutlierDetector` init."""
    # Tests default `__init__`.
    detect_outlier = TukeyOutlierDetector()
    # The default for `trim_percent` is different from Z-score.
    assert detect_outlier.trim_percent is None
    assert detect_outlier.diff_method is not None
    assert detect_outlier.lower_bound is None
    assert detect_outlier.upper_bound is None
    assert detect_outlier.fitted_param == {}
    assert detect_outlier.y is None
    assert detect_outlier.y_diffed is None
    assert detect_outlier.y_na_removed is None
    assert detect_outlier.y_trimmed is None
    assert detect_outlier.y_ready_to_fit is None
    assert detect_outlier.fitted == DetectionResult(scores=None, is_outlier=None)
    assert detect_outlier.y_new is None
    assert detect_outlier.y_new_ready_to_predict is None
    assert detect_outlier.predicted == DetectionResult(scores=None, is_outlier=None)
    # Specific to this class.
    assert detect_outlier.iqr_lower == 0.1
    assert detect_outlier.iqr_upper == 0.9
    assert detect_outlier.tukey_cutoff == 1.0

    # Tests `__init__` with parameters.
    detect_outlier = TukeyOutlierDetector(
        iqr_lower=0.05,
        iqr_upper=0.95,
        tukey_cutoff=0.2)

    assert detect_outlier.iqr_lower == 0.05
    assert detect_outlier.iqr_upper == 0.95
    assert detect_outlier.tukey_cutoff == 0.2


def test_z_score_outlier_detector(data):
    """Tests `ZScoreOutlierDetector` usage."""
    # Default setting and easy detection.
    detect_outlier = ZScoreOutlierDetector()

    # Easy detection example.
    y = data["y_easy_outlier"].copy()
    y_test = data["y_easy_outlier_test"].copy()

    detect_outlier.fit(y)
    assert abs(detect_outlier.fitted_param["trimmed_mean"] - 0.12188) < 0.1
    assert abs(detect_outlier.fitted_param["trimmed_sd"] - 2.202187) < 0.1

    detect_outlier.detect(y_test)

    fitted = detect_outlier.fitted
    predicted = detect_outlier.predicted

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="Easy detection Z-score")
    if FIG_SHOW:
        fig.show()
    assert fig is not None

    # We expect only an outlier in second position as per `data` definition.
    assert fitted.is_outlier[2]
    assert sum(fitted.is_outlier) == 1

    # We expect only an outlier in 5th position as per `data` definition.
    assert predicted.is_outlier[5]
    assert sum(predicted.is_outlier) == 1

    # Hard detection example without differencing.
    # Here we expect that the outlier is not removed.
    # This will showcase how without differencing anomaly is missed.
    y = data["y_hard_outlier"].copy()
    y_test = data["y_hard_outlier_test"].copy()

    detect_outlier = ZScoreOutlierDetector(diff_method=None)
    detect_outlier.fit(y)
    detect_outlier.detect(y_test)

    fitted = detect_outlier.fitted
    predicted = detect_outlier.predicted
    # We note that the trimmed mean and sd are quite large since no diffing is done.
    assert abs(detect_outlier.fitted_param["trimmed_mean"] - 50.5917) < 0.1
    assert abs(detect_outlier.fitted_param["trimmed_sd"] - 26.8908) < 0.1

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="Hard detection with Z-score and no baseline diffing.")
    if FIG_SHOW:
        fig.show()
    assert fig is not None

    # We expect no outlier is detected.
    assert not fitted.is_outlier[2]
    assert sum(fitted.is_outlier) == 0

    # We expect no outlier is detected.
    assert not predicted.is_outlier[5]
    assert sum(predicted.is_outlier) == 0

    # Hard detection example with differencing (default behavior).
    # Here we expect that the outlier is removed.
    # This will showcase how how differencing with appropriate baseline is helpful.
    y = data["y_hard_outlier"]
    y_test = data["y_hard_outlier_test"]

    detect_outlier = ZScoreOutlierDetector()
    detect_outlier.fit(y)
    detect_outlier.detect(y_test)

    fitted = detect_outlier.fitted
    predicted = detect_outlier.predicted
    # Due to diffing, the mean and sd are much smaller.
    assert abs(detect_outlier.fitted_param["trimmed_mean"] - 0.12188) < 0.1
    assert abs(detect_outlier.fitted_param["trimmed_sd"] - 2.202187) < 0.1

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="Hard detection with Z-score with diffing.")
    if FIG_SHOW:
        fig.show()
    assert fig is not None

    # We expect only an outlier in second position as per `data` definition.
    assert fitted.is_outlier[2]
    assert sum(fitted.is_outlier) == 1

    # We expect only an outlier in 5th position as per `data` definition.
    assert predicted.is_outlier[5]
    assert sum(predicted.is_outlier) == 1


def test_tukey_outlier_detector(data):
    """Tests `TukeyOutlierDetector` usage."""
    detect_outlier = TukeyOutlierDetector()

    # Easy detection example.
    y = data["y_easy_outlier"].copy()
    y_test = data["y_easy_outlier_test"].copy()

    detect_outlier.fit(y)
    detect_outlier.detect(y_test)
    fitted = detect_outlier.fitted
    predicted = detect_outlier.predicted

    assert abs(detect_outlier.fitted_param["quantile_value_lower"] - (-2.86)) < 0.5
    assert abs(detect_outlier.fitted_param["quantile_value_upper"] - 3.80) < 0.5
    assert abs(detect_outlier.fitted_param["iqr"] - 6.66) < 0.5

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="Easy detection Tukey")
    if FIG_SHOW:
        fig.show()
    assert fig is not None

    # We expect only an outlier in second position as per `data` definition.
    assert fitted.is_outlier[2]
    assert sum(fitted.is_outlier) == 1

    # We expect only an outlier in 5th position as per `data` definition.
    assert predicted.is_outlier[5]
    assert sum(predicted.is_outlier) == 1

    # Hard detection example without differencing.
    # Here we expect that the outlier is not removed.
    # This will showcase how without differencing anomaly is missed.
    y = data["y_hard_outlier"].copy()
    y_test = data["y_hard_outlier_test"].copy()

    detect_outlier = TukeyOutlierDetector(diff_method=None)
    detect_outlier.fit(y)
    detect_outlier.detect(y_test)

    fitted = detect_outlier.fitted
    predicted = detect_outlier.predicted

    assert abs(detect_outlier.fitted_param["quantile_value_lower"] - 11.8) < 0.5
    assert abs(detect_outlier.fitted_param["quantile_value_upper"] - 90.1) < 0.5
    assert abs(detect_outlier.fitted_param["iqr"] - 78.3) < 0.5

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="Hard detection with Tukey and no baseline diffing.")
    if FIG_SHOW:
        fig.show()
    assert fig is not None

    # We expect no outlier is detected.
    assert not fitted.is_outlier[2]
    assert sum(fitted.is_outlier) == 0

    # We expect no outlier is detected.
    assert not predicted.is_outlier[5]
    assert sum(predicted.is_outlier) == 0

    # Hard detection example with differencing (default behavior).
    # Here we expect that the outlier is removed.
    # This will showcase how how differencing with appropriate baseline is helpful.
    y = data["y_hard_outlier"].copy()
    y_test = data["y_hard_outlier_test"].copy()

    detect_outlier = TukeyOutlierDetector()
    detect_outlier.fit(y)
    detect_outlier.detect(y_test)

    fitted = detect_outlier.fitted
    predicted = detect_outlier.predicted

    assert abs(detect_outlier.fitted_param["quantile_value_lower"] - (-2.86)) < 0.5
    assert abs(detect_outlier.fitted_param["quantile_value_upper"] - 3.80) < 0.5
    assert abs(detect_outlier.fitted_param["iqr"] - 6.66) < 0.5

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="Hard detection with Tukey with diffing.")
    if FIG_SHOW:
        fig.show()
    assert fig is not None

    # We expect only an outlier in second position as per `data` definition.
    assert fitted.is_outlier[2]
    assert sum(fitted.is_outlier) == 1

    # We expect only an outlier in 5th position as per `data` definition.
    assert predicted.is_outlier[5]
    assert sum(predicted.is_outlier) == 1


def test_tukey_outlier_detector_corner_cases():
    """Tests `TukeyOutlierDetector` usage with corner cases."""
    detect_outlier = TukeyOutlierDetector()
    # The case where data is perfectly linear.
    y = np.arange(100)

    detect_outlier.fit(y)
    fitted = detect_outlier.fitted

    assert abs(detect_outlier.fitted_param["quantile_value_lower"] - 0) < 0.5
    assert abs(detect_outlier.fitted_param["quantile_value_upper"] - 0) < 0.5
    assert abs(detect_outlier.fitted_param["iqr"] - 0) < 0.5
    assert sum(fitted.is_outlier) == 0

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="Tukey: perfectly linear data")
    if FIG_SHOW:
        fig.show()
    assert fig is not None

    detect_outlier = TukeyOutlierDetector()
    # The case where data is constant.
    y = np.zeros(100)

    detect_outlier.fit(y)
    fitted = detect_outlier.fitted

    assert abs(detect_outlier.fitted_param["quantile_value_lower"] - 0) < 0.5
    assert abs(detect_outlier.fitted_param["quantile_value_upper"] - 0) < 0.5
    assert abs(detect_outlier.fitted_param["iqr"] - 0) < 0.5
    assert sum(fitted.is_outlier) == 0

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="Tukey: constant")
    if FIG_SHOW:
        fig.show()
    assert fig is not None


def test_z_score_outlier_detector_corner_cases():
    """Tests `TukeyOutlierDetector` usage with corner cases."""
    detect_outlier = ZScoreOutlierDetector()
    # The case where data is perfectly linear.
    y = np.arange(100)

    detect_outlier.fit(y)
    fitted = detect_outlier.fitted

    assert abs(detect_outlier.fitted_param["trimmed_mean"] - 0) < 0.5
    assert abs(detect_outlier.fitted_param["trimmed_sd"] - 0) < 0.5
    assert sum(fitted.is_outlier) == 0

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="ZScore: perfectly linear data")
    if FIG_SHOW:
        fig.show()
    assert fig is not None

    detect_outlier = ZScoreOutlierDetector()
    # The case where data is constant.
    y = np.zeros(100)

    detect_outlier.fit(y)
    fitted = detect_outlier.fitted

    assert abs(detect_outlier.fitted_param["trimmed_mean"] - 0) < 0.5
    assert abs(detect_outlier.fitted_param["trimmed_sd"] - 0) < 0.5
    assert sum(fitted.is_outlier) == 0

    fig = helper_plot_outliers(
        y=y,
        detection_result=fitted,
        title="ZScore: constant")
    if FIG_SHOW:
        fig.show()
    assert fig is not None
