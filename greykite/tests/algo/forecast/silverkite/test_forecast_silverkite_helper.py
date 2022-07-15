import datetime

import pandas as pd
import pytest

from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_default_changepoints_dict
from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_fourier_feature_col_names
from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_silverkite_uncertainty_dict
from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_similar_lag
from greykite.common.features.timeseries_features import fourier_series_multi_fcn
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests


@pytest.fixture
def hourly_data():
    """Generate 500 days of hourly data for tests"""
    return generate_df_for_tests(
        freq="H",
        periods=24 * 500,
        train_start_date=datetime.datetime(2018, 7, 1),
        conti_year_origin=2018)


def test_get_silverkite_uncertainty_dict():
    """Testing silverkite_uncertainty_dict."""
    # coverage provided, uncertainty None
    obtained_dict = get_silverkite_uncertainty_dict(
        uncertainty=None,
        coverage=0.95)
    expected_uncertainty = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    assert_equal(obtained_dict, expected_uncertainty)

    # coverage provided, uncertainty "auto" (same as above)
    obtained_dict = get_silverkite_uncertainty_dict(
        uncertainty="auto",
        coverage=0.95)
    assert_equal(obtained_dict, expected_uncertainty)

    # coverage provided, uncertainty missing quantiles
    coverage = 0.80
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals"
    }
    obtained_dict = get_silverkite_uncertainty_dict(
        uncertainty=uncertainty_dict,
        coverage=coverage)
    expected_dict = uncertainty_dict.copy()
    expected_dict["params"]["quantiles"] = [0.1, 0.9]
    assert_equal(obtained_dict, expected_dict)

    # coverage provided, consistent quantiles
    coverage = 0.80
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.1, 0.9],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 20,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    obtained_dict = get_silverkite_uncertainty_dict(
        uncertainty=uncertainty_dict,
        coverage=coverage)
    assert_equal(obtained_dict, uncertainty_dict)

    # coverage missing, uncertainty None
    assert get_silverkite_uncertainty_dict(
        uncertainty=None,
        coverage=None) is None

    # coverage missing, uncertainty "auto"
    obtained_dict = get_silverkite_uncertainty_dict(uncertainty="auto")
    expected_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}
    assert_equal(obtained_dict, expected_dict)

    # coverage missing, uncertainty has quantiles
    obtained_dict = get_silverkite_uncertainty_dict(uncertainty=expected_dict)
    assert_equal(obtained_dict, expected_dict)


def test_get_silverkite_uncertainty_dict_exception():
    """Testing silverkite_uncertainty_dict.
        Exception must be raised if `quantiles` and `coverage` are not consistent."""

    # note  param `quantiles = [0.02, 0.98]` will translate to coverage = 0.96
    # as opposed to the coverage specified below (0.95)
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.02, 0.98],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    expected_match_str = (
        "Coverage is specified/inferred both via `coverage`"
        " and via `uncertainty` input and values do not match."
        " Coverage specified via `coverage`: 0.95."
        " Coverage inferred via `uncertainty`: 0.96.")

    with pytest.raises(
            ValueError,
            match=expected_match_str):
        get_silverkite_uncertainty_dict(
            uncertainty=uncertainty_dict,
            coverage=0.95)

    # exception for quantiles not being an increasing sequence
    quantiles = [0.975, 0.025]
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": quantiles,
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    expected_match_str = (
        "`quantiles` is expected to be an increasing sequence"
        " of at least two elements.")

    with pytest.raises(
            ValueError,
            match=expected_match_str):
        get_silverkite_uncertainty_dict(
            uncertainty=uncertainty_dict,
            coverage=0.95)

    # exception check when quantiles are not available in `uncertainty`
    # and coverage is also None
    quantiles = None
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": quantiles,
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    expected_match_str = (
        "`quantiles` are not specified in `uncertainty`"
        " and `coverage` is not provided to infer them")

    with pytest.raises(
            ValueError,
            match=expected_match_str):
        get_silverkite_uncertainty_dict(
            uncertainty=uncertainty_dict,
            coverage=None)


def test_get_silverkite_uncertainty_dict_warning():
    """Testing silverkite_uncertainty_dict.
        Warning must be given if quantiles has more than two elements"""
    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.025, 0.1, 0.9, 0.975],  # 4 quantiles provided
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    with pytest.warns(Warning):
        get_silverkite_uncertainty_dict(
            uncertainty=uncertainty_dict,
            coverage=0.95)


def test_get_silverkite_uncertainty_dict_warning2():
    """Testing silverkite_uncertainty_dict.
        Checking for warnings when quantiles are asymmetric w.r.t to 0 and 1.
        We expect the 'upper quantile' to be '1 - lower quantile'"""

    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": ["dow_hr"],
            "quantiles": [0.1, 0.7],  # 2 quantiles provided and they are asymmetric
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    with pytest.warns(Warning):
        get_silverkite_uncertainty_dict(
            uncertainty=uncertainty_dict,
            coverage=0.6)


def test_get_similar_lag():
    """Tests get_similar_lag"""
    # Daily data
    assert get_similar_lag(freq_in_days=1) == 7
    # 2 days frequency
    assert get_similar_lag(freq_in_days=2) == 4
    # Weekly data
    assert get_similar_lag(freq_in_days=7) is None
    # Monthly data
    assert get_similar_lag(freq_in_days=30) is None
    # Hourly data
    assert get_similar_lag(freq_in_days=1/24) == 24*7
    # Two hour frequency
    assert get_similar_lag(freq_in_days=1/12) == 12*7


def test_get_default_changepoints_dict():
    """Tests ``get_default_changepoints_dict``."""
    change_points_dict = get_default_changepoints_dict(
        changepoints_method="uniform",
        num_days=365,
        forecast_horizon_in_days=7)

    for key in ["method", "n_changepoints", "continuous_time_col"]:
        assert change_points_dict[key] == {
                    "method": "uniform",
                    "n_changepoints": 12,
                    "continuous_time_col": "ct1"}[key]

    change_points_dict = get_default_changepoints_dict(
        changepoints_method="auto",
        num_days=365,
        forecast_horizon_in_days=7)

    assert change_points_dict == {
                "method": "auto",
                "yearly_seasonality_order": 10,
                "resample_freq": "7D",
                "regularization_strength": 0.8,
                "actual_changepoint_min_distance": "14D",
                "potential_changepoint_distance": "7D",
                "no_changepoint_distance_from_end": "14D"}

    change_points_dict = get_default_changepoints_dict(
        changepoints_method=None,
        num_days=365,
        forecast_horizon_in_days=7)

    assert change_points_dict is None

    change_points_dict = get_default_changepoints_dict(
        changepoints_method="auto",
        num_days=10,
        forecast_horizon_in_days=7)

    assert change_points_dict is None

    change_points_dict = get_default_changepoints_dict(
        changepoints_method="uniform",
        num_days=10,
        forecast_horizon_in_days=7)

    assert change_points_dict is None


def test_get_fourier_feature_col_names():
    """Tests getting Fourier feature column names."""
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", freq="D", periods=14)
    })
    fs_components_df = pd.DataFrame({
        "name": ["tod", "tow", "toy"],
        "period": [24.0, 7.0, 1.0],
        "order": [1, 2, 3],
        "seas_names": ["daily", "weekly", "yearly"]})
    fs_func = fourier_series_multi_fcn(
        col_names=fs_components_df.get("name"),
        periods=fs_components_df.get("period"),
        orders=fs_components_df.get("order"),
        seas_names=fs_components_df.get("seas_names")
    )
    fs_cols = get_fourier_feature_col_names(
        df=df,
        time_col="ts",
        fs_func=fs_func
    )
    assert fs_cols == [
        "sin1_tod_daily", "cos1_tod_daily",
        "sin1_tow_weekly", "cos1_tow_weekly",
        "sin2_tow_weekly", "cos2_tow_weekly",
        "sin1_toy_yearly", "cos1_toy_yearly",
        "sin2_toy_yearly", "cos2_toy_yearly",
        "sin3_toy_yearly", "cos3_toy_yearly",
    ]
