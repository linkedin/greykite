from datetime import timedelta

import pytest

from greykite.algo.changepoint.adalasso.auto_changepoint_params import generate_trend_changepoint_detection_params
from greykite.algo.changepoint.adalasso.auto_changepoint_params import get_actual_changepoint_min_distance
from greykite.algo.changepoint.adalasso.auto_changepoint_params import get_changepoint_resample_freq
from greykite.algo.changepoint.adalasso.auto_changepoint_params import get_no_changepoint_distance_from_end
from greykite.algo.changepoint.adalasso.auto_changepoint_params import get_potential_changepoint_n
from greykite.algo.changepoint.adalasso.auto_changepoint_params import get_regularization_strength
from greykite.algo.changepoint.adalasso.auto_changepoint_params import get_yearly_seasonality_order
from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.testing_utils import generate_df_for_tests


@pytest.fixture
def df_hourly():
    """Hourly data."""
    df = generate_df_for_tests(
        freq="H",
        periods=24*365
    )
    return df


@pytest.fixture
def df_daily():
    """Daily data."""
    df = generate_df_for_tests(
        freq="D",
        periods=365*2
    )
    return df


@pytest.fixture
def df_weekly():
    """Weekly data."""
    df = generate_df_for_tests(
        freq="W",
        periods=52*3
    )
    return df


@pytest.fixture
def df_monthly():
    """Monthly data."""
    df = generate_df_for_tests(
        freq="MS",
        periods=12*6
    )
    return df


def test_get_changepoint_resample_freq():
    """Tests get changepoint resample frequency."""
    # Hourly data.
    resample_freq = get_changepoint_resample_freq(
        n_points=24*365,
        min_increment=timedelta(hours=1)
    )
    assert resample_freq == "3D"

    # Daily data.
    resample_freq = get_changepoint_resample_freq(
        n_points=365 * 2,
        min_increment=timedelta(days=1)
    )
    assert resample_freq == "7D"

    # Weekly data.
    resample_freq = get_changepoint_resample_freq(
        n_points=52 * 2,
        min_increment=timedelta(days=7)
    )
    assert resample_freq is None

    # Override ``min_num_points_after_agg``.
    resample_freq = get_changepoint_resample_freq(
        n_points=365,
        min_increment=timedelta(days=1),
        min_num_points_after_agg=200
    )
    assert resample_freq == "D"

    # Override ``min_num_points_after_agg``.
    resample_freq = get_changepoint_resample_freq(
        n_points=365,
        min_increment=timedelta(days=1),
        min_num_points_after_agg=50
    )
    assert resample_freq == "7D"


def test_get_yearly_seasonality_order(df_daily):
    """Tests get yearly seasonality order."""
    yearly_seasonality_order = get_yearly_seasonality_order(
        df=df_daily["df"],
        time_col=TIME_COL,
        value_col=VALUE_COL,
        resample_freq="7D"
    )
    assert yearly_seasonality_order == 3


def test_get_potential_changepoint_n():
    """Tests get number of potential changepoints."""
    # Hourly data.
    n_changepoints = get_potential_changepoint_n(
        n_points=24*365,
        total_increment=timedelta(days=365),
        resample_freq="3D",
        yearly_seasonality_order=3,
        cap=200
    )
    assert n_changepoints == 114

    # Daily data.
    n_changepoints = get_potential_changepoint_n(
        n_points=365 * 2,
        total_increment=timedelta(days=365 * 2),
        resample_freq="7D",
        yearly_seasonality_order=15,
        cap=100
    )
    assert n_changepoints == 73

    # Monthly data.
    n_changepoints = get_potential_changepoint_n(
        n_points=12 * 5,
        total_increment=timedelta(days=365 * 5 + 1),
        resample_freq=None,
        yearly_seasonality_order=15,
        cap=100
    )
    assert n_changepoints == 29

    # Tests cap.
    n_changepoints = get_potential_changepoint_n(
        n_points=12 * 5,
        total_increment=timedelta(days=365 * 5 + 1),
        resample_freq=None,
        yearly_seasonality_order=15,
        cap=20
    )
    assert n_changepoints == 20


def test_get_no_changepoint_distance_from_end():
    """Tests get the distance for the end where no changepoints are placed."""
    # Hourly data.
    distance = get_no_changepoint_distance_from_end(
        min_increment=timedelta(hours=1),
        forecast_horizon=12
    )
    assert distance == "14D"
    distance = get_no_changepoint_distance_from_end(
        min_increment=timedelta(hours=1),
        forecast_horizon=24 * 7
    )
    assert distance == "28D"

    # Daily data.
    distance = get_no_changepoint_distance_from_end(
        min_increment=timedelta(days=1),
        forecast_horizon=1
    )
    assert distance == "30D"
    distance = get_no_changepoint_distance_from_end(
        min_increment=timedelta(days=1),
        forecast_horizon=14
    )
    assert distance == "56D"

    # Weekly data.
    distance = get_no_changepoint_distance_from_end(
        min_increment=timedelta(days=7),
        forecast_horizon=2
    )
    assert distance == "56D"
    distance = get_no_changepoint_distance_from_end(
        min_increment=timedelta(days=7),
        forecast_horizon=4
    )
    assert distance == "112D"

    # Monthly data.
    distance = get_no_changepoint_distance_from_end(
        min_increment=timedelta(days=28),
        forecast_horizon=3
    )
    assert distance == "252D"

    # Yearly data.
    distance = get_no_changepoint_distance_from_end(
        min_increment=timedelta(days=365),
        forecast_horizon=2
    )
    assert distance == "1460D"


def test_get_actual_changepoint_min_distance():
    """Tests get minimum distance between actual changepoints."""
    # Hourly data.
    distance = get_actual_changepoint_min_distance(
        min_increment=timedelta(hours=1)
    )
    assert distance == "14D"

    # Daily data.
    distance = get_actual_changepoint_min_distance(
        min_increment=timedelta(days=1)
    )
    assert distance == "30D"

    # Weekly data.
    distance = get_actual_changepoint_min_distance(
        min_increment=timedelta(days=7)
    )
    assert distance == "30D"

    # Monthly data.
    distance = get_actual_changepoint_min_distance(
        min_increment=timedelta(days=28)
    )
    assert distance == "56D"

    # Yearly data.
    distance = get_actual_changepoint_min_distance(
        min_increment=timedelta(days=365)
    )
    assert distance == "730D"


def test_get_regularization_strength():
    """Tests get regularization strength."""
    regularization = get_regularization_strength()
    assert regularization == 0.6


def test_generate_trend_changepoint_detection_params(df_hourly, df_daily, df_weekly, df_monthly):
    """Tests the overall parameter generation."""
    # Hourly data.
    params = generate_trend_changepoint_detection_params(
        df=df_hourly["df"],
        forecast_horizon=24
    )
    assert params == dict(
        yearly_seasonality_order=3,
        resample_freq="3D",
        regularization_strength=0.6,
        actual_changepoint_min_distance="14D",
        potential_changepoint_n=100,
        no_changepoint_distance_from_end="14D"
    )

    # Daily data.
    params = generate_trend_changepoint_detection_params(
        df=df_daily["df"],
        forecast_horizon=7
    )
    assert params == dict(
        yearly_seasonality_order=3,
        resample_freq="7D",
        regularization_strength=0.6,
        actual_changepoint_min_distance="30D",
        potential_changepoint_n=97,
        no_changepoint_distance_from_end="30D"
    )

    # Weekly data.
    params = generate_trend_changepoint_detection_params(
        df=df_weekly["df"],
        forecast_horizon=2
    )
    assert params == dict(
        yearly_seasonality_order=1,
        resample_freq=None,
        regularization_strength=0.6,
        actual_changepoint_min_distance="30D",
        potential_changepoint_n=100,
        no_changepoint_distance_from_end="56D"
    )

    # Monthly data.
    params = generate_trend_changepoint_detection_params(
        df=df_monthly["df"],
        forecast_horizon=3
    )
    assert params == dict(
        yearly_seasonality_order=1,
        resample_freq=None,
        regularization_strength=0.6,
        actual_changepoint_min_distance="56D",
        potential_changepoint_n=69,
        no_changepoint_distance_from_end="252D"
    )

    # No changepoints because data is too short.
    params = generate_trend_changepoint_detection_params(
        df=df_daily["df"].iloc[:60],
        forecast_horizon=1
    )
    assert params is None


def test_changepoint_detection(df_daily, df_weekly):
    """Tests changepoint detection with the auto generated parameters."""
    # Daily data.
    params = generate_trend_changepoint_detection_params(
        df=df_daily["df"],
        forecast_horizon=7
    )
    cd = ChangepointDetector()
    cd.find_trend_changepoints(
        df=df_daily["df"],
        time_col=TIME_COL,
        value_col=VALUE_COL,
        **params
    )

    # Weekly data.
    params = generate_trend_changepoint_detection_params(
        df=df_weekly["df"],
        forecast_horizon=7
    )
    cd = ChangepointDetector()
    cd.find_trend_changepoints(
        df=df_weekly["df"],
        time_col=TIME_COL,
        value_col=VALUE_COL,
        **params
    )
