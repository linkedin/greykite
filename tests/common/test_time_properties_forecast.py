import datetime
from datetime import timedelta

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.enums import SimpleTimeFrequencyEnum
from greykite.common.enums import TimeEnum
from greykite.common.features.timeseries_features import get_default_origin_for_time_vars
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.common.time_properties_forecast import get_default_horizon_from_period
from greykite.common.time_properties_forecast import get_forecast_time_properties
from greykite.common.time_properties_forecast import get_simple_time_frequency_from_period


def test_get_simple_time_frequency_from_period():
    """Tests get_simple_time_frequency_from_period function"""
    assert get_simple_time_frequency_from_period(1.0) == SimpleTimeFrequencyEnum.MINUTE
    assert get_simple_time_frequency_from_period(120.0) == SimpleTimeFrequencyEnum.MINUTE
    assert get_simple_time_frequency_from_period(3600.0) == SimpleTimeFrequencyEnum.HOUR
    assert get_simple_time_frequency_from_period(6*3600.0) == SimpleTimeFrequencyEnum.HOUR
    assert get_simple_time_frequency_from_period(24*3600.0) == SimpleTimeFrequencyEnum.DAY
    assert get_simple_time_frequency_from_period(7*24*3600.0) == SimpleTimeFrequencyEnum.WEEK
    assert get_simple_time_frequency_from_period(31*24*3600.0) == SimpleTimeFrequencyEnum.MONTH
    assert get_simple_time_frequency_from_period(366*24*3600.0) == SimpleTimeFrequencyEnum.YEAR
    assert get_simple_time_frequency_from_period(600*24*3600.0) == SimpleTimeFrequencyEnum.MULTIYEAR


def test_get_default_horizon_from_period():
    """Tests get_default_horizon_from_period function"""
    assert get_default_horizon_from_period(60.0) == 60  # 1 min
    assert get_default_horizon_from_period(300.0) == 60  # 5 min
    assert get_default_horizon_from_period(3600.0) == 24  # 1 hour
    assert get_default_horizon_from_period(6 * 3600.0) == 24  # 6 hours
    assert get_default_horizon_from_period(24 * 3600.0) == 30  # 1 day
    assert get_default_horizon_from_period(7 * 24 * 3600.0) == 12  # 1 week
    assert get_default_horizon_from_period(7 * 24 * 3600.0, num_observations=10) == 5
    assert get_default_horizon_from_period(7 * 24 * 3600.0, num_observations=30) == 12


def test_get_forecast_time_properties():
    """Tests get_forecast_time_properties"""
    num_training_points = 365  # one year of daily data
    data = generate_df_for_tests(freq="D", periods=num_training_points)
    df = data["df"]
    result = get_forecast_time_properties(
        df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="D",
        forecast_horizon=0)
    default_origin = get_default_origin_for_time_vars(df, TIME_COL)
    assert result == {
        "period": TimeEnum.ONE_DAY_IN_SECONDS.value,
        "simple_freq": SimpleTimeFrequencyEnum.DAY,
        "num_training_points": num_training_points,
        "num_training_days": num_training_points,
        "days_per_observation": 1,
        "forecast_horizon": 0,
        "forecast_horizon_in_timedelta": timedelta(days=0),
        "forecast_horizon_in_days": 0,
        "start_year": 2018,
        "end_year": 2019,
        "origin_for_time_vars": default_origin}

    # longer forecast_horizon
    result = get_forecast_time_properties(
        df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="D",
        forecast_horizon=365)
    default_origin = get_default_origin_for_time_vars(df, TIME_COL)
    assert result == {
        "period": TimeEnum.ONE_DAY_IN_SECONDS.value,
        "simple_freq": SimpleTimeFrequencyEnum.DAY,
        "num_training_points": num_training_points,
        "num_training_days": num_training_points,
        "days_per_observation": 1,
        "forecast_horizon": 365,
        "forecast_horizon_in_timedelta": timedelta(days=365),
        "forecast_horizon_in_days": 365,
        "start_year": 2018,
        "end_year": 2020,
        "origin_for_time_vars": default_origin}

    # two years of hourly data
    num_training_points = 2*365*24
    data = generate_df_for_tests(freq="H", periods=num_training_points)
    df = data["df"]
    result = get_forecast_time_properties(
        df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="H",
        forecast_horizon=0)
    default_origin = get_default_origin_for_time_vars(df, TIME_COL)
    assert result == {
        "period": TimeEnum.ONE_HOUR_IN_SECONDS.value,
        "simple_freq": SimpleTimeFrequencyEnum.HOUR,
        "num_training_points": num_training_points,
        "num_training_days": num_training_points / 24,
        "days_per_observation": 1/24,
        "forecast_horizon": 0,
        "forecast_horizon_in_timedelta": timedelta(days=0),
        "forecast_horizon_in_days": 0,
        "start_year": 2018,
        "end_year": 2020,
        "origin_for_time_vars": default_origin}

    # longer forecast_horizon
    result = get_forecast_time_properties(
        df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="H",
        forecast_horizon=365 * 24)
    default_origin = get_default_origin_for_time_vars(df, TIME_COL)
    assert result == {
        "period": TimeEnum.ONE_HOUR_IN_SECONDS.value,
        "simple_freq": SimpleTimeFrequencyEnum.HOUR,
        "num_training_points": num_training_points,
        "num_training_days": num_training_points / 24,
        "days_per_observation": 1/24,
        "forecast_horizon": 365*24,
        "forecast_horizon_in_timedelta": timedelta(days=365),
        "forecast_horizon_in_days": 365,
        "start_year": 2018,
        "end_year": 2021,
        "origin_for_time_vars": default_origin}

    # ``forecast_horizon=None``
    result = get_forecast_time_properties(
        df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="H",
        forecast_horizon=None)
    default_origin = get_default_origin_for_time_vars(df, TIME_COL)
    assert result == {
        "period": TimeEnum.ONE_HOUR_IN_SECONDS.value,
        "simple_freq": SimpleTimeFrequencyEnum.HOUR,
        "num_training_points": num_training_points,
        "num_training_days": num_training_points / 24,
        "days_per_observation": 1/24,
        "forecast_horizon": 24,
        "forecast_horizon_in_timedelta": timedelta(days=1),
        "forecast_horizon_in_days": 1,
        "start_year": 2018,
        "end_year": 2020,
        "origin_for_time_vars": default_origin}

    # weekly df with regressors
    num_training_points = 50
    data = generate_df_with_reg_for_tests(
        freq="W-SUN",
        periods=num_training_points,
        train_start_date=datetime.datetime(2018, 11, 30),
        remove_extra_cols=True,
        mask_test_actuals=True)
    df = data["df"]
    train_df = data["train_df"]
    forecast_horizon = data["fut_time_num"]
    regressor_cols = [col for col in df.columns if col not in [TIME_COL, VALUE_COL]]
    result = get_forecast_time_properties(
        df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="W-SUN",
        regressor_cols=regressor_cols,
        forecast_horizon=forecast_horizon)
    default_origin = get_default_origin_for_time_vars(df, TIME_COL)
    assert result == {
        "period": TimeEnum.ONE_WEEK_IN_SECONDS.value,
        "simple_freq": SimpleTimeFrequencyEnum.WEEK,
        "num_training_points": train_df.shape[0],  # size of training set
        "num_training_days": train_df.shape[0] * 7,
        "days_per_observation": 7,
        "forecast_horizon": 9,
        "forecast_horizon_in_timedelta": timedelta(days=63),
        "forecast_horizon_in_days": 63.0,
        "start_year": 2018,
        "end_year": 2019,
        "origin_for_time_vars": default_origin}

    # checks `num_training_days` with `train_end_date`
    data = generate_df_with_reg_for_tests(
        freq="H",
        periods=300*24,
        train_start_date=datetime.datetime(2018, 7, 1),
        remove_extra_cols=True,
        mask_test_actuals=True)
    df = data["df"]
    train_end_date = datetime.datetime(2019, 2, 1)
    result = get_forecast_time_properties(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="H",
        regressor_cols=data["regressor_cols"],
        train_end_date=train_end_date,
        forecast_horizon=forecast_horizon)
    period = 3600  # seconds between observations
    time_delta = (train_end_date - df[TIME_COL].min())  # train end - train start
    num_training_days = (time_delta.days + (time_delta.seconds + period) / TimeEnum.ONE_DAY_IN_SECONDS.value)
    assert result["num_training_days"] == num_training_days

    # checks `num_training_days` without `train_end_date`
    result = get_forecast_time_properties(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="H",
        regressor_cols=data["regressor_cols"],
        train_end_date=None,
        forecast_horizon=forecast_horizon)
    time_delta = (datetime.datetime(2019, 2, 26) - df[TIME_COL].min())  # by default, train end is the last date with nonnull value_col
    num_training_days = (time_delta.days + (time_delta.seconds + period) / TimeEnum.ONE_DAY_IN_SECONDS.value)
    assert result["num_training_days"] == num_training_days
