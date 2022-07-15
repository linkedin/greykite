from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset
from sklearn.base import RegressorMixin
from testfixtures import LogCapture

from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector
from greykite.algo.changepoint.adalasso.changepoint_detector import get_changepoints_dict
from greykite.algo.changepoint.adalasso.changepoint_detector import get_seasonality_changepoints
from greykite.common.data_loader import DataLoader
from greykite.common.logging import LOGGER_NAME
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_test_changepoint_df


@pytest.fixture
def hourly_data():
    """Generate 500 days of hourly data for tests"""
    return generate_df_for_tests(freq="H", periods=24 * 500)


def test_find_trend_changepoints(hourly_data):
    df = hourly_data["df"]
    dl = DataLoader()
    df_pt = dl.load_peyton_manning()

    model = ChangepointDetector()
    # test class variables are initialized as None
    assert model.trend_model is None
    assert model.trend_coef is None
    assert model.trend_intercept is None
    assert model.trend_changepoints is None
    assert model.trend_potential_changepoint_n is None
    assert model.trend_df is None
    assert model.y is None
    assert model.original_df is None
    assert model.value_col is None
    assert model.time_col is None
    assert model.adaptive_lasso_coef is None
    # model training with default values
    model.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y"
    )
    assert isinstance(model.trend_model, RegressorMixin)
    assert model.trend_model.coef_.shape[0] == 100 + 1 + 8 * 2
    assert model.trend_coef.shape[0] == 100 + 1 + 8 * 2
    assert model.trend_intercept is not None
    assert model.trend_changepoints is not None
    assert model.trend_potential_changepoint_n == 100
    assert model.trend_df.shape[1] == 100 + 1 + 8 * 2
    assert model.original_df.shape == df.shape
    assert model.time_col is not None
    assert model.value_col is not None
    assert model.adaptive_lasso_coef[1].shape[0] == 100 + 1 + 8 * 2
    assert model.y.index[0] not in model.trend_changepoints
    # model training with given values
    model = ChangepointDetector()
    model.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        potential_changepoint_n=50,
        yearly_seasonality_order=6,
        resample_freq="2D",
        trend_estimator="lasso",
        adaptive_lasso_initial_estimator="ols"
    )
    assert isinstance(model.trend_model, RegressorMixin)
    assert model.trend_model.coef_.shape[0] == 50 + 1 + 6 * 2
    assert model.trend_coef.shape[0] == 50 + 1 + 6 * 2
    assert model.trend_intercept is not None
    assert model.trend_changepoints is not None
    assert model.trend_potential_changepoint_n == 50
    assert model.trend_df.shape[1] == 50 + 1 + 6 * 2
    assert model.original_df.shape == df.shape
    assert model.time_col is not None
    assert model.value_col is not None
    assert model.adaptive_lasso_coef[1].shape[0] == 50 + 1 + 6 * 2
    assert model.y.index[0] not in model.trend_changepoints
    # test a given ``regularization_strength``
    model = ChangepointDetector()
    model.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        regularization_strength=1.0
    )
    assert isinstance(model.trend_model, RegressorMixin)
    assert model.trend_model.coef_.shape[0] == 100 + 1 + 8 * 2
    assert model.trend_coef.shape[0] == 100 + 1 + 8 * 2
    assert model.trend_intercept is not None
    assert model.trend_changepoints is not None
    assert model.trend_potential_changepoint_n == 100
    assert model.trend_df.shape[1] == 100 + 1 + 8 * 2
    assert model.original_df.shape == df.shape
    assert model.time_col is not None
    assert model.value_col is not None
    assert model.adaptive_lasso_coef[1].shape[0] == 100 + 1 + 8 * 2
    assert model.y.index[0] not in model.trend_changepoints
    # ``regularization_strength`` == 1.0 indicates no change point
    assert model.trend_changepoints == []
    model.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        regularization_strength=0.5
    )
    # ``regularization_strength`` between 0 and 1 indicates at least one change point
    assert len(model.trend_changepoints) > 0
    model.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        actual_changepoint_min_distance="D",
        regularization_strength=0.0
    )
    # ``regularization_strength`` == 0.0 indicates all potential change points are present
    assert len(model.trend_changepoints) == 100
    # test `potential_changepoint_distance`
    model = ChangepointDetector()
    model.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        potential_changepoint_distance="100D"
    )
    # test override `potential_changepoint_n`
    # df has length 500 days, with distance "100D", only 4 change points are placed.
    assert model.trend_potential_changepoint_n == 4
    with pytest.raises(ValueError,
                       match="In potential_changepoint_distance, the maximal unit is 'D', "
                             "i.e., you may use units no more than 'D' such as"
                             "'10D', '5H', '100T', '200S'. The reason is that 'W', 'M' "
                             "or higher has either cycles or indefinite number of days, "
                             "thus is not parsable by pandas as timedelta."):
        model.find_trend_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            potential_changepoint_distance="2M"
        )
    # test `no_changepoint_distance_from_end` with the Peyton Manning data
    model = ChangepointDetector()
    res = model.find_trend_changepoints(
        df=df_pt,
        time_col="ts",
        value_col="y",
        no_changepoint_distance_from_begin="730D",
        no_changepoint_distance_from_end="730D",
        regularization_strength=0
    )
    changepoints = res["trend_changepoints"]
    # test override `no_changepoint_proportion_from_end` and no change points in the last piece
    no_changepoint_proportion_from_end = timedelta(days=730) / (
            pd.to_datetime(df_pt["ts"].iloc[-1]) - pd.to_datetime(df_pt["ts"].iloc[0]))
    last_date_to_have_changepoint = pd.to_datetime(df_pt["ts"].iloc[int(
        df_pt.shape[0] * (1 - no_changepoint_proportion_from_end))])
    first_date_to_have_changepoint = pd.to_datetime(df_pt["ts"].iloc[int(
        df_pt.shape[0] * no_changepoint_proportion_from_end)])
    assert changepoints[-1] <= last_date_to_have_changepoint
    assert changepoints[0] >= first_date_to_have_changepoint
    # test value error
    with pytest.raises(ValueError,
                       match="In no_changepoint_distance_from_end, the maximal unit is 'D', "
                             "i.e., you may use units no more than 'D' such as"
                             "'10D', '5H', '100T', '200S'. The reason is that 'W', 'M' "
                             "or higher has either cycles or indefinite number of days, "
                             "thus is not parsable by pandas as timedelta."):
        model.find_trend_changepoints(
            df=df_pt,
            time_col="ts",
            value_col="y",
            no_changepoint_distance_from_end="2M"
        )
    # test `no_changepoint_proportion_from_end` and `actual_changepoint_min_distance`
    # generates a df with trend change points, ensuring we detect change points
    df_trend = generate_test_changepoint_df()
    model = ChangepointDetector()
    res = model.find_trend_changepoints(
        df=df_trend,
        time_col="ts",
        value_col="y",
        potential_changepoint_n=50,
        yearly_seasonality_order=0,
        adaptive_lasso_initial_estimator='lasso',
        no_changepoint_proportion_from_end=0.3,
        actual_changepoint_min_distance="10D"
    )
    changepoints = res["trend_changepoints"]
    # last changepoint in first 70% data
    assert changepoints[-1] <= df_trend["ts"][int(df_trend.shape[0] * 0.7)]
    assert all((changepoints[i + 1] - changepoints[i] >= to_offset("10D")) for i in range(len(changepoints) - 1))
    # test the asserts above are violated when not specifying `no_changepoint_proportion_from_end`
    model = ChangepointDetector()
    res = model.find_trend_changepoints(
        df=df_trend,
        time_col="ts",
        value_col="y",
        potential_changepoint_n=50,
        yearly_seasonality_order=0,
        adaptive_lasso_initial_estimator='ridge',
        no_changepoint_proportion_from_end=0.0,
        actual_changepoint_min_distance="1D"
    )
    changepoints = res["trend_changepoints"]
    # last changepoint after first 70% data
    assert changepoints[-1] > df_trend["ts"][int(df_trend.shape[0] * 0.7)]
    # negative potential_changepoint_n
    model = ChangepointDetector()
    with pytest.raises(ValueError, match="potential_changepoint_n can not be negative. "
                                         "A large number such as 100 is recommended"):
        model.find_trend_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            potential_changepoint_n=-1
        )
    # negative year_seasonality_order
    model = ChangepointDetector()
    with pytest.raises(ValueError, match="year_seasonality_order can not be negative. "
                                         "A number less than or equal to 10 is recommended"):
        model.find_trend_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            yearly_seasonality_order=-1
        )
    # negative regularization_strength
    with pytest.raises(ValueError, match="regularization_strength must be between 0.0 and 1.0."):
        model.find_trend_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            regularization_strength=-1
        )
    # estimator parameter combination not valid warning
    with pytest.warns(UserWarning) as record:
        model = ChangepointDetector()
        model.find_trend_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            trend_estimator="something"
        )
        assert "trend_estimator not in ['ridge', 'lasso', 'ols'], " \
               "estimating using ridge" in record[0].message.args[0]
    with pytest.warns(UserWarning) as record:
        model = ChangepointDetector()
        model.find_trend_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            trend_estimator="ols",
            yearly_seasonality_order=8
        )
        assert "trend_estimator = 'ols' with year_seasonality_order > 0 may create " \
               "over-fitting, trend_estimator has been set to 'ridge'." in record[0].message.args[0]
    with pytest.warns(UserWarning) as record:
        model = ChangepointDetector()
        model.find_trend_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            adaptive_lasso_initial_estimator="something"
        )
        assert "adaptive_lasso_initial_estimator not in ['ridge', 'lasso', 'ols'], " \
               "estimating with ridge" in record[0].message.args[0]
    # df sample size too small
    df = pd.DataFrame(
        data={
            "ts": pd.date_range(start='2020-1-1', end='2020-1-3', freq='D'),
            "y": [1, 2, 3]
        }
    )
    model = ChangepointDetector()
    with pytest.raises(ValueError, match="Change point detector does not work for less than "
                                         "5 observations. Please increase sample size."):
        model.find_trend_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
        )
    # test when training data has missing dates, the model drops na from resample
    df = pd.DataFrame(
        data={
            "ts": pd.date_range(start='2020-1-1', end='2020-1-9', freq='2D'),
            "y": [1, 2, 3, 4, 5]
        }
    )
    model = ChangepointDetector()
    model.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y"
    )
    assert model.y.isnull().sum().sum() == 0
    assert model.y.shape[0] == 5
    # tests varying yearly seasonality effect
    model = ChangepointDetector()
    model.find_trend_changepoints(
        df=df_pt,
        time_col="ts",
        value_col="y",
        yearly_seasonality_change_freq="365D"
    )
    assert model.trend_df.shape[1] > 100 + 1 + 8 * 2  # checks extra columns are created for varying yearly seasonality


def test_find_trend_changepoints_slow(hourly_data):
    """Tests the trend changepoint detection when fast trend estimation is turned off."""
    dl = DataLoader()
    df_pt = dl.load_peyton_manning()

    model = ChangepointDetector()
    model.find_trend_changepoints(
        df=df_pt,
        time_col="ts",
        value_col="y",
        fast_trend_estimation=False
    )
    assert isinstance(model.trend_model, RegressorMixin)
    assert model.trend_model.coef_.shape[0] == 100 + 1 + 8 * 2
    assert model.trend_coef.shape[0] == 100 + 1 + 8 * 2
    assert model.trend_intercept is not None
    assert model.trend_changepoints is not None
    assert model.trend_potential_changepoint_n == 100
    assert model.trend_df.shape[1] == 100 + 1 + 8 * 2
    assert model.original_df.shape == df_pt.shape
    assert model.time_col is not None
    assert model.value_col is not None
    assert model.adaptive_lasso_coef[1].shape[0] == 100 + 1 + 8 * 2
    assert model.y.index[0] not in model.trend_changepoints


def test_find_seasonality_changepoints(hourly_data):
    df = hourly_data["df"]
    dl = DataLoader()
    df_pt = dl.load_peyton_manning()

    # model training with given values
    model = ChangepointDetector()
    model.find_seasonality_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        potential_changepoint_n=80,
        resample_freq="2D",
        seasonality_components_df=pd.DataFrame({
            "name": ["tod", "tow", "conti_year"],
            "period": [24.0, 7.0, 1.0],
            "order": [3, 4, 5],
            "seas_names": ["daily", "weekly", "yearly"]})
    )
    # resample frequency is "2D", daily component is automatically removed from
    # seasonality_components_df
    assert model.seasonality_df.shape[1] == 18 * 81
    assert model.seasonality_changepoints is not None
    assert model.seasonality_estimation is not None
    assert model.seasonality_estimation.shape[0] == df.shape[0]
    # test a given ``regularization_strength``
    model = ChangepointDetector()
    model.find_seasonality_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        regularization_strength=1.0
    )
    # ``regularization_strength`` == 1.0 indicates no change point
    assert all([model.seasonality_changepoints[key] == [] for key in model.seasonality_changepoints.keys()])
    model.find_seasonality_changepoints(
        df=df_pt,
        time_col="ts",
        value_col="y",
        regularization_strength=0.0
    )
    # ``regularization_strength`` equals 0 indicates at least one change point
    assert any([model.seasonality_changepoints[key] != [] for key in model.seasonality_changepoints.keys()])
    # test `no_changepoint_distance_from_end` with the Peyton Manning data
    model = ChangepointDetector()
    res = model.find_seasonality_changepoints(
        df=df_pt,
        time_col="ts",
        value_col="y",
        no_changepoint_distance_from_end="730D",
        regularization_strength=0.1
    )
    changepoints_dict = res["seasonality_changepoints"]
    changepoints = []
    for key in changepoints_dict.keys():
        changepoints += changepoints_dict[key]
    # test override `no_changepoint_proportion_from_end` and no change points in the last piece
    no_changepoint_proportion_from_end = timedelta(days=730) / (
            pd.to_datetime(df_pt["ts"].iloc[-1]) - pd.to_datetime(df_pt["ts"].iloc[0]))
    last_date_to_have_changepoint = pd.to_datetime(df_pt["ts"].iloc[int(
        df_pt.shape[0] * (1 - no_changepoint_proportion_from_end))])
    assert changepoints[-1] <= last_date_to_have_changepoint
    # test daily data automatically drops daily seasonality components
    cd = ChangepointDetector()
    res = cd.find_seasonality_changepoints(
        df=df_pt,
        time_col="ts",
        value_col="y"
    )
    assert "daily" not in res["seasonality_changepoints"].keys()
    # test feeding the same df with different column names will not rerun trend estimation
    df2 = df_pt.copy().rename({"ts": "ts2", "y": "y2"}, axis=1)
    cd = ChangepointDetector()
    cd.find_seasonality_changepoints(
        df=df_pt,
        time_col="ts",
        value_col="y"
    )
    with pytest.warns(UserWarning) as record:
        cd.find_seasonality_changepoints(
            df=df2,
            time_col="ts2",
            value_col="y2"
        )
        assert ("Trend changepoints are already identified, using past trend estimation. "
                "If you would like to run trend change point detection again, "
                "please call ``find_trend_changepoints`` with desired parameters "
                "before calling ``find_seasonality_changepoints``.") in record[0].message.args[0]
    assert cd.time_col == "ts"
    assert cd.value_col == "y"
    # negative potential_changepoint_n
    model = ChangepointDetector()
    with pytest.raises(ValueError, match="potential_changepoint_n can not be negative. "
                                         "A large number such as 50 is recommended"):
        model.find_seasonality_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            potential_changepoint_n=-1
        )
    # negative regularization_strength
    with pytest.raises(ValueError, match="regularization_strength must be between 0.0 and 1.0."):
        model.find_seasonality_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            regularization_strength=-1
        )
    # test regularization_strength == None warning
    with pytest.warns(UserWarning) as record:
        model.find_seasonality_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            regularization_strength=None
        )
        assert ("regularization_strength is set to None. This will trigger cross-validation to "
                "select the tuning parameter which might result in too many change points. "
                "Keep the default value or tuning around it is recommended.") in record[0].message.args[0]
    # test existing trend estimation warning
    model = ChangepointDetector()
    model.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y"
    )
    with pytest.warns(UserWarning) as record:
        model.find_seasonality_changepoints(
            df=df,
            time_col="ts",
            value_col="y"
        )
        assert ("Trend changepoints are already identified, using past trend estimation. "
                "If you would like to run trend change point detection again, "
                "please call ``find_trend_changepoints`` with desired parameters "
                "before calling ``find_seasonality_changepoints``.") in record[0].message.args[0]
    # df sample size too small
    df_small = pd.DataFrame(
        data={
            "ts": pd.date_range(start='2020-1-1', end='2020-1-3', freq='D'),
            "y": [1, 2, 3]
        }
    )
    model = ChangepointDetector()
    with pytest.raises(ValueError, match="Change point detector does not work for less than "
                                         "5 observations. Please increase sample size."):
        model.find_seasonality_changepoints(
            df=df_small,
            time_col="ts",
            value_col="y",
        )
    # tests given trend changepoints
    cd = ChangepointDetector()
    cd.find_seasonality_changepoints(
        df=df_pt,
        time_col="ts",
        value_col="y",
        trend_changepoints=list(pd.to_datetime(["2016-01-01", "2017-02-05"]))
    )
    assert cd.trend_changepoints == list(pd.to_datetime(["2016-01-01", "2017-02-05"]))
    assert cd.original_df is not None
    assert cd.trend_estimation is not None
    assert cd.y is not None
    assert cd.time_col == "ts"
    assert cd.value_col == "y"


def test_plot(hourly_data):
    df = hourly_data['df']
    model = ChangepointDetector()
    model.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y"
    )
    # test empty plot
    with pytest.warns(UserWarning) as record:
        model.plot(
            observation=False,
            observation_original=False,
            trend_estimate=False,
            trend_change=False,
            yearly_seasonality_estimate=False,
            adaptive_lasso_estimate=False
        )
        assert "Figure is empty, at least one component has to be true." in record[0].message.args[0]
    # test plotting change without estimation
    with pytest.warns(UserWarning) as record:
        model = ChangepointDetector()
        model.plot(
            observation=False,
            observation_original=False,
            trend_estimate=False,
            trend_change=True,
            yearly_seasonality_estimate=False,
            adaptive_lasso_estimate=False
        )
        assert "You haven't run trend change point detection algorithm yet. " \
               "Please call find_trend_changepoints first." in record[0].message.args[0]
    # test plotting seasonality change or estimation without estimation
    with pytest.warns(UserWarning) as record:
        model = ChangepointDetector()
        model.plot(
            observation=False,
            observation_original=False,
            trend_estimate=False,
            trend_change=False,
            yearly_seasonality_estimate=False,
            adaptive_lasso_estimate=False,
            seasonality_change=True
        )
        assert ("You haven't run seasonality change point detection algorithm yet. "
                "Please call find_seasonality_changepoints first.") in record[0].message.args[0]
    with pytest.warns(UserWarning) as record:
        model = ChangepointDetector()
        model.plot(
            observation=False,
            observation_original=False,
            trend_estimate=False,
            trend_change=False,
            yearly_seasonality_estimate=False,
            adaptive_lasso_estimate=False,
            seasonality_estimate=True
        )
        assert ("You haven't run seasonality change point detection algorithm yet. "
                "Please call find_seasonality_changepoints first.") in record[0].message.args[0]


def test_get_changepoints_dict():
    dl = DataLoader()
    df_pt = dl.load_peyton_manning()

    changepoints_dict = {
        "method": "auto",
        "yearly_seasonality_order": 8,
        "resample_freq": "D",
        "trend_estimator": "ridge",
        "adaptive_lasso_initial_estimator": "ridge",
        "regularization_strength": None,
        "actual_changepoint_min_distance": "30D",
        "potential_changepoint_distance": None,
        "potential_changepoint_n": 100,
        "no_changepoint_distance_from_end": None,
        "no_changepoint_proportion_from_end": 0.0,
        "continuous_time_col": "ct1"
    }
    new_changepoints_dict, changepoint_detector = get_changepoints_dict(
        df=df_pt,
        time_col="ts",
        value_col="y",
        changepoints_dict=changepoints_dict
    )
    assert new_changepoints_dict["method"] == "custom"
    assert len(new_changepoints_dict["dates"]) > 0
    assert new_changepoints_dict["continuous_time_col"] == "ct1"
    assert changepoint_detector.trend_changepoints is not None
    # tests change point properties
    changepoints_dict = {
        "method": "auto",
        "yearly_seasonality_order": 8,
        "resample_freq": "D",
        "trend_estimator": "ridge",
        "adaptive_lasso_initial_estimator": "ridge",
        "regularization_strength": None,
        "actual_changepoint_min_distance": "100D",
        "potential_changepoint_distance": "50D",
        "potential_changepoint_n": 100,
        "no_changepoint_distance_from_end": None,
        "no_changepoint_proportion_from_end": 0.3,
        "continuous_time_col": "ct1",
        "dates": ["2001-01-01", "2010-01-01"]
    }
    new_changepoints_dict, changepoint_detector = get_changepoints_dict(
        df=df_pt,
        time_col="ts",
        value_col="y",
        changepoints_dict=changepoints_dict
    )
    changepoint_dates = new_changepoints_dict["dates"]
    # checks no change points at the end
    assert (changepoint_dates[-1] - pd.to_datetime(df_pt["ts"].iloc[0])) / \
           (pd.to_datetime(df_pt["ts"].iloc[-1]) - pd.to_datetime(df_pt["ts"].iloc[0])) <= 0.7
    # checks change point distance is good
    min_cp_dist = min([changepoint_dates[i] - changepoint_dates[i - 1] for i in range(1, len(changepoint_dates))])
    assert min_cp_dist >= timedelta(days=100)
    assert changepoint_detector.trend_changepoints is not None
    # checks additional custom changepoints are added
    assert pd.to_datetime("2001-01-01") not in changepoint_dates  # out of range
    assert pd.to_datetime("2010-01-01") in changepoint_dates
    # tests for None
    new_changepoints_dict, changepoint_detector = get_changepoints_dict(
        df=df_pt,
        time_col="ts",
        value_col="y",
        changepoints_dict=None
    )
    assert new_changepoints_dict is None
    assert changepoint_detector is None
    # tests for "custom"
    changepoints_dict = {
        "method": "custom",
        "dates": ["2020-01-01"]
    }
    new_changepoints_dict, changepoint_detector = get_changepoints_dict(
        df=df_pt,
        time_col="ts",
        value_col="y",
        changepoints_dict=changepoints_dict
    )
    assert new_changepoints_dict == changepoints_dict
    assert changepoint_detector is None
    # tests for uniform
    changepoints_dict = {
        "method": "uniform",
        "n_changepoints": 100
    }
    new_changepoints_dict, changepoint_detector = get_changepoints_dict(
        df=df_pt,
        time_col="ts",
        value_col="y",
        changepoints_dict=changepoints_dict
    )
    assert new_changepoints_dict == changepoints_dict
    assert changepoint_detector is None
    # tests unused keys
    changepoints_dict = {
        "method": "auto",
        "unused_key": "value"
    }
    with pytest.warns(UserWarning) as record:
        get_changepoints_dict(
            df=df_pt,
            time_col="ts",
            value_col="y",
            changepoints_dict=changepoints_dict
        )
        assert (f"The following keys in ``changepoints_dict`` are not recognized\n"
                f"{['unused_key']}") in record[0].message.args[0]


def test_get_seasonality_changepoints():
    # tests the functionality under cases that were not tested elsewhere
    df = pd.DataFrame({
        "ts": pd.date_range(start="2020-01-01", end="2020-01-30", freq="D"),
        "y": np.random.randn(30)
    })
    # tests uniform trend change point dictionary
    seasonality_changepoints_result = get_seasonality_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        trend_changepoints_dict={
            "method": "uniform",
            "n_changepoints": 5
        },
        trend_changepoint_dates=None,
        seasonality_changepoints_dict=None
    )
    assert "weekly" in seasonality_changepoints_result["seasonality_changepoints"].keys()
    assert "yearly" in seasonality_changepoints_result["seasonality_changepoints"].keys()
    assert isinstance(seasonality_changepoints_result["seasonality_changepoints"]["weekly"], list)
    assert isinstance(seasonality_changepoints_result["seasonality_changepoints"]["yearly"], list)
    # tests custom trend change point dictionary
    seasonality_changepoints_result = get_seasonality_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        trend_changepoints_dict={
            "method": "custom",
            "dates": ["2020-01-10"]
        },
        trend_changepoint_dates=None,
        seasonality_changepoints_dict={
            "seasonality_components_df": pd.DataFrame({
                "name": ["conti_year"],
                "period": [1.0],
                "order": [1],
                "seas_names": ["yearly"]
            }),
            "resample_freq": "D",
            "regularization_strength": 0.5,
            "actual_changepoint_min_distance": "D",
            "potential_changepoint_distance": "2D",
            "potential_changepoint_n": 15,
            "no_changepoint_distance_from_end": "5D",
            "no_changepoint_proportion_from_end": 0.1
        })
    assert "weekly" not in seasonality_changepoints_result["seasonality_changepoints"].keys()
    assert "yearly" in seasonality_changepoints_result["seasonality_changepoints"].keys()
    assert isinstance(seasonality_changepoints_result["seasonality_changepoints"]["yearly"], list)


def test_nan():
    df = pd.DataFrame({
        "ts": pd.date_range(start="2020-01-01", periods=366, freq="D"),
        "y": np.random.randn(366)
    })
    df.iloc[5:10, 1] = np.nan
    cd = ChangepointDetector()
    cd.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y"
    )
    cd.find_seasonality_changepoints(
        df=df,
        time_col="ts",
        value_col="y"
    )


def test_capping_potential_changepoints():
    df = pd.DataFrame({
        "ts": pd.date_range(start="2020-01-01", periods=366, freq="D"),
        "y": np.random.randn(366)
    })
    with LogCapture(LOGGER_NAME) as log_capture:
        cd = ChangepointDetector()
        cd.find_trend_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            resample_freq="D",
            potential_changepoint_distance="D",
            potential_changepoint_n_max=100
        )
        log_capture.check_present((
            LOGGER_NAME,
            "INFO",
            f"Number of potential changepoints is capped by 'potential_changepoint_n_max' "
            f"as 100. The 'potential_changepoint_distance' D is ignored. "
            f"The original number of changepoints was 365."
        ))
