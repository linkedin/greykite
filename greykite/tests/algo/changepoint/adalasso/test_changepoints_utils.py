from datetime import datetime as dt

import numpy as np
import pandas as pd
import pytest

from greykite.algo.changepoint.adalasso.changepoints_utils import adaptive_lasso_cv
from greykite.algo.changepoint.adalasso.changepoints_utils import build_seasonality_feature_df_from_detection_result
from greykite.algo.changepoint.adalasso.changepoints_utils import build_seasonality_feature_df_with_changes
from greykite.algo.changepoint.adalasso.changepoints_utils import build_trend_feature_df_with_changes
from greykite.algo.changepoint.adalasso.changepoints_utils import check_freq_unit_at_most_day
from greykite.algo.changepoint.adalasso.changepoints_utils import combine_detected_and_custom_trend_changepoints
from greykite.algo.changepoint.adalasso.changepoints_utils import compute_fitted_components
from greykite.algo.changepoint.adalasso.changepoints_utils import compute_min_changepoint_index_distance
from greykite.algo.changepoint.adalasso.changepoints_utils import estimate_seasonality_with_detected_changepoints
from greykite.algo.changepoint.adalasso.changepoints_utils import estimate_trend_with_detected_changepoints
from greykite.algo.changepoint.adalasso.changepoints_utils import filter_changepoints
from greykite.algo.changepoint.adalasso.changepoints_utils import find_neighbor_changepoints
from greykite.algo.changepoint.adalasso.changepoints_utils import get_changes_from_beta
from greykite.algo.changepoint.adalasso.changepoints_utils import get_seasonality_changepoint_df_cols
from greykite.algo.changepoint.adalasso.changepoints_utils import get_seasonality_changes_from_adaptive_lasso
from greykite.algo.changepoint.adalasso.changepoints_utils import get_trend_changepoint_dates_from_cols
from greykite.algo.changepoint.adalasso.changepoints_utils import get_trend_changes_from_adaptive_lasso
from greykite.algo.changepoint.adalasso.changepoints_utils import get_yearly_seasonality_changepoint_dates_from_freq
from greykite.algo.changepoint.adalasso.changepoints_utils import plot_change
from greykite.common.testing_utils import generate_df_for_tests


@pytest.fixture
def hourly_data():
    """Generate 500 days of hourly data for tests"""
    return generate_df_for_tests(freq="H", periods=24 * 100)


def test_check_freq_unit_at_most_day():
    # tests no error
    check_freq_unit_at_most_day("D", "name")
    check_freq_unit_at_most_day("50H", "name")
    # tests ValueError
    with pytest.raises(
            ValueError,
            match="In name, the maximal unit is 'D', "
                  "i.e., you may use units no more than 'D' such as"
                  "'10D', '5H', '100T', '200S'. The reason is that 'W', 'M' "
                  "or higher has either cycles or indefinite number of days, "
                  "thus is not parsable by pandas as timedelta."):
        check_freq_unit_at_most_day("2M", "name")


def test_build_trend_feature_df_with_changes(hourly_data):
    # test default parameters
    df = hourly_data["df"]
    df_trend = build_trend_feature_df_with_changes(
        df=df,
        time_col="ts"
    )
    assert df_trend.shape[0] == df.shape[0]
    assert df_trend.shape[1] == 101  # default value
    # test given parameters
    df_trend = build_trend_feature_df_with_changes(
        df=df,
        time_col="ts",
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 50
        }
    )
    assert df_trend.shape[0] == df.shape[0]
    assert df_trend.shape[1] == 51
    # test no change point
    df_trend = build_trend_feature_df_with_changes(
        df=df,
        time_col="ts",
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 0
        }
    )
    assert df_trend.shape[0] == df.shape[0]
    assert df_trend.shape[1] == 1
    # test result values
    df = pd.DataFrame(
        {
            "ts": [dt(2020, 1, 1),
                   dt(2020, 1, 2),
                   dt(2020, 1, 3),
                   dt(2020, 1, 4),
                   dt(2020, 1, 5),
                   dt(2020, 1, 6)],
            "y": [1, 2, 3, 4, 5, 6]
        }
    )
    df_trend = build_trend_feature_df_with_changes(
        df=df,
        time_col="ts",
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 2
        }
    )
    expected_df_trend = pd.DataFrame(
        {
            "changepoint0_2020_01_01_00": [0, 1 / 366, 2 / 366, 3 / 366, 4 / 366, 5 / 366],
            "changepoint1_2020_01_03_00": [0, 0, 0, 1 / 366, 2 / 366, 3 / 366],
            "changepoint2_2020_01_05_00": [0, 0, 0, 0, 0, 1 / 366]
        },
        index=[dt(2020, 1, 1),
               dt(2020, 1, 2),
               dt(2020, 1, 3),
               dt(2020, 1, 4),
               dt(2020, 1, 5),
               dt(2020, 1, 6)]
    )
    assert df_trend.round(3).equals(expected_df_trend.round(3))


def test_build_seasonality_feature_df_with_changes(hourly_data):
    # test default parameters
    df = hourly_data["df"]
    df_seasonality = build_seasonality_feature_df_with_changes(
        df=df,
        time_col="ts"
    )
    assert df_seasonality.shape[0] == df.shape[0]
    assert df_seasonality.shape[1] == 22  # default value
    # test given parameters
    df_seasonality = build_seasonality_feature_df_with_changes(
        df=df,
        time_col="ts",
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 10
        }
    )
    assert df_seasonality.shape[0] == df.shape[0]
    assert df_seasonality.shape[1] == 22 * 11
    # test given parameters
    df_seasonality = build_seasonality_feature_df_with_changes(
        df=df,
        time_col="ts",
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 10
        },
        fs_components_df=pd.DataFrame({
            "name": ["conti_year"],
            "period": [1.0],
            "order": [8],
            "seas_names": ["yearly"]})
    )
    assert df_seasonality.shape[0] == df.shape[0]
    assert df_seasonality.shape[1] == 16 * 11
    # test no change point
    df_seasonality = build_seasonality_feature_df_with_changes(
        df=df,
        time_col="ts",
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 0
        }
    )
    assert df_seasonality.shape[0] == df.shape[0]
    assert df_seasonality.shape[1] == 22
    # test result values
    df = pd.DataFrame(
        {
            "ts": [dt(2020, 1, 1),
                   dt(2020, 1, 2),
                   dt(2020, 1, 3),
                   dt(2020, 1, 4),
                   dt(2020, 1, 5)],
            "y": [1, 2, 3, 4, 5]
        }
    )
    df_seasonality = build_seasonality_feature_df_with_changes(
        df=df,
        time_col="ts",
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 1
        },
        fs_components_df=pd.DataFrame({
            "name": ["conti_year"],
            "period": [1.0],
            "order": [1],
            "seas_names": ["yearly"]})
    )
    freq = 2 * np.pi / 366
    expected_df_seasonality = pd.DataFrame(
        {
            "sin1_conti_year_yearly": [0, np.sin(freq * 1), np.sin(freq * 2), np.sin(freq * 3), np.sin(freq * 4)],
            "cos1_conti_year_yearly": [1, np.cos(freq * 1), np.cos(freq * 2), np.cos(freq * 3), np.cos(freq * 4)],
            "sin1_conti_year_yearly_2020_01_03_00": [0, 0, np.sin(freq * 2), np.sin(freq * 3), np.sin(freq * 4)],
            "cos1_conti_year_yearly_2020_01_03_00": [0, 0, np.cos(freq * 2), np.cos(freq * 3), np.cos(freq * 4)],
        },
        index=[dt(2020, 1, 1),
               dt(2020, 1, 2),
               dt(2020, 1, 3),
               dt(2020, 1, 4),
               dt(2020, 1, 5)]
    )
    pd.testing.assert_frame_equal(df_seasonality, expected_df_seasonality, check_names=False)


def test_build_seasonality_feature_df_with_detection_result():
    df = pd.DataFrame({
        "ts": pd.date_range(start="2020-01-06", end="2020-01-12", freq="D"),
        "y": list(range(7))
    })
    seasonality_changepoints = {
        "weekly": list(pd.to_datetime(["2020-01-08", "2020-01-11"])),
        "yearly": list(pd.to_datetime(["2020-01-09"]))
    }
    seasonality_components_df = pd.DataFrame({
        "name": ["tow", "conti_year"],
        "period": [7.0, 1.0],
        "order": [1, 2],
        "seas_names": ["weekly", "yearly"]
    })
    # we only assert values equal, since indices may differ
    # with overall block
    seasonality_df = build_seasonality_feature_df_from_detection_result(
        df=df,
        time_col="ts",
        seasonality_changepoints=seasonality_changepoints,
        seasonality_components_df=seasonality_components_df,
        include_original_block=True
    )
    # date index for changepoints
    week_cp1 = 1  # dates start from the 6th, first weekly cp is the 8th, last index before cp is 1 (from Monday)
    week_cp2 = 4  # dates start from the 6th, second weekly cp is the 11th, last index before cp is 4 (from Monday)
    year_cp1 = 7  # dates start from the 6th, first yearly cp is the 9th, last index before cp is 7 (from the 1st)
    expected_df = pd.DataFrame({
        "sin1_tow_weekly": np.sin([2 * np.pi * i / 7 for i in range(7)]),
        "cos1_tow_weekly": np.cos([2 * np.pi * i / 7 for i in range(7)]),
        "sin1_tow_weekly_2020_01_08_00": np.sin([2 * np.pi * i / 7 if i > week_cp1 else 0 for i in range(7)]),
        "cos1_tow_weekly_2020_01_08_00": np.cos([2 * np.pi * i / 7 if i > week_cp1 else np.pi / 2 for i in range(7)]),
        "sin1_tow_weekly_2020_01_11_00": np.sin([2 * np.pi * i / 7 if i > week_cp2 else 0 for i in range(7)]),
        "cos1_tow_weekly_2020_01_11_00": np.cos([2 * np.pi * i / 7 if i > week_cp2 else np.pi / 2 for i in range(7)]),
        # start date the 6th is 5 days from the 1st, and end date the 12th is 11 days from the 1st
        "sin1_conti_year_yearly": np.sin([2 * np.pi * i / 366 for i in range(5, 12)]),
        "cos1_conti_year_yearly": np.cos([2 * np.pi * i / 366 for i in range(5, 12)]),
        "sin2_conti_year_yearly": np.sin([2 * np.pi * i / 366 * 2 for i in range(5, 12)]),
        "cos2_conti_year_yearly": np.cos([2 * np.pi * i / 366 * 2 for i in range(5, 12)]),
        "sin1_conti_year_yearly_2020_01_09_00": np.sin(
            [2 * np.pi * i / 366 if i > year_cp1 else 0 for i in range(5, 12)]),
        "cos1_conti_year_yearly_2020_01_09_00": np.cos(
            [2 * np.pi * i / 366 if i > year_cp1 else np.pi / 2 for i in range(5, 12)]),
        "sin2_conti_year_yearly_2020_01_09_00": np.sin(
            [2 * np.pi * i / 366 * 2 if i > year_cp1 else 0 for i in range(5, 12)]),
        "cos2_conti_year_yearly_2020_01_09_00": np.cos(
            [2 * np.pi * i / 366 * 2 if i > year_cp1 else np.pi / 2 for i in range(5, 12)])
    },
        index=df["ts"])
    pd.testing.assert_frame_equal(seasonality_df, expected_df, check_names=False)
    # without overall block
    seasonality_df = build_seasonality_feature_df_from_detection_result(
        df=df,
        time_col="ts",
        seasonality_changepoints=seasonality_changepoints,
        seasonality_components_df=seasonality_components_df,
        include_original_block=False
    )
    expected_df = expected_df[[
        "sin1_tow_weekly_2020_01_08_00",
        "cos1_tow_weekly_2020_01_08_00",
        "sin1_tow_weekly_2020_01_11_00",
        "cos1_tow_weekly_2020_01_11_00",
        "sin1_conti_year_yearly_2020_01_09_00",
        "cos1_conti_year_yearly_2020_01_09_00",
        "sin2_conti_year_yearly_2020_01_09_00",
        "cos2_conti_year_yearly_2020_01_09_00"
    ]]
    pd.testing.assert_frame_equal(seasonality_df, expected_df, check_names=False)
    # with weekly only
    with pytest.warns(UserWarning) as record:
        seasonality_df = build_seasonality_feature_df_from_detection_result(
            df=df,
            time_col="ts",
            seasonality_changepoints=seasonality_changepoints,
            seasonality_components_df=seasonality_components_df,
            include_original_block=False,
            include_components=["weekly"]
        )
        assert (f"The following seasonality components have detected seasonality changepoints"
                f" but these changepoints are not included in the model,"
                f" because the seasonality component is not included in the model. {['yearly']}") in record[0].message.args[0]
    expected_df = expected_df[[
        "sin1_tow_weekly_2020_01_08_00",
        "cos1_tow_weekly_2020_01_08_00",
        "sin1_tow_weekly_2020_01_11_00",
        "cos1_tow_weekly_2020_01_11_00"
    ]]
    pd.testing.assert_frame_equal(seasonality_df, expected_df, check_names=False)


def test_compute_fitted_components():
    # test trend
    df = pd.DataFrame(
        {
            "changepoint0": [1, 1, 1, 1],
            "changepoint1": [0, 2, 2, 2],
            "sin1_conti_year_yearly_cp0": [np.sin(1), np.sin(2), np.sin(3), np.sin(4)],
            "cos1_conti_year_yearly_cp0": [np.cos(1), np.cos(2), np.cos(3), np.cos(4)],
            "sin1_conti_year_yearly_cp1": [0, 0, np.sin(3), np.sin(4)],
            "cos1_conti_year_yearly_cp1": [0, 0, np.cos(3), np.cos(4)]
        }
    )
    coef = np.array([1, 2, 3, 4, 5, 6])
    intercept = 1
    trend = compute_fitted_components(
        x=df,
        coef=coef,
        regex="^changepoint",
        include_intercept=True,
        intercept=intercept
    )
    expected_trend = pd.Series(
        [1 * 1 + 2 * 0 + 1] + [1 * 1 + 2 * 2 + 1] * 3
    )
    assert trend.equals(expected_trend)
    # test yearly seasonality
    df = pd.DataFrame(
        {
            "changepoint0": [1, 1, 1, 1],
            "changepoint1": [0, 2, 2, 2],
            "sin1_conti_year_yearly_cp0": [np.sin(1), np.sin(2), np.sin(3), np.sin(4)],
            "cos1_conti_year_yearly_cp0": [np.cos(1), np.cos(2), np.cos(3), np.cos(4)],
            "sin1_conti_year_yearly_cp1": [0, 0, np.sin(3), np.sin(4)],
            "cos1_conti_year_yearly_cp1": [0, 0, np.cos(3), np.cos(4)]
        }
    )
    coef = np.array([1, 1, 1, 1, 1, 1])
    seasonality = compute_fitted_components(
        x=df,
        coef=coef,
        regex="^.*yearly.*$",
        include_intercept=False
    )
    expected_seasonality = pd.Series(
        [
            np.sin(1) + np.cos(1),
            np.sin(2) + np.cos(2),
            np.sin(3) + np.cos(3) + np.sin(3) + np.cos(3),
            np.sin(4) + np.cos(4) + np.sin(4) + np.cos(4)
        ]
    )
    assert seasonality.equals(expected_seasonality)
    # tests ValueError
    with pytest.raises(
            ValueError,
            match="``intercept`` must be provided when ``include_intercept`` is True."):
        compute_fitted_components(
            x=df,
            coef=coef,
            regex="^.*yearly.*$",
            include_intercept=True
        )


def test_plot_change():
    observations = pd.Series(
        {
            "y": [1, 2, 3, 4, 5]
        },
        index=[
            dt(2020, 1, 1),
            dt(2020, 1, 2),
            dt(2020, 1, 3),
            dt(2020, 1, 4),
            dt(2020, 1, 5)
        ]
    )
    trend_estimate = pd.Series(
        [2, 3, 4, 5, 6],
        index=[
            dt(2020, 1, 1),
            dt(2020, 1, 2),
            dt(2020, 1, 3),
            dt(2020, 1, 4),
            dt(2020, 1, 5)
        ]
    )
    trend_change = [dt(2020, 1, 2), dt(2020, 1, 4)]
    year_seasonality_estimate = pd.Series(
        [1, 2, 1, 2, 1],
        index=[
            dt(2020, 1, 1),
            dt(2020, 1, 2),
            dt(2020, 1, 3),
            dt(2020, 1, 4),
            dt(2020, 1, 5)
        ]
    )
    adaptive_lasso_estimate = pd.Series(
        [3, 4, 5, 6, 7],
        index=[
            dt(2020, 1, 1),
            dt(2020, 1, 2),
            dt(2020, 1, 3),
            dt(2020, 1, 4),
            dt(2020, 1, 5)
        ]
    )
    seasonality_estimate = pd.Series(
        [1, 2, 3, 1, 2],
        index=[
            dt(2020, 1, 1),
            dt(2020, 1, 2),
            dt(2020, 1, 3),
            dt(2020, 1, 4),
            dt(2020, 1, 5)
        ]
    )
    seasonality_change_dict = {
        "weekly": [dt(2020, 1, 1)],
        "yearly": [dt(2020, 1, 3)]
    }
    seasonality_change_list = [dt(2020, 1, 1), dt(2020, 1, 3)]
    fig = plot_change(
        observation=observations,
        trend_estimate=trend_estimate,
        trend_change=trend_change,
        year_seasonality_estimate=year_seasonality_estimate,
        adaptive_lasso_estimate=adaptive_lasso_estimate
    )
    assert len(fig.data) == 6
    fig1 = plot_change(
        observation=None,
        trend_estimate=None,
        trend_change=trend_change,
        year_seasonality_estimate=None,
        adaptive_lasso_estimate=adaptive_lasso_estimate
    )
    assert len(fig1.data) == 3
    fig2 = plot_change(
        observation=observations,
        trend_estimate=trend_estimate,
        trend_change=trend_change,
        year_seasonality_estimate=year_seasonality_estimate,
        adaptive_lasso_estimate=adaptive_lasso_estimate,
        seasonality_estimate=seasonality_estimate,
        seasonality_change=seasonality_change_dict
    )
    assert len(fig2.data) == 9
    fig3 = plot_change(
        observation=observations,
        trend_estimate=trend_estimate,
        trend_change=trend_change,
        year_seasonality_estimate=year_seasonality_estimate,
        adaptive_lasso_estimate=adaptive_lasso_estimate,
        seasonality_estimate=seasonality_estimate,
        seasonality_change=seasonality_change_list
    )
    assert len(fig3.data) == 9
    # tests warning
    with pytest.warns(UserWarning) as record:
        seasonality_change_dict = {
            "weekly": [dt(2020, 1, 1)],
            "yearly": [dt(2020, 1, 3)],
            "test1": [dt(2020, 1, 3)],
            "test2": [dt(2020, 1, 3)],
            "test3": [dt(2020, 1, 3)],
            "test4": [dt(2020, 1, 3)],
            "test5": [dt(2020, 1, 3)],
            "test6": [dt(2020, 1, 3)],
            "test7": [dt(2020, 1, 3)],
        }
        plot_change(
            observation=observations,
            trend_estimate=trend_estimate,
            trend_change=trend_change,
            year_seasonality_estimate=year_seasonality_estimate,
            adaptive_lasso_estimate=adaptive_lasso_estimate,
            seasonality_estimate=seasonality_estimate,
            seasonality_change=seasonality_change_dict
        )
        assert ("Only the first 8 components with detected change points"
                "are plotted.") in record[0].message.args[0]
    with pytest.warns(UserWarning) as record:
        plot_change(
            observation=None,
            trend_estimate=None,
            trend_change=trend_change,
            year_seasonality_estimate=year_seasonality_estimate,
            adaptive_lasso_estimate=None,
            seasonality_estimate=None,
            seasonality_change=None
        )
    assert ("trend_change is not shown. Must provide observations, trend_estimate, "
            "adaptive_lasso_estimate or seasonality_estimate to plot trend_change.") in record[0].message.args[0]
    with pytest.warns(UserWarning) as record:
        plot_change(
            observation=None,
            trend_estimate=trend_estimate,
            trend_change=trend_change,
            year_seasonality_estimate=year_seasonality_estimate,
            adaptive_lasso_estimate=adaptive_lasso_estimate,
            seasonality_estimate=None,
            seasonality_change=seasonality_change_list
        )
        assert ("seasonality_change is not shown. Must provide observations or"
                " seasonality_estimate to plot seasonality_change.") in record[0].message.args[0]
    with pytest.raises(
            ValueError,
            match="seasonality_change must be either list or dict."):
        plot_change(
            observation=None,
            trend_estimate=trend_estimate,
            trend_change=trend_change,
            year_seasonality_estimate=year_seasonality_estimate,
            adaptive_lasso_estimate=adaptive_lasso_estimate,
            seasonality_estimate=seasonality_estimate,
            seasonality_change=np.array(seasonality_change_list)
        )


def test_adaptive_lasso_cv():
    # test "ridge" as initial estimator
    x = np.random.randn(20, 5)
    y = np.random.randn(20)
    intercept, coef = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef="ridge"
    )
    assert coef.shape[0] == 5
    assert intercept is not None
    # test "lasso" as initial estimator
    intercept, coef = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef="lasso"
    )
    assert coef.shape[0] == 5
    assert intercept is not None
    # test "ols" as initial estimator
    intercept, coef = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef="ols"
    )
    assert coef.shape[0] == 5
    assert intercept is not None
    # test `numpy.array` as initial estimator
    initial_coef = np.array([1, 2, 3, 4, 5])
    intercept, coef = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef=initial_coef
    )
    assert coef.shape[0] == 5
    assert intercept is not None
    # test given lambda for adaptive lasso
    intercept, coef = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef="ridge",
        regularization_strength=1.0
    )
    assert coef.shape[0] == 5
    assert intercept is not None
    # tests no change point when ``regularization_strength`` == 1.0
    assert (coef != 0).sum() == 0
    _, coef = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef="ridge",
        regularization_strength=0.0
    )
    # tests all change points present when ``regularization_strength`` == 0.0
    assert (coef != 0).sum() == 5
    # tests max_min_ratio
    # when we have greater max_min_ratio, the lambda that corresponds to regularization_strength=0.1
    # is smaller, thus we have more nonzero coefficients.
    _, coef1 = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef="ridge",
        regularization_strength=0.1
    )
    _, coef2 = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef="ridge",
        regularization_strength=0.1,
        max_min_ratio=1e100
    )
    nonzeros1 = (coef1 != 0).sum()
    nonzeros2 = (coef2 != 0).sum()
    assert nonzeros1 <= nonzeros2
    # tests value error
    with pytest.raises(ValueError,
                       match="regularization_strength must be between 0.0 and 1.0."):
        adaptive_lasso_cv(
            x=x,
            y=y,
            initial_coef="ridge",
            regularization_strength=-1
        )


def test_find_neighbor_changepoints():
    cp_idx = [1, 2, 5, 7, 8, 9, 10, 18, 22]
    neighbor_cps = find_neighbor_changepoints(
        cp_idx=cp_idx
    )
    assert neighbor_cps == [[1, 2], [5], [7, 8, 9, 10], [18], [22]]
    # test with greater neighbor distance
    cp_idx = [1, 2, 4, 7, 9, 12, 15, 20]
    neighbor_cps = find_neighbor_changepoints(
        cp_idx=cp_idx,
        min_index_distance=3
    )
    assert neighbor_cps == [[1, 2, 4], [7, 9], [12], [15], [20]]
    # test negative distance ValueError
    with pytest.raises(ValueError, match="`min_index_distance` must be positive."):
        find_neighbor_changepoints(
            cp_idx=cp_idx,
            min_index_distance=-1
        )
    # tests unsorted input
    cp_idx = [2, 1, 4, 7, 9, 12, 15, 20]
    with pytest.warns(UserWarning) as record:
        neighbor_cps = find_neighbor_changepoints(
            cp_idx=cp_idx,
            min_index_distance=3
        )
        assert ("The given `cp_idx` is not sorted. It has been sorted.") in record[0].message.args[0]
        assert neighbor_cps == [[1, 2, 4], [7, 9], [12], [15], [20]]


def test_get_trend_changes_from_adaptive_lasso():
    x = np.random.randn(20, 5)
    y = np.random.randn(20)
    intercept, coef = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef="ridge"
    )
    changepoipnt_dates = pd.Series(
        [dt(2020, 1, i) for i in range(1, 21)]
    )
    changepoints, coefs = get_trend_changes_from_adaptive_lasso(
        x=x,
        y=y,
        changepoint_dates=changepoipnt_dates,
        initial_coef="ridge"
    )
    assert intercept == coefs[0]
    assert np.array_equal(coef, coefs[1])
    assert changepoints is not None
    # test given adaptive lasso lambda
    intercept, coef = adaptive_lasso_cv(
        x=x,
        y=y,
        initial_coef="ridge",
        regularization_strength=1.0
    )
    changepoints, coefs = get_trend_changes_from_adaptive_lasso(
        x=x,
        y=y,
        changepoint_dates=changepoipnt_dates,
        initial_coef="ridge",
        regularization_strength=1.0
    )
    assert intercept == coefs[0]
    assert np.array_equal(coef, coefs[1])
    assert changepoints is not None


def test_compute_min_changepoint_index_distance():
    # test functionality
    df = pd.DataFrame(
        {
            "ts": pd.date_range(start="2020-01-01", end="2020-01-30"),
            "y": list(range(30))
        }
    )
    min_index_distance = compute_min_changepoint_index_distance(
        df=df,
        time_col="ts",
        n_changepoints=10,
        min_distance_between_changepoints="5D"
    )
    # two consecutive potential change points have distance 30/10 = 3 days
    # so need min_index_distance 2 to reach the desired distance "5D
    assert min_index_distance == 2
    # test no change points
    min_index_distance = compute_min_changepoint_index_distance(
        df=df,
        time_col="ts",
        n_changepoints=0,
        min_distance_between_changepoints="5D"
    )
    assert min_index_distance == df.shape[0]
    # test non-daily
    min_index_distance = compute_min_changepoint_index_distance(
        df=df,
        time_col="ts",
        n_changepoints=10,
        min_distance_between_changepoints="120H"
    )
    assert min_index_distance == 2  # same as the test above
    # test ValueError
    with pytest.raises(ValueError,
                       match="In min_distance_between_changepoints, the maximal unit is 'D', "
                             "i.e., you may use units no more than 'D' such as"
                             "'10D', '5H', '100T', '200S'. The reason is that 'W', 'M' "
                             "or higher has either cycles or indefinite number of days, "
                             "thus is not parsable by pandas as timedelta."):
        compute_min_changepoint_index_distance(
            df=df,
            time_col="ts",
            n_changepoints=10,
            min_distance_between_changepoints="W"
        )


def test_filter_changepoints():
    cp_blocks = [[1, 2, 4, 5], [8], [12, 14]]
    coef = np.array([0, 10, 5, 0, -5, 10, 0, 0, 10, 0, 0, 0, 10, 0, 5])
    min_index_distance = 3
    selected_changepoints = filter_changepoints(
        cp_blocks=cp_blocks,
        coef=coef,
        min_index_distance=min_index_distance
    )
    assert selected_changepoints == [1, 5, 8, 12]

    cp_blocks = [[1, 3, 5, 7, 9]]
    coef = np.array([0, 2, 0, 4, 0, 6, 0, 9, 0, 10])
    min_index_distance = 5
    selected_changepoints = filter_changepoints(
        cp_blocks=cp_blocks,
        coef=coef,
        min_index_distance=min_index_distance
    )
    assert selected_changepoints == [3, 9]

    cp_blocks = [[1, 3, 5, 7, 9]]
    coef = np.array([0, 2, 0, 4, 0, 6, 0, 9, 0, 10])
    min_index_distance = 3
    selected_changepoints = filter_changepoints(
        cp_blocks=cp_blocks,
        coef=coef,
        min_index_distance=min_index_distance
    )
    assert selected_changepoints == [1, 5, 9]

    cp_blocks = [[1, 3, 5, 7, 9]]
    coef = np.array([0, 2, 0, 4, 0, 6, 0, 9, 0, 10])
    min_index_distance = 2
    selected_changepoints = filter_changepoints(
        cp_blocks=cp_blocks,
        coef=coef,
        min_index_distance=min_index_distance
    )
    assert selected_changepoints == [1, 3, 5, 7, 9]

    cp_blocks = [[1, 3, 5, 7, 9]]
    coef = np.array([0, 2, 0, 4, 0, 6, 0, 9, 0, 10])
    min_index_distance = 1
    selected_changepoints = filter_changepoints(
        cp_blocks=cp_blocks,
        coef=coef,
        min_index_distance=min_index_distance
    )
    assert selected_changepoints == [1, 3, 5, 7, 9]

    cp_blocks = [[1, 2, 4, 7, 11, 12], [30, 32, 33], [41, 46], [59]]
    coef = np.array([0, 10, -5, 0, 8, 0, 0, -12, 0, 0, 0, -1, 8, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0,
                     3, -6, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, -18, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4])
    min_index_distance = 2
    selected_changepoints = filter_changepoints(
        cp_blocks=cp_blocks,
        coef=coef,
        min_index_distance=min_index_distance
    )
    assert selected_changepoints == [1, 4, 7, 12, 30, 33, 41, 46, 59]

    cp_blocks = [[1, 2, 4, 7, 11, 12], [30, 32, 33], [41, 46], [59]]
    coef = np.array([0, 10, -5, 0, 8, 0, 0, -12, 0, 0, 0, -1, 8, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0,
                     3, -6, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, -18, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4])
    min_index_distance = 3
    selected_changepoints = filter_changepoints(
        cp_blocks=cp_blocks,
        coef=coef,
        min_index_distance=min_index_distance
    )
    assert selected_changepoints == [1, 4, 7, 12, 30, 33, 41, 46, 59]

    cp_blocks = [[1, 2, 4, 7, 11, 12], [30, 32, 33], [41, 46], [59]]
    coef = np.array([0, 10, -5, 0, 8, 0, 0, -12, 0, 0, 0, -1, 8, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0,
                     3, -6, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, -18, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4])
    min_index_distance = 6
    selected_changepoints = filter_changepoints(
        cp_blocks=cp_blocks,
        coef=coef,
        min_index_distance=min_index_distance
    )
    assert selected_changepoints == [1, 7, 30, 46, 59]

    with pytest.raises(ValueError,
                       match="`min_index_distance` is the minimum distance between change point"
                             "indices to consider them separate, and must be positive."):
        filter_changepoints(
            cp_blocks=cp_blocks,
            coef=coef,
            min_index_distance=0
        )


def test_get_changes_from_beta():
    beta = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6])
    seasonality_components_df = pd.DataFrame({
        "name": ["tow", "conti_year"],
        "period": [7.0, 1.0],
        "order": [1, 2],
        "seas_names": ["weekly", "yearly"]})
    # tests changes with L2 norm on cusum values
    expected_components = ["weekly", "yearly"]
    expected_changes = [np.array([np.sqrt(5), np.sqrt(52) - np.sqrt(5)]),
                        np.array([np.sqrt(38), np.sqrt(264) - np.sqrt(38)])]
    result = get_changes_from_beta(beta, seasonality_components_df, True)  # with cusum
    components = list(result.keys())
    changes = list(result.values())
    assert components == expected_components
    assert all(all(changes[i] == expected_changes[i]) for i in range(len(changes)))
    # tests changes with L2 norm on change coefficients
    expected_changes = [np.array([np.sqrt(5), 5]), np.array([np.sqrt(38), np.sqrt(102)])]
    result = get_changes_from_beta(beta, seasonality_components_df, False)  # without cusum
    components = list(result.keys())
    changes = list(result.values())
    assert components == expected_components
    assert all(all(changes[i] == expected_changes[i]) for i in range(len(changes)))


def test_get_seasonality_changes_from_adaptive_lasso():
    x = np.random.randn(20, 12)
    y = np.random.randn(20)
    changepoipnt_dates = pd.Series(
        [dt(2020, 1, i) for i in range(1, 21)]
    )
    result = get_seasonality_changes_from_adaptive_lasso(
        x=x,
        y=y,
        changepoint_dates=changepoipnt_dates,
        initial_coef="ridge",
        seasonality_components_df=pd.DataFrame({
            "name": ["tow", "conti_year"],
            "period": [7.0, 1.0],
            "order": [1, 2],
            "seas_names": ["weekly", "yearly"]}),
        regularization_strength=0.6
    )
    assert "weekly" in result.keys()
    assert "yearly" in result.keys()
    assert result["weekly"] is not None
    assert result["yearly"] is not None


def test_estimate_trend_with_detected_changepoints():
    df = pd.DataFrame({
        "ts": pd.date_range(start="2020-01-01", end="2020-01-05", freq="D"),
        "y": [1, 2, 3, 4, 5]
    })
    changepoints = [dt(2020, 1, 3)]
    trend_estimate = estimate_trend_with_detected_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        changepoints=changepoints
    )
    assert isinstance(trend_estimate, pd.Series)
    assert trend_estimate.shape[0] == df.shape[0]
    assert (trend_estimate.index == df["ts"]).all()
    trend01 = np.round(trend_estimate[1] - trend_estimate[0], 5)
    trend12 = np.round(trend_estimate[2] - trend_estimate[1], 5)
    trend34 = np.round(trend_estimate[4] - trend_estimate[3], 5)
    assert trend01 == trend12
    assert trend12 != trend34
    # tests ValueError
    with pytest.raises(
            ValueError,
            match="estimator can only be either 'ridge' or 'ols'."):
        estimate_trend_with_detected_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            changepoints=changepoints,
            estimator="lasso"
        )


def test_estimate_seasonality_with_detected_changepoints():
    df = pd.DataFrame({
        "ts": pd.date_range(start="2020-01-01", end="2020-01-05", freq="D"),
        "y": [1, 2, 3, 4, 5]
    })
    seasonality_changepoints = {
        "weekly": [dt(2020, 1, 2)],
        "yearly": [dt(2020, 1, 4)]
    }
    seasonality_estimate = estimate_seasonality_with_detected_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        seasonality_changepoints=seasonality_changepoints,
        seasonality_components_df=pd.DataFrame({
            "name": ["tow", "conti_year"],
            "period": [7.0, 1.0],
            "order": [1, 2],
            "seas_names": ["weekly", "yearly"]})
    )
    assert isinstance(seasonality_estimate, pd.Series)
    assert seasonality_estimate.shape[0] == df.shape[0]
    assert (seasonality_estimate.index == df["ts"]).all()
    # tests ValueError
    with pytest.raises(
            ValueError,
            match="estimator can only be either 'ridge' or 'ols'."):
        estimate_seasonality_with_detected_changepoints(
            df=df,
            time_col="ts",
            value_col="y",
            seasonality_changepoints=seasonality_changepoints,
            seasonality_components_df=pd.DataFrame({
                "name": ["tow", "conti_year"],
                "period": [7.0, 1.0],
                "order": [1, 2],
                "seas_names": ["weekly", "yearly"]}),
            estimator="lasso"
        )


def test_get_seasonality_changepoint_df_cols():
    df = pd.DataFrame({
        "ts": pd.date_range(start="2020-01-01", end="2020-01-10", freq="D"),
        "y": list(range(10))
    })
    seasonality_changepoints = {
        "weekly": list(pd.to_datetime(["2020-01-05", "2020-01-07"])),
        "yearly": list(pd.to_datetime(["2020-01-08"]))
    }
    seasonality_components_df = pd.DataFrame({
        "name": ["tow", "conti_year"],
        "period": [7.0, 1.0],
        "order": [1, 2],
        "seas_names": ["weekly", "yearly"]})
    cols = get_seasonality_changepoint_df_cols(
        df=df,
        time_col="ts",
        seasonality_changepoints=seasonality_changepoints,
        seasonality_components_df=seasonality_components_df,
        include_original_block=True
    )
    assert cols == [
        "sin1_tow_weekly",
        "cos1_tow_weekly",
        "sin1_tow_weekly_2020_01_05_00",
        "cos1_tow_weekly_2020_01_05_00",
        "sin1_tow_weekly_2020_01_07_00",
        "cos1_tow_weekly_2020_01_07_00",
        "sin1_conti_year_yearly",
        "cos1_conti_year_yearly",
        "sin2_conti_year_yearly",
        "cos2_conti_year_yearly",
        "sin1_conti_year_yearly_2020_01_08_00",
        "cos1_conti_year_yearly_2020_01_08_00",
        "sin2_conti_year_yearly_2020_01_08_00",
        "cos2_conti_year_yearly_2020_01_08_00"
    ]
    cols = get_seasonality_changepoint_df_cols(
        df=df,
        time_col="ts",
        seasonality_changepoints=seasonality_changepoints,
        seasonality_components_df=seasonality_components_df,
        include_original_block=False
    )
    assert cols == [
        "sin1_tow_weekly_2020_01_05_00",
        "cos1_tow_weekly_2020_01_05_00",
        "sin1_tow_weekly_2020_01_07_00",
        "cos1_tow_weekly_2020_01_07_00",
        "sin1_conti_year_yearly_2020_01_08_00",
        "cos1_conti_year_yearly_2020_01_08_00",
        "sin2_conti_year_yearly_2020_01_08_00",
        "cos2_conti_year_yearly_2020_01_08_00",
    ]
    # with weekly only
    cols = get_seasonality_changepoint_df_cols(
        df=df,
        time_col="ts",
        seasonality_changepoints=seasonality_changepoints,
        seasonality_components_df=seasonality_components_df,
        include_original_block=False,
        include_components=["weekly"]
    )
    assert cols == [
        "sin1_tow_weekly_2020_01_05_00",
        "cos1_tow_weekly_2020_01_05_00",
        "sin1_tow_weekly_2020_01_07_00",
        "cos1_tow_weekly_2020_01_07_00"
    ]


def test_get_trend_changepoint_dates_from_cols():
    # hourly changepoints
    changepoint_cols = ["ct1", "changepoint0_2020_01_05_03", "changepoint1_2020_02_08_00"]
    changepoint_dates = get_trend_changepoint_dates_from_cols(changepoint_cols)
    expected_changepoint_dates = list(pd.to_datetime(["2020-01-05-03", "2020-02-08"]))
    assert changepoint_dates == expected_changepoint_dates
    # minute level changepoints
    changepoint_cols = ["ct1", "changepoint0_2020_01_05_03_05", "changepoint1_2020_02_08_00"]
    changepoint_dates = get_trend_changepoint_dates_from_cols(changepoint_cols)
    expected_changepoint_dates = list(pd.to_datetime(["2020-01-05 03:05", "2020-02-08"]))
    assert changepoint_dates == expected_changepoint_dates
    # no changepoints
    changepoint_cols = ["ct1"]
    changepoint_dates = get_trend_changepoint_dates_from_cols(changepoint_cols)
    assert changepoint_dates == []


def test_get_yearly_seasonality_changepoint_dates_from_freq():
    df = pd.DataFrame({
        "ts": pd.date_range(start="2020-01-01", end="2025-01-01", freq="D")
    })
    # tests None
    result = get_yearly_seasonality_changepoint_dates_from_freq(
        df=df,
        time_col="ts",
        yearly_seasonality_change_freq=None
    )
    assert result == []
    # tests regular input
    result = get_yearly_seasonality_changepoint_dates_from_freq(
        df=df,
        time_col="ts",
        yearly_seasonality_change_freq="365D"
    )
    assert result == list(pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]))
    # tests least_training_length
    result = get_yearly_seasonality_changepoint_dates_from_freq(
        df=df,
        time_col="ts",
        yearly_seasonality_change_freq="365D",
        min_training_length="400D"
    )
    assert result == list(pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31"]))
    # tests warnings
    with pytest.warns(UserWarning) as record:
        get_yearly_seasonality_changepoint_dates_from_freq(
            df=df,
            time_col="ts",
            yearly_seasonality_change_freq="180D"
        )
        assert ("yearly_seasonality_change_freq is less than a year. It might be too short "
                "to fit accurate yearly seasonality." in record[0].message.args[0])
    with pytest.warns(UserWarning) as record:
        get_yearly_seasonality_changepoint_dates_from_freq(
            df=df,
            time_col="ts",
            yearly_seasonality_change_freq="10000D"
        )
        assert ("No yearly seasonality changepoint added. Either data length is too short "
                "or yearly_seasonality_change_freq is too long." in record[0].message.args[0])


def test_combine_detected_and_custom_trend_changepoints():
    detected_changepoints = ["2020-01-01", "2020-02-05", "2020-07-03"]
    custom_changepoints = ["2020-01-05", "2020-04-13"]
    # tests no ``min_distance``
    combined_changepoints = combine_detected_and_custom_trend_changepoints(
        detected_changepoint_dates=detected_changepoints,
        custom_changepoint_dates=custom_changepoints,
        min_distance=None,
        keep_detected=False
    )
    assert all(combined_changepoints == pd.to_datetime(
        ["2020-01-01", "2020-01-05", "2020-02-05", "2020-04-13", "2020-07-03"]
    ))
    # tests ``min_distance`` with ``keep_detected=False``
    combined_changepoints = combine_detected_and_custom_trend_changepoints(
        detected_changepoint_dates=detected_changepoints,
        custom_changepoint_dates=custom_changepoints,
        min_distance="5D",
        keep_detected=False
    )
    assert all(combined_changepoints == pd.to_datetime(
        ["2020-01-05", "2020-02-05", "2020-04-13", "2020-07-03"]
    ))
    # tests ``min_distance`` with ``keep_detected=True``
    combined_changepoints = combine_detected_and_custom_trend_changepoints(
        detected_changepoint_dates=detected_changepoints,
        custom_changepoint_dates=custom_changepoints,
        min_distance="5D",
        keep_detected=True
    )
    assert all(combined_changepoints == pd.to_datetime(
        ["2020-01-01", "2020-02-05", "2020-04-13", "2020-07-03"]
    ))
    # tests empty input
    combined_changepoints = combine_detected_and_custom_trend_changepoints(
        detected_changepoint_dates=detected_changepoints,
        custom_changepoint_dates=[],
        min_distance="5D",
        keep_detected=False
    )
    assert all(combined_changepoints == pd.to_datetime(
        ["2020-01-01", "2020-02-05", "2020-07-03"]
    ))
    combined_changepoints = combine_detected_and_custom_trend_changepoints(
        detected_changepoint_dates=[],
        custom_changepoint_dates=custom_changepoints,
        min_distance="5D",
        keep_detected=False
    )
    assert all(combined_changepoints == pd.to_datetime(
        ["2020-01-05", "2020-04-13"]
    ))
