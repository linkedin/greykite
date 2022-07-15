import pandas as pd

from greykite.algo.common.col_name_utils import add_category_cols
from greykite.algo.common.col_name_utils import create_pred_category
from greykite.algo.common.col_name_utils import filter_coef_summary
from greykite.algo.common.col_name_utils import simplify_changepoints
from greykite.algo.common.col_name_utils import simplify_event
from greykite.algo.common.col_name_utils import simplify_name
from greykite.algo.common.col_name_utils import simplify_pred_cols
from greykite.algo.common.col_name_utils import simplify_time_features


def test_simplify_event():
    name = simplify_event("C(Q('events_Chinese New Year_minus_1'), levels=['', 'event'])[T.event]")
    assert name == "events_Chinese New Year-1"
    name = simplify_event("C(Q('events_Chinese New Year_plus_1'), levels=['', 'event'])[T.event]")
    assert name == "events_Chinese New Year+1"
    name = simplify_event("no_prefix")
    assert name == "no_prefix"


def test_simplify_time_feature():
    name = simplify_time_features("C(Q('str_dow'), levels=['1-Mon', '2-Tue', '3-Wed', '4-Thu', '5-Fri', '6-Sat', '7-Sun'])[T.2-Tue]")
    assert name == "str_dow_2-Tue"
    name = simplify_time_features("str_dow[T.7-Sun]")
    assert name == "str_dow_7-Sun"
    name = simplify_time_features("toy")
    assert name == "toy"
    name = simplify_time_features("no_prefix")
    assert name == "no_prefix"


def test_simplify_changepoints():
    name = simplify_changepoints("changepoint1_2018_09_02_00")
    assert name == "cp1_2018_09_02_00"
    name = simplify_changepoints("no_prefix")
    assert name == "no_prefix"


def test_simplify_name():
    name = simplify_name("C(Q('events_Chinese New Year_minus_1'), levels=['', 'event'])[T.event]")
    assert name == "events_Chinese New Year-1"
    name = simplify_name("C(Q('str_dow'), levels=['1-Mon', '2-Tue', '3-Wed', '4-Thu', '5-Fri', '6-Sat', '7-Sun'])[T.2-Tue]")
    assert name == "str_dow_2-Tue"
    name = simplify_name("changepoint1_2018_09_02_00")
    assert name == "cp1_2018_09_02_00"
    name = simplify_name("cos1_ct1_yearly_2019_02_05_00")
    assert name == "cos1_ct1_yearly_2019_02_05_00"
    name = simplify_name("no_prefix")
    assert name == "no_prefix"


def test_simplify_pred_cols():
    pred_cols = [
        "C(Q('events_Chinese New Year_minus_1'), levels=['', 'event'])[T.event]",
        "C(Q('str_dow'), levels=['1-Mon', '2-Tue', '3-Wed', '4-Thu', '5-Fri', '6-Sat', '7-Sun'])[T.2-Tue]",
        "changepoint1_2018_09_02_00",
        "cos1_ct1_yearly_2019_02_05_00",
        "no_prefix"]
    new_pred_cols = simplify_pred_cols(pred_cols)
    assert new_pred_cols == [
        "events_Chinese New Year-1",
        "str_dow_2-Tue",
        "cp1_2018_09_02_00",
        "cos1_ct1_yearly_2019_02_05_00",
        "no_prefix"]


def test_add_category_cols():
    coef_summary = pd.DataFrame({
        "Pred_col": [
            "Intercept",
            "ct1",
            "sin1_toy_yearly",
            "y_lag7",
            "ct1:sin1_toy_yearly",
            "C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]",
            "x"
        ]
    })
    pred_category = {
        "intercept": ["Intercept"],
        "time_features": ["ct1", "ct1:sin1_toy_yearly"],
        "event_features": ["C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]"],
        "trend_features": ["ct1", "ct1:sin1_toy_yearly"],
        "seasonality_features": ["sin1_toy_yearly", "ct1:sin1_toy_yearly"],
        "lag_features": ["y_lag7"],
        "regressor_features": ["x"],
        "interaction_features": ["ct1:sin1_toy_yearly"]
    }
    new_df = add_category_cols(coef_summary, pred_category)
    pd.testing.assert_frame_equal(new_df, pd.DataFrame({
        "Pred_col": [
            "Intercept",
            "ct1",
            "sin1_toy_yearly",
            "y_lag7",
            "ct1:sin1_toy_yearly",
            "C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]",
            "x"
        ],
        "is_intercept": [1, 0, 0, 0, 0, 0, 0],
        "is_time_feature": [0, 1, 0, 0, 1, 0, 0],
        "is_event": [0, 0, 0, 0, 0, 1, 0],
        "is_trend": [0, 1, 0, 0, 1, 0, 0],
        "is_seasonality": [0, 0, 1, 0, 1, 0, 0],
        "is_lag": [0, 0, 0, 1, 0, 0, 0],
        "is_regressor": [0, 0, 0, 0, 0, 0, 1],
        "is_interaction": [0, 0, 0, 0, 1, 0, 0]
    }))


def test_create_pred_category():
    pred_cols = ["Intercept",
                 "ct1",
                 "sin1_toy_yearly",
                 "y_lag7",
                 "ct1:sin1_toy_yearly",
                 "C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]",
                 "x"]
    extra_pred_cols = ["x"]
    pred_category = create_pred_category(
        pred_cols,
        extra_pred_cols,
        df_cols=["ts", "y", "x"])
    expected_pred_category = {
        "intercept": ["Intercept"],
        "time_features": ["ct1", "ct1:sin1_toy_yearly"],
        "event_features": ["C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]"],
        "trend_features": ["ct1", "ct1:sin1_toy_yearly"],
        "seasonality_features": ["sin1_toy_yearly", "ct1:sin1_toy_yearly"],
        "lag_features": ["y_lag7"],
        "regressor_features": ["x"],
        "interaction_features": ["ct1:sin1_toy_yearly"]
    }
    assert pred_category == expected_pred_category


def test_create_pred_category_regressor():
    """Tests regressors are classified into the correct category,
    even their names matches some specific patterns such as "cp" or "lag".
    """
    pred_cols = ["Intercept",
                 "ct1",
                 "sin1_toy_yearly",
                 "y_lag7",
                 "ct1:sin1_toy_yearly",
                 "C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]",
                 "x",
                 "weather_wghtd_avg_cld_cvr_tot_pct_max_mid",
                 "media_total_spend_lag4"]
    extra_pred_cols = ["x", "weather_wghtd_avg_cld_cvr_tot_pct_max_mid", "media_total_spend_lag4"]
    pred_category = create_pred_category(
        pred_cols,
        extra_pred_cols,
        df_cols=["ts", "y", "x", "weather_wghtd_avg_cld_cvr_tot_pct_max_mid", "media_total_spend_lag4"])
    expected_pred_category = {
        "intercept": ["Intercept"],
        "time_features": ["ct1", "ct1:sin1_toy_yearly"],
        "event_features": ["C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]"],
        "trend_features": ["ct1", "ct1:sin1_toy_yearly"],
        "seasonality_features": ["sin1_toy_yearly", "ct1:sin1_toy_yearly"],
        "lag_features": ["y_lag7"],
        "regressor_features": ["x", "weather_wghtd_avg_cld_cvr_tot_pct_max_mid", "media_total_spend_lag4"],
        "interaction_features": ["ct1:sin1_toy_yearly"]
    }
    assert pred_category == expected_pred_category


def test_filter_coef_summary():
    coef_summary = pd.DataFrame({
        "Pred_col": [
            "Intercept",
            "cp1_2018_09_02_00",
            "sin1_toy_yearly",
            "y_lag7",
            "ct1:sin1_toy_yearly",
            "C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]",
            "x"
        ]})
    pred_category = create_pred_category(coef_summary["Pred_col"].tolist(), [], ["ts", "y"])
    x = filter_coef_summary(
        coef_summary=coef_summary,
        pred_category=pred_category,
        is_intercept=True,
        is_time_feature=False,
        is_trend=True,
        is_interaction=False)
    pd.testing.assert_frame_equal(x, pd.DataFrame({
        "Pred_col": [
            "Intercept",
            "cp1_2018_09_02_00"
        ],
        "is_intercept": [1, 0],
        "is_time_feature": [0, 0],
        "is_event": [0, 0],
        "is_trend": [0, 1],
        "is_seasonality": [0, 0],
        "is_lag": [0, 0],
        "is_regressor": [0, 0],
        "is_interaction": [0, 0]
    }))
