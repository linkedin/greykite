import pandas as pd
import pytest

from greykite.algo.common.holiday_grouper import HolidayGrouper
from greykite.algo.common.holiday_utils import HOLIDAY_NAME_COL
from greykite.algo.common.holiday_utils import get_weekday_weekend_suffix
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader
from greykite.common.features.timeseries_features import get_holidays
from greykite.common.python_utils import assert_equal


@pytest.fixture
def daily_df():
    df = DataLoader().load_peyton_manning()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


@pytest.fixture
def holiday_df():
    holiday_df = get_holidays(countries=["US"], year_start=2010, year_end=2016)["US"]
    # Only keeps non-observed holidays.
    holiday_df = holiday_df[~holiday_df["event_name"].str.contains("Observed")]
    return holiday_df.sort_values(by=[EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL]).reset_index(drop=True)


HOLIDAY_IMPACT_DICT = {
    "Christmas Day": (4, 3),               # Always 12/25.
    "Halloween": (1, 1),                   # Always 10/31.
    "Independence Day": (4, 4),            # Always 7/4.
    "Labor Day": (3, 1),                   # Monday.
    "Martin Luther King Jr. Day": (3, 1),  # Monday.
    "Memorial Day": (3, 1),                # Monday.
    "New Year's Day": (3, 4),              # Always 1/1.
    "Thanksgiving": (1, 4),                # Thursday.
}


def test_expand_holiday_df_with_suffix(holiday_df):
    """Tests `expand_holiday_df_with_suffix` function."""
    # Tests the case when no change is made.
    expanded_holiday_df = HolidayGrouper.expand_holiday_df_with_suffix(
        holiday_df=holiday_df,
        holiday_date_col=EVENT_DF_DATE_COL,
        holiday_name_col=EVENT_DF_LABEL_COL,
        holiday_impact_pre_num_days=0,
        holiday_impact_post_num_days=0,
        holiday_impact_dict=None,
        get_suffix_func=None
    ).sort_values(by=[EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL]).reset_index(drop=True)

    assert_equal(expanded_holiday_df, holiday_df)

    # When unknown holidays are present in `holiday_impact_dict`, result remains the same.
    expanded_holiday_df = HolidayGrouper.expand_holiday_df_with_suffix(
        holiday_df=holiday_df,
        holiday_date_col=EVENT_DF_DATE_COL,
        holiday_name_col=EVENT_DF_LABEL_COL,
        holiday_impact_pre_num_days=0,
        holiday_impact_post_num_days=0,
        holiday_impact_dict={"unknown": [1, 1]},
        get_suffix_func=None
    ).sort_values(by=[EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL]).reset_index(drop=True)

    assert_equal(expanded_holiday_df, holiday_df)

    # Tests the case when only neighboring days are added and only through `holiday_impact_pre_num_days` and
    # `holiday_impact_post_num_days`.
    expanded_holiday_df = HolidayGrouper.expand_holiday_df_with_suffix(
        holiday_df=holiday_df,
        holiday_date_col=EVENT_DF_DATE_COL,
        holiday_name_col=EVENT_DF_LABEL_COL,
        holiday_impact_pre_num_days=1,
        holiday_impact_post_num_days=2,
        holiday_impact_dict=None,
        get_suffix_func=None
    ).sort_values(by=[EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL]).reset_index(drop=True)

    # Spot checks a few events are being correctly added.
    assert "Christmas Day_minus_1" in expanded_holiday_df[EVENT_DF_LABEL_COL].tolist()
    assert "New Year's Day_plus_2" in expanded_holiday_df[EVENT_DF_LABEL_COL].tolist()

    # Checks the expected total number of events.
    expected_diff = len(holiday_df) * (1+2)
    assert len(expanded_holiday_df) - len(holiday_df) == expected_diff

    # Tests the case when only neighboring days are added and only through `holiday_impact_dict`.
    expanded_holiday_df = HolidayGrouper.expand_holiday_df_with_suffix(
        holiday_df=holiday_df,
        holiday_date_col=EVENT_DF_DATE_COL,
        holiday_name_col=EVENT_DF_LABEL_COL,
        holiday_impact_pre_num_days=0,
        holiday_impact_post_num_days=0,
        holiday_impact_dict=HOLIDAY_IMPACT_DICT,
        get_suffix_func=None
    ).sort_values(by=[EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL]).reset_index(drop=True)

    # Spot checks a few events are being correctly added.
    assert "Christmas Day_minus_4" in expanded_holiday_df[EVENT_DF_LABEL_COL].tolist()
    assert "New Year's Day_plus_4" in expanded_holiday_df[EVENT_DF_LABEL_COL].tolist()

    # Checks the expected total number of events.
    expected_diff = 0
    for event, (pre, post) in HOLIDAY_IMPACT_DICT.items():
        count = (holiday_df[EVENT_DF_LABEL_COL] == event).sum()
        additional_days = (pre + post) * count
        expected_diff += additional_days
    assert len(expanded_holiday_df) - len(holiday_df) == expected_diff

    # Tests the case when neighboring days are added through `holiday_impact_pre_num_days`,
    # `holiday_impact_post_num_days` and `holiday_impact_dict`.
    expanded_holiday_df = HolidayGrouper.expand_holiday_df_with_suffix(
        holiday_df=holiday_df,
        holiday_date_col=EVENT_DF_DATE_COL,
        holiday_name_col=EVENT_DF_LABEL_COL,
        holiday_impact_pre_num_days=8,
        holiday_impact_post_num_days=0,
        holiday_impact_dict=HOLIDAY_IMPACT_DICT,
        get_suffix_func=None
    ).sort_values(by=[EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL]).reset_index(drop=True)

    # Spot checks a few events are being correctly added or not added.
    assert "Veterans Day_minus_8" in expanded_holiday_df[EVENT_DF_LABEL_COL].tolist()
    assert "Veterans Day_plus_1" not in expanded_holiday_df[EVENT_DF_LABEL_COL].tolist()
    assert "New Year's Day_plus_4" in expanded_holiday_df[EVENT_DF_LABEL_COL].tolist()
    assert "New Year's Day_minus_8" not in expanded_holiday_df[EVENT_DF_LABEL_COL].tolist()

    # Checks the expected total number of events.
    expected_diff = len(holiday_df[~holiday_df[EVENT_DF_LABEL_COL].isin(HOLIDAY_IMPACT_DICT.keys())]) * 8
    for event, (pre, post) in HOLIDAY_IMPACT_DICT.items():
        count = (holiday_df[EVENT_DF_LABEL_COL] == event).sum()
        additional_days = (pre + post) * count
        expected_diff += additional_days
    assert len(expanded_holiday_df) - len(holiday_df) == expected_diff

    # Tests the case when both neighboring days and suffixes are added.
    expanded_holiday_df = HolidayGrouper.expand_holiday_df_with_suffix(
        holiday_df=holiday_df,
        holiday_date_col=EVENT_DF_DATE_COL,
        holiday_name_col=EVENT_DF_LABEL_COL,
        holiday_impact_pre_num_days=0,
        holiday_impact_post_num_days=0,
        holiday_impact_dict=HOLIDAY_IMPACT_DICT,
        get_suffix_func=get_weekday_weekend_suffix
    ).sort_values(by=[EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL]).reset_index(drop=True)

    # Checks an instance where New Year's Day falls on Friday.
    idx = expanded_holiday_df["event_name"].str.contains("New Year's Day_WD_plus_1_WE")
    assert expanded_holiday_df.loc[idx, EVENT_DF_DATE_COL].values[0] == pd.to_datetime("2010-01-02")
    assert expanded_holiday_df.loc[idx, EVENT_DF_DATE_COL].values[1] == pd.to_datetime("2016-01-02")

    # Checks an instance where Christmas Day falls on Sunday and is observed on Monday.
    idx = expanded_holiday_df["event_name"].str.contains("Christmas Day_WE_plus_1_WD")
    assert expanded_holiday_df.loc[idx, EVENT_DF_DATE_COL].values[0] == pd.to_datetime("2011-12-26")
    assert expanded_holiday_df.loc[idx, EVENT_DF_DATE_COL].values[1] == pd.to_datetime("2016-12-26")

    # Checks an instance where Labor Day always fall on Monday for all years.
    idx = expanded_holiday_df["event_name"].str.contains("Labor Day_WD_minus_1_WE")
    assert idx.sum() == 7

    # Tests unknown `get_suffix_func`.
    with pytest.raises(NotImplementedError, match="is not supported"):
        HolidayGrouper.expand_holiday_df_with_suffix(
            holiday_df=holiday_df,
            holiday_date_col=EVENT_DF_DATE_COL,
            holiday_name_col=EVENT_DF_LABEL_COL,
            holiday_impact_pre_num_days=0,
            holiday_impact_post_num_days=0,
            holiday_impact_dict=None,
            get_suffix_func="unknown"
        )


def test_holiday_grouper_init(daily_df, holiday_df):
    """Tests the initialization of `HolidayGrouper`."""
    # Tests pre-processing.
    hg = HolidayGrouper(
        df=daily_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        holiday_df=holiday_df,
        holiday_date_col=EVENT_DF_DATE_COL,
        holiday_name_col=EVENT_DF_LABEL_COL,
        holiday_impact_pre_num_days=0,
        holiday_impact_post_num_days=0,
        holiday_impact_dict=None,
        get_suffix_func=None
    )
    # After initialization, a new column (is not already exists)
    # will be added to `holiday_df` and `expanded_holiday_df`.
    assert HOLIDAY_NAME_COL in hg.holiday_df.columns
    assert HOLIDAY_NAME_COL in hg.expanded_holiday_df.columns


def test_group_holidays(daily_df, holiday_df):
    """Tests `get_holiday_scores` and `group_holidays` functions."""
    default_get_suffix_func = "wd_we"
    default_baseline_offsets = (-7, 7)
    default_use_relative_score = True

    # Initializes the holiday grouper.
    hg = HolidayGrouper(
        df=daily_df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        holiday_df=holiday_df,
        holiday_date_col=EVENT_DF_DATE_COL,
        holiday_name_col=EVENT_DF_LABEL_COL,
        holiday_impact_pre_num_days=0,
        holiday_impact_post_num_days=0,
        holiday_impact_dict=HOLIDAY_IMPACT_DICT,
        get_suffix_func=default_get_suffix_func
    )
    assert hg.get_suffix_func == default_get_suffix_func
    assert hg.baseline_offsets is None
    assert hg.use_relative_score is None

    # Runs the holiday grouper with KDE-based clustering.
    min_n_days = 2
    min_same_sign_ratio = 0.66
    min_abs_avg_score = 0.02
    bandwidth_multiplier = 0.2

    hg.group_holidays(
        min_n_days=min_n_days,
        min_same_sign_ratio=min_same_sign_ratio,
        min_abs_avg_score=min_abs_avg_score,
        clustering_method="kde",
        bandwidth_multiplier=bandwidth_multiplier
    )

    # Checks the attributes are overriden.
    assert hg.baseline_offsets == default_baseline_offsets
    assert hg.use_relative_score == default_use_relative_score
    assert hg.clustering_method == "kde"
    assert hg.result_dict is not None
    assert hg.bandwidth is not None
    assert hg.bandwidth_multiplier == bandwidth_multiplier
    assert hg.kde is not None

    # Checks correctness of the grouping results.
    result_dict = hg.result_dict.copy()

    expected_keys = [
        "holiday_inferrer",
        "score_result_original",
        "score_result_avg_original",
        "score_result",
        "score_result_avg",
        "daily_event_df_dict_with_score",
        "daily_event_df_dict",
        "kde_cutoffs",
        "kde_res",
        "kde_plot"
    ]
    for key in expected_keys:
        assert result_dict[key] is not None

    assert len(result_dict["score_result_original"]) == len(result_dict["score_result_avg_original"])
    assert len(result_dict["score_result"]) == len(result_dict["score_result_avg"])

    # Checks if the pruning works as expected.
    for key, value in result_dict["score_result_avg"].items():
        assert abs(result_dict["score_result_avg"][key]) >= min_abs_avg_score
        assert len(result_dict["score_result"][key]) >= min_n_days

    # Checks the grouped holidays output.
    for event_df in result_dict["daily_event_df_dict"].values():
        assert event_df.shape[1] == 2
        assert EVENT_DF_DATE_COL in event_df.columns
        assert EVENT_DF_LABEL_COL in event_df.columns

    for event_df in result_dict["daily_event_df_dict_with_score"].values():
        assert event_df.shape[1] == 4
        assert EVENT_DF_DATE_COL in event_df.columns
        assert EVENT_DF_LABEL_COL in event_df.columns

    assert len(result_dict["daily_event_df_dict"]) == 4

    # Runs again with a bigger bandwidth for clustering.
    new_bandwidth_multiplier = 1

    hg.group_holidays(
        min_n_days=min_n_days,
        min_same_sign_ratio=min_same_sign_ratio,
        min_abs_avg_score=min_abs_avg_score,
        clustering_method="kde",
        bandwidth_multiplier=new_bandwidth_multiplier
    )
    assert hg.bandwidth_multiplier == new_bandwidth_multiplier

    # Checks results.
    new_result_dict = hg.result_dict.copy()

    # New grouping has fewer groups due to a relaxed bandwidth.
    assert len(new_result_dict["daily_event_df_dict"]) == 3

    # Scoring results have not changed.
    for key in [
        "score_result",
        "score_result_avg"
    ]:
        assert new_result_dict[key] == result_dict[key]

    # Grouping results changed.
    for key in [
        "daily_event_df_dict_with_score",
        "daily_event_df_dict",
        "kde_cutoffs",
        "kde_res",
        "kde_plot"
    ]:
        with pytest.raises(AssertionError):
            assert_equal(new_result_dict[key], result_dict[key])

    # Runs holiday grouper with k-means clustering.
    n_clusters = 5
    hg.group_holidays(
        min_n_days=min_n_days,
        min_same_sign_ratio=min_same_sign_ratio,
        min_abs_avg_score=min_abs_avg_score,
        clustering_method="kmeans",
        n_clusters=n_clusters,
        include_diagnostics=True
    )

    # Checks the attributes are overriden.
    assert hg.baseline_offsets == default_baseline_offsets
    assert hg.use_relative_score == default_use_relative_score
    assert hg.clustering_method == "kmeans"
    assert hg.result_dict is not None
    assert hg.n_clusters == n_clusters
    assert hg.kmeans is not None

    # Checks results.
    new_result_dict = hg.result_dict.copy()
    expected_keys = [
        "holiday_inferrer",
        "score_result_original",
        "score_result_avg_original",
        "score_result",
        "score_result_avg",
        "daily_event_df_dict_with_score",
        "daily_event_df_dict",
        "kmeans_diagnostics",
        "kmeans_plot"
    ]
    for key in expected_keys:
        assert new_result_dict[key] is not None
    # Checks the number of groups matches the input.
    assert len(new_result_dict["daily_event_df_dict"]) == n_clusters

    # Checks invalid clustering method.
    # Tests unknown `get_suffix_func`.
    with pytest.raises(NotImplementedError, match="is not supported"):
        hg.group_holidays(
            min_n_days=min_n_days,
            min_same_sign_ratio=min_same_sign_ratio,
            min_abs_avg_score=min_abs_avg_score,
            clustering_method="unknown",
            bandwidth_multiplier=new_bandwidth_multiplier
        )
