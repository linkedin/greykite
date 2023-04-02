import pandas as pd

from greykite.algo.common.holiday_utils import get_dow_grouped_suffix
from greykite.algo.common.holiday_utils import get_weekday_weekend_suffix


def test_get_dow_grouped_suffix():
    """Tests `get_dow_grouped_suffix` function."""
    date = pd.to_datetime("2023-01-01")
    assert get_dow_grouped_suffix(date) == "_Sun"

    date = pd.to_datetime("2023-01-02")
    assert get_dow_grouped_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-03")
    assert get_dow_grouped_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-04")
    assert get_dow_grouped_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-05")
    assert get_dow_grouped_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-06")
    assert get_dow_grouped_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-07")
    assert get_dow_grouped_suffix(date) == "_Sat"


def test_get_weekday_weekend_suffix():
    """Tests `get_weekday_weekend_suffix` function."""
    date = pd.to_datetime("2023-01-01")
    assert get_weekday_weekend_suffix(date) == "_WE"

    date = pd.to_datetime("2023-01-02")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-03")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-04")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-05")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-06")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-07")
    assert get_weekday_weekend_suffix(date) == "_WE"
