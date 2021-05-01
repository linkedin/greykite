import pytest

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.python_utils import assert_equal
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS


def test_init():
    dl = DataLoaderTS()
    assert dl.available_datasets == dl.get_data_inventory()


def test_load_peyton_manning_ts():
    dl = DataLoaderTS()
    ts = dl.load_peyton_manning_ts()
    assert ts.original_time_col == TIME_COL
    assert ts.original_value_col == VALUE_COL
    assert ts.freq == "1D"
    assert_equal(ts.df[VALUE_COL], ts.y)


def test_load_hourly_parking_ts():
    dl = DataLoaderTS()
    ts = dl.load_parking_ts(system_code_number=None)
    assert ts.original_time_col == "LastUpdated"
    assert ts.original_value_col == "OccupancyRatio"
    assert ts.freq == "30min"
    assert ts.df.shape == (3666, 4)

    ts = dl.load_parking_ts(system_code_number="NIA South")
    assert ts.original_time_col == "LastUpdated"
    assert ts.original_value_col == "OccupancyRatio"
    assert ts.freq == "30min"
    assert ts.df.shape == (3522, 5)


def test_load_hourly_bikesharing_ts():
    dl = DataLoaderTS()
    ts = dl.load_bikesharing_ts()
    assert ts.original_time_col == "ts"
    assert ts.original_value_col == "count"
    assert ts.freq == "H"
    assert ts.regressor_cols == ["tmin", "tmax", "pn"]
    assert_equal(ts.df[VALUE_COL], ts.y)


def test_load_hourly_beijing_pm_ts():
    dl = DataLoaderTS()
    ts = dl.load_beijing_pm_ts()
    assert ts.original_time_col == TIME_COL
    assert ts.original_value_col == "pm"
    assert ts.freq == "H"
    assert ts.regressor_cols == ["dewp", "temp", "pres", "cbwd", "iws", "is", "ir"]
    assert_equal(ts.df[VALUE_COL], ts.y)


def test_load_data_ts():
    dl = DataLoaderTS()
    ts = dl.load_data_ts(data_name="daily_peyton_manning")
    expected_ts = dl.load_peyton_manning_ts()
    assert_equal(ts.df, expected_ts.df)

    ts = dl.load_data_ts(data_name="hourly_parking", system_code_number="Shopping")
    expected_ts = dl.load_parking_ts(system_code_number="Shopping")
    assert_equal(ts.df, expected_ts.df)

    # Error due to unavailable data name
    data_name = "dummy"
    data_inventory = dl.get_data_inventory()
    with pytest.raises(ValueError, match=fr"Input data name '{data_name}' is not recognized. "
                                         fr"Must be one of \{data_inventory}\."):
        dl.load_data_ts(data_name=data_name)
