"""
Test for dtype_column_selector.py
"""
import pytest

from greykite.common.constants import TIME_COL
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.sklearn.transform.dtype_column_selector import DtypeColumnSelector


@pytest.fixture
def data():
    """Generates dataset for test cases
    :return: pd.DataFrame with columns of type:
        datetime, number, number, boolean, object, category
    """
    df = generate_df_with_reg_for_tests(
        freq="D",
        periods=50,
        remove_extra_cols=False)["df"]
    df["dow_categorical"] = df["str_dow"].astype("category")
    df = df[[TIME_COL, "regressor1", "regressor2", "regressor_bool", "str_dow", "dow_categorical"]]
    return df


def test_column_selector_numeric(data):
    """Tests if column selector works on numeric"""
    cols = ["regressor1", "regressor2"]
    selector = DtypeColumnSelector(include="number")
    assert selector.include == "number"
    assert selector.exclude is None

    result = selector.fit_transform(data)
    assert result.equals(data[cols])


def test_column_selector_bool(data):
    """Tests if column selector works on boolean"""
    cols = ["regressor_bool"]
    selector = DtypeColumnSelector(include="bool")
    assert selector.include == "bool"
    assert selector.exclude is None

    result = selector.fit_transform(data)
    assert result.equals(data[cols])


def test_column_selector_datetime(data):
    """Tests if column selector works on datetime"""
    cols = [TIME_COL]
    selector = DtypeColumnSelector(include="datetime")
    assert selector.include == "datetime"
    assert selector.exclude is None

    result = selector.fit_transform(data)
    assert result.equals(data[cols])


def test_column_selector_object(data):
    """Tests if column selector works on string"""
    cols = ["str_dow"]
    selector = DtypeColumnSelector(include="object")
    assert selector.include == "object"
    assert selector.exclude is None

    result = selector.fit_transform(data)
    assert result.equals(data[cols])


def test_column_selector_categorical(data):
    """Tests if column selector works on categorical"""
    cols = ["dow_categorical"]
    selector = DtypeColumnSelector(include="category")
    assert selector.include == "category"
    assert selector.exclude is None

    result = selector.fit_transform(data)
    assert result.equals(data[cols])


def test_column_selector_exclude(data):
    """Tests column selector exclude parameter"""
    cols = ["regressor1", "regressor2"]
    selector = DtypeColumnSelector(exclude="number")
    assert selector.include is None
    assert selector.exclude == "number"

    result = selector.fit_transform(data)
    assert result.equals(data.drop(cols, axis=1))


def test_column_selector_multiple(data):
    """Tests if column selector works on multiple types"""
    cols = ["regressor1", "regressor2", "dow_categorical"]
    selector = DtypeColumnSelector(include=["number", "category"], exclude=["bool"])
    assert selector.include == ["number", "category"]
    assert selector.exclude == ["bool"]

    result = selector.fit_transform(data)
    assert result.equals(data[cols])
