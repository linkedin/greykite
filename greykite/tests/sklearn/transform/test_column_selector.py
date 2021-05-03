"""
Test for column_selector.py
"""
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from greykite.sklearn.transform.column_selector import ColumnSelector


@pytest.fixture
def data():
    """
    Generates iris dataset for test cases
    :return:
    """
    return load_iris(return_X_y=True)[0]


def test_column_selector(data):
    """
    Checks if column selector works
    :param data: iris dataset from pytest.fixture
    """
    df = pd.DataFrame(data, columns=list("abcd"))
    columns = ["a", "d"]
    selector = ColumnSelector(columns)
    assert selector.column_names == columns

    result = selector.fit_transform(df)
    assert result.equals(result[columns])


def test_column_selector_empty(data):
    """
    Checks if column selector works
    :param data: iris dataset from pytest.fixture
    """
    df = pd.DataFrame(data, columns=list("abcd"))
    columns = []
    selector = ColumnSelector(columns)

    result = selector.fit_transform(df)
    assert result.shape == (df.shape[0], 0)
    assert selector.column_names == columns
