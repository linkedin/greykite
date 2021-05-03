import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.sklearn.estimator.null_model import DummyEstimator


def test_init():
    """Tests model initialization"""
    model = DummyEstimator()
    assert model.strategy == "mean"
    assert model.constant is None
    assert model.quantile is None
    assert model.score_func == mean_squared_error
    assert model.coverage is None

    model = DummyEstimator(strategy="quantile", quantile=0.9, score_func=mean_absolute_error)
    assert model.strategy == "quantile"
    assert model.constant is None
    assert model.quantile == 0.9
    assert model.score_func == mean_absolute_error

    model = DummyEstimator(strategy="median")
    assert model.strategy == "median"
    assert model.constant is None
    assert model.quantile is None
    # No additional variables should be set in init
    assert model.model is None
    assert model.time_col_ is None
    assert model.value_col_ is None

    # set_params must be able to replicate the init
    model2 = DummyEstimator()
    model2.set_params(strategy="median")
    assert model2.__dict__ == model.__dict__


def test_fit_predict():
    """Tests training mean estimator"""
    model = DummyEstimator()
    X = pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=3, freq="1D"),
        VALUE_COL: [2, 3, 4]
    })

    model.fit(X)
    predicted = model.predict(X)

    expected = pd.DataFrame({
        TIME_COL: X[TIME_COL],
        PREDICTED_COL: np.repeat(3.0, X.shape[0])
    })

    assert predicted.equals(expected)

    # with np.nan value
    model = DummyEstimator()
    X = pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=4, freq="1D"),
        VALUE_COL: [2, 3, np.nan, 4]
    })

    model.fit(X)
    predicted = model.predict(X)

    expected = pd.DataFrame({
        TIME_COL: X[TIME_COL],
        PREDICTED_COL: np.repeat(3.0, X.shape[0])
    })

    assert predicted.equals(expected)


def test_fit_predict1():
    """Tests sample_weight parameter and different train/test set"""
    model = DummyEstimator()

    X = pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=3, freq="1D"),
        VALUE_COL: [2, 3, 5]
    })

    df_test = pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=4, freq="1D")
    })

    model.fit(X, sample_weight=[1, 1, 2])
    predicted = model.predict(df_test)

    expected = pd.DataFrame({
        TIME_COL: df_test[TIME_COL],
        PREDICTED_COL: np.repeat(3.75, df_test.shape[0])
    })

    assert predicted.equals(expected)


def test_constant_model():
    """Tests constant model"""
    constant = 1.0
    model = DummyEstimator(strategy="constant", constant=constant)

    X = pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=3, freq="1D"),
        VALUE_COL: [2, 3, 4]
    })

    model.fit(X)
    predicted = model.predict(X)

    expected = pd.DataFrame({
        TIME_COL: X[TIME_COL],
        PREDICTED_COL: np.repeat(constant, X.shape[0])
    })

    assert predicted.equals(expected)


def test_quantile_model():
    """Tests quantile model with custom column names"""
    model = DummyEstimator(strategy="quantile", quantile=0.8)

    X = pd.DataFrame({
        "time_name": pd.date_range("2018-01-01", periods=11, freq="D"),
        "value_name": np.arange(11)
    })

    model.fit(X, time_col="time_name", value_col="value_name")
    predicted = model.predict(X)

    expected = pd.DataFrame({
        TIME_COL: X["time_name"],
        PREDICTED_COL: np.repeat(8.0, X.shape[0])
    })

    assert predicted.equals(expected)


def test_summary():
    """Tests summary function returns without error"""
    model = DummyEstimator(strategy="quantile", constant=None, quantile=0.99)
    model.summary()


def test_score():
    """Tests score function"""
    model = DummyEstimator(strategy="mean", score_func=mean_absolute_error)
    X = pd.DataFrame({
        TIME_COL: pd.date_range("2018-01-01", periods=3, freq="1D"),
        VALUE_COL: [2.0, 3.0, 4.0]
    })
    model.fit(X)  # prediction is 3.0

    y = pd.Series([1.0, 2.0, 3.0])

    assert model.score(X, y) == mean_absolute_error(y, np.repeat(3.0, X.shape[0]))
