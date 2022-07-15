"""Tests the L1 norm regularized quantile regression."""

import numpy as np
import pytest
from testfixtures import LogCapture

from greykite.algo.common.l1_quantile_regression import QuantileRegression
from greykite.algo.common.l1_quantile_regression import l1_quantile_regression
from greykite.algo.common.l1_quantile_regression import ordinary_quantile_regression
from greykite.common.logging import LOGGER_NAME


@pytest.fixture
def data():
    """Sample data for regression."""
    np.random.seed(123)
    x = np.random.randn(200, 20)
    beta = np.array([50] * 18 + [0] * 2)
    alpha = np.random.randn(1)
    y = x @ beta + alpha + np.random.randn(200)
    return {
        "x": x,
        "y": y
    }


def test_ordinary_quantile_regression(data):
    """Tests ordinary quantile regression."""
    coef = ordinary_quantile_regression(
        x=np.concatenate([data["x"], np.ones([len(data["x"]), 1])], axis=1),
        y=data["y"],
        q=0.1,
        sample_weight=np.ones(len(data["y"])),
        max_iter=200,
        tol=1e-3
    )
    pred = np.concatenate([data["x"], np.ones([len(data["x"]), 1])], axis=1) @ coef
    coef1 = ordinary_quantile_regression(
        x=np.concatenate([data["x"], np.ones([len(data["x"]), 1])], axis=1),
        y=data["y"],
        q=0.9,
        sample_weight=np.ones(len(data["y"])),
        max_iter=200,
        tol=1e-3
    )
    pred1 = np.concatenate([data["x"], np.ones([len(data["x"]), 1])], axis=1) @ coef1
    assert all(pred1 > pred)


def test_l1_quantile_regression(data):
    """Tests L1 norm regularized quantile regression."""
    coef = l1_quantile_regression(
        x=data["x"],
        y=data["y"],
        q=0.1,
        alpha=1,
        sample_weight=np.ones(len(data["y"])),
        feature_weight=np.ones(data["x"].shape[1]),
        include_intercept=True
    )
    pred = data["x"] @ coef["coef"] + coef["intercept"]
    coef1 = l1_quantile_regression(
        x=data["x"],
        y=data["y"],
        q=0.9,
        alpha=1,
        sample_weight=np.ones(len(data["y"])),
        feature_weight=np.ones(data["x"].shape[1]),
        include_intercept=True
    )
    pred1 = data["x"] @ coef1["coef"] + coef1["intercept"]
    assert all(pred1 > pred)


def test_quantile_regression_init():
    """Tests the QuantileRegression class instantiating."""
    qr = QuantileRegression()
    assert qr.quantile == 0.9
    assert qr.alpha == 0.001
    assert qr.sample_weight is None
    assert qr.feature_weight is None
    assert qr.max_iter == 100
    assert qr.tol == 1e-2
    assert qr.fit_intercept is True
    assert qr.n is None
    assert qr.p is None
    assert qr.constant_cols is None
    assert qr.nonconstant_cols is None
    assert qr.intercept_ is None
    assert qr.coef_ is None


def test_quantile_regression_fit_predict_ordinary(data):
    """Tests fitting quantile regression fit and predict with no alpha."""
    qr = QuantileRegression(alpha=0)
    qr.fit(data["x"], data["y"])
    pred = qr.predict(data["x"])
    assert round(sum(pred > data["y"]) / len(data["y"]), 1) == 0.9


def test_quantile_regression_fit_predict_l1(data):
    """Tests fitting quantile regression fit and predict with alpha."""
    qr = QuantileRegression(alpha=0.1)
    qr.fit(data["x"], data["y"])
    pred = qr.predict(data["x"])
    assert round(sum(pred > data["y"]) / len(data["y"]), 1) == 0.9


def test_errors(data):
    """Tests errors."""
    # y is not a column vector.
    with LogCapture(LOGGER_NAME) as log_capture:
        qr = QuantileRegression()
        qr.fit(data["x"], data["y"].reshape(-1, 1))
        log_capture.check_present((
            LOGGER_NAME,
            "WARNING",
            "A column-vector y was passed when a 1d array was expected. "
            "Please change the shape of y to (n_samples,), "
            "for example using ravel()."
        ))
    # Shapes of x and y do not match.
    with pytest.raises(
            ValueError,
            match="The shapes of x and y do not match. "):
        qr = QuantileRegression()
        qr.fit(data["x"], data["y"][:5])
    # Quantile not in range.
    with pytest.raises(
            ValueError,
            match="Quantile q must be between 0.0 and 1.0,"):
        qr = QuantileRegression(quantile=-0.1)
        qr.fit(data["x"], data["y"])
    # Sample weight length wrong.
    with pytest.raises(
            ValueError,
            match="The length of sample weight must match the number of observations"):
        qr = QuantileRegression(sample_weight=[1, 2])
        qr.fit(data["x"], data["y"])
    # Feature weight length wrong.
    with pytest.raises(
            ValueError,
            match="The length of feature weight must match the number of features"):
        qr = QuantileRegression(feature_weight=[1, 2])
        qr.fit(data["x"], data["y"])
    # Max iterations not in range.
    with pytest.raises(
            ValueError,
            match="max_iter must be a positive integer,"):
        qr = QuantileRegression(max_iter=0)
        qr.fit(data["x"], data["y"])
    # Tolerance not in range.
    with pytest.raises(
            ValueError,
            match="tol must be a positive number, found"):
        qr = QuantileRegression(tol=-1)
        qr.fit(data["x"], data["y"])
