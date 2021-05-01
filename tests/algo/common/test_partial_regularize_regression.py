import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from greykite.algo.common.partial_regularize_regression import PartialRegularizeRegression
from greykite.algo.common.partial_regularize_regression import PartialRegularizeRegressionCV
from greykite.algo.common.partial_regularize_regression import constant_col_finder
from greykite.common.python_utils import assert_equal


def test_constant_col_finder():
    x = np.concatenate([np.zeros([100, 2]), np.ones([100, 2]), np.ones([100, 1]) * 2, np.random.randn(100, 5)], axis=1)
    constant_cols = constant_col_finder(x, exclude_cols=[3])
    assert constant_cols == [2, 4]


def test_partial_regularize_regression():
    np.random.seed(123)
    x = np.random.randn(1000, 10) * 10 + 1
    beta = np.random.randn(10) * 20
    zero_index = [1, 2]
    beta[zero_index] = 0
    intercept = 5
    y = x @ beta + intercept

    # Tests functionality.
    model = PartialRegularizeRegression()
    model.fit(x, y)
    model.predict(x)

    # Tests mixed regularization.
    model = PartialRegularizeRegression(
        l1_index=[0, 1, 2, 3, 4],
        l2_index=[5, 6, 7],
        l1_alpha=1,
        l2_alpha=0.1,
    )
    model.fit(x[:800], y[:800])
    # Asserts prediction is ok.
    assert_equal(model.predict(x[800:]), y[800:], rel=1e-1)
    # No constants detected.
    assert model.has_constant == []
    # L1 norm penalized features are sparse.
    assert any(model.coef_[[0, 1, 2, 3, 4]] == 0)
    # L2 norm penalized and non-penalized features are not sparse.
    assert all(model.coef_[[5, 6, 7, 8, 9]] != 0)

    # Tests l1 index only.
    model = PartialRegularizeRegression(
        l1_index=list(range(x.shape[1])),
    )
    model.fit(x, y)

    # Tests l2 index only.
    model = PartialRegularizeRegression(
        l2_index=list(range(x.shape[1])),
    )
    model.fit(x, y)

    # Tests with constants.
    xc = np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)
    model = PartialRegularizeRegression(
        l1_index=[0, 1, 2]
    )
    model.fit(xc, y)
    model.predict(xc)
    assert model.has_constant == [10]

    # Tests input pandas DataFrame.
    model.fit(pd.DataFrame(x), pd.Series(y))
    model.predict(x)

    # Tests the parameter of unpenalized parameters is greater in magnitude than Ridge.
    model = PartialRegularizeRegression(
        l2_index=[0, 1, 2, 3, 4, 5, 6],
        l2_alpha=0.1
    )
    model.fit(x, y)
    coef = model.coef_
    ridge = Ridge(
        alpha=0.1
    )
    ridge.fit(x, y)
    ridge_coef = ridge.coef_
    assert (np.abs(coef[7:]) >= np.abs(ridge_coef[7:])).all()

    # Tests CV
    model = PartialRegularizeRegressionCV(
        l1_index=[0, 1, 2, 3, 4, 5, 6],
        l2_index=[9],
    )
    model.fit(x[:800], y[:800])
    # It identifies the correct zero sets.
    assert all(model.coef_[zero_index] == 0)
    assert all(model.coef_[[i for i in range(x.shape[1]) if i not in zero_index]] != 0)
    # Asserts prediction is ok.
    assert_equal(model.predict(x[800:]), y[800:], rel=1e-1)

    # Test correlated case.
    np.random.seed(123)
    x = np.random.randn(1000, 2) * 10 + 1
    u = np.random.randn(1000, 1) * 100
    x += u  # Around 99% correlation.
    beta = np.array([20, 20])
    intercept = 5
    y = x @ beta + intercept
    # Huge penalization on the second beta.
    model = PartialRegularizeRegressionCV(
        l1_index=[1],
        l1_alphas=[10000]
    )
    model.fit(x, y)
    assert_equal(model.coef_[0], 40, rel=1e-2)
    assert model.coef_[1] == 0
    # No regularization.
    model = PartialRegularizeRegressionCV()
    model.fit(x, y)
    assert_equal(model.coef_[0], 20, rel=1e-2)
    assert_equal(model.coef_[1], 20, rel=1e-2)
