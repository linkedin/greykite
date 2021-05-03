import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV

from greykite.algo.common.model_summary_utils import Bootstrapper
from greykite.algo.common.model_summary_utils import format_summary_df
from greykite.algo.common.model_summary_utils import get_info_dict_lm
from greykite.algo.common.model_summary_utils import print_summary
from greykite.algo.common.model_summary_utils import process_intercept
from greykite.algo.common.model_summary_utils import round_numbers


def test_process_intercept():
    x = np.array([[1, 2], [1, 3], [1, 4]])
    beta = np.array([0, 1])
    intercept = 1
    pred_cols = ["Intercept", "x"]
    # tests all with intercept term
    x_new, beta_new, pred_cols_new = process_intercept(
        x=x,
        beta=beta,
        intercept=intercept,
        pred_cols=pred_cols)
    assert np.array_equal(x_new, x)
    assert np.array_equal(beta_new, np.array([1, 1]))
    assert pred_cols_new == pred_cols
    # x does not have intercept term
    x_no_intercept = np.array([[2], [3], [4]])
    x_new, beta_new, pred_cols_new = process_intercept(
        x=x_no_intercept,
        beta=beta,
        intercept=intercept,
        pred_cols=pred_cols)
    assert np.array_equal(x_new, x)
    assert np.array_equal(beta_new, np.array([1, 1]))
    assert pred_cols_new == pred_cols
    # beta does not have intercept term either
    beta_no_intercept = np.array([1])
    x_new, beta_new, pred_cols_new = process_intercept(
        x=x_no_intercept,
        beta=beta_no_intercept,
        intercept=intercept,
        pred_cols=pred_cols)
    assert np.array_equal(x_new, x)
    assert np.array_equal(beta_new, np.array([1, 1]))
    assert pred_cols_new == pred_cols
    # pred_col does not have intercept term
    pred_cols_no_intercept = ["x"]
    x_new, beta_new, pred_cols_new = process_intercept(
        x=x_no_intercept,
        beta=beta_no_intercept,
        intercept=intercept,
        pred_cols=pred_cols_no_intercept)
    assert np.array_equal(x_new, x)
    assert np.array_equal(beta_new, np.array([1, 1]))
    assert pred_cols_new == pred_cols
    # tests error
    x = np.array([[1, 2], [1, 3], [1, 4]])
    beta = np.array([0, 1, 2])
    intercept = 1
    pred_cols = ["Intercept", "x"]
    with pytest.raises(
            ValueError,
            match="The shape of x and beta do not match. "
                  f"x has shape \\(3, 2\\) with the intercept column, "
                  f"but beta has length 3."):
        process_intercept(
            x=x,
            beta=beta,
            intercept=intercept,
            pred_cols=pred_cols)
    # tests second error
    x = np.array([[1, 2], [1, 3], [1, 4]])
    beta = np.array([0, 1])
    intercept = 1
    pred_cols = ["(Intercept)", "x", "x2"]
    with pytest.raises(
            ValueError,
            match="The length of pred_cols does not match the shape of x. "
                  f"x has shape \\(3, 2\\) with the intercept column, "
                  f"but pred_cols has length 3."):
        process_intercept(
            x=x,
            beta=beta,
            intercept=intercept,
            pred_cols=pred_cols)


def test_round_numbers():
    numbers = [0.0000043152, -1.889, 614003, 0]
    formatted_numbers = round_numbers(numbers, 3)
    assert formatted_numbers[0] == "4.320e-06"
    assert formatted_numbers[1] == "-1.89"
    assert formatted_numbers[2] == "6.140e+05"
    assert formatted_numbers[3] == "0."


def test_format_summary_df():
    # tests short column names
    summary_df = pd.DataFrame({
        "Pred_col": ["Intercept", "x"],
        "Estimate": [123456, 0.00545],
        "Std. Err": [12346, 0.00072],
        "t value": [14.588889, 3.6799934],
        "z value": [14.588889, 3.6799934],
        "Pr(>|t|)": [1e-18, 0.00066],
        "Pr(>|Z|)": [1e-18, 0.00066],
        "Pr(>)_boot": [1e-18, 0.00066],
        "Pr(>)_split": [1e-18, 0.00066],
        "sig. code": ["***", "***"],
        "95%CI": [[120000, 130000], [0.005, 0.006]],
        "Feature importance": [123456, 0.00545]})
    formatted_summary_df = format_summary_df(summary_df)
    assert formatted_summary_df.equals(
        pd.DataFrame({
            "Pred_col": ["Intercept", "x"],
            "Estimate": ["1.235e+05", "0.00545"],
            "Std. Err": ["1.235e+04", "0.00072"],
            "t value": ["14.59", "3.68"],
            "z value": ["14.59", "3.68"],
            "Pr(>|t|)": ["<2e-16", "6.60e-04"],
            "Pr(>|Z|)": ["<2e-16", "6.60e-04"],
            "Pr(>)_boot": ["<2e-16", "6.60e-04"],
            "Pr(>)_split": ["<2e-16", "6.60e-04"],
            "sig. code": ["***", "***"],
            "95%CI": [("1.200e+05", "1.300e+05"), ("0.005", "0.006")],
            "Feature importance": ["1.235e+05", "0.00545"]}))
    # tests long column names
    summary_df = pd.DataFrame({
        "Pred_col": ["Intercept", "i_have_longer_than_20_characters"],
        "Estimate": [123456, 0.00545],
        "Std. Err": [12346, 0.00072],
        "t value": [14.588889, 3.6799934],
        "Pr(>|t|)": [1e-18, 0.00066],
        "sig. code": ["***", "***"],
        "95%CI": [[120000, 130000], [0.005, 0.006]]
    })
    formatted_summary_df = format_summary_df(summary_df)
    assert formatted_summary_df.equals(
        pd.DataFrame({
            "Pred_col": ["Intercept", "i_have_l...aracters"],
            "Estimate": ["1.235e+05", "0.00545"],
            "Std. Err": ["1.235e+04", "0.00072"],
            "t value": ["14.59", "3.68"],
            "Pr(>|t|)": ["<2e-16", "6.60e-04"],
            "sig. code": ["***", "***"],
            "95%CI": [("1.200e+05", "1.300e+05"), ("0.005", "0.006")]}))


def test_get_bootstrapper():
    bootstrap_idx = Bootstrapper(
        sample_size=100,
        bootstrap_size=100,
        num_bootstrap=5)
    idx = []
    for i in bootstrap_idx:
        idx.append(i)
    assert len(idx) == 5  # 5 bootstrap samples
    assert max([max(i) for i in idx]) < 100  # max index is less than 10
    assert len(set(idx[0])) < 100  # draw with replacement


def test_degenerate_columns():
    x = np.random.randn(100, 2)
    x = np.concatenate([np.ones([100, 1]), x, np.ones([100, 2])], axis=1)  # the last two columns are degenerate
    beta = np.random.rand(5) * 10
    y = x @ beta + np.random.randn(100)
    ml_model = RidgeCV().fit(x, y)
    pred_cols = ["Intercept", "x1", "x2", "x3", "x4"]
    fit_algorithm = "ridge"
    beta = ml_model.coef_
    beta[0] += ml_model.intercept_
    info_dict = get_info_dict_lm(
        x=x,
        y=y,
        beta=beta,
        ml_model=ml_model,
        fit_algorithm=fit_algorithm,
        pred_cols=pred_cols)
    assert info_dict["beta_var_cov"].shape == (3, 3)
    print_summary(info_dict)
