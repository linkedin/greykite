import pytest

from greykite.algo.uncertainty.conditional.conf_interval import conf_interval
from greykite.algo.uncertainty.conditional.conf_interval import predict_ci
from greykite.common.constants import ERR_STD_COL
from greykite.common.testing_utils import gen_sliced_df


@pytest.fixture
def data():
    """Generate data for tests"""
    df = gen_sliced_df()
    df = df[["x", "z_categ", "y", "residual"]]
    new_df = df.iloc[[1, 100, 150, 200, 250, 300, 305, 400, 405, 500, 550, 609]].copy()
    return {"df": df, "new_df": new_df}


def test_conf_interval_ecdf_method(data):
    """Testing "conf_interval" function with "ecdf" method
    """
    df = data["df"]
    new_df = data["new_df"]

    # ``quantile_estimation_method = "ecdf"``
    ci_model = conf_interval(
        df=df,
        value_col="y",
        residual_col="residual",
        conditional_cols=["x"],
        quantiles=[0.005, 0.025, 0.975, 0.995],
        quantile_estimation_method="ecdf",
        sample_size_thresh=5,
        small_sample_size_method="std_quantiles",
        small_sample_size_quantile=0.95,
        min_admissible_value=None,
        max_admissible_value=None)

    pred_df = predict_ci(
        new_df,
        ci_model)

    assert list(pred_df.columns) == ["x", "y_quantile_summary", ERR_STD_COL], (
        "pred_df does not have the expected column names")
    pred_df["y_quantile_summary"] = pred_df["y_quantile_summary"].apply(
        lambda x: tuple(round(e, 2) for e in x))
    pred_df[ERR_STD_COL] = round(pred_df[ERR_STD_COL], 2)
    assert pred_df["y_quantile_summary"].values[5] == (289.32, 289.38, 291.3, 291.34), (
        "quantiles are incorrect")
    assert pred_df["y_quantile_summary"].values[11] == (-5.63, -5.56, -4.13, -4.08), (
        "quantiles are incorrect")
    expected_stds = [0.29, 0.42, 0.42, 0.42, 0.42, 0.58, 0.58, 0.58, 0.58, 0.58,
                     0.58, 0.42]
    assert list(pred_df[ERR_STD_COL].values) == expected_stds


def test_conf_interval_normal_method(data):
    """Testing "conf_interval" function, normal method"""
    df = data["df"]
    new_df = data["new_df"]
    # ``quantile_estimation_method = "normal_fit"``
    ci_model = conf_interval(
        df=df,
        value_col="y",
        residual_col="residual",
        conditional_cols=["x"],
        quantiles=[0.005, 0.025, 0.975, 0.995],
        quantile_estimation_method="normal_fit",
        sample_size_thresh=5,
        small_sample_size_method="std_quantiles",
        small_sample_size_quantile=0.95,
        min_admissible_value=None,
        max_admissible_value=None)

    pred_df = predict_ci(
        new_df,
        ci_model)
    assert list(pred_df.columns) == ["x", "y_quantile_summary", ERR_STD_COL], (
        "pred_df does not have the expected column names")
    pred_df["y_quantile_summary"] = pred_df["y_quantile_summary"].apply(
        lambda x: tuple(round(e, 2) for e in x))
    assert pred_df["y_quantile_summary"].values[5] == (289.9, 290.25, 292.54, 292.9), (
        "quantiles are incorrect")
    assert pred_df["y_quantile_summary"].values[11] == (-5.14, -4.88, -3.24, -2.98), (
        "quantiles are incorrect")


def test_conf_interval_normal_method_with_bounds(data):
    """Testing "conf_interval" function, normal method"""
    df = data["df"]
    new_df = data["new_df"]
    # ``quantile_estimation_method = "normal_fit"``
    # with enforced lower limit (``min_admissible_value``)
    ci_model = conf_interval(
        df=df,
        value_col="y",
        residual_col="residual",
        conditional_cols=["x"],
        quantiles=[0.005, 0.025, 0.975, 0.995],
        quantile_estimation_method="normal_fit",
        sample_size_thresh=5,
        small_sample_size_method="std_quantiles",
        small_sample_size_quantile=0.95,
        min_admissible_value=290.0,
        max_admissible_value=None)

    pred_df = predict_ci(
        new_df,
        ci_model)
    assert list(pred_df.columns) == ["x", "y_quantile_summary", ERR_STD_COL], (
        "pred_df does not have the expected column names")
    pred_df["y_quantile_summary"] = pred_df["y_quantile_summary"].apply(
        lambda x: tuple(round(e, 2) for e in x))
    assert pred_df["y_quantile_summary"].values[5] == (290.0, 290.25, 292.54, 292.9), (
        "quantiles are incorrect")
    assert pred_df["y_quantile_summary"].values[11] == (290.0, 290.0, 290.0, 290.0), (
        "quantiles are incorrect")


def test_conf_interval_normal_method_fallback(data):
    """Testing "conf_interval" function, normal method,
    no slices have enough samples"""
    df = data["df"]
    df = df.sample(n=10)
    new_df = data["new_df"]

    # ``quantile_estimation_method = "normal_fit"``
    # fallback expected for all slices as df is small (10)
    # and ``sample_size_thresh`` is large (20)
    with pytest.warns(Warning):
        ci_model = conf_interval(
            df=df,
            value_col="y",
            residual_col="residual",
            conditional_cols=["x"],
            quantiles=[0.005, 0.025, 0.975, 0.995],
            quantile_estimation_method="normal_fit",
            sample_size_thresh=20,
            small_sample_size_method="std_quantiles",
            small_sample_size_quantile=0.95,
            min_admissible_value=None,
            max_admissible_value=None)

    pred_df = predict_ci(
        new_df,
        ci_model)
    assert list(pred_df.columns) == ["x", "y_quantile_summary", ERR_STD_COL], (
        "pred_df does not have the expected column names")
    pred_df["y_quantile_summary"] = pred_df["y_quantile_summary"].apply(
        lambda x: tuple(round(e, 2) for e in x))
    assert pred_df["y_quantile_summary"].values[5] == (290.31, 290.57, 292.23, 292.49), (
        "quantiles are incorrect")
    assert pred_df["y_quantile_summary"].values[11] == (-5.15, -4.89, -3.23, -2.97), (
        "quantiles are incorrect")


def test_conf_interval_normal_method_multivar_conditionals(data):
    """Testing ``conf_interval`` function, normal method,
    multivariate conditional columns
    """
    df = data["df"]
    new_df = data["new_df"]
    # ``quantile_estimation_method = "normal_fit"``
    # with multi-variate ``conditional_cols``
    ci_model = conf_interval(
        df=df,
        value_col="y",
        residual_col="residual",
        conditional_cols=["x", "z_categ"],
        quantiles=[0.005, 0.025, 0.975, 0.995],
        quantile_estimation_method="normal_fit",
        sample_size_thresh=5,
        small_sample_size_method="std_quantiles",
        small_sample_size_quantile=0.95,
        min_admissible_value=None,
        max_admissible_value=None)

    pred_df = predict_ci(
        new_df,
        ci_model)
    assert list(pred_df.columns) == ["x", "z_categ", "y_quantile_summary", ERR_STD_COL], (
        "pred_df does not have the expected column names")
    pred_df["y_quantile_summary"] = pred_df["y_quantile_summary"].apply(
        lambda x: tuple(round(e, 2) for e in x))
    assert pred_df["y_quantile_summary"].values[5] == (289.9, 290.26, 292.54, 292.9), (
        "quantiles are incorrect")
    assert pred_df["y_quantile_summary"].values[11] == (-5.15, -4.89, -3.23, -2.97), (
        "quantiles are incorrect")


def test_conf_interval_normal_method_no_conditionals(data):
    """Testing "conf_interval" function, normal method, with no conditioning."""
    df = data["df"]
    new_df = data["new_df"]
    # ``quantile_estimation_method = "normal_fit"``;
    # with no ``conditional_cols``
    ci_model = conf_interval(
        df=df,
        value_col="y",
        residual_col="residual",
        conditional_cols=None,
        quantiles=[0.005, 0.025, 0.975, 0.995],
        quantile_estimation_method="normal_fit",
        sample_size_thresh=5,
        small_sample_size_method="std_quantiles",
        small_sample_size_quantile=0.95,
        min_admissible_value=None,
        max_admissible_value=None)

    pred_df = predict_ci(
        new_df,
        ci_model)
    assert list(pred_df.columns) == ["y_quantile_summary", ERR_STD_COL], (
        "pred_df does not have the expected column names")
    pred_df["y_quantile_summary"] = pred_df["y_quantile_summary"].apply(
        lambda x: tuple(round(e, 2) for e in x))
    assert pred_df["y_quantile_summary"].values[5] == (290.05, 290.37, 292.42, 292.74), (
        "quantiles are incorrect")
    assert pred_df["y_quantile_summary"].values[11] == (-5.41, -5.08, -3.04, -2.72), (
        "quantiles are incorrect")


def test_conf_interval_normal_method_no_small_sample_calc(data):
    """Testing "conf_interval" function, normal method,
       no small sample size calculation"""
    df = data["df"]
    new_df = data["new_df"]
    # ``quantile_estimation_method = "normal_fit"``;
    # with no small sample size calculation
    ci_model = conf_interval(
        df=df,
        value_col="y",
        residual_col="residual",
        conditional_cols=["x"],
        quantiles=[0.005, 0.025, 0.975, 0.995],
        quantile_estimation_method="normal_fit",
        sample_size_thresh=None,
        small_sample_size_method=None,
        small_sample_size_quantile=None,
        min_admissible_value=None,
        max_admissible_value=None)

    pred_df = predict_ci(
        new_df,
        ci_model)
    assert list(pred_df.columns) == ["x", "y_quantile_summary", ERR_STD_COL], (
        "pred_df does not have the expected column names")
    pred_df["y_quantile_summary"] = pred_df["y_quantile_summary"].apply(
        lambda x: tuple(round(e, 2) for e in x))
    assert pred_df["y_quantile_summary"].values[5] == (289.9, 290.25, 292.54, 292.9), (
        "quantiles are incorrect")
    assert pred_df["y_quantile_summary"].values[11] == (-5.64, -5.26, -2.86, -2.49), (
        "quantiles are incorrect")


def test_conf_interval_normal_method_exception(data):
    """Testing "conf_interval" function, non-existing small sample method"""
    df = data["df"]
    # non-implemented ``small_sample_size_method``
    with pytest.raises(
            Exception,
            match="small_sample_size_method non-implemented-method is not implemented."):
        conf_interval(
            df=df,
            value_col="y",
            residual_col="residual",
            conditional_cols=None,
            quantiles=[0.005, 0.025, 0.975, 0.995],
            quantile_estimation_method="normal_fit",
            sample_size_thresh=5,
            small_sample_size_method="non-implemented-method",
            small_sample_size_quantile=0.95,
            min_admissible_value=None,
            max_admissible_value=None)
