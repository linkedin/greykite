import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal

from greykite.algo.common.ml_models import breakdown_regression_based_prediction
from greykite.algo.common.ml_models import design_mat_from_formula
from greykite.algo.common.ml_models import fit_ml_model
from greykite.algo.common.ml_models import fit_ml_model_with_evaluation
from greykite.algo.common.ml_models import fit_model_via_design_matrix
from greykite.algo.common.ml_models import predict_ml
from greykite.algo.common.ml_models import predict_ml_with_uncertainty
from greykite.algo.uncertainty.conditional.conf_interval import predict_ci
from greykite.common.constants import ERR_STD_COL
from greykite.common.constants import QUANTILE_SUMMARY_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import calc_pred_err
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import gen_sliced_df


@pytest.fixture
def design_mat_info():
    """Training data in design matrix form"""
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = 2 * x1
    x3 = np.array([1, 3, 5, 11, 12])
    y = 10 + 0 * x1 + 2 * x2 + 4 * x3

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    model_formula_str = "y ~ x1 + x2 + x3"
    return design_mat_from_formula(
        df,
        model_formula_str,
        pred_cols=None,
        y_col=None)


@pytest.fixture
def data_with_weights():
    """Training data with weights"""
    n = 10000
    np.random.seed(666)
    x1 = np.random.normal(loc=0.0, scale=1.0, size=n)
    x2 = np.random.normal(loc=0.0, scale=1.0, size=n)
    x1 = x1 / np.std(x1)
    x2 = x2 / np.std(x2)
    y = 10 + (5 * x1) + (20 * x2)
    y[(n//2):] = 20 + (-20 * x2[(n//2):])
    w = np.array([0]*(n//2) + [1]*(n//2))
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "w": w})
    model_formula_str = "y ~ -1 + x1 + x2"
    return {
        "design_mat_info": design_mat_from_formula(
            df=df,
            model_formula_str=model_formula_str,
            pred_cols=None,
            y_col=None),
        "df": df,
        "model_formula_str": model_formula_str}


def test_design_mat_from_formula(design_mat_info):
    """Tests design_mat_from_formula"""
    assert design_mat_info["x_mat"]["x1"][0] == 1
    assert design_mat_info["y_col"] == "y"


def test_fit_model_via_design_matrix(design_mat_info):
    """Tests fit_model_via_design_matrix"""
    x_train = design_mat_info["x_mat"]
    y_train = design_mat_info["y"]
    sample_weight = np.array([1, 2, 3, 4, 5])

    # Linear
    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="linear")

    assert ml_model.coef_[0].round() == 10.0
    assert np.round(ml_model.intercept_, 1) == 0.0

    # Ridge without weights
    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="ridge",
        sample_weight=None)

    assert ml_model.coef_[0].round() == 0.0
    assert np.round(ml_model.intercept_, 1) == 10.0

    # Ridge with weights
    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="ridge",
        sample_weight=sample_weight)

    assert ml_model.coef_[0].round() == 0.0
    assert np.round(ml_model.intercept_, 1) == 10.0

    # statsmodels_wls with weights
    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="statsmodels_wls",
        sample_weight=sample_weight)

    assert ml_model.coef_[0].round() == 10.0
    assert np.round(ml_model.intercept_, 1) == 0.0

    with pytest.raises(
            ValueError,
            match="sample weights are passed."):
        fit_model_via_design_matrix(
            x_train=x_train,
            y_train=y_train,
            fit_algorithm="lasso",
            sample_weight=sample_weight)


def test_fit_model_via_design_matrix_with_weights(data_with_weights):
    df = data_with_weights["df"]
    design_mat_info = data_with_weights["design_mat_info"]
    x_train = design_mat_info["x_mat"]
    y_train = design_mat_info["y"]
    sample_weight = df["w"]

    # Ridge without weights
    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="ridge",
        sample_weight=None)

    assert np.round(ml_model.intercept_, 0) == 15.0
    assert ml_model.coef_[0].round() == 0.0
    assert ml_model.coef_[1].round() == 0.0

    # Ridge with weights
    # Here we expect to get the coeffcients from: ``y[(n//2):] = 20 + -20 * x2[(n//2):]``
    # This is becauase the weights are zero in the first half
    # Therefore only the second equation (given above) will be relevant
    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="ridge",
        sample_weight=sample_weight)

    """
    # commented out graphical test
    # we expect to see two trends for y w.r.t x2
    from plotly import graph_objects as go
    trace = go.Scatter(
                x=df['x2'].values,
                y=df['y'].values,
                mode='markers')
    data = [trace]
    fig = go.Figure(data)
    fig.show()
    """
    assert np.round(ml_model.intercept_, 0) == 20.0
    assert ml_model.coef_[0].round() == 0.0
    assert ml_model.coef_[1].round() == -2.0


def test_fit_model_via_design_matrix_stats_models():
    """Testing the model fits via statsmodels module"""
    df = generate_test_data_for_fitting(
        n=50,
        seed=41,
        heteroscedastic=False)["df"]
    df["y"] = df["y"].abs() + 1
    model_formula_str = "y ~ x1_categ + x2 + x3"
    design_mat_info = design_mat_from_formula(
        df,
        model_formula_str,
        pred_cols=None,
        y_col=None)

    x_train = design_mat_info["x_mat"]
    y_train = design_mat_info["y"]

    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="statsmodels_ols")

    expected = [14.0, 1.0, -3.1, -2.1, -0.3, 2.3, 0.7]
    assert list(round(ml_model.params, 1).values) == expected

    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="statsmodels_wls")
    assert list(round(ml_model.params, 1).values) == expected

    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="statsmodels_gls")
    assert list(round(ml_model.params, 1).values) == expected

    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="statsmodels_glm")
    assert list(round(ml_model.params, 1).values) == [0.1, 0, 0, 0, 0, 0, 0]


def test_fit_model_via_design_matrix2(design_mat_info):
    """Tests fit_model_via_design_matrix with elastic_net algorithm
        and fit_algorithm_params"""
    x_train = design_mat_info["x_mat"]
    y_train = design_mat_info["y"]

    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="elastic_net",
        fit_algorithm_params=dict(n_alphas=100, eps=1e-2))

    assert ml_model.coef_[0].round() == 0
    assert ml_model.n_alphas == 100
    assert ml_model.eps == 1e-2
    assert ml_model.cv == 5  # from default parameters


def test_fit_model_via_design_matrix3(design_mat_info):
    """Tests fit_model_via_design_matrix with
        elastic_net fit_algorithm and fit_algorithm_params"""
    x_train = design_mat_info["x_mat"]
    y_train = design_mat_info["y"]

    ml_model = fit_model_via_design_matrix(
        x_train=x_train,
        y_train=y_train,
        fit_algorithm="lasso_lars",
        fit_algorithm_params=dict(max_n_alphas=100, eps=1e-2, cv=2))

    assert ml_model.coef_[0].round() == 0
    assert ml_model.max_n_alphas == 100
    assert ml_model.eps == 1e-2
    assert ml_model.cv == 2  # override default


def test_fit_model_via_design_matrix_error(design_mat_info):
    """Tests fit_model_via_design_matrix with """
    x_train = design_mat_info["x_mat"]
    y_train = design_mat_info["y"]

    with pytest.raises(
            ValueError,
            match="The fit algorithm requested was not found"):
        fit_model_via_design_matrix(
            x_train=x_train,
            y_train=y_train,
            fit_algorithm="unknown_model")


def generate_test_data_for_fitting(
        n=1000,
        seed=41,
        heteroscedastic=False):
    """Generate data for testing the fitting algorithm with different parameters
        e.g. linear and random forest.
    :param n: integer
        sample size for both the generated training data (df) and the test data
        (df_test)
    :param seed: integer
        random number generator seed
    :param heteroscedastic: bool
        if True, response has heteroscedastic error
    :return: dict
        "df": pd.DataFrame
            the dataframe of generated input data for training of size n
        "df_test": pd.DataFrame
            the test dataframe of the same size (n)
        "y_test": pd.Series
            the response values for the test set
        "model_formula_str": str
            the model formula string
    """
    m = 2 * n
    np.random.seed(seed)
    x1 = np.random.normal(size=m)
    x2 = np.random.normal(size=m)
    x2 = np.sort(x2)
    x3 = np.random.normal(size=m)
    x4 = np.random.normal(size=m)
    err = np.random.normal(size=m)
    y = 10 + 2*x2 + 8*x4 + 2*err

    df0 = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4})

    if heteroscedastic:
        err_hetero = np.random.normal(size=m)
        df0["y"] = y + 5 * abs(x1) * err_hetero

    # define a categorical variable based on x1 values
    # this is useful for testing models which use categorical features
    df0["x1_categ"] = "C" + (2.0 * df0["x1"]).abs().round().map(int).astype(str)

    df = df0[1:n]
    df_test = df0[n:(2*n)]
    y_test = df_test["y"].copy()

    model_formula_str = "y~x1+x2+x3+x4"  # note: x1_categ is not included by default

    return {
        "df": df,
        "df_test": df_test,
        "y_test": y_test,
        "model_formula_str": model_formula_str
    }


def test_fit_ml_model():
    """Tests ``fit_ml_model``"""
    data = generate_test_data_for_fitting()
    df = data["df"]
    model_formula_str = data["model_formula_str"]
    y_test = data["y_test"]
    df_test = data["df_test"]
    y_col = "y"

    trained_model = fit_ml_model(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="sgd",
        fit_algorithm_params={"alpha": 0.1})

    assert list(trained_model.keys()) == [
        "y",
        "y_mean",
        "y_std",
        "x_design_info",
        "ml_model",
        "uncertainty_model",
        "ml_model_summary",
        "y_col",
        "x_mat",
        "min_admissible_value",
        "max_admissible_value",
        "normalize_df_func",
        "regression_weight_col",
        "fitted_df"]

    assert (trained_model["y"] == df["y"]).all()
    assert trained_model["y_mean"] == np.mean(df["y"])
    assert trained_model["y_std"] == np.std(df["y"])

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)

    pred_df = pred_res["fut_df"]

    input_cols = ["x1", "x2", "x3", "x4", "x1_categ"]
    assert_frame_equal(
        pred_df[input_cols].reset_index(drop=True),
        df_test[input_cols].reset_index(drop=True))

    y_test_pred = pred_df[y_col]

    assert trained_model["ml_model"].alpha == 0.1

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert round(err[enum.get_metric_name()]) == 6.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()]) == 7.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5

    # Tests if ``fitted_df`` returned is correct
    pred_res = predict_ml(
        fut_df=df,
        trained_model=trained_model)

    fitted_df_via_predict = pred_res["fut_df"]
    x_mat_via_predict = pred_res["x_mat"]

    assert trained_model["fitted_df"].equals(fitted_df_via_predict)
    # Tests if the design matrix in the prediction time is correct
    # by comparing to the ``x_mat`` from fit phase using training data
    assert trained_model["x_mat"].reset_index(drop=True).equals(
        x_mat_via_predict.reset_index(drop=True))

    ml_model = trained_model["ml_model"]
    ml_model_coef = ml_model.coef_
    intercept = ml_model.intercept_
    x_mat_via_predict_weighted = x_mat_via_predict * ml_model_coef
    # Checks to see if the manually calculated forecast is consistent
    # Note that intercept from the regression based ML model needs to be aded
    calculated_pred = x_mat_via_predict_weighted.sum(axis=1) + intercept
    assert max(abs(calculated_pred - fitted_df_via_predict["y"])) < 1e-5

    # Tests actual values for a smaller set
    y_test_pred = predict_ml(
        fut_df=df_test[:10],
        trained_model=trained_model)["fut_df"][y_col]

    expected_values = [9.0, 9.0, 7.0, 10.0, 10.0, 10.0, 11.0, 9.0, 8.0, 6.0]
    assert list(y_test_pred.round()) == expected_values

    ml_model_summary = trained_model["ml_model_summary"].round(2)
    assert list(ml_model_summary["variable"].values) == [
        "Intercept", "x1", "x2", "x3", "x4"]
    assert list(ml_model_summary["coef"].round().values) == [-0.0, -0.0, 2.0, 1.0, 10.0]

    # Testing the summary returned from statsmodels.
    # The summary in this case is very informative with several tables.
    # `table[1]` inlcudes the cofficients and p-values.
    # Parameters without p-values are available directly
    # through `trained_model["ml_model"].params` (names and coefficients).
    # However `summary` does include more information (eg p-values) which is desirable.
    # Here we test those values even though it is harder to get them through `summary`.
    trained_model = fit_ml_model(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="statsmodels_ols",
        fit_algorithm_params={"alpha": 0.1})

    ml_model_summary = trained_model["ml_model_summary"]
    ml_model_summary_table = ml_model_summary.tables[1]
    assert ml_model_summary_table[0].data == (
        ["", "coef", "std err", "t", "P>|t|", "[0.025", "0.975]"])
    assert ml_model_summary_table[1].data == (
        ["Intercept", "  -26.5445", "    0.456", "  -58.197", " 0.000", "  -27.440", "  -25.649"])
    assert ml_model_summary_table[2].data == (
        ["x1", "    0.5335", "    0.409", "    1.304", " 0.192", "   -0.269", "    1.336"])


def test_fit_ml_model_normalization():
    """Tests ``fit_ml_model`` with and without normalization"""

    np.random.seed(seed=123)
    n = 1000
    x1 = np.random.normal(loc=0.0, scale=1.0, size=n)
    x2 = np.random.normal(loc=0.0, scale=1.0, size=n)
    y = 3 + 2 * x1 - 2 * x2
    x1_range_length = max(x1) - min(x1)
    x2_range_length = max(x2) - min(x2)
    y_mean = y.mean()
    y = y - y_mean

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    model_formula_str = "y ~ x1 + x2"

    # without normalization
    trained_model = fit_ml_model(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="linear",
        normalize_method=None)

    ml_model_summary = trained_model["ml_model_summary"].round()
    obtained_coefs = np.array(ml_model_summary["coef"].round())
    expected_coefs = np.array([0, 2, -2])
    assert np.array_equal(obtained_coefs, expected_coefs)

    # with normalization
    trained_model = fit_ml_model(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="linear",
        normalize_method="zero_to_one")

    ml_model_summary = trained_model["ml_model_summary"].round()
    obtained_coefs = np.array(ml_model_summary["coef"].round())
    # Because first we subtract the mean of y from y, which does not necessarily equal to
    # the original intercept, and second in normalizing, we subtract the min values from each x,
    # an intercept is brought in, which equals
    # (original_intercept - y_mean) + beta_1 * x1_min + beta_2 * x2_min
    expected_intercept = 3 - y_mean + 2 * x1.min() - 2 * x2.min()
    # since normalization divides variables by their range (max - min)
    # we expect the regression coefficients to be multilpied by that value
    expected_coefs = np.array(
        [expected_intercept, 2*x1_range_length, -2*x2_range_length]).round()

    assert np.array_equal(obtained_coefs, expected_coefs)


def test_fit_ml_model_with_uncertainty():
    """Tests fit_ml_model, with uncertainty intervals"""
    data = generate_test_data_for_fitting(
        n=1000,
        seed=41,
        heteroscedastic=False)

    df = data["df"]
    model_formula_str = data["model_formula_str"]
    y_test = data["y_test"]
    df_test = data["df_test"].reset_index(drop=True)
    fut_df = df_test.copy()
    fut_df["y"] = None

    trained_model = fit_ml_model(
        df=df,
        model_formula_str=model_formula_str,
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 10,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.8}})

    pred_res = predict_ml_with_uncertainty(
        fut_df=fut_df,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df["y"]

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5

    # Tests if `fitted_df` returned is correct
    pred_res = predict_ml_with_uncertainty(
        fut_df=df,
        trained_model=trained_model)

    fitted_df_via_predict = pred_res["fut_df"]
    assert trained_model["fitted_df"].equals(fitted_df_via_predict)

    # Tests actual values for a smaller set
    expected_values = [8.36, 11.19, 1.85, 15.57, 16.84, 14.44, 21.07, 9.02,
                       1.81, -7.32]
    assert list(y_test_pred[:10].round(2)) == expected_values

    # calculate coverage of the CI
    # first add true values to fut_df
    fut_df["y_true"] = df_test["y"]
    fut_df["inside_95_ci"] = fut_df.apply(
        lambda row: (
            (row["y_true"] <= row[QUANTILE_SUMMARY_COL][1])
            and (row["y_true"] >= row[QUANTILE_SUMMARY_COL][0])),
        axis=1)

    ci_coverage = 100.0 * fut_df["inside_95_ci"].mean()
    assert round(ci_coverage) == 95, (
        "95 percent CI coverage is not as expected")


def test_fit_ml_model_with_uncertainty_heteroscedastic():
    """Testing the uncertainty model fits using homoscedastic
        and heteroscedastic"""
    data = generate_test_data_for_fitting(
        n=1000,
        seed=41,
        heteroscedastic=True)

    df = data["df"]
    model_formula_str = data["model_formula_str"]
    df_test = data["df_test"].reset_index(drop=True)
    fut_df = df_test.copy()
    fut_df["y"] = None

    def ci_width_and_coverage(conditional_cols, df, fut_df):
        """Fits two types of uncertainty model depending
        on the input of conditional_cols"""
        trained_model = fit_ml_model(
            df=df,
            model_formula_str=model_formula_str,
            uncertainty_dict={
                "uncertainty_method": "simple_conditional_residuals",
                "params": {
                    "conditional_cols": conditional_cols,
                    "quantiles": [0.025, 0.975],
                    "quantile_estimation_method": "normal_fit",
                    "sample_size_thresh": 50,
                    "small_sample_size_method": "std_quantiles",
                    "small_sample_size_quantile": 0.95}})

        pred_res = predict_ml_with_uncertainty(
            fut_df=fut_df,
            trained_model=trained_model)
        fut_df = pred_res["fut_df"]
        y_test_pred = fut_df["y"]

        # testing actual values for a small set
        ind = [1, 300, 500, 700, 950]
        expected_values = [11.01, 9.88, 14.32, 17.68, 10.88]
        assert list(y_test_pred.iloc[ind].round(2)) == expected_values, (
            "predicted values are not as expected.")

        # calculate coverage of the CI
        # first add true values to fut_df
        fut_df["y_true"] = df_test["y"]
        fut_df["inside_95_ci"] = fut_df.apply(
            lambda row: (
                (row["y_true"] <= row[QUANTILE_SUMMARY_COL][1])
                and (row["y_true"] >= row[QUANTILE_SUMMARY_COL][0])),
            axis=1)

        fut_df["ci_width"] = fut_df.apply(
            lambda row: (
                (row[QUANTILE_SUMMARY_COL][1] - row[QUANTILE_SUMMARY_COL][0])),
            axis=1)
        ci_width_avg = fut_df["ci_width"].mean()

        fut_df[QUANTILE_SUMMARY_COL] = fut_df[QUANTILE_SUMMARY_COL].apply(
            lambda x: tuple(round(e, 2) for e in x))
        ci_coverage = 100.0 * fut_df["inside_95_ci"].mean()

        return {
            "ci_width_avg": ci_width_avg,
            "ci_coverage": ci_coverage}

    # fitting homoscedastic (without conditioning) uncertainty model
    ci_info = ci_width_and_coverage(
        conditional_cols=None,
        df=df,
        fut_df=fut_df)
    ci_coverage = ci_info["ci_coverage"]
    ci_width_avg = ci_info["ci_width_avg"]
    assert round(ci_coverage, 1) == 94.7, (
        "95 percent CI coverage is not as expected")
    assert round(ci_width_avg, 1) == 22.4, (
        "95 percent CI coverage average width is not as expected")

    # fitting heteroscedastic (with conditioning) uncertainty model
    ci_info = ci_width_and_coverage(
        conditional_cols=["x1_categ"],
        df=df,
        fut_df=fut_df)
    ci_coverage = ci_info["ci_coverage"]
    ci_width_avg = ci_info["ci_width_avg"]
    # we observe better coverage is higher and ci width is narrower with
    # heteroscedastic model than before
    assert round(ci_coverage, 1) == 96.3, (
        "95 percent CI coverage is not as expected")
    assert round(ci_width_avg, 1) == 20.3, (
        "95 percent CI coverage average width is not as expected")


def test_fit_ml_model_with_evaluation_with_test_set():
    """Tests fit_ml_model_with_evaluation, with test set"""
    data = generate_test_data_for_fitting()
    df = data["df"]
    model_formula_str = data["model_formula_str"]
    y_test = data["y_test"]
    df_test = data["df_test"]
    y_col = "y"

    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="sgd",
        fit_algorithm_params={"alpha": 0.1})

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    assert trained_model["ml_model"].alpha == 0.1
    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert round(err[enum.get_metric_name()]) == 6.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()]) == 7.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5

    # testing actual values for a smaller set
    pred_res = predict_ml(
        fut_df=df_test[:10],
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    expected_values = [9.0, 9.0, 7.0, 10.0, 10.0, 10.0, 11.0, 9.0, 8.0, 6.0]
    assert list(y_test_pred.round()) == expected_values


def test_fit_ml_model_with_evaluation_with_weights():
    """Tests fit_ml_model_with_evaluation, with test set"""
    data = generate_test_data_for_fitting()
    df = data["df"]
    model_formula_str = data["model_formula_str"]
    y_test = data["y_test"]
    df_test = data["df_test"]
    y_col = "y"
    df["weights"] = range(1, len(df)+1)

    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="ridge",
        regression_weight_col="weights")

    assert trained_model["regression_weight_col"] == "weights"

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert round(err[enum.get_metric_name()]) == 2.0

    # Checks for raising exception if weights have negative values
    df["weights"] = -df["weights"]
    with pytest.raises(
            ValueError,
            match="Weights can not be negative."):
        fit_ml_model_with_evaluation(
            df=df,
            model_formula_str=model_formula_str,
            fit_algorithm="ridge",
            regression_weight_col="weights")


def test_fit_ml_model_with_evaluation_with_uncertainty():
    """Tests fit_ml_model_with_evaluation with uncertainty intervals"""
    df = gen_sliced_df(
        sample_size_dict={"a": 200, "b": 340, "c": 300, "d": 8, "e": 800},
        seed_dict={"a": 301, "b": 167, "c": 593, "d": 893, "e": 191, "z": 397},
        err_magnitude_coef=8.0)

    df = df[["x", "z_categ", "y_hat"]]
    df.rename(columns={"y_hat": "y"}, inplace=True)
    model_formula_str = "y~x+z_categ"
    y_col = "y"
    # test_df
    fut_df = df.copy()
    # we change the name of the column of true values in fut_df
    # to be able to keep track of true values later
    fut_df.rename(columns={"y": "y_true"}, inplace=True)
    y_test = fut_df["y_true"]
    # create a small dataframe for testing values only
    small_sample_index = [1, 500, 750, 1000]

    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        uncertainty_dict={
            "uncertainty_method": "simple_conditional_residuals",
            "params": {
                "quantiles": [0.025, 0.975],
                "quantile_estimation_method": "normal_fit",
                "sample_size_thresh": 10,
                "small_sample_size_method": "std_quantiles",
                "small_sample_size_quantile": 0.8}})

    pred_res = predict_ml(
        fut_df=fut_df,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    y_test_pred_small = y_test_pred[small_sample_index]

    # testing predictions
    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 10.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 10.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5

    # testing actual values for a smaller set
    assert list(y_test_pred_small.round(1)) == [99.7, 201.5, 303.5, 7.3], (
        "predictions are not correct")

    # Testing uncertainty
    # Assigns the predicted y to the response in fut_df
    fut_df["y"] = y_test_pred
    new_df_with_uncertainty = predict_ci(
        fut_df,
        trained_model["uncertainty_model"])
    assert list(new_df_with_uncertainty.columns) == list(fut_df.columns) + [QUANTILE_SUMMARY_COL, ERR_STD_COL], (
        "column names are not as expected")
    fut_df[QUANTILE_SUMMARY_COL] = new_df_with_uncertainty[QUANTILE_SUMMARY_COL]

    # Calculates coverage of the CI
    fut_df["inside_95_ci"] = fut_df.apply(
        lambda row: (
            (row["y_true"] <= row[QUANTILE_SUMMARY_COL][1])
            and (row["y_true"] >= row[QUANTILE_SUMMARY_COL][0])),
        axis=1)

    ci_coverage = 100.0 * fut_df["inside_95_ci"].mean()
    assert 94.0 < ci_coverage < 96.0, (
        "95 percent CI coverage is not between 94 and 96")

    # testing uncertainty_method not being implemented but passed
    with pytest.raises(
            Exception,
            match="uncertainty method: non_existing_method is not implemented"):
        fit_ml_model_with_evaluation(
            df=df,
            model_formula_str=model_formula_str,
            uncertainty_dict={
                "uncertainty_method": "non_existing_method",
                "params": {
                    "quantiles": [0.025, 0.975],
                    "quantile_estimation_method": "normal_fit",
                    "sample_size_thresh": 10,
                    "small_sample_size_method": "std_quantiles",
                    "small_sample_size_quantile": 0.8}})


def test_fit_ml_model_with_evaluation_with_user_provided_bounds():
    """Tests fit_ml_model_with_evaluation
        with min_admissible_value and max_admissible_value"""
    data = generate_test_data_for_fitting()
    df = data["df"]
    model_formula_str = data["model_formula_str"]
    y_test = data["y_test"]
    df_test = data["df_test"]
    y_col = "y"

    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        min_admissible_value=-7,
        max_admissible_value=20.00)

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5

    # testing actual values on a smaller set
    pred_res = predict_ml(
        fut_df=df_test[:10],
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    expected_values = [8.36, 11.19, 1.85, 15.57, 16.84, 14.44, 20.00, 9.02,
                       1.81, -7.00]
    assert list(y_test_pred.round(2)) == expected_values


def test_fit_ml_model_with_evaluation_skip_test():
    """Tests fit_ml_model_with_evaluation, on linear model,
        skipping test set"""
    data = generate_test_data_for_fitting()
    df = data["df"]
    model_formula_str = data["model_formula_str"]
    y_test = data["y_test"]
    df_test = data["df_test"]
    y_col = "y"

    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        training_fraction=1.0)

    assert len(trained_model["y_test"]) == 0
    assert trained_model["y_test_pred"] is None
    assert trained_model["test_evaluation"] is None
    assert trained_model["plt_compare_test"] is None

    pred_res = predict_ml(
        fut_df=df,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    arr1 = fut_df[y_col].tolist()
    arr2 = trained_model["y_train_pred"]

    assert np.array_equal(arr1, arr2)

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5


def test_fit_ml_model_with_evaluation_random_forest():
    """Tests fit_ml_model_with_evaluation, on random forest model"""
    data = generate_test_data_for_fitting()
    df = data["df"]
    model_formula_str = data["model_formula_str"]
    y_test = data["y_test"]
    df_test = data["df_test"]
    y_col = "y"

    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="rf")

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 4.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 4.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5


def test_fit_ml_model_with_evaluation_elastic_net():
    """Tests fit_ml_model_with_evaluation, on elastic net model"""
    data = generate_test_data_for_fitting()
    df = data["df"]
    model_formula_str = data["model_formula_str"]
    y_test = data["y_test"]
    df_test = data["df_test"]
    y_col = "y"

    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="elastic_net")

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5


def test_fit_ml_model_with_evaluation_sgd():
    """Tests fit_ml_model_with_evaluation, on sgd model"""
    res = generate_test_data_for_fitting()
    df = res["df"]
    model_formula_str = res["model_formula_str"]
    y_test = res["y_test"]
    df_test = res["df_test"]
    y_col = "y"

    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="sgd",
        fit_algorithm_params={"penalty": "none"})

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5

    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm="sgd",
        fit_algorithm_params={
            "penalty": "elasticnet",
            "alpha": 0.01,
            "l1_ratio": 0.2})

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert round(err[enum.get_metric_name()]) == 3.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert round(err[enum.get_metric_name()]) == 3.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5


def test_dummy():
    df = pd.DataFrame({"a": [1, 2, 1], "b": [1, 3, 1], "c": ["a", "b", "a"]})
    df = pd.get_dummies(df)
    df["y"] = [1, 5, 4]
    model_formula_str = "y~a+b+c_a+c_b"
    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        training_fraction=1.0)
    expected_coefs = np.array([0., 1., 1., -1., 1.])
    obtained_coefs = np.array(trained_model["ml_model"].coef_).round()
    np.array_equal(expected_coefs, obtained_coefs)


def test_fit_ml_model_with_evaluation_nan():
    """Tests if NaNs are dropped before fitting."""
    df = pd.DataFrame({"a": [1, 2, 3, 2], "b": [1, 3, 1, 2], "c": ["a", "b", "a", "b"]})
    df = pd.get_dummies(df)
    df["y"] = [1, 5, np.nan, 3]
    model_formula_str = "y~a+b+c_a+c_b"
    with pytest.raises(ValueError, match="Model training requires at least 3 observations"):
        fit_ml_model_with_evaluation(
            df=df.head(3),
            model_formula_str=model_formula_str,
            training_fraction=1.0)

    with pytest.warns(UserWarning) as record:
        trained_model = fit_ml_model_with_evaluation(
            df=df,
            model_formula_str=model_formula_str,
            training_fraction=1.0)
        assert "The data frame included 1 row(s) with NAs which were removed for model fitting."\
               in record[0].message.args[0]
        assert_equal(trained_model["y"], df["y"].loc[(0, 1, 3), ])


def test_fit_ml_model_with_evaluation_constant_column():
    """Tests ``fit_ml_model_with_evaluation``
    when some regressors are constant"""
    res = generate_test_data_for_fitting(n=80)
    df = res["df"]
    y_test = res["y_test"]
    df_test = res["df_test"]
    y_col = "y"

    # add constant columns
    new_cols = []
    for i in range(300):
        col = f"cst{i}"
        df[col] = 0
        df_test[col] = 2
        new_cols.append(col)
    df["cst_event"] = "string"
    df_test["cst_event"] = "string"
    new_cols.append("cst_event")

    model_formula_str = "+".join([res["model_formula_str"]] + new_cols)

    fit_algorithm = "linear"
    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm=fit_algorithm,
        normalize_method="zero_to_one")

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    # intercept, x1, x2, x3, x4, [constant columns]
    expected_values = [-23.0, 1.0, 4.0, 0.0, 44.0, 0.0, 0.0]
    assert list(pd.Series(trained_model["ml_model"].coef_)[:7].round()) == expected_values

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5

    # testing actual values for a smaller set
    y_test_pred = predict_ml(
        fut_df=df_test[:10],
        trained_model=trained_model)["fut_df"][y_col]
    expected_values = [18.0, 19.0, 16.0, 13.0, 9.0, 1.0, 23.0, 14.0, 12.0, 14.0]
    assert list(y_test_pred.round()) == expected_values


def test_fit_ml_model_with_evaluation_constant_column_sgd():
    """Tests fit_ml_model_with_evaluation using sgd with
    no penalty when some regressors are constant
    With limited data, the models converge to slightly different predictions
    than the linear model"""
    res = generate_test_data_for_fitting(n=80)
    df = res["df"]
    y_test = res["y_test"]
    df_test = res["df_test"]
    y_col = "y"

    # add constant columns
    new_cols = []
    for i in range(300):
        col = f"cst{i}"
        df[col] = 0
        df_test[col] = 2
        new_cols.append(col)
    df["cst_event"] = "string"
    df_test["cst_event"] = "string"
    new_cols.append("cst_event")

    model_formula_str = "+".join([res["model_formula_str"]] + new_cols)

    fit_algorithm = "sgd"
    trained_model = fit_ml_model_with_evaluation(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm=fit_algorithm,
        fit_algorithm_params={"tol": 1e-5, "penalty": "none"})

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)
    fut_df = pred_res["fut_df"]
    y_test_pred = fut_df[y_col]

    expected_values = [-8.0, 2.0, 3.0, -2.0, 38.0, 0.0, 0.0]
    assert list(pd.Series(trained_model["ml_model"].coef_)[:7].round()) == expected_values

    err = calc_pred_err(y_test, y_test_pred)
    enum = EvaluationMetricEnum.MeanAbsoluteError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.RootMeanSquaredError
    assert err[enum.get_metric_name()] < 3.0
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.5

    # testing actual values for a smaller set
    y_test_pred = predict_ml(
        fut_df=df_test[:10],
        trained_model=trained_model)["fut_df"][y_col]
    expected_values = [17.0, 18.0, 16.0, 13.0, 9.0, 2.0, 20.0, 12.0, 12.0, 13.0]
    assert list(y_test_pred.round()) == expected_values


def test_breakdown_regression_based_prediction():
    "Tests ``breakdown_regression_based_prediction``."
    data = generate_test_data_for_fitting()
    df = data["df"]

    # Adds a few more columns
    df["var1"] = df["x1"]**2
    df["var2"] = df["x2"]**2
    df["y_lag1"] = df["x1"]**3
    df["y_lag2"] = df["x2"]**3
    df["u1"] = (df["x1"] + 3)**2
    df["u2"] = (df["x2"] - 3)**2

    df_train = df[:800].reset_index(drop=True)
    df_test = df[800:].reset_index(drop=True)

    trained_model = fit_ml_model(
        df=df_train,
        model_formula_str="y~x1+x2+x3+x4+var1+var2+x1_categ+y_lag1+y_lag2+u1+u2",
        fit_algorithm="sgd",
        fit_algorithm_params={"alpha": 0.1})

    pred_res = predict_ml(
        fut_df=df_test,
        trained_model=trained_model)

    pred_df = pred_res["fut_df"]
    x_mat = pred_res["x_mat"]

    grouping_regex_patterns_dict = {
        "A": ".*categ",
        "B": "var",
        "C": "x",
        "D": ".*_lag.*"
    }

    # Example 1: ``center_components=False``
    result = breakdown_regression_based_prediction(
        trained_model=trained_model,
        x_mat=x_mat,
        grouping_regex_patterns_dict=grouping_regex_patterns_dict,
        remainder_group_name="OTHER",
        center_components=False)

    column_grouping_result = result["column_grouping_result"]
    breakdown_df = result["breakdown_df"]
    breakdown_fig = result["breakdown_fig"]
    assert breakdown_fig.layout.title.text == "prediction breakdown"
    assert len(breakdown_fig.data) == 6
    assert list(breakdown_df.columns) == ["Intercept", "A", "B", "C", "D", "OTHER"]

    # Note that if a variable/column is already picked in a step,
    # it will be taken out from the columns list and will not appear
    # in next groups.
    assert column_grouping_result == {
        "str_groups": [
            [
                "x1_categ[T.C1]",
                "x1_categ[T.C2]",
                "x1_categ[T.C3]",
                "x1_categ[T.C4]",
                "x1_categ[T.C5]",
                "x1_categ[T.C6]",
                "x1_categ[T.C7]"],
            ["var1", "var2"],
            ["x1", "x2", "x3", "x4"],
            ["y_lag1", "y_lag2"]],
        "remainder": ["u1", "u2"]}

    ml_model = trained_model["ml_model"]
    ml_model_coef = ml_model.coef_
    intercept = ml_model.intercept_
    x_mat_weighted = x_mat * ml_model_coef
    y_mean = trained_model["y_mean"]

    pred_raw = round(pred_df["y"], 5)
    pred_raw_sum = round(x_mat_weighted.sum(axis=1) + intercept, 5)
    pred_from_breakdown = round(breakdown_df.sum(axis=1), 5)

    assert pred_raw_sum.equals(pred_from_breakdown)
    assert pred_raw.equals(pred_from_breakdown)

    # Example 2: ``remainder_group_name="REMAINDER"``
    result = breakdown_regression_based_prediction(
        trained_model=trained_model,
        x_mat=x_mat,
        grouping_regex_patterns_dict=grouping_regex_patterns_dict,
        remainder_group_name="REMAINDER")

    column_grouping_result = result["column_grouping_result"]
    breakdown_df = result["breakdown_df"]
    assert list(breakdown_df.columns) == ["Intercept", "A", "B", "C", "D", "REMAINDER"]

    assert column_grouping_result == {
        "str_groups": [
            [
                "x1_categ[T.C1]",
                "x1_categ[T.C2]",
                "x1_categ[T.C3]",
                "x1_categ[T.C4]",
                "x1_categ[T.C5]",
                "x1_categ[T.C6]",
                "x1_categ[T.C7]"],
            ["var1", "var2"],
            ["x1", "x2", "x3", "x4"],
            ["y_lag1", "y_lag2"]],
        "remainder": ["u1", "u2"]}

    # Example 3: ``center_components=True``
    result = breakdown_regression_based_prediction(
        trained_model=trained_model,
        x_mat=x_mat,
        grouping_regex_patterns_dict=grouping_regex_patterns_dict,
        remainder_group_name="OTHER",
        center_components=True)

    column_grouping_result = result["column_grouping_result"]
    breakdown_df = result["breakdown_df"]
    assert list(breakdown_df.columns) == ["Intercept", "A", "B", "C", "D", "OTHER"]

    assert column_grouping_result == {
        "str_groups": [
            [
                "x1_categ[T.C1]",
                "x1_categ[T.C2]",
                "x1_categ[T.C3]",
                "x1_categ[T.C4]",
                "x1_categ[T.C5]",
                "x1_categ[T.C6]",
                "x1_categ[T.C7]"],
            ["var1", "var2"],
            ["x1", "x2", "x3", "x4"],
            ["y_lag1", "y_lag2"]],
        "remainder": ["u1", "u2"]}

    pred_from_breakdown = round(breakdown_df.sum(axis=1), 5)

    assert pred_raw_sum.equals(pred_from_breakdown)
    assert pred_raw.equals(pred_from_breakdown)

    # Checks to see if components are centered
    for col in ["A", "B", "C", "D", "OTHER"]:
        assert round(breakdown_df[col].mean(), 5) == 0

    # Example 4: ``center_components=True``
    result_denom = breakdown_regression_based_prediction(
        trained_model=trained_model,
        x_mat=x_mat,
        grouping_regex_patterns_dict=grouping_regex_patterns_dict,
        remainder_group_name="OTHER",
        center_components=True,
        denominator="abs_y_mean")

    column_grouping_result = result_denom["column_grouping_result"]
    breakdown_df_denom = result_denom["breakdown_df"]
    assert list(breakdown_df_denom.columns) == ["Intercept", "A", "B", "C", "D", "OTHER"]

    assert column_grouping_result == {
        "str_groups": [
            [
                "x1_categ[T.C1]",
                "x1_categ[T.C2]",
                "x1_categ[T.C3]",
                "x1_categ[T.C4]",
                "x1_categ[T.C5]",
                "x1_categ[T.C6]",
                "x1_categ[T.C7]"],
            ["var1", "var2"],
            ["x1", "x2", "x3", "x4"],
            ["y_lag1", "y_lag2"]],
        "remainder": ["u1", "u2"]}

    pred_from_breakdown = round(breakdown_df_denom.sum(axis=1) * abs(y_mean), 5)
    assert pred_raw_sum.equals(pred_from_breakdown)
    assert pred_raw.equals(pred_from_breakdown)

    # Checks to see if components are centered
    for col in ["A", "B", "C", "D", "OTHER"]:
        assert round(breakdown_df_denom[col].mean(), 5) == 0

    # Checks to see if components are divided by absolute mean
    for col in ["A", "B", "C", "D", "OTHER"]:
        assert max(abs(breakdown_df_denom[col] * abs(y_mean) - breakdown_df[col])) < 0.0001

    with pytest.raises(
            NotImplementedError,
            match=f"quantile is not an admissable denominator"):
        breakdown_regression_based_prediction(
            trained_model=trained_model,
            x_mat=x_mat,
            grouping_regex_patterns_dict=grouping_regex_patterns_dict,
            remainder_group_name="OTHER",
            center_components=True,
            denominator="quantile")
