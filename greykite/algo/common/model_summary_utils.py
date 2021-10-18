# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Kaixu Yang
"""Utilities to provide summaries of sklearn and statsmodels
regression models.
"""

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from scipy.stats import f
from scipy.stats import norm
from scipy.stats import t
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from greykite.algo.common.col_name_utils import INTERCEPT
from greykite.algo.common.col_name_utils import simplify_pred_cols


def process_intercept(x, beta, intercept, pred_cols):
    """Processes the intercept term.

    Merges intercept in beta and dummy column in x if they are not there.
    Appends "(Intercept)" to the front of ``pred_cols`` if it is not there.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    beta : `numpy.array`
        The estimated coefficients.
    intercept : `float`
        The estimated intercept.
    pred_cols : `list` [ `str` ]
        The names for predictors.

    Returns
    -------
    x : `numpy.array`
        The design matrix with dummy column.
    beta : `numpy.array`
        The estimated coefficients with intercept in the first position.
    pred_cols : `list` [ `str` ]
        List of names of predictors, with "Intercept" in the first position.
    """
    beta = np.copy(beta.ravel())
    x = np.copy(x)

    # Checks if x has an intercept column
    if not all(x[:, 0] == 1):
        x = np.concatenate([np.ones([x.shape[0], 1]), x], axis=1)
    # Checks beta has intercept column.
    # In this case, the coefficient for intercept column should be zero.
    # Use intercept as the coefficient for beta column
    if x.shape[1] == beta.shape[0]:
        beta[0] += intercept  # combines the intercept term
    # If beta does not have intercept term, concatenate the intercept term.
    elif x.shape[1] == beta.shape[0] + 1:
        beta = np.concatenate([[intercept], beta])
    else:
        raise ValueError("The shape of x and beta do not match. "
                         f"x has shape ({x.shape[0]}, {x.shape[1]}) with the intercept column, "
                         f"but beta has length {len(beta)}.")
    # Processes pred_cols
    if len(pred_cols) == x.shape[1] - 1:
        pred_cols = [INTERCEPT] + pred_cols
    elif len(pred_cols) == x.shape[1] and "Intercept" in pred_cols[0]:
        pass
    else:
        raise ValueError("The length of pred_cols does not match the shape of x. "
                         f"x has shape ({x.shape[0]}, {x.shape[1]}) with the intercept column, "
                         f"but pred_cols has length {len(pred_cols)}.")
    return x, beta, pred_cols


def round_numbers(numbers, sig_digits=4):
    """Rounds numbers to significant figure.

    Rounds numbers to a specified significant figure, and uses string to format it.
    The number will be displayed in scientific mode if there are leading/trailing zeros.
    Displays zero as 0.

    Parameters
    ----------
    numbers : `iterable` [ `float` or `int` ]
        Numbers to be formatted.
    sig_digits : `int`, default 4.
        Significant figure.

    Returns
    -------
    formatted_numbers : `numpy.array`
        Formatted numbers.
    """

    def round_number(number):
        if abs(number) < 1e-40:
            new_num = "0."
        elif ~np.isfinite(number):
            new_num = "nan"
        else:
            num_digit = -int(np.floor(np.log10(abs(number)))) + sig_digits - 1
            new_num = np.round(number, num_digit)
            new_num = ("{:.3e}".format(new_num)
                       if abs(new_num) > 10 ** sig_digits or abs(new_num) < 10 ** (-sig_digits)
                       else str(new_num))
        return new_num

    return np.vectorize(round_number)(numbers)


def create_info_dict_lm(x, y, beta, ml_model, fit_algorithm, pred_cols):
    """Creates a information dictionary for linear model results.

    Only basic information will be created in this function.
    A series of these functions are used in a flow to get all information needed
    in linear model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_params_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_ss_lm`,
    `~greykite.algo.common.model_summary_utils.add_beta_var_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_significance_lm`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    beta : `numpy.arrar`
        The estimated coefficients.
    ml_model : `class`
        The trained machine learning model class, see
        `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
    fit_algorithm : `str`
        The name of fit algorithm, see
        `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
    pred_cols : `list` [ `str` ]
        A list of predictor names.

    Returns
    -------
    info_dict : `dict`
        A dictionary of basic information with the following keys:

            "x" : `numpy.array`
                The design matrix.
            "y" : `numpy.array`
                The response vector.
            "beta" : `numpy.array`
                The estimated coefficients.
            "ml_model" : `class`
                The trained machine learning model class, see
                `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
            "fit_algorithm" : `str`
                The name of fit algorithm, see
                `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
            "pred_cols" : `list` [ `str` ]
                A list of predictor names.
            "degenerate_index" : `numpy.array`
                The indices where the x columns are degenerate.
            "n_sample" : `int`
                Number of observations.
            "n_feature" : `int`
                Number of features.
            "nonzero_index" : `numpy.array`
                Indices of nonzero coefficients.
            "n_feature_nonzero" : `int`
                Number of nonzero coefficients.
            "y_pred" : `numpy.array`
                The predicted values.
            "y_mean" : `float`
                The mean of response.
            "residual" : `numpy.array`
                Residuals.
            "residual_summary" : `numpy.array`
                Five number summary of residuals.
    """
    info_dict = dict()
    # Passes through the given information
    info_dict["x"] = x
    info_dict["y"] = y.ravel()
    info_dict["beta"] = beta.ravel()
    info_dict["ml_model"] = ml_model
    info_dict["fit_algorithm"] = fit_algorithm
    info_dict["pred_cols"] = pred_cols
    # Derives direct information of the input
    info_dict["degenerate_index"] = np.arange(x.shape[1])[x.var(axis=0) == 0]
    info_dict["n_sample"] = x.shape[0]
    info_dict["n_feature"] = x.shape[1]
    info_dict["nonzero_index"] = np.nonzero(beta.ravel())[0]
    info_dict["n_feature_nonzero"] = len(info_dict["nonzero_index"])
    info_dict["y_pred"] = x @ beta
    info_dict["y_mean"] = np.mean(y)
    info_dict["residual"] = info_dict["y"] - info_dict["y_pred"]
    info_dict["residual_summary"] = np.percentile(info_dict["residual"], [0, 25, 50, 75, 100])
    return info_dict


def add_model_params_lm(info_dict):
    """Adds model parameter information to ``info_dict`` for linear models.

    Only model-related information will be added in this function.
    A series of these functions are used in a flow to get all information needed
    in linear model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_params_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_ss_lm`,
    `~greykite.algo.common.model_summary_utils.add_beta_var_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_lm`,
    `cadd_model_significance_lm`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.create_info_dict_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "model" : `str`
                The model name.
            "weights" : `numpy.array`, optional
                The weight matrix.
            "family" : `str`, optional
                The distribution family in generalized linear model.
            "link_function" : `str`, optional
                The link function used in generalized linear model.
            "alpha" : `float`, optional
                The regularization parameter in regularized methods.
            "l1_ratio" : `float`, optional
                The l1 norm ratio in Elastic Net methods.
    """
    fit_algorithm = info_dict["fit_algorithm"]
    ml_model = info_dict["ml_model"]
    n_sample = info_dict["n_sample"]
    # Dictionary for model parameters
    valid_linear_fit_algorithms = ["linear", "statsmodels_ols", "statsmodels_wls", "statsmodels_gls", "statsmodels_glm",
                                   "ridge", "lasso", "lars", "lasso_lars", "sgd", "elastic_net"]
    if fit_algorithm in valid_linear_fit_algorithms:
        # Adds special parameters
        if fit_algorithm in ["linear", "statsmodels_ols"]:
            info_dict["model"] = "Ordinary least squares"
        elif fit_algorithm == "statsmodels_wls":
            info_dict["model"] = "Weighted least squares"
            info_dict["weights"] = ml_model.model.weights
        elif fit_algorithm == "statsmodels_gls":
            info_dict["model"] = "Generalized least squares"
            info_dict["weights"] = ml_model.model.sigma
        elif fit_algorithm == "statsmodels_glm":
            info_dict["model"] = "Generalized linear model"
            info_dict["family"] = ml_model.model.family.__class__.__name__
            info_dict["link_function"] = ml_model.model.family.link.__class__.__name__
        elif fit_algorithm in ["ridge", "lasso", "lars", "lasso_lars"]:
            info_dict["model"] = fit_algorithm.capitalize() + " regression"
            info_dict["alpha"] = ml_model.alpha_
        elif fit_algorithm == "elastic_net":
            info_dict["model"] = "Elastic Net regression"
            info_dict["alpha"] = ml_model.alpha_
            info_dict["l1_ratio"] = ml_model.l1_ratio_
        elif fit_algorithm == "sgd":
            info_dict["model"] = "Elastic Net regression via SGD"
            info_dict["alpha"] = ml_model.alpha  # does not have underscore, because this is not cv
            if info_dict["ml_model"].penalty == "l1":
                info_dict["l1_ratio"] = 1.0
            elif info_dict["ml_model"].penalty == "l2":
                info_dict["l1_ratio"] = 0.0
            else:
                info_dict["l1_ratio"] = ml_model.l1_ratio

        # Processes parameters
        if "weights" in info_dict:
            if info_dict["weights"] is None:
                weights = np.eye(info_dict["n_sample"])
            else:
                weights = np.array(info_dict["weights"])
                if weights.shape == ():  # checks if weights is a scalar
                    weights = np.diag(np.repeat(weights, n_sample))
                elif weights.shape == (n_sample,) or weights.shape == (n_sample, 1):  # checks if weights is a vector
                    weights = np.diag(weights.ravel())
                elif weights.shape == (n_sample, n_sample):
                    weights = weights
                else:
                    raise ValueError("The shape of weights does not match the design matrix. "
                                     f"The design matrix has length {n_sample}. "
                                     f"The weights has shape {weights.shape}.")
            info_dict["weights"] = weights
    else:
        raise ValueError(f"{fit_algorithm} is not a valid algorithm, it must be in "
                         f"{valid_linear_fit_algorithms}.")
    return info_dict


def add_model_df_lm(info_dict):
    """Adds degrees of freedom of the regression model to ``info_dict`` for linear models.

    The df is defined as the trace of hat matrix. In general, we have
    H=X(X'WX+a*I)^-1X'W and df=trace(H), where X is the design matrix, W is the weight matrix,
    a is the regularization parameter, and I it the identity matrix. For non-weighted methods,
    W is identity matrix, and for non-regularized methods, a is 0.
    For sparse solutions, this is calculated on the nonzero predictors.

    Only degree of freedom information will be added in this function.
    A series of these functions are used in a flow to get all information needed
    in linear model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_params_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_ss_lm`,
    `~greykite.algo.common.model_summary_utils.add_beta_var_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_significance_lm`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_model_params_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "x_nz" : `numpy.array`
                The design matrix with columns corresponding to nonzero estimated coefficients.
            "condition_number" : `float`
                The condition number for sample covariance matrix (weighted, adjusted).
            "xtwx_alphai_inv" : `numpy.array`
                (X'WX+a*I)^-1
            "reg_df" : `float`
                The regression degree of freedom, defined as the trace of hat matrix.
            "df_sse" : `float`
                The degree of freedom of residuals, defined as n_sample - reg_df.
            "df_ssr" : `float`
                The degree of freedom of the regression, defined as reg_df - 1.
            "df_sst" : `int`
                The degree of freedom of total, defined as n_sample - 1.
    """
    x = info_dict["x"]
    n_sample = info_dict["n_sample"]
    w = info_dict.get("weights")
    if w is None:
        w = np.eye(n_sample)
    alpha = info_dict.get("alpha", 0)
    l1_ratio = info_dict.get("l1_ratio", 0)
    # In ElasticNet type models (ElasticNetCV, SGD with penalty == "elasticnet"), the penalty is
    # alpha * l1_ratio * ||beta||_1 + alpha * (1 - l1_ratio) * ||beta||_2^2.
    # For ridge regression, we have l1_ratio = 0, so the l2_norm regularization parameter is alpha.
    # For lasso regression, we have l1_ratio = 1, so the l2_norm regularization parameter is zero.
    alpha = alpha * (1 - l1_ratio)  # if model has l1_ratio, the l2 norm regularization parameter is alpha * (1 - l1_ratio)
    if info_dict["fit_algorithm"] in ["lasso", "lars", "lasso_lars"]:
        alpha = 0  # alpha is specifically for l2 norm regularization in calculating df
    nonzero_idx = info_dict["nonzero_index"]
    x_nz = x[:, nonzero_idx]
    xtwx = x_nz.T @ w @ x_nz
    xtwx_alphai = xtwx + alpha * np.eye(x_nz.shape[1])
    xtwx_alphai_inv = np.linalg.pinv(xtwx_alphai)
    # The hat matrix was x_nz @ xtwx_alphai_inv @ x_nz.T @ w.
    # The regression degrees of freedom is the trace of this hat matrix.
    # This matrix is n x n, which causes memory overflow for large n.
    # Using the trace property trace(ABCD)=trace(CDAB),
    # we compute x_nz.T @ w @ x_nz @ xtwx_alphai_inv,
    # which has dimension p x p.
    trace = np.trace(x_nz.T @ w @ x_nz @ xtwx_alphai_inv)
    info_dict["x_nz"] = x_nz
    info_dict["condition_number"] = np.linalg.cond(xtwx_alphai)
    info_dict["xtwx_alphai_inv"] = xtwx_alphai_inv  # (X'WX+aI)^-1
    info_dict["reg_df"] = trace
    info_dict["df_sse"] = n_sample - info_dict["reg_df"]
    info_dict["df_ssr"] = info_dict["reg_df"] - 1
    info_dict["df_sst"] = n_sample - 1
    return info_dict


def add_model_ss_lm(info_dict):
    """Adds model sum of squared errors to ``info_dict`` for linear models.

    Only sum of squared error information will be added in this function.
    A series of these functions are used in a flow to get all information needed
    in linear model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_params_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_ss_lm`,
    `~greykite.algo.common.model_summary_utils.add_beta_var_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_significance_lm`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_model_df_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "sse" : `float`
                Sum of squared errors from residuals.
            "mse" : `float`
                ``sse`` divided by its degree of freedom.
            "ssr" : `float`
                Sum of squared errors from regression.
            "msr" : `float`
                ``ssr`` divided by its degree of freedom.
            "sst" : `float`
                Sum of squared errors from total.
            "mst" : `float`
                ``sst`` divided by its degree of freedom.
    """
    residual = info_dict["residual"]
    weights = info_dict.get("weights")
    if weights is None:
        weights = np.eye(info_dict["n_sample"])
    y = info_dict["y"]
    y_pred = info_dict["y_pred"]
    y_mean = info_dict["y_mean"]
    info_dict["sse"] = residual.T @ weights @ residual
    info_dict["mse"] = info_dict["sse"] / info_dict["df_sse"]
    info_dict["ssr"] = (y_pred - y_mean).T @ weights @ (y_pred - y_mean)
    info_dict["msr"] = info_dict["ssr"] / info_dict["df_ssr"]
    info_dict["sst"] = (y - y_mean).T @ weights @ (y - y_mean)
    info_dict["mst"] = info_dict["sst"] / info_dict["df_sst"]
    return info_dict


def add_beta_var_lm(info_dict):
    """Adds the covariance matrix for estimated coefficients to `info_dict` for linear models.

    The covariance matrix for estimated coefficients is defined as

    .. code-block:: none

        mse * (X'WX+aI)^-1X'X(X'WX+aI)

    for linear models, and defined as the inverse Fisher information matrix
    divided by ``n_sample`` for generalized linear models.

    Only variance of beta hat information will be added in this function.
    A series of these functions are used in a flow to get all information needed
    in linear model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_params_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_ss_lm`,
    `~greykite.algo.common.model_summary_utils.add_beta_var_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_significance_lm`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_model_ss_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "beta_var_cov" : `numpy.array` or `None`
                The covariance matrix of estimated coefficients. The square root of
                its diagonal elements are standard errors of the estimated coefficients.
                Set as ``None`` for sparse solutions.
    """
    if (info_dict["fit_algorithm"] in ["statsmodels_ols", "statsmodels_wls", "statsmodels_gls", "linear", "ridge"]
            or info_dict["fit_algorithm"] in ["sgd", "elastic_net"] and info_dict["l1_ratio"] == 0):
        xtwx_alphai_inv = info_dict["xtwx_alphai_inv"]
        mse = info_dict["mse"]
        # Variance of estimated coefficients mse * (X'WX+aI)^-1X'X(X'WX+aI)^-1
        x_nz = info_dict["x_nz"]
        info_dict["beta_var_cov"] = xtwx_alphai_inv @ x_nz.T @ x_nz @ xtwx_alphai_inv * mse
    elif info_dict["fit_algorithm"] in ["statsmodels_glm"]:
        # Variance of mle is approximately I(beta_hat)^-1/n, where I is Fisher's information matrix.
        assert isinstance(info_dict["ml_model"], statsmodels.genmod.generalized_linear_model.GLMResultsWrapper)
        info_dict["beta_var_cov"] = info_dict["ml_model"].cov_params()
    else:
        info_dict["beta_var_cov"] = None  # The covariance matrix for beta is not support with sparse solution
    return info_dict


def get_ls_coef_df(info_dict):
    """Gets the coefficients dataframe for least squared models.

    The dataframe includes the estimated values, the standard errors, the t-test
    values, the t-test p-values, the significance code and 95% confidence t-intervals.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_beta_var_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "coef_summary_df" : `pandas.DataFrame`
                The summary df for estimated coefficients.
            "significance_code_legend" : `str`
                The significance code legend.
    """
    beta = info_dict["beta"]
    std_err = np.sqrt(np.abs(np.diag(info_dict["beta_var_cov"])))
    if len(std_err) != len(beta):
        full_std_err = np.full(beta.shape, np.inf)
        full_std_err[info_dict["nonzero_index"]] = std_err
        full_std_err[full_std_err == 0] = np.inf  # These are the degenerate columns.
        std_err = full_std_err
    # Gets t-values
    t_values = beta / std_err

    # Gets p-values and significance codes
    p_values = 2 * t.cdf(-np.abs(t_values), info_dict["df_sse"]).ravel()
    sig_codes = ["***" if p < 0.001 else
                 "**" if p < 0.01 else
                 "*" if p < 0.05 else
                 "." if p < 0.1 else
                 " " for p in p_values]
    # Gets 95% confidence intervals
    t_975 = t.ppf(0.975, info_dict["df_sse"])
    ci_lb = beta - t_975 * std_err
    ci_ub = beta + t_975 * std_err
    ci = list(zip(ci_lb, ci_ub))
    # Makes full df
    info_dict["coef_summary_df"] = pd.DataFrame({
        "Pred_col": info_dict["pred_cols"],
        "Estimate": beta,
        "Std. Err": std_err,
        "t value": t_values,
        "Pr(>|t|)": p_values,
        "sig. code": sig_codes,
        "95%CI": ci
    })
    info_dict["significance_code_legend"] = "0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
    return info_dict


def get_glm_coef_df(info_dict):
    """Gets the coefficients dataframe for generalized linear models.

    The dataframe includes the estimated values, the standard errors, the Z-test
    values, the Z-test p-values, the significance code and 95% confidence Z-intervals.
    The tests and intervals are based on the asymptotic normality of MLE.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_beta_var_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "coef_summary_df" : `pandas.DataFrame`
                The summary df for estimated coefficients.
            "significance_code_legend" : `str`
                The significance code legend.
    """
    beta = info_dict["beta"]
    std_err = np.sqrt(np.abs(np.diag(info_dict["beta_var_cov"])))
    # Gets z-values
    z_values = beta / std_err
    # Gets p-values and significance codes
    p_values = 2 * norm.cdf(-np.abs(z_values)).ravel()
    sig_codes = ["***" if p < 0.001 else
                 "**" if p < 0.01 else
                 "*" if p < 0.05 else
                 "." if p < 0.1 else
                 " " for p in p_values]
    # Gets 95% confidence intervals
    z_975 = norm.ppf(0.975)
    ci_lb = beta - z_975 * std_err
    ci_ub = beta + z_975 * std_err
    ci = list(zip(ci_lb, ci_ub))
    # Makes full df
    info_dict["coef_summary_df"] = pd.DataFrame({
        "Pred_col": info_dict["pred_cols"],
        "Estimate": beta,
        "Std. Err": std_err,
        "z value": z_values,
        "Pr(>|Z|)": p_values,
        "sig. code": sig_codes,
        "95%CI": ci
    })
    info_dict["significance_code_legend"] = "0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
    return info_dict


class Bootstrapper(object):
    """Bootstrap generator, an iterable object.

    The generator for bootstrap samples. Iterating along the object will
    give the bootstrap sample indices.

    Attributes
    ----------
    sample_size : `int`
        The sample size of data.
    bootstrap_size : `int`
        The bootstrap sample size to be randomly drawn.
    num_bootstrap : `int`
        The number of bootstrap samples to be drawn.
    sample_num : `int`
        The index of number of samples generated.
    """

    def __init__(self, sample_size, bootstrap_size, num_bootstrap):
        self.sample_size = sample_size
        self.bootstrap_size = bootstrap_size
        self.num_bootstrap = num_bootstrap
        self.sample_num = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.sample_num < self.num_bootstrap:
            sample_index = np.random.choice(
                a=self.sample_size,
                size=self.bootstrap_size,
                replace=True)
            self.sample_num += 1
            return sample_index
        else:
            raise StopIteration


def get_ridge_summary_df_by_bootstrap(x, y, beta, alpha, pred_cols):
    """Gets the ridge regression coefficients p-values and confidence intervals.

    The tests and confidence intervals are bootstrap p-values and confidence intervals.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    beta : `numpy.array`
        The estimated coefficients.
    alpha : `float`
        The regularization parameters.
    pred_cols : `list` [ `str` ]
        The list of predictor names.

    Returns
    -------
    coef_summary_df : `pandas.DataFrame`
        The coefficients summary dataframe.
    significance_code_legend : `str`
        The significance code legend.
    """
    num_bootstrap = 500
    bootstrap_idx = Bootstrapper(
        sample_size=x.shape[0],
        bootstrap_size=x.shape[0],
        num_bootstrap=num_bootstrap)
    beta_estimates = None
    intercepts = []
    for idx in bootstrap_idx:
        x_b = x[idx, :]
        y_b = y[idx]
        model = Ridge(alpha=alpha, fit_intercept=True).fit(x_b, y_b)  # Fits intercept separately, does not penalize intercept
        intercepts.append(model.intercept_)
        # ``beta_estimates`` stores the bootstrapped beta: col = bootstrap indices; rows=betas
        if beta_estimates is None:
            beta_estimates = model.coef_.reshape(-1, 1)
        else:
            beta_estimates = np.concatenate([beta_estimates, model.coef_.reshape(-1, 1)], axis=1)
    beta_estimates[0] += np.array(intercepts).reshape(beta_estimates[0].shape)  # adds intercepts
    # Gets coefficient standard errors
    std_err = beta_estimates.std(axis=1)

    # Reference: An introduction to Bootstrap, Efron 1993

    # Gets p-values and significance codes
    # For an estimated coefficient, its p-value is the proportion of bootstrap estimates whose distances
    # to the bootstrap distribution mean are at least the absolute value of the estimated coefficient.
    p_values = np.apply_along_axis(
        func1d=lambda row: len(row[:-1][abs(row[:-1] - row[:-1].mean()) >= abs(row[-1])]),
        axis=1,
        arr=np.concatenate([beta_estimates, beta.reshape(-1, 1)], axis=1)) / num_bootstrap
    sig_codes = ["***" if p < 0.001 else
                 "**" if p < 0.01 else
                 "*" if p < 0.05 else
                 "." if p < 0.1 else
                 " " for p in p_values]

    # Gets 95% confidence intervals
    ci_lb = np.percentile(beta_estimates, 2.5, axis=1)
    ci_ub = np.percentile(beta_estimates, 97.5, axis=1)
    ci = [[lb, ub] for lb, ub in zip(ci_lb, ci_ub)]

    # Makes full df
    coef_summary_df = pd.DataFrame({
        "Pred_col": pred_cols,
        "Estimate": beta,
        "Std. Err": std_err,
        "Pr(>)_boot": p_values,
        "sig. code": sig_codes,
        "95%CI": ci
    })
    return coef_summary_df, "0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"


def get_ridge_coef_df(info_dict):
    """Gets the coefficients dataframe for ridge regression models.

    The dataframe includes the estimated values, the standard errors,
    the significance test p-values, the significance code and 95% confidence intervals.
    The tests and confidence intervals are based on bootstrap techniques.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_beta_var_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "coef_summary_df" : `pandas.DataFrame`
                The summary df for estimated coefficients.
            "significance_code_legend" : `str`
                The significance code legend.
    """
    info_dict["coef_summary_df"], info_dict["significance_code_legend"] = get_ridge_summary_df_by_bootstrap(
        x=info_dict["x"],
        y=info_dict["y"],
        beta=info_dict["beta"],
        alpha=info_dict["alpha"],
        pred_cols=info_dict["pred_cols"]
    )
    return info_dict


def get_lasso_coef_by_single_sample_split(info_dict):
    """Gets the lasso coefficients' and metrics on a single sample-splitting.

    Randomly splits the data into two parts of equal size. Performs the same
    lasso regression method as ``info_dict["ml_model"]`` to train on
    one part to identify the significant features, then trains OLS on the other
    part with the selected features. Returns the estimated coefficients and the
    standard errors.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_beta_var_lm`.

    Returns
    -------
    beta : `numpy.array`
        The estimated coefficients. For those features that are selected with
        the first half data, the coefficients are estimated with OLS on the other
        half data. For those features that have coefficients zero with the first
        half data, their coefficients are zero.
    std_err : `numpy.array`
        The standard errors for the estimated coefficients. For zero coefficients,
        the standard errors are set to 1 for the convenience of future calculations,
        they are not the real standard errors.
    df_mse : `float`
        The degree of freedom of the error.
    """
    # Gets indices for the split
    x_0, x_1, y_0, y_1 = train_test_split(
        info_dict["x"],
        info_dict["y"],
        test_size=0.5,
        shuffle=True)
    # Performs lasso on x_0 and y_0
    model = clone(info_dict["ml_model"]).fit(x_0, y_0)
    # Gets nonzero features' index
    coef = model.coef_
    coef[0] = model.intercept_
    nonzero_index = (model.coef_ != 0)
    # Trains regular linear regression on the nonzero features with x_1 and y_1
    x_2 = x_1[:, nonzero_index]  # nonzero features only
    nonzero_beta = sm.OLS(y_1, x_2).fit().params
    beta = np.zeros(len(coef))
    beta[nonzero_index] = nonzero_beta
    df_mse = x_2.shape[0] - np.trace(x_2 @ np.linalg.pinv(x_2.T @ x_2) @ x_2.T)
    y_pred = x_2 @ nonzero_beta
    residuals = y_1 - y_pred
    mse = np.sum(residuals ** 2) / df_mse
    covariance = mse * np.linalg.pinv(x_2.T @ x_2)
    standard_err = np.sqrt(np.diag(covariance))  # nonzero standard errors
    # The standard errors for the zero betas are not one, but since they should not be
    # used and to avoid warnings for possible divisions, we set them to be one.
    # Remember these are not the real standard errors for the zero beta.
    std_err = np.ones(len(beta))
    std_err[nonzero_index] = standard_err
    return beta, std_err, df_mse


def get_lasso_coef_df_by_multi_sample_split(info_dict):
    """Gets the lasso coefficients' p-values and confidence intervals by multi sample-splitting.

    Repeats the single sample-splitting in
    `~greykite.algo.common.model_summary_utils.get_lasso_coef_by_single_sample_split`
    for multiple times, and aggregate the p-values and confidence intervals according to
    "High-Dimensional Inference: Confidence Intervals, p-Values and R-Software hdi"
    by Dezeure, Buhlmann, Meier and Meinshausen

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_beta_var_lm`.
        Note that the "ml_model" in ``info_dict`` should be a lasso method
        that supports cloning.

    Returns
    -------
    coef_summary_df : `pandas.DataFrame`
        The coefficients summary dataframe.
    significance_code_legend : `str`
        The significance code legend.
    """
    # Keeping ``num_splits`` as 100 is a good choice.
    # Less number of splits make some values inaccurate (prob_nonzero, p_value, etc.).
    # Greater number of splits takes longer to run (each split runs a LassoCV).
    num_splits = 100
    beta_estimates = None
    standard_errs = None
    df_mses = np.array([])
    # beta_estimates includes the betas estimated by multi sample-splitting.
    # Each row is a coefficient, and each column is an estimation from a sample-splitting.
    for i in range(num_splits):
        beta, standard_err, df_mse = get_lasso_coef_by_single_sample_split(info_dict)
        if beta_estimates is None:
            beta_estimates = beta.reshape(-1, 1)
        else:
            beta_estimates = np.concatenate([beta_estimates, beta.reshape(-1, 1)], axis=1)
        if standard_errs is None:
            standard_errs = standard_err.reshape(-1, 1)
        else:
            standard_errs = np.concatenate([standard_errs, standard_err.reshape(-1, 1)], axis=1)
        df_mses = np.append(df_mses, df_mse)
    # Calculates nonzero probability for each feature coefficient
    beta_nonzero_prob = np.count_nonzero(beta_estimates, axis=1) / beta_estimates.shape[1]

    # Calculates individual p-values
    beta_estimates_p_values = np.where(beta_estimates != 0,
                                       2 * t.cdf(-np.abs(beta_estimates / standard_errs), df_mses),
                                       1)  # p-value is 1.0 for zero coefficients
    # Aggregates indivudual p-values
    # Reference: P-Values for High-dimensional Regression
    # by Meinshausen, Meier, Buhlmann
    # Link: https://arxiv.org/pdf/0811.2177.pdf
    gamma_min = 0.05  # a lower bound for the p-value percentile, typically 0.05
    gammas = np.arange(gamma_min, 1, 0.01)
    qjs = None
    # For each percentile gamma, the qj is defined as the gamma_th percentile of p-values divided by gamma
    # Set to 1 if the value exceeds 1.
    for gamma in gammas:
        qj = np.percentile(beta_estimates_p_values, q=gamma * 100, axis=1) / gamma
        qj = np.where(qj < 1, qj, 1)  # set to 1 if the value exceeds 1
        if qjs is None:
            qjs = qj.reshape(-1, 1)
        else:
            qjs = np.concatenate([qjs, qj.reshape(-1, 1)], axis=1)
    p_values = np.min(qjs, axis=1) * (1 - np.log(gamma_min))
    p_values = np.where(p_values < 1, p_values, 1)
    sig_codes = ["***" if p < 0.001 else
                 "**" if p < 0.01 else
                 "*" if p < 0.05 else
                 "." if p < 0.1 else
                 " " for p in p_values]

    # Calculates p-value rankings, plus one because we want ranking to start with 1
    p_value_rankings = (beta_estimates_p_values.argsort(axis=1) + 1) / beta_estimates_p_values.shape[1]
    # Calculates the confidence intervals
    # Reference: High-Dimensional Inference: Confidence Intervals, p-Values and R-Software hdi
    # by Dezeure, Buhlmann, Meier and Meinshausen
    alpha = 0.05
    # Divides by 2 because we need half of the area on each side
    quantile = 1 - alpha * p_value_rankings / (1 - np.log(gamma_min)) / 2
    # Raveled dfs are applied along columns
    ordinary_confidence_interval_lb = beta_estimates - t.ppf(quantile, df_mses.ravel()) * standard_errs
    ordinary_confidence_interval_ub = beta_estimates + t.ppf(quantile, df_mses.ravel()) * standard_errs

    confidence_intervals = []
    for i in range(beta_estimates.shape[0]):
        if (beta_estimates[i] == 0).all():
            confidence_intervals.append([0., 0.])
        else:
            check_index = (p_value_rankings[i] > gamma_min) & (beta_estimates[i] != 0)
            if check_index.sum() == 0:
                confidence_intervals.append([0., 0.])
            else:
                lb = np.min(ordinary_confidence_interval_lb[i, check_index])
                ub = np.max(ordinary_confidence_interval_ub[i, check_index])
                confidence_intervals.append([lb, ub])

    # Makes summary df
    coef_summary_df = pd.DataFrame({
        "Pred_col": info_dict["pred_cols"],
        "Estimate": info_dict["beta"],
        "Pr(>)_split": p_values,
        "sig. code": sig_codes,
        "95%CI": confidence_intervals,
        "Prob_nonzero": beta_nonzero_prob
    })
    return coef_summary_df, "0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"


def get_lasso_coef_df(info_dict):
    """Gets the coefficients dataframe for lasso regression models.

    The dataframe includes the estimated values, the probability a coefficient is nonzero,
    the significance test p-values, the significance code and 95% confidence intervals.
    The tests and confidence intervals are based on multi sample-splitting techniques.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_beta_var_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "coef_summary_df" : `pandas.DataFrame`
                The summary df for estimated coefficients.
            "significance_code_legend" : `str`
                The significance code legend.
    """
    info_dict["coef_summary_df"], info_dict["significance_code_legend"] = get_lasso_coef_df_by_multi_sample_split(info_dict)
    return info_dict


def get_elasticnet_coef_df(info_dict):
    """Gets the coefficients dataframe for Elastic Net regression models.

    Currently only returns the predictor names and their corresponding
    estimated coefficients.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_beta_var_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "coef_summary_df" : `pandas.DataFrame`
                The summary df for estimated coefficients.
    """
    info_dict["coef_summary_df"] = pd.DataFrame({
        "Pred_col": info_dict["pred_cols"],
        "Estimate": info_dict["beta"]
    })
    return info_dict


def add_model_coef_df_lm(info_dict):
    """Adds the tests and confidence intervals for estimated coefficients
       to `info_dict` for linear models.

    Only tests and confidence intervals information will be added in this function.
    A series of these functions are used in a flow to get all information needed
    in linear model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_params_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_ss_lm`,
    `~greykite.algo.common.model_summary_utils.add_beta_var_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_significance_lm`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_beta_var_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "coef_summary_df" : `pandas.DataFrame`
                The summary df for estimated coefficients.
            "significance_code_legend" : `str`, optional
                The significance code legend.
    """
    if info_dict["fit_algorithm"] in ["statsmodels_ols", "statsmodels_wls", "statsmodels_gls", "linear"]:
        info_dict = get_ls_coef_df(info_dict)
    elif info_dict["fit_algorithm"] in ["statsmodels_glm"]:
        info_dict = get_glm_coef_df(info_dict)
    elif (info_dict["fit_algorithm"] in ["ridge"]
          or (info_dict["fit_algorithm"] in ["sgd", "elastic_net"] and info_dict["l1_ratio"] == 0)):
        info_dict = get_ridge_coef_df(info_dict)
    elif (info_dict["fit_algorithm"] in ["lasso", "lars", "lasso_lars"]
          or (info_dict["fit_algorithm"] in ["elastic_net"] and info_dict["l1_ratio"] == 1)):
        info_dict = get_lasso_coef_df(info_dict)
    elif info_dict["fit_algorithm"] in ["sgd", "elastic_net"]:
        info_dict = get_elasticnet_coef_df(info_dict)
    return info_dict


def add_model_significance_lm(info_dict):
    """Adds model significance metrics to `info_dict` for linear models.

    Only model significance metrics information will be added in this function.
    A series of these functions are used in a flow to get all information needed
    in linear model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_params_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_ss_lm`,
    `~greykite.algo.common.model_summary_utils.add_beta_var_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_significance_lm`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_model_coef_df_lm`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "f_value" : `float`
                The F-ratio for model significance.
                Defined as ``msr / mse``
            "f_p_value" : `float`
                The p-value of F-ratio using degrees of freedom
                ``df_ssr`` and ``df_sse``.
            "r2" : `float`
                The coefficient of determination.
                Defined as ``ssr / sst``
            "r2_adj" : `float`
                The adjusted coefficient of determination.
                Defined as ``msr / mst``
            "aic" : `float`
                The AIC of model, with the constants depending only on
                ``n_sample`` removed.
            "bic" : `float`
                The BIC of model, with the constants depending only on
                ``n_sample`` removed.
    """
    # Gets F-value and its p-value
    info_dict["f_value"] = info_dict["msr"] / info_dict["mse"]
    info_dict["f_p_value"] = 1 - f.cdf(info_dict["f_value"], info_dict["df_ssr"], info_dict["df_sse"])
    # Gets R^2 and adjusted R^2
    info_dict["r2"] = 1 - info_dict["sse"] / info_dict["sst"]
    info_dict["r2_adj"] = 1 - info_dict["mse"] / info_dict["mst"]
    # Gets AIC and BIC
    if info_dict["fit_algorithm"] == "statsmodels_glm":
        info_dict["aic"] = info_dict["ml_model"].aic
        info_dict["bic"] = info_dict["ml_model"].bic
    else:
        info_dict["aic"] = 2 * info_dict["reg_df"] + info_dict["n_sample"] * np.log(info_dict["sse"])  # Ignores the constants
        info_dict["bic"] = np.log(info_dict["n_sample"]) * info_dict["reg_df"] + info_dict["n_sample"] * np.log(info_dict["sse"])  # Ignores the constants
    return info_dict


def get_info_dict_lm(x, y, beta, ml_model, fit_algorithm, pred_cols):
    """Get the ``info_dict`` dictionary for linear models.

    A series of functions are used in a flow to get all information needed
    in linear model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_params_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_ss_lm`,
    `~greykite.algo.common.model_summary_utils.add_beta_var_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_lm`,
    `~greykite.algo.common.model_summary_utils.add_model_significance_lm`.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    beta : `numpy.array`
        The estimated coefficients.
    ml_model : `class`
        The trained machine learning model class, see
        `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
    fit_algorithm : `str`
        The name of fit algorithm, see
        `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
    pred_cols : `list` [ `str` ]
        A list of predictor names.

    Returns
    -------
    info_dict : `dict`
        The dictionary of linear model summary information with the following keys:

            "x" : `numpy.array`
                The design matrix.
            "y" : `numpy.array`
                The response vector.
            "beta" : `numpy.array`
                The estimated coefficients.
            "ml_model" : `class`
                The trained machine learning model class, see
                `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
            "fit_algorithm" : `str`
                The name of fit algorithm, see
                `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
            "pred_cols" : `list` [ `str` ]
                A list of predictor names.
            "degenerate_index" : `numpy.array`
                The indices where the x columns are degenerate.
            "n_sample" : `int`
                Number of observations.
            "n_feature" : `int`
                Number of features.
            "nonzero_index" : `numpy.array`
                Indices of nonzero coefficients.
            "n_feature_nonzero" : `int`
                Number of nonzero coefficients.
            "y_pred" : `numpy.array`
                The predicted values.
            "y_mean" : `float`
                The mean of response.
            "residual" : `numpy.array`
                Residuals.
            "residual_summary" : `numpy.array`
                Five number summary of residuals.
            "model" : `str`
                The model name.
            "weights" : `numpy.array`, optional
                The weight matrix.
            "family" : `str`, optional
                The distribution family in generalized linear model.
            "link_function" : `str`, optional
                The link function used in generalized linear model.
            "alpha" : `float`, optional
                The regularization parameter in regularized methods.
            "l1_ratio" : `float`, optional
                The l1 norm ratio in Elastic Net methods.
            "x_nz" : `numpy.array`
                The design matrix with columns corresponding to nonzero estimated coefficients.
            "condition_number" : `float`
                The condition number for sample covariance matrix (weighted, adjusted).
            "xtwx_alphai_inv" : `numpy.array`
                (X'WX+a*I)^-1
            "reg_df" : `float`
                The regression degree of freedom, defined as the trace of hat matrix.
            "df_sse" : `float`
                The degree of freedom of residuals, defined as n_sample - reg_df.
            "df_ssr" : `float`
                The degree of freedom of the regression, defined as reg_df - 1.
            "df_sst" : `int`
                The degree of freedom of total, defined as n_sample - 1.
            "sse" : `float`
                Sum of squared errors from residuals.
            "mse" : `float`
                ``sse`` divided by its degree of freedom.
            "ssr" : `float`
                Sum of squared errors from regression.
            "msr" : `float`
                ``ssr`` divided by its degree of freedom.
            "sst" : `float`
                Sum of squared errors from total.
            "mst" : `float`
                ``sst`` divided by its degree of freedom.
            "beta_var_cov" : `numpy.array`
                The covariance matrix of estimated coefficients. The squared root of
                its diagonal elements are standard errors of the estimated coefficients.
                Set as ``None`` for sparse solutions.
            "coef_summary_df" : `pandas.DataFrame`
                The summary df for estimated coefficients.
            "significance_code_legend" : `str`, optional
                The significance code legend.
            "f_value" : `float`
                The F-ratio for model significance.
                Defined as ``msr / mse``
            "f_p_value" : `float`
                The p-value of F-ratio using degrees of freedom
                ``df_ssr`` and ``df_sse``.
            "r2" : `float`
                The coefficient of determination.
                Defined as ``ssr / sst``
            "r2_adj" : `float`
                The adjusted coefficient of determination.
                Defined as ``msr / mst``
            "aic" : `float`
                The AIC of model, with the constants depending only on
                ``n_sample`` removed.
            "bic" : `float`
                The BIC of model, with the constants depending only on
                ``n_sample`` removed.
            "model_type" : `str`
                Equals "lm".
    """
    info_dict = create_info_dict_lm(x, y, beta, ml_model, fit_algorithm, pred_cols)
    info_dict = add_model_params_lm(info_dict)
    info_dict = add_model_df_lm(info_dict)
    info_dict = add_model_ss_lm(info_dict)
    info_dict = add_beta_var_lm(info_dict)
    info_dict = add_model_coef_df_lm(info_dict)
    info_dict = add_model_significance_lm(info_dict)
    info_dict["model_type"] = "lm"
    return info_dict


def create_info_dict_tree(x, y, ml_model, fit_algorithm, pred_cols):
    """Creates a information dictionary for tree model results.

    Only basic information will be created in this function.
    A series of these functions are used in a flow to get all information needed
    in tree model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_tree`,
    `~greykite.algo.common.model_summary_utils.add_model_params_tree`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_tree`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    ml_model : `class`
        The trained machine learning model class, see
        `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
    fit_algorithm : `str`
        The name of fit algorithm, see
        `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
    pred_cols : `list` [ `str` ]
        A list of predictor names.

    Returns
    -------
    info_dict : `dict`
        A dictionary of basic information with the following keys:

            "x" : `numpy.array`
                The design matrix.
            "y" : `numpy.array`
                The response vector.
            "ml_model" : `class`
                The trained machine learning model class, see
                `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
            "fit_algorithm" : `str`
                The name of fit algorithm, see
                `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
            "pred_cols" : `list` [ `str` ]
                A list of predictor names.
            "degenerate_index" : `numpy.array`
                The indices where the x columns are degenerate.
            "n_sample" : `int`
                Number of observations.
            "n_feature" : `int`
                Number of features.
            "y_pred" : `numpy.array`
                The predicted values.
            "y_mean" : `float`
                The mean of response.
            "residual" : `numpy.array`
                Residuals.
            "residual_summary" : `numpy.array`
                Five number summary of residuals.
    """
    info_dict = dict()
    # Passes through the given information
    info_dict["x"] = x
    info_dict["y"] = y
    info_dict["ml_model"] = ml_model
    info_dict["fit_algorithm"] = fit_algorithm
    info_dict["pred_cols"] = pred_cols
    # Derives simple information
    info_dict["n_sample"] = x.shape[0]
    info_dict["n_feature"] = x.shape[1]
    info_dict["y_pred"] = ml_model.predict(x)
    info_dict["y_mean"] = np.mean(y)
    info_dict["residual"] = info_dict["y"] - info_dict["y_pred"]
    info_dict["residual_summary"] = np.percentile(info_dict["residual"], [0, 25, 50, 75, 100])
    return info_dict


def add_model_params_tree(info_dict):
    """Adds model parameters to `info_dict` for tree models.

    Only model-related parameter information will be added in this function.
    A series of these functions are used in a flow to get all information needed
    in tree model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_tree`,
    `~greykite.algo.common.model_summary_utils.add_model_params_tree`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_tree`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.create_info_dict_tree`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "model" : `str`
                The model name.
            "num_tree" : `int`
                Number of trees used.
            "criterion" : `str`
                The criterion to be minimized.
            "max_depth" : `int` or `None`
                The maximal tree depth.
            "subsample" : `float`, `int` or `None`
                The subsampling proportion or a whole number to sample.
            "max_features" : `int` or `None`
                The maximal number of features to be used in a single split.
    """
    fit_algorithm = info_dict["fit_algorithm"]
    ml_model = info_dict["ml_model"]
    valid_tree_fit_algorithms = ["rf", "gradient_boosting"]
    if fit_algorithm in valid_tree_fit_algorithms:
        if fit_algorithm == "gradient_boosting":
            info_dict["model"] = "Gradient Boosting"
            info_dict["num_tree"] = ml_model.n_estimators_
            info_dict["criterion"] = ml_model.criterion
            info_dict["max_depth"] = ml_model.max_depth
            info_dict["subsample"] = ml_model.subsample
            info_dict["max_features"] = ml_model.max_features_
        elif fit_algorithm == "rf":
            info_dict["model"] = "Random Forest"
            info_dict["num_tree"] = ml_model.n_estimators
            info_dict["criterion"] = ml_model.criterion
            info_dict["max_depth"] = ml_model.max_depth
            info_dict["subsample"] = ml_model.max_samples
            info_dict["max_features"] = ml_model.max_features
    else:
        raise ValueError(f"{fit_algorithm} is not a valid algorithm, it must be in "
                         f"{valid_tree_fit_algorithms}.")
    return info_dict


def add_model_coef_df_tree(info_dict):
    """Adds coefficient summary df to `info_dict` for tree models.

    Only coefficient summary information will be added in this function.
    A series of these functions are used in a flow to get all information needed
    in tree model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_tree`,
    `~greykite.algo.common.model_summary_utils.add_model_params_tree`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_tree`.
    For flow implementation, see
    `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.

    Parameters
    ----------
    info_dict : `dict`
        The information dictionary returned by
        `~greykite.algo.common.model_summary_utils.add_model_params_tree`.

    Returns
    -------
    info_dict : `dict`
        The information dictionary with the following keys added:

            "coef_summary_df" : `pandas.DataFrame`
                The model summary df with the following columns:

                    "Pred_col" : `str`
                        The predictor names.
                    "Feature importance" : `float`
                        The impurity-based feature importance.
                    "Importance rank" : `int`
                        The rank of feature importance.
    """
    feature_importance = info_dict["ml_model"].feature_importances_
    # Gets ranks
    temp = feature_importance.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(feature_importance))
    coef_df = pd.DataFrame({
        "Pred_col": info_dict["pred_cols"],
        "Feature importance": feature_importance,
        "Importance rank": len(feature_importance) - ranks  # Reverse order
    })
    info_dict["coef_summary_df"] = coef_df
    return info_dict


def get_info_dict_tree(x, y, ml_model, fit_algorithm, pred_cols):
    """Get the ``info_dict`` dictionary for tree models.

    A series of functions are used in a flow to get all information needed
    in tree model summary. The flow is
    `~greykite.algo.common.model_summary_utils.create_info_dict_tree`,
    `~greykite.algo.common.model_summary_utils.add_model_params_tree`,
    `~greykite.algo.common.model_summary_utils.add_model_coef_df_tree`.

    Parameters
    ----------
    x : `numpy.array`
        The design matrix.
    y : `numpy.array`
        The response vector.
    ml_model : `class`
        The trained machine learning model class, see
        `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
    fit_algorithm : `str`
        The name of fit algorithm, see
        `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
    pred_cols : `list` [ `str` ]
        A list of predictor names.

    Returns
    -------
    info_dict : `dict`
        The dictionary of tree model summary information with the following keys:

            "x" : `numpy.array`
                The design matrix.
            "y" : `numpy.array`
                The response vector.
            "ml_model" : `class`
                The trained machine learning model class, see
                `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
            "fit_algorithm" : `str`
                The name of fit algorithm, see
                `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
            "pred_cols" : `list` [ `str` ]
                A list of predictor names.
            "degenerate_index" : `numpy.array`
                The indices where the x columns are degenerate.
            "n_sample" : `int`
                Number of observations.
            "n_feature" : `int`
                Number of features.
            "y_pred" : `numpy.array`
                The predicted values.
            "y_mean" : `float`
                The mean of response.
            "residual" : `numpy.array`
                Residuals.
            "residual_summary" : `numpy.array`
                Five number summary of residuals.
            "model" : `str`
                The model name.
            "num_tree" : `int`
                Number of trees used.
            "criterion" : `str`
                The criterion to be minimized.
            "max_depth" : `int` or `None`
                The maximal tree depth.
            "subsample" : `float` or `None`
                The subsampling proportion.
            "max_features" : `int` or `None`
                The maximal number of features to be used in a single split.
            "coef_summary_df" : `pandas.DataFrame`
                The model summary df with the following columns:

                    "Pred_col" : `str`
                        The predictor names.
                    "Feature importance" : `float`
                        The impurity-based feature importance.
                    "Importance rank" : `int`
                        The rank of feature importance.

            "model_type" : `str`
                Equals "tree" for tree models.
    """
    info_dict = create_info_dict_tree(x, y, ml_model, fit_algorithm, pred_cols)
    info_dict = add_model_params_tree(info_dict)
    info_dict = add_model_coef_df_tree(info_dict)
    info_dict["model_type"] = "tree"
    return info_dict


def format_summary_df(summary_df, max_colwidth=20):
    """Formats the summary df for printing.

    Rounds numbers so they look good.
    Formats p-values to make them display the same way as the lm function in R.
    Replaces column names with index if they are too long.

    Parameters
    ----------
    summary_df : `pandas.DataFrame`
        The summary df with the following columns:

            "Pred_col" : `str`
                Names for predictors.
            "Col_index" : `int`
                Indices for predictors. Replaces ``Pred_col`` if predictor names are too long.
            "Estimate" : `float`
                The estimation of coefficients.
            "Std. Err" : `float`, optional
                The standard error for estimations.
            "t value" : `float`, optional
                The t test statistics.
            "z value" : `float`, optional
                The z test statistics. (for glm)
            "Pr(>|t|)" : `float`, optional
                The t test p-values.
            "Pr(>|Z|)" : `float`, optional
                The z test p-value. (for glm)
            "Pr(>)_boot" : `float`, optional
                The bootstrap p-value. (for ridge)
            "sig. code" : `str`, optional
                The significance code for p-values.
            "95%CI" : `list` [ `float` ], optional
                The 95% confidence intervals.

    max_colwidth : `int`
        The maximum length for predictors to be shown in their original name.
        If the maximum length of predictors exceeds this parameter, all
        predictors name will be suppressed and only indices are shown.

    Returns
    -------
    formatted_summary_df : `pandas.DataFrame`
        Formatted summary dataframe.
    """
    summary_df_print = summary_df.copy()
    # Displays column index if the maximum length of column names is too long.
    if summary_df.shape[0] > 0:
        max_colname_length = max([len(name) for name in summary_df["Pred_col"]])
        summary_df_print["Pred_col"] = simplify_pred_cols(summary_df_print["Pred_col"])
        if max_colname_length > max_colwidth:
            summary_df_print["Pred_col"] = [col if len(col) <= max_colwidth
                                            else col[:int((max_colwidth - 3) // 2)]
                                            + "..."
                                            + col[-int((max_colwidth - 3) // 2):]
                                            for col in summary_df_print["Pred_col"]]
        # Rounds columns
        if "Estimate" in summary_df_print.columns:
            summary_df_print["Estimate"] = round_numbers(summary_df_print["Estimate"], 4)
        if "Std. Err" in summary_df_print.columns:
            summary_df_print["Std. Err"] = round_numbers(summary_df_print["Std. Err"], 4)
        if "t value" in summary_df_print.columns:
            summary_df_print["t value"] = round_numbers(summary_df_print["t value"], 4)
        if "z value" in summary_df_print.columns:
            summary_df_print["z value"] = round_numbers(summary_df_print["z value"], 4)
        # Formats p-values
        if "Pr(>|t|)" in summary_df_print.columns:
            summary_df_print["Pr(>|t|)"] = ["<2e-16" if val < 2e-16 else
                                            "{:.2e}".format(val) if val < 1e-3 else
                                            "{0:.3f}".format(val)
                                            for val in summary_df_print["Pr(>|t|)"]]
        if "Pr(>|Z|)" in summary_df_print.columns:
            summary_df_print["Pr(>|Z|)"] = ["<2e-16" if val < 2e-16 else
                                            "{:.2e}".format(val) if val < 1e-3 else
                                            "{0:.3f}".format(val)
                                            for val in summary_df_print["Pr(>|Z|)"]]
        if "Pr(>)_boot" in summary_df_print.columns:
            summary_df_print["Pr(>)_boot"] = ["<2e-16" if val < 2e-16 else
                                              "{:.2e}".format(val) if val < 1e-3 else
                                              "{0:.3f}".format(val)
                                              for val in summary_df_print["Pr(>)_boot"]]
        if "Pr(>)_split" in summary_df_print.columns:
            summary_df_print["Pr(>)_split"] = ["<2e-16" if val < 2e-16 else
                                               "{:.2e}".format(val) if val < 1e-3 else
                                               "{0:.3f}".format(val)
                                               for val in summary_df_print["Pr(>)_split"]]
        # Formats confidence intervals
        if "95%CI" in summary_df_print.columns:
            lbs = round_numbers([ci[0] for ci in summary_df_print["95%CI"]], 4)
            ubs = round_numbers([ci[1] for ci in summary_df_print["95%CI"]], 4)
            summary_df_print["95%CI"] = list(zip(lbs, ubs))
        # Formats feature importance for tree
        if "Feature importance" in summary_df_print.columns:
            summary_df_print["Feature importance"] = round_numbers(summary_df_print["Feature importance"], 4)
    return summary_df_print


def create_title_section():
    """Creates the title section for model summary.

    Returns
    -------
    content : `str`
        Title section.
    """
    content = " Model Summary ".center(80, "=") + "\n\n"
    return content


def create_model_parameter_section(info_dict):
    """Creates the model parameter section for model summary.

    Parameters
    ----------
    info_dict : `dict`
        The dictionary returned by
        `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.
        or
        `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.

    Returns
    -------
    content : `str`
        The model parameter section.
    """
    content = f"Number of observations: {info_dict['n_sample']}"
    content += ",   "
    content += f"Number of features: {info_dict['n_feature']}"
    content += "\n"
    content += f"Method: {info_dict['model']}"
    content += "\n"
    if info_dict["model_type"] == "lm":
        content += f"Number of nonzero features: {info_dict['n_feature_nonzero']}"
        content += "\n"
        if info_dict["fit_algorithm"] == "statsmodels_glm":
            content += f"Family: {info_dict['family']}"
            content += ",   "
            content += f"Link function: {info_dict['link_function']}"
            content += "\n"
        elif info_dict["fit_algorithm"] in ["ridge", "lasso", "lars", "lasso_lars",
                                            "elastic_net", "sgd"]:
            content += f"Regularization parameter: {round_numbers(info_dict['alpha'], 4)}"
            if info_dict["fit_algorithm"] in ["elastic_net", "sgd"]:
                content += ",   "
                content += f"l1_ratio: {round_numbers(info_dict['l1_ratio'], 4)}"
            content += "\n"
    elif info_dict["model_type"] == "tree":
        content += f"Number of Trees: {info_dict['num_tree']}"
        content += ",   "
        content += f"Criterion: {info_dict['criterion'].upper()}"
        content += "\n"
        content += f"Subsample: {info_dict['subsample']}"
        content += ",   "
        content += f"Max features: {info_dict['max_features']}"
        content += ",   "
        content += f"Max depth: {info_dict['max_depth']}"
        content += "\n"
    content += "\n"
    return content


def create_residual_section(info_dict):
    """Creates the residual section for model summary.

    Parameters
    ----------
    info_dict : `dict`
        The dictionary returned by
        `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.
        or
        `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.

    Returns
    -------
    content : `str`
        The residual section.
    """
    residual_summary = round_numbers(info_dict["residual_summary"], 4)
    content = "Residuals:\n"
    content += "{:>12} {:>12} {:>12} {:>12} {:>12}".format("Min", "1Q", "Median", "3Q", "Max") + "\n"
    content += "{:>12} {:>12} {:>12} {:>12} {:>12}".format(*list(residual_summary)) + "\n"
    content += "\n"
    return content


def create_coef_df_section(info_dict, max_colwidth=20):
    """Creates the coefficient summary df section for model summary.

    Parameters
    ----------
    info_dict : `dict`
        The dictionary returned by
        `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.
        or
        `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.
    max_colwidth : `int`
        The maximum length for predictors to be shown in their original name.
        If the maximum length of predictors exceeds this parameter, all
        predictors name will be suppressed and only indices are shown.

    Returns
    -------
    content : `str`
        The coefficient summary df section.
    """
    content = format_summary_df(info_dict["coef_summary_df"], max_colwidth).to_string(index=False)
    content += "\n"
    if "significance_code_legend" in info_dict.keys():
        content += f"Signif. Code: {info_dict['significance_code_legend']}"
        content += "\n"
    content += "\n"
    return content


def create_significance_section(info_dict):
    """Creates the model sifnificance section for model summary.

    Parameters
    ----------
    info_dict : `dict`
        The dictionary returned by
        `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.
        or
        `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.

    Returns
    -------
    content : `str`
        The model significance section.
    """
    if info_dict["model_type"] == "lm":
        content = f"Multiple R-squared: {round_numbers(info_dict['r2'], 4)},"
        content += "   "
        content += f"Adjusted R-squared: {round_numbers(info_dict['r2_adj'], 4)}"
        content += "\n"
        content += f"F-statistic: {round_numbers(info_dict['f_value'], 5)} " \
                   f"on {int(info_dict['df_ssr'])} and" \
                   f" {int(info_dict['df_sse'])} DF,"
        content += "   "
        content += f"p-value: {round_numbers(info_dict['f_p_value'], 4)}"
        content += "\n"
        content += f"Model AIC: {round_numbers(info_dict['aic'], 5)},"
        content += "   "
        content += f"model BIC: {round_numbers(info_dict['bic'], 5)}"
        content += "\n\n"
        return content
    else:
        return None


def create_warning_section(info_dict):
    """Creates the warning section for model summary.

    The following warnings are possible to be included:

        Condition number is too big.
        F-ratio and p-value on regularized methods.
        R-squared and F-ratio in glm.
        Zero coefficients in non-sparse regression methods due to degenerate columns.

    Parameters
    ----------
    info_dict : `dict`
        The dictionary returned by
        `~greykite.algo.common.model_summary_utils.get_info_dict_lm`.
        or
        `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.

    Returns
    -------
    content : `str`
        The model warnings section.
    """
    if info_dict["model_type"] == "lm":
        content = ""
        if info_dict["condition_number"] > 1000:
            content += f"WARNING: the condition number is large, {'{:.2e}'.format(info_dict['condition_number'])}. " \
                       f"This might indicate that there are strong multicollinearity " \
                       f"or other numerical problems."
            content += "\n"
        if info_dict["fit_algorithm"] in ["ridge", "lasso", "lars", "lasso_lars",
                                          "sgd", "elastic_net"]:
            content += "WARNING: the F-ratio and its p-value on regularized methods might be misleading, " \
                       "they are provided only for reference purposes."
            content += "\n"
        if info_dict["fit_algorithm"] in ["statsmodels_glm"]:
            content += "WARNING: the R-squared and F-statistics on glm might be misleading, " \
                       "they are provided only for reference purposes."
            content += "\n"
        if (len(info_dict["nonzero_index"]) < info_dict["n_feature"]
            and
            info_dict["fit_algorithm"] in ["linear", "ridge", "statsmodels_ols",
                                           "statsmodels_wls", "statsmodels_gls",
                                           "statsmodels_glm"]):
            content += "WARNING: the following columns have estimated coefficients equal to zero, " \
                       f"while {info_dict['fit_algorithm']} is not supposed to have zero estimates. " \
                       f"This is probably because these columns are degenerate in the design matrix. " \
                       f"Make sure these columns do not have constant values." + "\n" +\
                       f"{[col for i, col in enumerate(info_dict['pred_cols']) if i not in info_dict['nonzero_index']]}"
            content += "\n"
        if len(info_dict["degenerate_index"]) > 1:
            content += "WARNING: the following columns are degenerate, do you really want to include them in your model? " \
                       "This may cause some of them to show unrealistic significance. Consider using the `drop_degenerate` transformer." + "\n" +\
                       f"{[col for i, col in enumerate(info_dict['pred_cols']) if i in info_dict['degenerate_index']]}"
            content += "\n"
        if content == "":
            content = None
    else:
        content = None
    return content


def print_summary(info_dict, max_colwidth=20):
    """Creates the content for printing the summary.

    Parameters
    ----------
    info_dict : `dict`
        The output summary dictionary from
        `~greykite.algo.common.model_summary.ModelSummary._get_summary`,
        which is originally from
        `~greykite.algo.common.model_summary_utils.get_info_dict_lm`
        or
        `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.
    max_colwidth : `int`
        The maximum length for predictors to be shown in their original name.
        If the maximum length of predictors exceeds this parameter, all
        predictors name will be suppressed and only indices are shown.

    Returns
    -------
    summary_content : `str`
        The content to be printed.
    """
    content = create_title_section()
    model_param_section = create_model_parameter_section(info_dict)
    if model_param_section is not None:
        content += model_param_section
    residual_section = create_residual_section(info_dict)
    if residual_section is not None:
        content += residual_section
    coef_df_section = create_coef_df_section(info_dict, max_colwidth)
    if coef_df_section is not None:
        content += coef_df_section
    signif_section = create_significance_section(info_dict)
    if signif_section is not None:
        content += signif_section
    warning_section = create_warning_section(info_dict)
    if warning_section is not None:
        content += warning_section
    return content
