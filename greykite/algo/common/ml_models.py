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
# original author: Reza Hosseini
"""Functions to fit a machine learning model
and use it for prediction.
"""

import random
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LarsCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor

from greykite.algo.common.l1_quantile_regression import QuantileRegression
from greykite.algo.uncertainty.conditional.conf_interval import conf_interval
from greykite.algo.uncertainty.conditional.conf_interval import predict_ci
from greykite.common.constants import RESIDUAL_COL
from greykite.common.constants import R2_null_model_score
from greykite.common.evaluation import calc_pred_err
from greykite.common.evaluation import r2_null_model_score
from greykite.common.features.normalize import normalize_df
from greykite.common.python_utils import group_strs_with_regex_patterns
from greykite.common.viz.timeseries_plotting import plot_multivariate


matplotlib.use("agg")  # noqa: E402
import matplotlib.pyplot as plt  # isort:skip


register_matplotlib_converters()


def design_mat_from_formula(
        df,
        model_formula_str,
        y_col=None,
        pred_cols=None):
    """ Given a formula it extracts the response vector (y)
    and builds the design matrix (x_mat)

    :param df: pd.DataFrame
        A dataframe with the response vector (y) and the feature columns (x_mat)
    :param model_formula_str: str
        A formula string e.g. "y~x1+x2+x3*x4".
        This is similar to R formulas.
        See https://patsy.readthedocs.io/en/latest/formulas.html#how-formulas-work.
    :param y_col: str
        The column name which has the value of interest to be forecasted.
        If the model_formula_str is not passed, y_col e.g. ["y"] is used
        as the response vector column
    :param pred_cols: List[str]
        The names of the feature columns
        If the model_formula_str is not passed, pred_cols e.g.
        ["x1", "x2", "x3"] is used as the design matrix columns
    :return: dict
        "y": The response vector
        "y_col": Name of the response column (y)
        "x_mat": A design matrix
        "pred_cols": Name of the columns of the design matrix (x_mat)
        "x_design_info": Information for design matrix
    """
    if model_formula_str is not None:
        y, x_mat = patsy.dmatrices(
            model_formula_str,
            data=df,
            return_type="dataframe")
        pred_cols = list(x_mat.columns)
        # get the response column name using "~" location
        y_col = re.search("(.*)~", model_formula_str).group(1).strip(" ")
        y = y[y.columns[0]]
        x_design_info = x_mat.design_info
    elif y_col is not None and pred_cols is not None:
        y = df[y_col]
        x_mat = df[pred_cols]
        x_design_info = None
    else:
        raise Exception(
            f"Either provide a model expression or both y_col and pred_cols.")

    return {
        "y": y,
        "y_col": y_col,
        "x_mat": x_mat,
        "pred_cols": pred_cols,
        "x_design_info": x_design_info}


def fit_model_via_design_matrix(
        x_train,
        y_train,
        fit_algorithm,
        sample_weight=None,
        fit_algorithm_params=None):
    """Fits the predictive model and returns the prediction function.

    Parameters
    ----------
    x_train : `numpy.array`
        The design matrix
    y_train : `numpy.array`
        The vector of responses
    fit_algorithm : `str`
        The type of predictive model used in fitting
        (implemented by sklearn and statsmodels).

        Available models are:

            - ``"statsmodels_ols"``   : `statsmodels.regression.linear_model.OLS`
            - ``"statsmodels_wls"``   : `statsmodels.regression.linear_model.WLS`
            - ``"statsmodels_gls"``   : `statsmodels.regression.linear_model.GLS`
            - ``"statsmodels_glm"``   : `statsmodels.genmod.generalized_linear_model.GLM`
            - ``"linear"``            : `statsmodels.regression.linear_model.OLS`
            - ``"elastic_net"``       : `sklearn.linear_model.ElasticNetCV`
            - ``"ridge"``             : `sklearn.linear_model.RidgeCV`
            - ``"lasso"``             : `sklearn.linear_model.LassoCV`
            - ``"sgd"``               : `sklearn.linear_model.SGDRegressor`
            - ``"lars"``              : `sklearn.linear_model.LarsCV`
            - ``"lasso_lars"``        : `sklearn.linear_model.LassoLarsCV`
            - ``"rf"``                : `sklearn.ensemble.RandomForestRegressor`
            - ``"gradient_boosting"`` : `sklearn.ensemble.GradientBoostingRegressor`
            - ``"quantile_regression"`` : `~greykite.algo.common.l1_quantile_regression.QuantileRegression`

        See `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
        for the sklearn and statsmodels classes that implement these methods, and their parameters.

        "linear" is the same as "statsmodels_ols", because `statsmodels.regression.linear_model.OLS`
        is more stable than `sklearn.linear_model.LinearRegression`.
    sample_weight : `numpy.array` or None, default None
        The vector of weights to be used in weighted models.
        These weights will be used to weigh each loss potentially differently.
    fit_algorithm_params : `dict` or None, default None
        Parameters passed to the requested ``fit_algorithm``.
        If None, uses the defaults defined in this function.

    Returns
    -------
    ml_model : `class`
        a trained predictive model with available predict method
    """
    fit_algorithm_dict = {
        "statsmodels_ols": sm.OLS,  # ordinary least squares
        "statsmodels_wls": sm.WLS,  # weighted least squares
        "statsmodels_gls": sm.GLS,  # generalized least squares
        "statsmodels_glm": sm.GLM,  # generalized linear models
        # "linear" has been redirected to statsmodels OLS instead of sklearn LinearRegression
        # We've found sklearn LinearRegression's solution to be unstable under some cases.
        # The reason could be that sklearn calls lapack backend that depends on the build environment.
        # statsmodels uses simple QR decomposition/pseudo inverse to compute the solution,
        # and provides stabler solutions.
        "linear": sm.OLS,
        "elastic_net": ElasticNetCV,
        "ridge": RidgeCV,
        "lasso": LassoCV,
        "sgd": SGDRegressor,  # fits linear, elastic_net, ridge, lasso via SGD. Default is ridge with alpha = 0.0001
        "lars": LarsCV,
        "lasso_lars": LassoLarsCV,
        "rf": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "quantile_regression": QuantileRegression
    }

    # for our purposes, we may want different defaults from those provided in the classes
    # sets the default `cv` and `n_estimators`
    default_fit_algorithm_params = {
        "statsmodels_ols": dict(),
        "statsmodels_wls": dict(),
        "statsmodels_gls": dict(),
        "statsmodels_glm": dict(family=sm.families.Gamma()),  # default is gamma distribution
        "linear": dict(),
        "elastic_net": dict(cv=5),
        "ridge": dict(cv=5, alphas=np.logspace(-5, 5, 30)),  # by default RidgeCV only has 3 candidate alphas, not enough
        "lasso": dict(cv=5),
        "sgd": dict(),
        "lars": dict(cv=5),
        "lasso_lars": dict(cv=5),
        "rf": dict(n_estimators=100),
        "gradient_boosting": dict(),
        "quantile_regression": dict(quantile=0.5, alpha=0)  # unregularized version modeling median
    }

    # Re-standardizes the weights so that they sum up to data length
    # Note that in the case of no weights, each weight will be 1,
    # which also sums up to data length
    if sample_weight is not None:
        if fit_algorithm not in ["ridge", "statsmodels_wls"]:
            raise ValueError(
                "sample weights are passed. "
                f"However {fit_algorithm} does not support weighted regression.")
        sample_weight = len(sample_weight) * sample_weight / sum(sample_weight)

    if fit_algorithm not in fit_algorithm_dict.keys():
        raise ValueError(f"The fit algorithm requested was not found: {fit_algorithm}. "
                         f"Must be one of {list(fit_algorithm_dict.keys())}")

    # overwrites default params with those provided by user
    params = default_fit_algorithm_params.get(fit_algorithm, {})
    if fit_algorithm_params is not None:
        params.update(fit_algorithm_params)

    # ``ml_model`` refers to fitted machine-learning model object
    if "statsmodels" in fit_algorithm or fit_algorithm == "linear":
        if fit_algorithm == "statsmodels_wls" and sample_weight is not None:
            ml_model = fit_algorithm_dict[fit_algorithm](
                endog=y_train,
                exog=x_train,
                weights=sample_weight,
                **params)
        else:
            ml_model = fit_algorithm_dict[fit_algorithm](
                endog=y_train,
                exog=x_train,
                **params)
        ml_model = ml_model.fit()
        # Adds .coef_ and .intercept_ to statsmodels, so we could fetch parameters from .coef_ for all models.
        # Intercept is already included in params, setting .intercept_=0 in case it is needed.
        ml_model.coef_ = ml_model.params
        ml_model.intercept_ = 0.
    else:
        ml_model = fit_algorithm_dict[fit_algorithm](**params)
        if fit_algorithm == "ridge":
            ml_model.fit(
                X=x_train,
                y=y_train,
                sample_weight=sample_weight)
        else:
            ml_model.fit(
                X=x_train,
                y=y_train)

    return ml_model


def fit_ml_model(
        df,
        model_formula_str=None,
        fit_algorithm="linear",
        fit_algorithm_params=None,
        y_col=None,
        pred_cols=None,
        min_admissible_value=None,
        max_admissible_value=None,
        uncertainty_dict=None,
        normalize_method="zero_to_one",
        regression_weight_col=None):
    """Fits predictive ML (machine learning) models to continuous
    response vector (given in ``y_col``)
    and returns fitted model.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with the response vector (y) and the feature columns
        (``x_mat``).
    model_formula_str : str
        The prediction model formula string e.g. "y~x1+x2+x3*x4".
        This is similar to R formulas.
        See https://patsy.readthedocs.io/en/latest/formulas.html#how-formulas-work.
    fit_algorithm : `str`, optional, default "linear"
        The type of predictive model used in fitting.

        See `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
        for available options and their parameters.
    fit_algorithm_params : `dict` or None, optional, default None
        Parameters passed to the requested fit_algorithm.
        If None, uses the defaults in `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
    y_col : str
        The column name which has the value of interest to be forecasted
        If the model_formula_str is not passed, ``y_col`` e.g. ["y"]
        is used as the response vector column
    pred_cols : List[str]
        The names of the feature columns
        If the ``model_formula_str`` is not passed, ``pred_cols`` e.g.
        ["x1", "x2", "x3"] is used as the design matrix columns
    min_admissible_value : Optional[Union[int, float, double]]
        the minimum admissible value for the ``predict`` function to return
    max_admissible_value : Optional[Union[int, float, double]]
        the maximum admissible value for the ``predict`` function to return
    uncertainty_dict : `dict` or None
        If passed as a dictionary an uncertainty model will be fit.
        The items in the dictionary are:

            ``"uncertainty_method"`` : `str`
                the title of the method
                as of now only "simple_conditional_residuals" is implemented
                which calculates CIs by using residuals
            ``"params"`` : `dict`
                A dictionary of parameters needed for the ``uncertainty_method``
                requested

    normalize_method : `str` or None, default "zero_to_one"
        If a string is provided, it will be used as the normalization method
        in `~greykite.common.features.normalize.normalize_df`, passed via
        the argument ``method``.
        Available options are: "zero_to_one", "statistical", "minus_half_to_half", "zero_at_origin".
        If None, no normalization will be performed.
        See that function for more details.
    regression_weight_col : `str` or None, default None
        The column name for the weights to be used in weighted regression version
        of applicable machine-learning models.

    Returns
    -------
    trained_model : `dict`
        Trained model dictionary with keys:

            - "y" : response values
            - "x_design_info" : design matrix information
            - "ml_model" : A trained model with predict method
            - "uncertainty_model" : `dict`
                The returned uncertainty_model dict from
                `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`.
            - "ml_model_summary": model summary
            - "y_col" : response columns
            - "x_mat ": design matrix
            - "min_admissible_value" : minimum acceptable value
            - "max_admissible_value" : maximum acceptable value
            - "normalize_df_func" : normalization function
            - "regression_weight_col" : regression weight column

    """

    # build model matrices
    res = design_mat_from_formula(
        df=df,
        model_formula_str=model_formula_str,
        y_col=y_col,
        pred_cols=pred_cols)

    y = res["y"]
    y_mean = np.mean(y)
    y_std = np.std(y)
    x_mat = res["x_mat"]
    y_col = res["y_col"]
    x_design_info = res["x_design_info"]

    normalize_df_func = None
    if normalize_method is not None:
        if "Intercept" in (x_mat.columns):
            cols = [col for col in list(x_mat.columns) if col != "Intercept"]
        else:
            cols = list(x_mat.columns)
        normalize_info = normalize_df(
            df=x_mat[cols],
            method=normalize_method,
            drop_degenerate_cols=False,
            replace_zero_denom=True)
        x_mat[cols] = normalize_info["normalized_df"]
        x_mat = x_mat.fillna(value=0)
        normalize_df_func = normalize_info["normalize_df_func"]

    sample_weight = None
    if regression_weight_col is not None:
        if df[regression_weight_col].min() < 0:
            raise ValueError(
                "Weights can not be negative. "
                f"The column {regression_weight_col} includes negative values.")
        sample_weight = df[regression_weight_col]

    # prediction model generated by using all observed data
    ml_model = fit_model_via_design_matrix(
        x_train=x_mat,
        y_train=y,
        fit_algorithm=fit_algorithm,
        fit_algorithm_params=fit_algorithm_params,
        sample_weight=sample_weight)

    # uncertainty model is fitted if uncertainty_dict is passed
    uncertainty_model = None
    if uncertainty_dict is not None:
        uncertainty_method = uncertainty_dict["uncertainty_method"]
        if uncertainty_method == "simple_conditional_residuals":
            # reset index to match behavior of predict before assignment
            new_df = df.reset_index(drop=True)
            (new_x_mat,) = patsy.build_design_matrices(
                [x_design_info],
                data=new_df,
                return_type="dataframe")
            if normalize_df_func is not None:
                if "Intercept" in list(x_mat.columns):
                    cols = [col for col in list(x_mat.columns) if col != "Intercept"]
                else:
                    cols = list(x_mat.columns)
                new_x_mat[cols] = normalize_df_func(new_x_mat[cols])
            new_x_mat = new_x_mat.fillna(value=0)
            new_df[f"{y_col}_pred"] = ml_model.predict(new_x_mat)
            new_df[RESIDUAL_COL] = new_df[y_col] - new_df[f"{y_col}_pred"]

            # re-assign some param defaults for function conf_interval
            # with values best suited to this case
            conf_interval_params = {
                "quantiles": [0.025, 0.975],
                "sample_size_thresh": 10}

            if uncertainty_dict["params"] is not None:
                conf_interval_params.update(uncertainty_dict["params"])
            uncertainty_model = conf_interval(
                df=new_df,
                distribution_col=RESIDUAL_COL,
                offset_col=y_col,
                min_admissible_value=min_admissible_value,
                max_admissible_value=max_admissible_value,
                **conf_interval_params)
        else:
            raise NotImplementedError(
                f"uncertainty method: {uncertainty_method} is not implemented")

    # We get the model summary for a subset of models
    # where summary is available (statsmodels module),
    # or summary can be constructed (a subset of models from sklearn).
    ml_model_summary = None
    if "statsmodels" in fit_algorithm:
        ml_model_summary = ml_model.summary()
    elif hasattr(ml_model, "coef_"):
        var_names = list(x_mat.columns)
        coefs = ml_model.coef_
        ml_model_summary = pd.DataFrame({
            "variable": var_names,
            "coef": coefs})

    trained_model = {
        "y": y,
        "y_mean": y_mean,
        "y_std": y_std,
        "x_design_info": x_design_info,
        "ml_model": ml_model,
        "uncertainty_model": uncertainty_model,
        "ml_model_summary": ml_model_summary,
        "y_col": y_col,
        "x_mat": x_mat,
        "min_admissible_value": min_admissible_value,
        "max_admissible_value": max_admissible_value,
        "normalize_df_func": normalize_df_func,
        "regression_weight_col": regression_weight_col}

    if uncertainty_dict is None:
        fitted_df = predict_ml(
            fut_df=df,
            trained_model=trained_model)["fut_df"]
    else:
        fitted_df = predict_ml_with_uncertainty(
            fut_df=df,
            trained_model=trained_model)["fut_df"]

    trained_model["fitted_df"] = fitted_df

    return trained_model


def predict_ml(
        fut_df,
        trained_model):
    """Returns predictions on new data using the machine-learning (ml) model
    fitted via ``fit_ml_model``.

    :param fut_df: `pd.DataFrame`
        Input data for prediction.
        Must have all columns used for training,
        specified in ``model_formula_str`` or ``pred_cols``
    :param trained_model: `dict`
        A trained model returned from ``fit_ml_model``
    :return: `dict`
        A dictionary with following keys

        - "fut_df": `pd.DataFrame`
            Input data with ``y_col`` set to the predicted values
        - "x_mat": `patsy.design_info.DesignMatrix`
            Design matrix of the predictive model

    """
    y_col = trained_model["y_col"]
    ml_model = trained_model["ml_model"]
    x_design_info = trained_model["x_design_info"]
    min_admissible_value = trained_model["min_admissible_value"]
    max_admissible_value = trained_model["max_admissible_value"]

    # reset indices to avoid issues when adding new cols
    fut_df = fut_df.reset_index(drop=True)
    (x_mat,) = patsy.build_design_matrices(
        [x_design_info],
        data=fut_df,
        return_type="dataframe")
    if trained_model["normalize_df_func"] is not None:
        if "Intercept" in list(x_mat.columns):
            cols = [col for col in list(x_mat.columns) if col != "Intercept"]
        else:
            cols = list(x_mat.columns)
        x_mat[cols] = trained_model["normalize_df_func"](x_mat[cols])
    x_mat = x_mat.fillna(value=0)
    y_pred = ml_model.predict(x_mat)
    if min_admissible_value is not None or max_admissible_value is not None:
        y_pred = np.clip(
            a=y_pred,
            a_min=min_admissible_value,
            a_max=max_admissible_value)
    fut_df[y_col] = y_pred.tolist()

    return {
        "fut_df": fut_df,
        "x_mat": x_mat}


def predict_ml_with_uncertainty(
        fut_df,
        trained_model):
    """Returns predictions and prediction intervals on new data using
    the machine-learning (ml) model
    fitted via ``fit_ml_model`` and the uncertainty model fitted via
    ``greykite.algo.uncertainty.conditional.conf_interval.conf_interval``

    :param fut_df: `pd.DataFrame`
        Input data for prediction.
        Must have all columns specified by
        ``model_formula_str`` or ``pred_cols``
    :param trained_model: `dict`
        A trained model returned from ``fit_ml_model``
    :return: `dict`
        A dictionary with following keys

        - "fut_df": `pd.DataFrame`
            Input data with ``y_col`` set to the predicted values
        - "x_mat": `patsy.design_info.DesignMatrix`
            Design matrix of the predictive model
    """
    # gets point predictions
    fut_df = fut_df.reset_index(drop=True)
    y_col = trained_model["y_col"]
    pred_res = predict_ml(
        fut_df=fut_df,
        trained_model=trained_model)

    y_pred = pred_res["fut_df"][y_col]
    x_mat = pred_res["x_mat"]

    fut_df[y_col] = y_pred.tolist()

    # apply uncertainty model
    pred_df_with_uncertainty = predict_ci(
        fut_df,
        trained_model["uncertainty_model"])

    return {
        "fut_df": pred_df_with_uncertainty,
        "x_mat": x_mat}


def fit_ml_model_with_evaluation(
        df,
        model_formula_str=None,
        y_col=None,
        pred_cols=None,
        fit_algorithm="linear",
        fit_algorithm_params=None,
        ind_train=None,
        ind_test=None,
        training_fraction=0.9,
        randomize_training=False,
        min_admissible_value=None,
        max_admissible_value=None,
        uncertainty_dict=None,
        normalize_method="zero_to_one",
        regression_weight_col=None):
    """Fits prediction models to continuous response vector (y)
    and report results.

    Parameters
    ----------
    df : `pandas.DataFrame`
        A data frame with the response vector (y) and the feature columns (``x_mat``)
    model_formula_str : `str`
        The prediction model formula e.g. "y~x1+x2+x3*x4".
        This is similar to R language (https://www.r-project.org/) formulas.
        See https://patsy.readthedocs.io/en/latest/formulas.html#how-formulas-work.
    y_col : `str`
        The column name which has the value of interest to be forecasted
        If the ``model_formula_str`` is not passed, ``y_col`` e.g. ["y"] is used as the response
        vector column
    pred_cols : `list` [`str`]
        The names of the feature columns
        If the ``model_formula_str`` is not passed, ``pred_cols`` e.g. ["x1", "x2", "x3"]
        is used as the design matrix columns
    fit_algorithm : `str`, optional, default "linear"
        The type of predictive model used in fitting.

        See `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
        for available options and their parameters.
    fit_algorithm_params : `dict` or None, optional, default None
        Parameters passed to the requested fit_algorithm.
        If None, uses the defaults in `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.
    ind_train : `list` [`int`]
        The index (row number) of the training set
    ind_test : `list` [`int`]
        The index (row number) of the test set
    training_fraction : `float`, between 0.0 and 1.0
        The fraction of data used for training
        This is invoked if ind_train and ind_test are not passed
        If this is also None or 1.0, then we skip testing
        and train on the entire dataset
    randomize_training : `bool`
        If True, then the training and the test sets will be randomized
        rather than in chronological order
    min_admissible_value : Optional[Union[int, float, double]]
        The minimum admissible value for the ``predict`` function to return
    max_admissible_value : Optional[Union[int, float, double]]
        The maximum admissible value for the ``predict`` function to return
    uncertainty_dict: `dict` or None
        If passed as a dictionary an uncertainty model will be fit.
        The items in the dictionary are:

            ``"uncertainty_method"`` : `str`
                the title of the method
                as of now only "simple_conditional_residuals" is implemented
                which calculates CIs by using residuals
            ``"params"`` : `dict`
                A dictionary of parameters needed for the ``uncertainty_method``
                requested

    normalize_method : `str` or None, default "zero_to_one"
        If a string is provided, it will be used as the normalization method
        in `~greykite.common.features.normalize.normalize_df`, passed via
        the argument ``method``.
        Available options are: "zero_to_one", "statistical", "minus_half_to_half", "zero_at_origin".
        If None, no normalization will be performed.
        See that function for more details.
    regression_weight_col : `str` or None, default None
        The column name for the weights to be used in weighted regression version
        of applicable machine-learning models.

    Returns
    -------
    trained_model : `dict`
        Trained model dictionary with the following keys.

            "ml_model": A trained model object
            "summary": Summary of the final model trained on all data
            "x_mat": Feature vectors matrix used for training of full data (rows of ``df`` with NA are dropped)
            "y": Response vector for training and testing (rows of ``df`` with NA are dropped).
                The index corresponds to selected rows in the input ``df``.
            "y_train": Response vector used for training
            "y_train_pred": Predicted values of ``y_train``
            "training_evaluation": score function value of ``y_train`` and ``y_train_pred``
            "y_test": Response vector used for testing
            "y_test_pred": Predicted values of ``y_test``
            "test_evaluation": score function value of ``y_test`` and ``y_test_pred``
            "uncertainty_model": `dict`
                The returned uncertainty_model dict from
                `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`.
            "plt_compare_test": plot function to compare ``y_test`` and ``y_test_pred``,
            "plt_pred": plot function to compare
                ``y_train``, ``y_train_pred``, ``y_test`` and ``y_test_pred``.

    """
    # to avoid pandas unnecessary warnings due to chain assignment
    pd.options.mode.chained_assignment = None

    # dropping NAs
    if df.isnull().values.any():
        nrows_original = df.shape[0]
        df = df.dropna(subset=df.columns)  # preserves index
        nrows = df.shape[0]
        warnings.warn(
            f"The data frame included {nrows_original-nrows} row(s) with NAs which were removed for model fitting.",
            UserWarning)
        if nrows <= 2:
            raise ValueError(
                f"Model training requires at least 3 observations, but the dataframe passed "
                f"to training has {nrows} rows after removing NAs."
                f"Sometimes this can be caused by unnecessary columns in your training data "
                f"which contain NAs. Make sure to remove unnecessary columns from data before "
                f"passing it to the function.")

    # an internal function for fitting model
    # this is wrapped into a function since we can do evaluations
    trained_model = fit_ml_model(
        df=df,
        model_formula_str=model_formula_str,
        fit_algorithm=fit_algorithm,
        fit_algorithm_params=fit_algorithm_params,
        y_col=y_col,
        pred_cols=pred_cols,
        min_admissible_value=min_admissible_value,
        max_admissible_value=max_admissible_value,
        uncertainty_dict=uncertainty_dict,
        normalize_method=normalize_method,
        regression_weight_col=regression_weight_col)

    # we store the obtained ``y_col`` from the function in a new variable (``y_col_final``)
    # this is done since the input y_col could be None
    # in which case we extract ``y_col`` from the formula (``model_formula_str``)
    y_col_final = trained_model["y_col"]

    # determining what should be the training and test sets
    n = len(df)
    skip_test = False
    if ind_train is not None and ind_test is not None:
        if max(ind_train) >= min(ind_test):
            raise Exception("Test set indices should start after training set indices.")
        elif max(ind_test) >= n:
            warnings.warn(
                "Testing set indices exceed the size of the dataset."
                "Setting max index of the Test set "
                "equal to the max index of the dataset.")
            ind_test = [x for x in ind_test if x < n]
    elif ind_train is None and training_fraction is not None and training_fraction < 1.0:
        k = round(n * training_fraction)
        k = int(k)
        ind_train = range(k)
        if randomize_training:
            ind_train = random.sample(range(n), k)
            ind_train = np.sort(ind_train)
        ind_test = list(set(range(n)) - set(ind_train))
        ind_test = np.sort(ind_test)
    else:
        ind_train = range(n)
        ind_test = []
        skip_test = True

    df_train = df.iloc[ind_train]
    df_test = df.iloc[ind_test]
    y_train = df_train[y_col_final]
    y_test = df_test[y_col_final]

    if skip_test:
        y_test_pred = None
        test_evaluation = None
        plt_compare_test = None
        y_train_pred = predict_ml(
            fut_df=df,
            trained_model=trained_model,
            )["fut_df"][y_col_final].tolist()

        def plt_pred():
            plt.plot(ind_train, y_train, label="full data", alpha=0.4)
            plt.plot(ind_train, y_train_pred, label="fit", alpha=0.4)
            plt.xlabel("index")
            plt.ylabel("value")
            plt.title("fit on the whole dataset")
            plt.legend()
    else:
        # validation: fit with df_train only and predict with df_test
        # first remove responses from df_test
        df_test[y_col_final] = None
        trained_model_tr = fit_ml_model(
            df=df_train,
            model_formula_str=model_formula_str,
            fit_algorithm=fit_algorithm,
            y_col=y_col,
            pred_cols=pred_cols,
            min_admissible_value=min_admissible_value,
            max_admissible_value=max_admissible_value,
            uncertainty_dict=uncertainty_dict,
            normalize_method=normalize_method,
            regression_weight_col=regression_weight_col)

        y_train_pred = predict_ml(
            fut_df=df_train,
            trained_model=trained_model_tr)["fut_df"][y_col_final].tolist()

        y_test_pred = predict_ml(
            fut_df=df_test,
            trained_model=trained_model_tr)["fut_df"][y_col_final].tolist()

        test_evaluation = calc_pred_err(
            y_true=y_test,
            y_pred=y_test_pred)

        test_evaluation[R2_null_model_score] = r2_null_model_score(
            y_true=y_test,
            y_pred=y_test_pred,
            y_pred_null=y_train.mean(),
            y_train=None)

        def plt_compare_test():
            plt.scatter(y_test, y_test_pred, color="red", alpha=0.05)
            plt.xlabel("observed")
            plt.ylabel("predicted")
            plt.title("test set")

        def plt_pred():
            plt.plot(ind_train, y_train, label="train", alpha=0.4)
            plt.plot(ind_train, y_train_pred, label="fit", alpha=0.4)
            plt.plot(ind_test, y_test, label="observed test set", alpha=0.4)
            plt.plot(ind_test, y_test_pred, label="predicted test set", alpha=0.4)
            plt.xlabel("index")
            plt.ylabel("value")
            plt.title("training and test fits")
            plt.legend()

    training_evaluation = calc_pred_err(
        y_true=y_train,
        y_pred=y_train_pred)
    training_evaluation[R2_null_model_score] = r2_null_model_score(
        y_true=y_train,
        y_pred=y_train_pred,
        y_pred_null=y_train.mean(),
        y_train=None)

    trained_model["summary"] = None
    trained_model["y"] = df[y_col_final]
    trained_model["y_train"] = y_train
    trained_model["y_train_pred"] = y_train_pred
    trained_model["training_evaluation"] = training_evaluation
    trained_model["y_test"] = y_test
    trained_model["y_test_pred"] = y_test_pred
    trained_model["test_evaluation"] = test_evaluation
    trained_model["plt_compare_test"] = plt_compare_test
    trained_model["plt_pred"] = plt_pred

    return trained_model


def breakdown_regression_based_prediction(
        trained_model,
        x_mat,
        grouping_regex_patterns_dict,
        remainder_group_name="OTHER",
        center_components=False,
        denominator=None,
        index_values=None,
        index_col="index_col",
        plt_title="prediction breakdown"):
    """Given a regression based ML model (``ml_model``) and a design matrix
    (``x_mat``), and a string based grouping rule (``grouping_regex_patterns_dict``)
    for the design matrix columnns, constructs a dataframe with columns corresponding
    to the weighted (according to ML model regression coefficient) sum of the columns in each group.
    Note that if a variable/column is already picked in a step, it will be taken
    out from the columns list and will not appear in next groups.

    Parameters
    ----------
    trained_model : `dict`
        A trained machine-learning model which includes items:

        - ml_model : `sklearn.base.BaseEstimator`
            sklearn ML estimator/model of various form.
            We require this object to have ``.coef_`` and ``.intercept`` attributes.
        - y_mean : `float`
            Observed mean of the response
        - y_std : `float`
            Observed standard deviation of the response

    x_mat :`pandas.DataFrame`
        Design matrix of the regression model
    grouping_regex_patterns_dict : `dict` {`str`: `str`}
        A dictionary with group names as keys and regexes as values.
        This dictinary is used to partition the columns into various groups
    remainder_group_name : `str`, default "OTHER"
        In case some columns are left and not assigned to any groups, a group
        with this name will be added to breakdown dataframe and includes the
        weighted some of the remaining columns.
    center_components : `bool`, default False
        It determines if components should be centered at their mean and the mean
        be added to the intercept. More concretely, if a componet is "x" then it will
        be mapped to "x - mean(x)"; and "mean(x)" will be added to the intercept so
        that the sum of the components remains the same.
    denominator : `str`, default None
        If not None, it will specify a way to divide the components. There are
        two options implemented:

        - "abs_y_mean" : `float`
            The absolute value of the observed mean of the response
        - "y_std" : `float`
            The standard deviation of the observed response

        This will be useful if we want to make the components scale free.
        Dividing by the absolute mean value of the response, is particularly
        useful to understand how much impact each component has for an average
        response.

    index_values : `list`, default None
        The values added as index which can of any types that can be used for
        plotting the x axis in plotly eg `int` or `datetime`.
        This is useful for plotting or if later this data to be joined
        with other data. For example in forecasting context timestamps can be added.
    index_col : `str`, default "index_col"
        The name of the added column to breakdown data to keep track of index
    plt_title : `str`, default "prediction breakdown"
        The title of generated plot

    Returns
    -------
    result : `dict`
        A dictionary with the following keys.

        - "breakdown_df" : `pandas.DataFrame`
            A dataframe which includes the sums for each group / component
        - "breakdown_df_with_index_col" : `pandas.DataFrame`
            Same as ``breakdown_df`` with an added column to keep track of index
        - "breakdown_fig" : `plotly.graph_objs._figure.Figure`
            plotly plot overlaying various components
        - "column_grouping_result" : `dict`
            A dictionary which includes information for the generated groups.
            See `~greykite.common.python_utils.group_strs_with_regex_patterns`
            for more details.

    """
    if index_values is not None:
        assert len(index_values) == len(x_mat), "the number of indices must match the size of data"

    ml_model = trained_model["ml_model"]
    # The dataframe which includes the group sums
    breakdown_df = pd.DataFrame()
    ml_model_coef = ml_model.coef_
    intercept = ml_model.intercept_
    x_mat_weighted = x_mat * ml_model_coef
    data_len = len(x_mat)
    cols = list(x_mat.columns)

    breakdown_df["Intercept"] = np.repeat(intercept, data_len)
    if "Intercept" in cols:
        breakdown_df["Intercept"] += x_mat_weighted["Intercept"]

    # If Intercept appear in the columns, we remove it
    # Note that this column was utilized and added to intercept
    if "Intercept" in cols:
        del x_mat_weighted["Intercept"]
        cols.remove("Intercept")

    regex_patterns = list(grouping_regex_patterns_dict.values())
    group_names = list(grouping_regex_patterns_dict.keys())

    column_grouping_result = group_strs_with_regex_patterns(
        strings=cols,
        regex_patterns=regex_patterns)

    col_groups = column_grouping_result["str_groups"]
    remainder = column_grouping_result["remainder"]

    assert len(col_groups) == len(grouping_regex_patterns_dict)

    for i, group_name in enumerate(group_names):
        group_elements = col_groups[i]
        if len(group_elements) != 0:
            breakdown_df[group_name] = x_mat_weighted[group_elements].sum(axis=1)
        else:
            breakdown_df[group_name] = 0.0

    if len(remainder) != 0:
        breakdown_df[remainder_group_name] = x_mat_weighted[remainder].sum(axis=1)
        group_names.append(remainder_group_name)

    if center_components:
        for col in group_names:
            col_mean = breakdown_df[col].mean()
            breakdown_df[col] += -col_mean
            breakdown_df["Intercept"] += col_mean

    if denominator == "abs_y_mean":
        d = abs(trained_model["y_mean"])
    elif denominator == "y_std":
        d = trained_model["y_std"]
    elif denominator is None:
        d = 1.0
    else:
        raise NotImplementedError(f"{denominator} is not an admissable denominator")

    for col in group_names + ["Intercept"]:
        breakdown_df[col] /= d

    if index_values is None:
        index_values = range(len(breakdown_df))

    breakdown_df_with_index_col = breakdown_df.copy()
    breakdown_df_with_index_col[index_col] = index_values

    breakdown_fig = plot_multivariate(
        df=breakdown_df_with_index_col,
        x_col=index_col,
        title=plt_title,
        ylabel="component")

    return {
        "breakdown_df": breakdown_df,
        "breakdown_df_with_index_col": breakdown_df_with_index_col,
        "breakdown_fig": breakdown_fig,
        "column_grouping_result": column_grouping_result
    }
