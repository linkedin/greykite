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
"""Provides summaries of sklearn and statsmodels
regression models.
"""

from greykite.algo.common.col_name_utils import filter_coef_summary
from greykite.algo.common.model_summary_utils import format_summary_df
from greykite.algo.common.model_summary_utils import get_info_dict_lm
from greykite.algo.common.model_summary_utils import get_info_dict_tree
from greykite.algo.common.model_summary_utils import print_summary
from greykite.algo.common.model_summary_utils import process_intercept


class ModelSummary:
    """A class to store regression model summary statistics.

    The class can be printed to get a well formatted model summary.

    Attributes
    ----------
    x : `numpy.array`
        The design matrix.
    beta : `numpy.array`
        The estimated coefficients.
    y : `numpy.array`
        The response.
    pred_cols : `list` [ `str` ]
        List of predictor names.
    pred_category : `dict`
        Predictor category, returned by
        `~greykite.algo.common.col_name_utils.create_pred_category`.
    fit_algorithm : `str`
        The name of algorithm to fit the regression.
    ml_model : `class`
        The trained machine learning model class.
    max_colwidth : `int`
        The maximum length for predictors to be shown in their original name.
        If the maximum length of predictors exceeds this parameter, all
        predictors name will be suppressed and only indices are shown.
    info_dict : `dict`
        The model summary dictionary, output of
        `~greykite.algo.common.model_summary.ModelSummary._get_summary`
    html_str : `str`
        An html formatting of the string representation of the model summary.
    """

    def __init__(
            self,
            x,
            y,
            pred_cols,
            pred_category,
            fit_algorithm,
            ml_model,  # needs to support cloning if ml_model implements a lasso method
            max_colwidth=20):
        # process x, beta and pred_cols to include the intercept term
        beta = getattr(ml_model, "coef_", None)
        intercept = getattr(ml_model, "intercept_", None)
        if beta is not None:
            x, beta, pred_cols = process_intercept(x, beta, intercept, pred_cols)
        self.x = x
        self.beta = beta
        self.y = y
        self.pred_cols = pred_cols
        self.pred_category = pred_category
        self.fit_algorithm = fit_algorithm
        self.ml_model = ml_model
        self.max_colwidth = max_colwidth
        self.info_dict = self._get_summary()
        self.html_str = f"<pre>{self.__str__()}</pre>"

    def __str__(self):
        """print method.
        """
        return print_summary(self.info_dict, self.max_colwidth)

    def __repr__(self):
        """print method
        """
        return print_summary(self.info_dict, self.max_colwidth)

    def _get_summary(self):
        """Gets the model summary from input.
        This function is called during initialization.

        Returns
        -------
        info_dict : `dict`
            Includes direct and derived metrics about the trained model. For detailed keys, refer to
            `~greykite.algo.common.model_summary_utils.get_info_dict_lm`
            or
            `~greykite.algo.common.model_summary_utils.get_info_dict_tree`.
        """
        if self.fit_algorithm in ["linear", "ridge", "lasso", "lars", "lasso_lars",
                                  "sgd", "elastic_net", "statsmodels_ols",
                                  "statsmodels_wls", "statsmodels_gls", "statsmodels_glm"]:
            info_dict = get_info_dict_lm(
                x=self.x,
                y=self.y,
                beta=self.beta,
                ml_model=self.ml_model,
                fit_algorithm=self.fit_algorithm,
                pred_cols=self.pred_cols)
        elif self.fit_algorithm in ["rf", "gradient_boosting"]:
            info_dict = get_info_dict_tree(
                x=self.x,
                y=self.y,
                ml_model=self.ml_model,
                fit_algorithm=self.fit_algorithm,
                pred_cols=self.pred_cols)
        else:
            raise NotImplementedError(f"{self.fit_algorithm} is not recognized, "
                                      f"summary is not implemented.")
        return info_dict

    def get_coef_summary(
            self,
            is_intercept=None,
            is_time_feature=None,
            is_event=None,
            is_trend=None,
            is_seasonality=None,
            is_lag=None,
            is_regressor=None,
            is_interaction=None,
            return_df=False):
        """Gets the coefficient summary filtered by conditions.

        Parameters
        ----------
        is_intercept : `bool` or `None`, default `None`
            Intercept or not.
        is_time_feature : `bool` or `None`, default `None`
            Time features or not.
            Time features belong to `~greykite.common.constants.TimeFeaturesEnum`.
        is_event : `bool` or `None`, default `None`
            Event features or not.
            Event features have `~greykite.common.constants.EVENT_PREFIX`.
        is_trend : `bool` or `None`, default `None`
            Trend features or not.
            Trend features have `~greykite.common.constants.CHANGEPOINT_COL_PREFIX` or "cp\\d".
        is_seasonality : `bool` or `None`, default `None`
            Seasonality feature or not.
            Seasonality features have `~greykite.common.constants.SEASONALITY_REGEX`.
        is_lag : `bool` or `None`, default `None`
            Lagged features or not.
            Lagged features have "lag".
        is_regressor : 0 or 1
            Extra features provided by users.
            They are provided through ``extra_pred_cols`` in the fit function.
        is_interaction : `bool` or `None`, default `None`
            Interaction feature or not.
            Interaction features have ":".
        return_df : `bool`, default `False`
           If True, the filtered coefficient summary df is also returned.
            Otherwise, the filtered coefficient summary df is printed only.

        Returns
        -------
        filtered_coef_summary : `pandas.DataFrame` or `None`
            If ``return_df`` is set to True, returns the filtered coefficient summary
            df filtered by the given conditions.
        """
        filtered_coef_summary = filter_coef_summary(
            coef_summary=self.info_dict["coef_summary_df"],
            pred_category=self.pred_category,
            is_intercept=is_intercept,
            is_time_feature=is_time_feature,
            is_event=is_event,
            is_trend=is_trend,
            is_seasonality=is_seasonality,
            is_lag=is_lag,
            is_regressor=is_regressor,
            is_interaction=is_interaction)
        # excludes column categories
        cols = [col for col in filtered_coef_summary.columns if "is_" not in col]
        print(format_summary_df(filtered_coef_summary[cols]).to_string(index=False))
        if return_df:
            return filtered_coef_summary
