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
"""Defines the simple conditional residuals uncertainty model."""

from typing import Dict
from typing import List
from typing import Optional

import pandas as pd

from greykite.algo.uncertainty.conditional.conf_interval import conf_interval
from greykite.algo.uncertainty.conditional.conf_interval import predict_ci
from greykite.common import constants as cst
from greykite.common.features.timeseries_features import add_time_features_df
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.sklearn.uncertainty.base_uncertainty_model import BaseUncertaintyModel
from greykite.sklearn.uncertainty.exceptions import UncertaintyError


class SimpleConditionalResidualsModel(BaseUncertaintyModel):
    """The simple conditional residuals uncertainty model.
    For more details, see `~greykite.algo.uncertainty.conditional.conf_interval`.

    Attributes
    ----------
    UNCERTAINTY_METHOD : `str`
        The name for the uncertainty method.
    REQUIRED_PARAMS : `list` [`str`]
        A list of required parameters for the method.
    coverage : `float`
        The coverage of the uncertainty intervals.
        Will be used to calculate the quantiles.
    time_col : `str`
        The column name for timestamps in ``train_df`` and ``fut_df``.
    value_col : `str` or None
        The column name for values in ``train_df`` and ``fut_df``.
    residual_col : `str` or None
        The column name for residuals in ``train_df``.
    conditional_cols : `list` [`str`]
        The conditional columns when calculating the standard errors of the residuals.
    """

    UNCERTAINTY_METHOD = "simple_conditional_residuals"
    # Parameters required by the core algo function
    # `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`.
    # The class will try to infer this parameter from constants,
    # but it's preferred to provide this parameter to ensure correctness.
    REQUIRED_PARAMS = ["value_col"]

    def __init__(
            self,
            uncertainty_dict: Dict[str, any],
            coverage: Optional[float] = None,
            time_col: Optional[str] = None,
            **kwargs):
        super().__init__(
            uncertainty_dict=uncertainty_dict,
            **kwargs)

        self.coverage = coverage
        self.time_col = time_col

        # Set by ``fit`` method.
        self.value_col: Optional[str] = None
        self.residual_col: Optional[str] = None
        self.conditional_cols: Optional[List[str]] = None

    def _check_input(self):
        """Checks that necessary input are provided in ``self.uncertainty_dict`` and ``self.train_df``.
        This check only raises
        `~greykite.sklearn.uncertainty.exceptions.UncertaintyError`.
        """
        super()._check_input()
        # Checks the uncertainty method matches.
        self.uncertainty_method = self.uncertainty_dict.get("uncertainty_method")
        if self.uncertainty_method != self.UNCERTAINTY_METHOD:
            raise UncertaintyError(
                f"The uncertainty method {self.uncertainty_method} is not as expected {self.UNCERTAINTY_METHOD}."
            )
        # Gets the parameters.
        self.params = self.uncertainty_dict.get("params", {})
        # Tries to populate value column if it is not given.
        if self.params.get("value_col", None) is None and cst.VALUE_COL in self.train_df.columns:
            self.params["value_col"] = cst.VALUE_COL
        # Checks all required parameters are given.
        for required_param in self.REQUIRED_PARAMS:
            if required_param not in self.params:
                raise UncertaintyError(
                    f"The parameter {required_param} is required but not found in "
                    f"`uncertainty['params']` {self.params}. "
                    f"The required parameters are {self.REQUIRED_PARAMS}."
                )
        # Checks value column.
        self.value_col = self.params.get("value_col")
        if self.value_col is None or not isinstance(self.value_col, str):
            raise UncertaintyError(f"`value_col` has to be a string, but found {self.value_col}.")
        if self.value_col not in self.train_df:
            raise UncertaintyError(f"`value_col` {self.value_col} not found in `train_df`.")
        # Checks residual column.
        self.residual_col = self.params.get("residual_col")
        if self.residual_col is not None and not isinstance(self.residual_col, str):
            raise UncertaintyError(
                f"`residual_col` has to be a string or None, but found {self.residual_col}.")
        # Residuals are calculated if ``PREDICT_COL`` exists.
        # The method in this class always looks for using residual based approach.
        if self.residual_col is not None and self.residual_col not in self.train_df.columns:
            if cst.PREDICTED_COL in self.train_df.columns:
                log_message(
                    message=f"`residual_col` {self.residual_col} is given but not found in `train_df.columns`, "
                            f"however, the prediction column {cst.PREDICTED_COL} is found. "
                            f"Calculating residuals based on the prediction column.",
                    level=LoggingLevelEnum.INFO
                )
                self.train_df[self.residual_col] = self.train_df[self.value_col] - self.train_df[cst.PREDICTED_COL]
            else:
                raise UncertaintyError(
                    f"`residual_col` {self.residual_col} not found in `train_df.columns`."
                )
        # Tries to build conditional features based on time column.
        if self.time_col is not None and self.time_col in self.train_df.columns:
            self.train_df = add_time_features_df(
                df=self.train_df,
                time_col=self.time_col,
                conti_year_origin=0  # this only affects the ``ct`` columns, which are not expected as conditional cols
            )
        elif cst.TIME_COL in self.train_df.columns:
            self.train_df = add_time_features_df(
                df=self.train_df,
                time_col=cst.TIME_COL,
                conti_year_origin=0  # this only affects the ``ct`` columns, which are not expected as conditional cols
            )
        # Checks conditional columns.
        self.conditional_cols = self.params.get("conditional_cols")
        if isinstance(self.conditional_cols, str):
            self.conditional_cols = [self.conditional_cols]
        if self.conditional_cols is not None:
            if (not isinstance(self.conditional_cols, list)
                    or any([not isinstance(col, str) for col in self.conditional_cols])):
                raise UncertaintyError(
                    f"`conditional_cols` {self.conditional_cols} must be a list of strings."
                )
            cols_not_in_df = [col for col in self.conditional_cols if col not in self.train_df.columns]
            if cols_not_in_df:
                raise UncertaintyError(
                    f"The following conditional columns are not found in `train_df`: {cols_not_in_df}."
                )
        # If ``self.coverage`` is given and ``quantiles`` is not given in ``params``,
        # the coverage is used to calculate quantiles.
        if self.coverage is not None:
            if self.coverage <= 0 or self.coverage >= 1:
                raise UncertaintyError(
                    f"Coverage must be between 0 and 1, found {self.coverage}"
                )
            if self.params is not None:
                quantiles = self.params.get("quantiles")
                if quantiles is None:
                    alpha = (1 + self.coverage) / 2
                    self.params["quantiles"] = (1 - alpha, alpha)

    def fit(
            self,
            train_df: pd.DataFrame):
        """Fits the uncertainty model.

        Parameters
        ----------
        train_df : `pandas.DataFrame`
            The data used to fit the uncertainty model.
            Must have the following columns:

                value_col: this is required
                time_col: if ``time_col`` is given.
                residual_col: if ``residual_col`` is given but ``PREDICT_COL`` does not exist.
                conditional_cols: if ``conditional_cols`` is given.

        Returns
        -------
        None
        """
        super().fit(train_df=train_df)
        self._check_input()

        self.uncertainty_model = conf_interval(
            df=self.train_df,
            **self.params)

    def predict(
            self,
            fut_df: pd.DataFrame):
        """Predicts the interval.

        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The dataframe used for prediction.
            Must have the following columns:

                value_col: the value column.
                PREDICT_COL: if this column exists, it will be used instead of ``value_col``.

        Returns
        -------
        result_df : `pandas.DataFrame`
            The ``fut_df`` augmented with prediction intervals.
        """
        fut_df = fut_df.copy()
        # Checks ``value_col`` in ``fut_df``.
        if self.value_col not in fut_df.columns:
            raise UncertaintyError(
                f"The value column {self.value_col} is not found in `fut_df`."
            )
        # Records the originals columns to be output.
        output_cols = list(fut_df.columns)
        # Tries to build conditional features based on time column.
        if self.time_col is not None and self.time_col in fut_df:
            fut_df = add_time_features_df(
                df=fut_df,
                time_col=self.time_col,
                conti_year_origin=0  # this only affects the ``ct`` columns, which are not expected as conditional cols
            )
        elif cst.TIME_COL in fut_df.columns:
            fut_df = add_time_features_df(
                df=fut_df,
                time_col=cst.TIME_COL,
                conti_year_origin=0  # this only affects the ``ct`` columns, which are not expected as conditional cols
            )

        # If ``PREDICT_COL`` is in the df,
        # it will be used instead of ``value_col``.
        # The original ``value_col`` will be recorded and appended in the output.
        value_col = None
        if cst.PREDICTED_COL in fut_df.columns:
            if self.value_col in fut_df.columns:
                value_col = fut_df[self.value_col]
            fut_df[self.value_col] = fut_df[cst.PREDICTED_COL]

        # Predict.
        pred_df_with_uncertainty = predict_ci(
            fut_df,
            self.uncertainty_model)
        # Adds uncertainty column to df
        pred_df_with_uncertainty.reset_index(drop=True, inplace=True)
        fut_df.reset_index(drop=True, inplace=True)
        fut_df[f"{self.value_col}_quantile_summary"] = (
            pred_df_with_uncertainty[f"{self.value_col}_quantile_summary"])
        fut_df[cst.ERR_STD_COL] = pred_df_with_uncertainty[cst.ERR_STD_COL]
        fut_df[cst.PREDICTED_LOWER_COL] = fut_df[f"{self.value_col}_quantile_summary"].str[0]
        fut_df[cst.PREDICTED_UPPER_COL] = fut_df[f"{self.value_col}_quantile_summary"].str[-1]
        output_cols += [cst.PREDICTED_LOWER_COL, cst.PREDICTED_UPPER_COL, cst.ERR_STD_COL]
        if value_col is not None:
            fut_df[self.value_col] = value_col
        return fut_df[output_cols]
