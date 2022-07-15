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

from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_silverkite_uncertainty_dict
from greykite.algo.uncertainty.conditional.conf_interval import conf_interval
from greykite.algo.uncertainty.conditional.conf_interval import predict_ci
from greykite.common import constants as cst
from greykite.common.features.timeseries_features import add_time_features_df
from greykite.sklearn.uncertainty.base_uncertainty_model import BaseUncertaintyModel
from greykite.sklearn.uncertainty.exceptions import UncertaintyError


class SimpleConditionalResidualsModel(BaseUncertaintyModel):
    """The simple conditional residuals uncertainty model.
    For more details, see `~greykite.algo.uncertainty.conditional.conf_interval`.

    Attributes
    ----------
    uncertainty_dict : `dict` [`str`, any]
        The uncertainty model specification. It should have the following keys:

                "uncertainty_method": a string that is in
                    `~greykite.sklearn.uncertainty.uncertainty_methods.UncertaintyMethodEnum`.
                "params": a dictionary that includes any additional parameters needed by the uncertainty method.

    UNCERTAINTY_METHOD : `str`
        The name for the uncertainty method.
    REQUIRED_PARAMS : `list` [`str`]
        A list of required parameters for the method.
    coverage : `float`
        The coverage of the uncertainty intervals.
        Will be used to calculate the quantiles.
    time_col : `str`
        The column name for timestamps in ``train_df`` and ``fut_df``.
    value_col : `str`
        The column name for values in ``train_df`` and ``fut_df``.
    distribution_col : `str` or None
        The column name for the column used to fit the distribution in ``train_df`` and ``fut_df``.
    residual_col : `str` or None
        The column name for residuals in ``train_df``.
    conditional_cols : `list` [`str`]
        The conditional columns when calculating the standard errors of the residuals.
    offset_col : `str` or None
        The column name for the column used to offset predicted intervals.
    is_residual_based : `bool` or None
        Whether to fit the original values or residuals.
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
        self.distribution_col: Optional[str] = None
        self.value_col: Optional[str] = None
        self.predicted_col: Optional[str] = None
        self.residual_col: Optional[str] = None
        self.conditional_cols: Optional[List[str]] = None
        self.offset_col: Optional[str] = None
        self.is_residual_based: Optional[bool] = None

        # Set by ``predict`` method
        self.pred_df = None

    def _check_input(self):
        """Checks that necessary input are provided in ``self.uncertainty_dict`` and ``self.train_df``.
        This check only raises
        `~greykite.sklearn.uncertainty.exceptions.UncertaintyError`.
        """
        super()._check_input()

        # Gets whether it's residual based. If not provided, the default is residual based.
        self.is_residual_based = self.params.get("is_residual_based", True)
        if not self.is_residual_based:
            raise UncertaintyError(f"'is_residual_based' must be True when "
                                   f"the uncertainty method is {self.UNCERTAINTY_METHOD}.")

        # Checks value column.
        self.value_col = self.params.get("value_col")
        if self.value_col not in self.train_df:
            raise UncertaintyError(f"`value_col` {self.value_col} not found in `train_df`.")

        # Checks columns needed when ``is_residual_based = True``.
        # If residual based, ``predicted_col`` must be provided.
        # Also looks for ``cst.PREDICTED_COL`` if the above is not provided.
        self.distribution_col = self.value_col
        if self.is_residual_based:
            # Tries to get predicted column from parameters and set ``offset_col``.
            self.predicted_col = self.params.get("predicted_col")
            if self.predicted_col is None:
                self.predicted_col = cst.PREDICTED_COL
            self.offset_col = self.predicted_col

            # Creates ``residual_col`` with ``value_col`` and ``predicted_col``.
            self.residual_col = cst.RESIDUAL_COL
            self.train_df[cst.RESIDUAL_COL] = self.train_df[self.value_col] - self.train_df[self.predicted_col]
            self.distribution_col = self.residual_col

        self.params["distribution_col"] = self.distribution_col
        self.params["offset_col"] = self.offset_col

        # Gets the valid parameters for ``conf_interval``.
        valid_params = ["distribution_col", "offset_col", "conditional_cols",
                        "quantiles", "quantile_estimation_method",
                        "sample_size_thresh", "small_sample_size_method",
                        "small_sample_size_quantile", "min_admissible_value",
                        "max_admissible_value"]
        self.params = {k: v for k, v in self.params.items() if k in valid_params}

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

        if self.coverage is None:
            self.coverage = self.DEFAULT_COVERAGE
        # If ``self.coverage`` is given and ``quantiles`` is not given in ``params``,
        # the coverage is used to calculate quantiles.
        uncertainty_dict = get_silverkite_uncertainty_dict(
            uncertainty=self.uncertainty_dict,
            coverage=self.coverage)
        self.quantiles = uncertainty_dict["params"]["quantiles"]
        self.params["quantiles"] = self.quantiles

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
        fut_df = fut_df.reset_index(drop=True)
        # Checks ``self.offset_col`` in ``fut_df``.
        if self.offset_col is not None and self.offset_col not in fut_df.columns:
            raise UncertaintyError(
                f"The offset column {self.offset_col} is not found in `fut_df`."
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

        # Predict.
        fut_df = predict_ci(
            fut_df,
            self.uncertainty_model)
        fut_df[cst.PREDICTED_LOWER_COL] = fut_df[cst.QUANTILE_SUMMARY_COL].str[0]
        fut_df[cst.PREDICTED_UPPER_COL] = fut_df[cst.QUANTILE_SUMMARY_COL].str[-1]
        self.pred_df = fut_df
        output_cols += [cst.PREDICTED_LOWER_COL, cst.PREDICTED_UPPER_COL, cst.ERR_STD_COL]

        return fut_df[output_cols]
