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
"""Defines the quantile regression based uncertainty model."""

from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from sklearn.exceptions import NotFittedError

from greykite.algo.common.l1_quantile_regression import QuantileRegression
from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_fourier_feature_col_names
from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_silverkite_uncertainty_dict
from greykite.common import constants as cst
from greykite.common.features.timeseries_features import fourier_series_multi_fcn
from greykite.common.features.timeseries_features import get_default_origin_for_time_vars
from greykite.sklearn.uncertainty.base_uncertainty_model import BaseUncertaintyModel
from greykite.sklearn.uncertainty.exceptions import UncertaintyError


class QuantileRegressionUncertaintyModel(BaseUncertaintyModel):
    """The quantile regression based uncertainty model.
    Predicts quantiles as the prediction intervals.
    The quantiles are calculated based on the ``coverage`` parameter.
    For example, a 90% prediction interval is the 0.05 quantile and 0.95 quantile predictions.

    There are four scenarios:

        - ``is_residual_based = False`` and ``x_mat`` is provided.
          In this case, the method fits two quantile regression models based on the ``value_col`` and ``x_mat``.
          Depending on the forecast model, the generated prediction intervals may not cover the forecasted values.
        - ``is_residual_based = False`` and ``x_mat`` is not provided.
          In this case, a ``x_mat`` is generated based on ``time_col``
          including linear growth and yearly/weekly seasonality.
          Currently we do not support customizing the generation of this feature matrix.
          The method fits two quantile regression models based on ``value_col`` and the generated ``x_mat``.
          Depending on the forecast model, the generated prediction intervals may not cover the forecasted values.
        - ``is_residual_based = True`` and ``x_mat`` is provided.
          In this case, the ``predicted_col`` needs to be provided
          in addition to ``value_col`` to calculate the residuals.
          The method fits two quantile regression models based on the residuals and ``x_mat``.
          The fitted residual quantiles are added to the forecasted values.
        - ``is_residual_based = True`` and ``x_mat`` is not provided.
          In this case, the ``predicted_col`` needs to be provided
          in addition to ``value_col`` to calculate the residuals.
          In this case, a ``x_mat`` is generated based on ``time_col``
          including linear growth and yearly/weekly seasonality.
          Currently we do not support customizing the generation of this feature matrix.
          The method fits two quantile regression models based on residuals and the generated ``x_mat``.
          The fitted residual quantiles are added to the forecasted values.

    If ``x_mat`` is not provided, ``time_col`` must be provided to build the feature matrix.

    If ``is_residual_based = False``, then ``value_col`` must be provided to train the model.
    If ``is_residual_based = True``, then both ``value_col`` and ``predicted_col`` must be provided.
    In this case, the method also tries to find ``cst.PREDICTED_COL`` in the training data
    too if ``predicted_col`` is not provided.

    Attributes
    ----------
    uncertainty_dict : `dict` [`str`, any]
        The uncertainty model specification. It should have the following keys:

                "uncertainty_method": a string that is in
                    `~greykite.sklearn.uncertainty.uncertainty_methods.UncertaintyMethodEnum`.
                "params": a dictionary that includes any additional parameters needed by the uncertainty method.
                          Expected keys include: "value_col", "predicted_col", "is_residual_based"
                          and "quantiles".

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
    predicted_col : `str` or None
        The column name for predicted values in ``train_df`` and ``fut_df``.
    residual_col : `str` or None
        The column name for residuals in ``train_df``.
    quantiles : `list` [`float`] or None
        A list of quantiles to be fitted.
        This is also derived from the ``coverage`` parameter.
    is_residual_based : `bool` or None
        Whether to fit the original values or residuals.
    models : `list` [`QuantileRegression`] or None
        A list of trained quantile regression models.
    x_mat : `pandas.DataFrame` or None
        The feature matrix used to fit the model.
    build_x_mat : `bool` or None
        Whether the feature matrix is built. If False, it is provided.
    distribution_col : `str` or None
        The column name for the column used as response in fitting the quantile regression models.
    offset_col : `str` or None
        The column name for the column used to offset the predicted intervals.
    """

    UNCERTAINTY_METHOD = "quantile_regression"
    # Required parameters.
    # We need the value column
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

        # Derived column names.
        self.distribution_col: Optional[str] = None
        self.offset_col: Optional[str] = None

        # Set by ``fit`` method.
        self.is_residual_based: Optional[bool] = None
        self.x_mat: Optional[pd.DataFrame] = None
        self.build_x_mat: Optional[bool] = None
        self.value_col: Optional[str] = None
        self.residual_col: Optional[str] = None
        self.quantiles: Optional[List[float]] = None
        self.models: Optional[List[QuantileRegression]] = None

    def _check_input(self):
        """Checks that necessary input are provided in ``self.uncertainty_dict`` and ``self.train_df``.
        This check only raises
        `~greykite.sklearn.uncertainty.exceptions.UncertaintyError`.
        """
        super()._check_input()

        # Gets whether it's residual based.
        # If not provided, the default is not residual based.
        self.is_residual_based = self.params.get("is_residual_based", False)

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
                if cst.PREDICTED_COL in self.train_df.columns:
                    self.predicted_col = cst.PREDICTED_COL
                else:
                    raise UncertaintyError(f"`predicted_col` is needed for offsetting predictions when "
                                           f"`is_residual_based=True`, but it is not provided.")
            self.offset_col = self.predicted_col

            # Creates ``residual_col`` with ``value_col`` and ``predicted_col``.
            self.residual_col = cst.RESIDUAL_COL
            self.distribution_col = self.residual_col
            self.train_df[cst.RESIDUAL_COL] = self.train_df[self.value_col] - self.train_df[self.predicted_col]

        # Checks columns needed when ``x_mat`` is not provided.
        # This includes the time column.
        self.build_x_mat = False
        if self.x_mat is None:
            self.build_x_mat = True  # Flag to build x mat in the prediction phase too
            if self.time_col is None:
                self.time_col = self.params.get("time_col")
            if self.time_col is None and cst.TIME_COL in self.train_df:
                self.time_col = cst.TIME_COL
            if self.time_col is None:
                raise UncertaintyError(
                    "Time column must be provided when `x_mat` is not given. "
                    "It is used to generate necessary features for the uncertainty model."
                )

        if self.coverage is None:
            self.coverage = self.DEFAULT_COVERAGE
        # If ``self.coverage`` is given and ``quantiles`` is not given in ``params``,
        # the coverage is used to calculate quantiles.
        uncertainty_dict = get_silverkite_uncertainty_dict(
            uncertainty=self.uncertainty_dict,
            coverage=self.coverage)
        self.quantiles = uncertainty_dict["params"]["quantiles"]

    def _build_quantile_regression_features(
            self,
            training_phase: bool = True,
            fut_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Builds a default ``self.x_mat`` if it is not given.
        Currently the class does not support customizing the built feature matrix.

        The built matrix has the following components:

            - growth: linear
            - seasonality: yearly/weekly

        Parameters
        ----------
        training_phase : `bool`, default True
            Whether it's training phase or prediction phase.
            For training phase, it uses ``self.train_df``.
        fut_df : `pandas.DataFrame` or None, default None
            If ``training_phase`` is False, this must be provided.
            Used to build the feature matrix.
            Includes the ``time_col`` to build features.

        Returns
        -------
        x_mat : `pandas.DataFrame`
            The feature matrix.
        """
        columns = ["ct1"]
        if training_phase:
            df = self.train_df[[self.time_col]].copy()
        else:
            if fut_df is None:
                raise UncertaintyError(
                    "Creating quantile regression feature matrix for future phase "
                    "requires `fut_df`."
                )
            df = fut_df[[self.time_col]].copy()

        # Includes yearly/weekly seasonality features.
        fs_components_df = pd.DataFrame({
            "name": [
                cst.TimeFeaturesEnum.tow.value,
                cst.TimeFeaturesEnum.toy.value],
            "period": [7.0, 1.0],
            "order": [4, 10],
            "seas_names": ["weekly", "yearly"]})
        fs_func = fourier_series_multi_fcn(
            col_names=fs_components_df.get("name"),
            periods=fs_components_df.get("period"),
            orders=fs_components_df.get("order"),
            seas_names=fs_components_df.get("seas_names")
        )
        # Gets the fourier columns
        fs_cols = get_fourier_feature_col_names(
            df=df,
            time_col=self.time_col,
            fs_func=fs_func
        )
        columns += fs_cols

        silverkite = SilverkiteForecast()
        x_mat = silverkite._SilverkiteForecast__build_silverkite_features(
            df=df,
            time_col=self.time_col,
            origin_for_time_vars=get_default_origin_for_time_vars(
                df=self.train_df[[self.time_col]].copy(),
                time_col=self.time_col
            ),
            continuous_time_col="ct1",
            growth_func=lambda x: x,
            fs_func=fs_func)
        return x_mat[columns]

    def fit(
            self,
            train_df: pd.DataFrame,
            x_mat: Optional[pd.DataFrame] = None) -> QuantileRegressionUncertaintyModel:
        """Fits the uncertainty model.

        Parameters
        ----------
        train_df : `pandas.DataFrame`
            The data used to fit the uncertainty model.
            Must have the following columns:

                value_col: this is required
                time_col: if ``x_mat`` is not provided.

            If ``is_residual_based = True``, ``train_df`` must include predicted values
            to calculate residuals. The predicted value column is ``predicted_col``
            if specified in ``uncertainty_dict``, default ``cst.PREDICTED_COL`` if not.

        x_mat : `pandas.DataFrame` or None, default None
            The feature matrix used to train the quantile regression model.
            Must have the same length as ``train_df``.
            If not provided, a feature matrix will be automatically generated to include
            linear growth and yearly/weekly seasonality.

        Returns
        -------
        self
        """
        super().fit(train_df=train_df)
        self.x_mat = x_mat
        self._check_input()

        # Builds the feature matrix if needed.
        if self.x_mat is None:
            self.x_mat = self._build_quantile_regression_features(
                training_phase=True
            )
        else:
            if len(train_df) != len(x_mat):
                raise UncertaintyError(f"The size of `train_df` {len(train_df)} must be the same "
                                       f"as the size of `x_mat` {len(x_mat)}.")

        # Trains the quantile regression model.
        self.models = [
            QuantileRegression(
                quantile=quantile,
                alpha=0
            ).fit(self.x_mat, self.train_df[self.distribution_col]) for quantile in self.quantiles
        ]

        return self

    def predict(
            self,
            fut_df: pd.DataFrame,
            x_mat: Optional[pd.DataFrame] = None):
        """Predicts the interval.

        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The dataframe used for prediction.
            Must have the following columns:

                time_col: if ``x_mat`` is not given during the training phase.
                predicted_col: if ``is_residual_based=True``, used to offset the predictions.

        x_mat : `pandas.DataFrame` or None, default None
            The feature matrix used for prediction.
            If an ``x_mat`` was provided during training, it should be provided during prediction too.

        Returns
        -------
        result_df : `pandas.DataFrame`
            The ``fut_df`` augmented with prediction intervals.
        """
        if self.models is None:
            raise NotFittedError("Please train the uncertainty model first.")
        fut_df = fut_df.copy()
        # Records the originals columns to be output.
        output_cols = list(fut_df.columns)

        # Builds x mat if needed.
        if self.build_x_mat:
            if self.time_col not in fut_df:
                raise UncertaintyError(f"Time column {self.time_col} not found in `fut_df`.")
            x_mat = self._build_quantile_regression_features(
                training_phase=False,
                fut_df=fut_df
            )
        else:
            if x_mat is None:
                raise UncertaintyError("Please provide `x_mat` for prediction.")
            if len(fut_df) != len(x_mat):
                raise UncertaintyError(f"The size of `fut_df` {len(fut_df)} must be the same "
                                       f"as the size of `x_mat` {len(x_mat)}.")
            try:
                x_mat = x_mat[self.x_mat.columns]
            except KeyError:
                raise UncertaintyError(f"The `x_mat` does not have all columns used to train the uncertainty model. "
                                       f"The following columns are missing: "
                                       f"{[col for col in self.x_mat.columns if col not in x_mat.columns]}.")

        # Makes predictions to the quantiles.
        predictions = [model.predict(x_mat) for model in self.models]

        # Calculates the offset values.
        if self.is_residual_based:
            offset_col = fut_df[self.offset_col].values
        else:
            offset_col = 0

        # Creates the quantile columns.
        for quantile, prediction in zip(self.quantiles, predictions):
            fut_df[f"{cst.PREDICTED_COL}_{round(quantile, 3)}"] = prediction + offset_col

        # Sets the upper/lower bounds.
        fut_df[cst.PREDICTED_LOWER_COL] = fut_df[f"{cst.PREDICTED_COL}_{round(min(self.quantiles), 3)}"]
        fut_df[cst.PREDICTED_UPPER_COL] = fut_df[f"{cst.PREDICTED_COL}_{round(max(self.quantiles), 3)}"]

        output_cols += [cst.PREDICTED_LOWER_COL, cst.PREDICTED_UPPER_COL]
        output_cols += [f"{cst.PREDICTED_COL}_{round(quantile, 3)}" for quantile in self.quantiles]

        return fut_df[output_cols]
