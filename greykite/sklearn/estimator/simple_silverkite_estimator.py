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
# original author: Albert Chen
"""sklearn estimator for ``simple_forecast_silverkite``"""

from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from sklearn.metrics import mean_squared_error

from greykite.algo.forecast.silverkite.forecast_simple_silverkite import SimpleSilverkiteForecast
from greykite.common import constants as cst
from greykite.common.python_utils import update_dictionary
from greykite.sklearn.estimator.base_silverkite_estimator import BaseSilverkiteEstimator
from greykite.sklearn.estimator.silverkite_diagnostics import SilverkiteDiagnostics
from greykite.sklearn.uncertainty.uncertainty_methods import UncertaintyMethodEnum


class SimpleSilverkiteEstimator(BaseSilverkiteEstimator):
    """Wrapper for `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.forecast_simple_silverkite`.

    Parameters
    ----------
    score_func : callable, optional, default mean_squared_error
        See `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`.
    coverage : `float` between [0.0, 1.0] or None, optional
        See `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`.
    null_model_params : `dict` or None, optional
        Dictionary with arguments to define ``DummyRegressor`` null model, default is `None`.
        See `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`.
    fit_algorithm_dict : `dict` or None, optional
        How to fit the model. A dictionary with the following optional keys.

            ``"fit_algorithm"`` : `str`, optional, default "ridge"
                The type of predictive model used in fitting.

                See `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
                for available options and their parameters.
            ``"fit_algorithm_params"`` : `dict` or None, optional, default None
                Parameters passed to the requested fit_algorithm.
                If None, uses the defaults in `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.

    uncertainty_dict : `dict` or `str` or None, optional
        How to fit the uncertainty model.
        See `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
        Note that this is allowed to be "auto". If None or "auto", will be set to
        a default value by ``coverage`` before calling ``forecast_silverkite``.
        See ``BaseForecastEstimator`` for details.

    kwargs : additional parameters

        Other parameters are the same as in
        `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.forecast_simple_silverkite`.

        See source code ``__init__`` for the parameter names, and refer to
        `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.forecast_simple_silverkite` for
        their description.

        If this Estimator is called from
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`,
        ``train_test_thresh`` and ``training_fraction`` should almost
        always be `None`, because train/test is handled outside
        this Estimator.

    Notes
    -----
    Attributes match those of
    `~greykite.sklearn.estimator.base_silverkite_estimator.BaseSilverkiteEstimator`.

    See Also
    --------
    `~greykite.sklearn.estimator.base_silverkite_estimator.BaseSilverkiteEstimator`
        For attributes and details on fit, predict, and component plots.
    `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.forecast_simple_silverkite`
        Function to transform the parameters to call ``forecast_silverkite`` fit.
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`
        Functions performing the fit and predict.
    """
    def __init__(
            self,
            silverkite: SimpleSilverkiteForecast = SimpleSilverkiteForecast(),
            silverkite_diagnostics: SilverkiteDiagnostics = SilverkiteDiagnostics(),
            score_func: callable = mean_squared_error,
            coverage: float = None,
            null_model_params: Optional[Dict] = None,
            time_properties: Optional[Dict] = None,
            freq: Optional[str] = None,
            forecast_horizon: Optional[int] = None,
            origin_for_time_vars: Optional[float] = None,
            train_test_thresh: Optional[datetime] = None,
            training_fraction: Optional[float] = None,
            fit_algorithm_dict: Optional[Dict] = None,
            auto_holiday: bool = False,
            holidays_to_model_separately: Optional[Union[str, List[str]]] = "auto",
            holiday_lookup_countries: Optional[Union[str, List[str]]] = "auto",
            holiday_pre_num_days: int = 2,
            holiday_post_num_days: int = 2,
            holiday_pre_post_num_dict: Optional[Dict] = None,
            daily_event_df_dict: Optional[Dict] = None,
            auto_growth: bool = False,
            changepoints_dict: Optional[Dict] = None,
            auto_seasonality: bool = False,
            yearly_seasonality: Union[bool, str, int] = "auto",
            quarterly_seasonality: Union[bool, str, int] = "auto",
            monthly_seasonality: Union[bool, str, int] = "auto",
            weekly_seasonality: Union[bool, str, int] = "auto",
            daily_seasonality: Union[bool, str, int] = "auto",
            max_daily_seas_interaction_order: Optional[int] = None,
            max_weekly_seas_interaction_order: Optional[int] = None,
            autoreg_dict: Optional[Dict] = None,
            past_df: Optional[pd.DataFrame] = None,
            lagged_regressor_dict: Optional[Dict] = None,
            seasonality_changepoints_dict: Optional[Dict] = None,
            min_admissible_value: Optional[float] = None,
            max_admissible_value: Optional[float] = None,
            uncertainty_dict: Optional[Dict] = None,
            normalize_method: Optional[str] = None,
            growth_term: Optional[str] = cst.GrowthColEnum.linear.name,
            regressor_cols: Optional[List[str]] = None,
            feature_sets_enabled: Optional[Union[bool, Dict[str, bool]]] = None,
            extra_pred_cols: Optional[List[str]] = None,
            drop_pred_cols: Optional[List[str]] = None,
            explicit_pred_cols: Optional[List[str]] = None,
            regression_weight_col: Optional[str] = None,
            simulation_based: Optional[bool] = False,
            simulation_num: int = 10,
            fast_simulation: bool = False):
        # every subclass of BaseSilverkiteEstimator must call super().__init__
        super().__init__(
            silverkite=silverkite,
            silverkite_diagnostics=silverkite_diagnostics,
            score_func=score_func,
            coverage=coverage,
            null_model_params=null_model_params,
            uncertainty_dict=uncertainty_dict)

        # necessary to set parameters, to ensure get_params() works
        # (used in grid search)
        self.score_func = score_func
        self.coverage = coverage
        self.null_model_params = null_model_params
        self.time_properties = time_properties
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.origin_for_time_vars = origin_for_time_vars
        self.train_test_thresh = train_test_thresh
        self.training_fraction = training_fraction
        self.fit_algorithm_dict = fit_algorithm_dict
        self.auto_holiday = auto_holiday
        self.holidays_to_model_separately = holidays_to_model_separately
        self.holiday_lookup_countries = holiday_lookup_countries
        self.holiday_pre_num_days = holiday_pre_num_days
        self.holiday_post_num_days = holiday_post_num_days
        self.holiday_pre_post_num_dict = holiday_pre_post_num_dict
        self.daily_event_df_dict = daily_event_df_dict
        self.auto_growth = auto_growth
        self.changepoints_dict = changepoints_dict
        self.auto_seasonality = auto_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.quarterly_seasonality = quarterly_seasonality
        self.monthly_seasonality = monthly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.max_daily_seas_interaction_order = max_daily_seas_interaction_order
        self.max_weekly_seas_interaction_order = max_weekly_seas_interaction_order
        self.autoreg_dict = autoreg_dict
        self.past_df = past_df
        self.lagged_regressor_dict = lagged_regressor_dict
        self.seasonality_changepoints_dict = seasonality_changepoints_dict
        self.min_admissible_value = min_admissible_value
        self.max_admissible_value = max_admissible_value
        self.uncertainty_dict = uncertainty_dict
        self.normalize_method = normalize_method
        self.growth_term = growth_term
        self.regressor_cols = regressor_cols
        self.feature_sets_enabled = feature_sets_enabled
        self.extra_pred_cols = extra_pred_cols
        self.drop_pred_cols = drop_pred_cols
        self.explicit_pred_cols = explicit_pred_cols
        self.regression_weight_col = regression_weight_col
        self.simulation_based = simulation_based
        self.simulation_num = simulation_num
        self.fast_simulation = fast_simulation
        # ``forecast_simple_silverkite`` generates a ``fs_components_df`` to call
        # ``forecast_silverkite`` that is compatible with ``BaseSilverkiteEstimator``.
        # Unlike ``SilverkiteEstimator``, this does not need to call ``validate_inputs``.

    def fit(
            self,
            X,
            y=None,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            **fit_params):
        """Fits ``Silverkite`` forecast model.

        Parameters
        ----------
        X: `pandas.DataFrame`
            Input timeseries, with timestamp column,
            value column, and any additional regressors.
            The value column is the response, included in
            ``X`` to allow transformation by `sklearn.pipeline`.
        y: ignored
            The original timeseries values, ignored.
            (The ``y`` for fitting is included in ``X``).
        time_col: `str`
            Time column name in ``X``.
        value_col: `str`
            Value column name in ``X``.
        fit_params: `dict`
            additional parameters for null model.

        Returns
        -------
        self : self
            Fitted model is stored in ``self.model_dict``.
        """
        # Initializes `fit_algorithm_dict` with default values.
        # This cannot be done in __init__ to remain compatible
        # with sklearn grid search.
        default_fit_algorithm_dict = {
            "fit_algorithm": "ridge",
            "fit_algorithm_params": None}
        self.fit_algorithm_dict = update_dictionary(
            default_fit_algorithm_dict,
            overwrite_dict=self.fit_algorithm_dict)

        # Fits null model
        super().fit(
            X=X,
            y=y,
            time_col=time_col,
            value_col=value_col,
            **fit_params)

        # The uncertainty method has been filled as "simple_conditional_residuals" in ``super().fit`` above if
        # ``coverage`` is given but ``uncertainty_dict`` is not given.
        # In the case that the method is "simple_conditional_residuals",
        # we use SimpleSilverkiteForecast to fit it, because under the situation of AR simulation,
        # those information are needed in generating the prediction intervals.
        # In all other cases, we fit the uncertainty model separately.
        uncertainty_dict = None
        if self.uncertainty_dict is not None:
            uncertainty_method = self.uncertainty_dict.get("uncertainty_method", None)
            if uncertainty_method == UncertaintyMethodEnum.simple_conditional_residuals.name:
                uncertainty_dict = self.uncertainty_dict

        self.model_dict = self.silverkite.forecast_simple(
            df=X,
            time_col=time_col,
            value_col=value_col,
            time_properties=self.time_properties,
            freq=self.freq,
            forecast_horizon=self.forecast_horizon,
            origin_for_time_vars=self.origin_for_time_vars,
            train_test_thresh=self.train_test_thresh,
            training_fraction=self.training_fraction,
            fit_algorithm=self.fit_algorithm_dict["fit_algorithm"],
            fit_algorithm_params=self.fit_algorithm_dict["fit_algorithm_params"],
            auto_holiday=self.auto_holiday,
            holidays_to_model_separately=self.holidays_to_model_separately,
            holiday_lookup_countries=self.holiday_lookup_countries,
            holiday_pre_num_days=self.holiday_pre_num_days,
            holiday_post_num_days=self.holiday_post_num_days,
            holiday_pre_post_num_dict=self.holiday_pre_post_num_dict,
            daily_event_df_dict=self.daily_event_df_dict,
            auto_growth=self.auto_growth,
            changepoints_dict=self.changepoints_dict,
            auto_seasonality=self.auto_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            quarterly_seasonality=self.quarterly_seasonality,
            monthly_seasonality=self.monthly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            max_daily_seas_interaction_order=self.max_daily_seas_interaction_order,
            max_weekly_seas_interaction_order=self.max_weekly_seas_interaction_order,
            autoreg_dict=self.autoreg_dict,
            past_df=self.past_df,
            lagged_regressor_dict=self.lagged_regressor_dict,
            seasonality_changepoints_dict=self.seasonality_changepoints_dict,
            min_admissible_value=self.min_admissible_value,
            max_admissible_value=self.max_admissible_value,
            uncertainty_dict=uncertainty_dict,
            normalize_method=self.normalize_method,
            growth_term=self.growth_term,
            regressor_cols=self.regressor_cols,
            feature_sets_enabled=self.feature_sets_enabled,
            extra_pred_cols=self.extra_pred_cols,
            drop_pred_cols=self.drop_pred_cols,
            explicit_pred_cols=self.explicit_pred_cols,
            regression_weight_col=self.regression_weight_col,
            simulation_based=self.simulation_based,
            simulation_num=self.simulation_num,
            fast_simulation=self.fast_simulation)

        # Fits the uncertainty model if not already fit.
        if self.uncertainty_dict is not None and uncertainty_dict is None:
            # The quantile regression model.
            if uncertainty_method == UncertaintyMethodEnum.quantile_regression.name:
                fit_df = self.silverkite.predict(
                    X,
                    trained_model=self.model_dict
                )["fut_df"].rename(
                    columns={self.value_col_: cst.PREDICTED_COL}
                )[[self.time_col_, cst.PREDICTED_COL]]
                fit_df[self.value_col_] = X[self.value_col_].values
                x_mat = self.model_dict["x_mat"]

                default_params = {"is_residual_based": False}
                params = self.uncertainty_dict.get("params", {})
                default_params.update(params)

                fit_params = {"x_mat": x_mat}

                self.fit_uncertainty(
                    df=fit_df,
                    uncertainty_dict=self.uncertainty_dict,
                    fit_params=fit_params,
                    **default_params
                )

        # Sets attributes based on ``self.model_dict``
        super().finish_fit()
        return self
