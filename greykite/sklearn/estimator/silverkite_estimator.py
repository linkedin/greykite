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
"""sklearn estimator for ``forecast_silverkite``"""

import pandas as pd
from sklearn.metrics import mean_squared_error

from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.common import constants as cst
from greykite.common.python_utils import update_dictionary
from greykite.sklearn.estimator.base_silverkite_estimator import BaseSilverkiteEstimator
from greykite.sklearn.estimator.silverkite_diagnostics import SilverkiteDiagnostics


class SilverkiteEstimator(BaseSilverkiteEstimator):
    """Wrapper for `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

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

            ``"fit_algorithm"`` : `str`, optional, default "linear"
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
    fs_components_df : `pandas.DataFrame` or None, optional
        A dataframe with information about fourier series generation.
        If provided, it must contain columns with following names:

        - `"name"`: name of the timeseries feature (e.g. ``tod``, ``tow`` etc.).
        - `"period"`: Period of the fourier series.
        - `"order"`: Order of the fourier series.
          `"seas_names"`: Label for the type of seasonality (e.g. ``daily``, ``weekly`` etc.)
          and should be unique.
        - `~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator.validate_fs_components_df`
          checks for it, so that component plots don't have duplicate y-axis labels.

        This differs from the expected input of `forecast_silverkite` where `"period"`, `"order"`
        and `"seas_names"` are optional. This restriction is to facilitate appropriate computation
        of component (e.g. trend, seasonalities and holidays) effects. See Notes section in this
        docstring for a more detailed explanation with examples.


    Other parameters are the same as in
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

    If this Estimator is called from
    :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`,
    ``train_test_thresh`` and ``training_fraction`` should almost
    always be `None`, because train/test is handled outside
    this Estimator.

    The attributes are the same as
    `~greykite.sklearn.estimator.base_silverkite_estimator.BaseSilverkiteEstimator`.

    See Also
    --------
    `~greykite.sklearn.estimator.base_silverkite_estimator.BaseSilverkiteEstimator`
        For details on fit, predict, and component plots.
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`
        Functions performing the fit and predict.
    """
    def __init__(
            self,
            silverkite: SilverkiteForecast = SilverkiteForecast(),
            silverkite_diagnostics: SilverkiteDiagnostics = SilverkiteDiagnostics(),
            score_func=mean_squared_error,
            coverage=None,
            null_model_params=None,
            freq=None,
            origin_for_time_vars=None,
            extra_pred_cols=None,
            drop_pred_cols=None,
            explicit_pred_cols=None,
            train_test_thresh=None,
            training_fraction=None,
            fit_algorithm_dict=None,
            daily_event_df_dict=None,
            fs_components_df=pd.DataFrame({
                "name": [
                    cst.TimeFeaturesEnum.tod.value,
                    cst.TimeFeaturesEnum.tow.value,
                    cst.TimeFeaturesEnum.conti_year.value],
                "period": [24.0, 7.0, 1.0],
                "order": [3, 3, 5],
                "seas_names": ["daily", "weekly", "yearly"]}),
            autoreg_dict=None,
            past_df=None,
            lagged_regressor_dict=None,
            changepoints_dict=None,
            seasonality_changepoints_dict=None,
            changepoint_detector=None,
            min_admissible_value=None,
            max_admissible_value=None,
            uncertainty_dict=None,
            normalize_method=None,
            adjust_anomalous_dict=None,
            impute_dict=None,
            regression_weight_col=None,
            forecast_horizon=None,
            simulation_based=False,
            simulation_num=10,
            fast_simulation=False):
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
        self.freq = freq
        self.origin_for_time_vars = origin_for_time_vars
        self.extra_pred_cols = extra_pred_cols
        self.drop_pred_cols = drop_pred_cols
        self.explicit_pred_cols = explicit_pred_cols
        self.train_test_thresh = train_test_thresh
        self.fit_algorithm_dict = fit_algorithm_dict
        self.training_fraction = training_fraction
        self.daily_event_df_dict = daily_event_df_dict
        self.fs_components_df = fs_components_df
        self.autoreg_dict = autoreg_dict
        self.past_df = past_df
        self.lagged_regressor_dict = lagged_regressor_dict
        self.changepoints_dict = changepoints_dict
        self.seasonality_changepoints_dict = seasonality_changepoints_dict
        self.changepoint_detector = changepoint_detector
        self.min_admissible_value = min_admissible_value
        self.max_admissible_value = max_admissible_value
        self.uncertainty_dict = uncertainty_dict
        self.normalize_method = normalize_method
        self.adjust_anomalous_dict = adjust_anomalous_dict
        self.impute_dict = impute_dict
        self.regression_weight_col = regression_weight_col
        self.forecast_horizon = forecast_horizon
        self.simulation_based = simulation_based
        self.simulation_num = simulation_num
        self.fast_simulation = fast_simulation
        self.validate_inputs()

    def validate_inputs(self):
        """Validates the inputs to ``SilverkiteEstimator``."""
        # verifies that user-provided ``fs_components_df`` satisfies
        # the requirements of BaseSilverkiteEstimator
        if self.fs_components_df is not None:
            self.validate_fs_components_df(self.fs_components_df)

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
        """
        # Initializes `fit_algorithm_dict` with default values.
        # This cannot be done in __init__ to remain compatible
        # with sklearn grid search.
        default_fit_algorithm_dict = {
            "fit_algorithm": "linear",
            "fit_algorithm_params": None}
        self.fit_algorithm_dict = update_dictionary(
            default_fit_algorithm_dict,
            overwrite_dict=self.fit_algorithm_dict)

        # fits null model
        super().fit(
            X=X,
            y=y,
            time_col=time_col,
            value_col=value_col,
            **fit_params)

        self.model_dict = self.silverkite.forecast(
            df=X,
            time_col=time_col,
            value_col=value_col,
            freq=self.freq,
            origin_for_time_vars=self.origin_for_time_vars,
            extra_pred_cols=self.extra_pred_cols,
            drop_pred_cols=self.drop_pred_cols,
            explicit_pred_cols=self.explicit_pred_cols,
            train_test_thresh=self.train_test_thresh,
            training_fraction=self.training_fraction,
            fit_algorithm=self.fit_algorithm_dict["fit_algorithm"],
            fit_algorithm_params=self.fit_algorithm_dict["fit_algorithm_params"],
            daily_event_df_dict=self.daily_event_df_dict,
            fs_components_df=self.fs_components_df,
            autoreg_dict=self.autoreg_dict,
            past_df=self.past_df,
            lagged_regressor_dict=self.lagged_regressor_dict,
            changepoints_dict=self.changepoints_dict,
            seasonality_changepoints_dict=self.seasonality_changepoints_dict,
            changepoint_detector=self.changepoint_detector,
            min_admissible_value=self.min_admissible_value,
            max_admissible_value=self.max_admissible_value,
            uncertainty_dict=self.uncertainty_dict,
            normalize_method=self.normalize_method,
            adjust_anomalous_dict=self.adjust_anomalous_dict,
            impute_dict=self.impute_dict,
            regression_weight_col=self.regression_weight_col,
            forecast_horizon=self.forecast_horizon,
            simulation_based=self.simulation_based,
            simulation_num=self.simulation_num,
            fast_simulation=self.fast_simulation)
        # sets attributes based on ``self.model_dict``
        super().finish_fit()

        return self

    @staticmethod
    def validate_fs_components_df(fs_components_df):
        """Validates the inputs of a fourier series components dataframe
        called by ``SilverkiteEstimator`` to validate the input ``fs_components_df``.

        Parameters
        ----------
        fs_components_df : `pandas.DataFrame`
            A DataFrame with information about fourier series generation.
            Must contain columns with following names:

            - "name": name of the timeseries feature (e.g. "tod", "tow" etc.)
            - "period": Period of the fourier series
            - "order": Order of the fourier series
            - "seas_names": seas_name corresponding to the name (e.g. "daily", "weekly" etc.).

        """
        fs_cols_expected = ["name", "period", "order", "seas_names"]
        fs_cols_not_found = [col for col in fs_cols_expected if col not in fs_components_df.columns]
        if fs_cols_not_found:
            raise ValueError(f"fs_components_df is missing the following columns: {fs_cols_not_found}")

        if any(fs_components_df.duplicated(subset=["name", "seas_names"])):
            raise ValueError("Found multiple rows in fs_components_df with the same `names` and "
                             "`seas_names`. Make sure these are unique.")
