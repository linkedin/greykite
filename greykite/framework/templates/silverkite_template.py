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
"""Template for `greykite.sklearn.estimator.silverkite_estimator`.
Takes input data and forecast config,
and returns parameters to call
:func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
"""

import dataclasses
import functools
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd

from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.common.constants import TimeFeaturesEnum
from greykite.common.features.timeseries_lags import build_autoreg_df_multi
from greykite.common.python_utils import dictionaries_values_to_lists
from greykite.common.python_utils import unique_in_list
from greykite.common.python_utils import update_dictionaries
from greykite.common.python_utils import update_dictionary
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.base_template import BaseTemplate
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator
from greykite.sklearn.estimator.silverkite_diagnostics import SilverkiteDiagnostics
from greykite.sklearn.estimator.silverkite_estimator import SilverkiteEstimator


def get_extra_pred_cols(model_components=None):
    """Gets extra predictor columns from the model components for
    :func:`~greykite.framework.templates.silverkite_templates.silverkite_template`.

    Parameters
    ----------
    model_components : :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` or None, default None
        Configuration of model growth, seasonality, events, etc.
        See :func:`~greykite.framework.templates.silverkite_templates.silverkite_template`
        for details.

    Returns
    -------
    extra_pred_cols : `list` [`str`]
        All extra predictor columns used in any hyperparameter set
        requested by ``model_components.custom["extra_pred_cols]``.
        Regressors are included in this list.
        None if there are no extra predictor columns.
    """
    if model_components is not None and model_components.custom is not None:
        # ``extra_pred_cols`` is a list of strings to initialize
        # SilverkiteEstimator.extra_pred_cols, or a list of
        # such lists.
        extra_pred_cols = model_components.custom.get("extra_pred_cols", [])
    else:
        extra_pred_cols = []
    return unique_in_list(
        array=extra_pred_cols,
        ignored_elements=(None,))


def apply_default_model_components(
        model_components=None,
        time_properties=None):
    """Sets default values for ``model_components``.

    Parameters
    ----------
    model_components : :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` or None, default None
        Configuration of model growth, seasonality, events, etc.
        See :func:`~greykite.framework.templates.silverkite_templates.silverkite_template` for details.
    time_properties : `dict` [`str`, `any`] or None, default None
        Time properties dictionary (likely produced by
        `~greykite.common.time_properties_forecast.get_forecast_time_properties`)
        with keys:

        ``"period"`` : `int`
            Period of each observation (i.e. minimum time between observations, in seconds).
        ``"simple_freq"`` : `SimpleTimeFrequencyEnum`
            ``SimpleTimeFrequencyEnum`` member corresponding to data frequency.
        ``"num_training_points"`` : `int`
            Number of observations for training.
        ``"num_training_days"`` : `int`
            Number of days for training.
        ``"start_year"`` : `int`
            Start year of the training period.
        ``"end_year"`` : `int`
            End year of the forecast period.
        ``"origin_for_time_vars"`` : `float`
            Continuous time representation of the first date in ``df``.

    Returns
    -------
    model_components : :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`
        The provided ``model_components`` with default values set
    """
    if model_components is None:
        model_components = ModelComponentsParam()
    else:
        # makes a copy to avoid mutating input
        model_components = dataclasses.replace(model_components)

    # sets default values
    default_seasonality = {
        "fs_components_df": [pd.DataFrame({
            "name": [
                TimeFeaturesEnum.tod.value,
                TimeFeaturesEnum.tow.value,
                TimeFeaturesEnum.tom.value,
                TimeFeaturesEnum.toq.value,
                TimeFeaturesEnum.toy.value],
            "period": [24.0, 7.0, 1.0, 1.0, 1.0],
            "order": [3, 3, 1, 1, 5],
            "seas_names": ["daily", "weekly", "monthly", "quarterly", "yearly"]})],
    }
    model_components.seasonality = update_dictionary(
        default_seasonality,
        overwrite_dict=model_components.seasonality,
        allow_unknown_keys=False)

    # model_components.growth must be empty.
    # Pass growth terms via `extra_pred_cols` instead.
    default_growth = {}
    model_components.growth = update_dictionary(
        default_growth,
        overwrite_dict=model_components.growth,
        allow_unknown_keys=False)

    default_events = {
        "daily_event_df_dict": [None],
    }
    model_components.events = update_dictionary(
        default_events,
        overwrite_dict=model_components.events,
        allow_unknown_keys=False)

    default_changepoints = {
        "changepoints_dict": [None],
        "seasonality_changepoints_dict": [None],
        # Not allowed, to prevent leaking future information
        # into the past. Pass `changepoints_dict` with method="auto" for
        # automatic detection.
        # "changepoint_detector": [None],
    }
    model_components.changepoints = update_dictionary(
        default_changepoints,
        overwrite_dict=model_components.changepoints,
        allow_unknown_keys=False)

    default_autoregression = {
        "autoreg_dict": [None],
        "simulation_num": [10],
        "fast_simulation": [False]
    }
    model_components.autoregression = update_dictionary(
        default_autoregression,
        overwrite_dict=model_components.autoregression,
        allow_unknown_keys=False)

    default_regressors = {}
    model_components.regressors = update_dictionary(
        default_regressors,
        overwrite_dict=model_components.regressors,
        allow_unknown_keys=False)

    default_lagged_regressors = {
        "lagged_regressor_dict": [None],
    }
    model_components.lagged_regressors = update_dictionary(
        default_lagged_regressors,
        overwrite_dict=model_components.lagged_regressors,
        allow_unknown_keys=False)

    default_uncertainty = {
        "uncertainty_dict": [None],
    }
    model_components.uncertainty = update_dictionary(
        default_uncertainty,
        overwrite_dict=model_components.uncertainty,
        allow_unknown_keys=False)

    if time_properties is not None:
        origin_for_time_vars = time_properties.get("origin_for_time_vars")
    else:
        origin_for_time_vars = None

    default_custom = {
        "silverkite": [SilverkiteForecast()],  # NB: sklearn creates a copy in grid search
        "silverkite_diagnostics": [SilverkiteDiagnostics()],
        # The same origin for every split, based on start year of full dataset.
        # To use first date of each training split, set to `None` in model_components.
        "origin_for_time_vars": [origin_for_time_vars],
        "extra_pred_cols": [TimeFeaturesEnum.ct1.value],  # linear growth
        "drop_pred_cols": [None],
        "explicit_pred_cols": [None],
        "fit_algorithm_dict": [{
            "fit_algorithm": "linear",
            "fit_algorithm_params": None,
        }],
        "min_admissible_value": [None],
        "max_admissible_value": [None],
        "regression_weight_col": [None],
        "normalize_method": [None]
    }
    model_components.custom = update_dictionary(
        default_custom,
        overwrite_dict=model_components.custom,
        allow_unknown_keys=False)

    # sets to {} if None, for each item if
    # `model_components.hyperparameter_override` is a list of dictionaries
    model_components.hyperparameter_override = update_dictionaries(
        {},
        overwrite_dicts=model_components.hyperparameter_override)

    return model_components


class SilverkiteTemplate(BaseTemplate):
    """A template for :class:`~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator`.

    Takes input data and optional configuration parameters
    to customize the model. Returns a set of parameters to call
    :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.

    Notes
    -----
    The attributes of a `~greykite.framework.templates.autogen.forecast_config.ForecastConfig` for
    :class:`~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator` are:

        computation_param: `ComputationParam` or None, default None
            How to compute the result. See
            :class:`~greykite.framework.templates.autogen.forecast_config.ComputationParam`.
        coverage: `float` or None, default None
            Intended coverage of the prediction bands (0.0 to 1.0).
            Same as coverage in ``forecast_pipeline``.
            You may tune how the uncertainty is computed
            via `model_components.uncertainty["uncertainty_dict"]`.
        evaluation_metric_param: `EvaluationMetricParam` or None, default None
            What metrics to evaluate. See
            :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam`.
        evaluation_period_param: `EvaluationPeriodParam` or None, default None
            How to split data for evaluation. See
            :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationPeriodParam`.
        forecast_horizon: `int` or None, default None
            Number of periods to forecast into the future. Must be > 0
            If None, default is determined from input data frequency
            Same as forecast_horizon in `forecast_pipeline`
        metadata_param: `MetadataParam` or None, default None
            Information about the input data. See
            :class:`~greykite.framework.templates.autogen.forecast_config.MetadataParam`.
        model_components_param: `ModelComponentsParam` or None, default None
            Parameters to tune the model. See
            :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`.
            The fields are dictionaries with the following items.

            See inline comments on which values accept lists for grid search.

            seasonality: `dict` [`str`, `any`] or None, optional
                How to model the seasonality. A dictionary with keys corresponding to
                parameters in `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

                Allowed keys: ``"fs_components_df"``.
            growth: `dict` [`str`, `any`] or None, optional
                How to model the growth.

                Allowed keys: None. (Use ``model_components.custom["extra_pred_cols"]`` to specify
                growth terms.)
            events: `dict` [`str`, `any`] or None, optional
                How to model the holidays/events. A dictionary with keys corresponding to
                parameters in `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

                Allowed keys: ``"daily_event_df_dict"``.

                .. note::

                    Event names derived from ``daily_event_df_dict`` must be specified via
                    ``model_components.custom["extra_pred_cols"]`` to be included in the model.
                    This parameter has no effect on the model unless event names are passed to
                    ``extra_pred_cols``.

                    The function
                    `~greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper.get_event_pred_cols`
                    can be used to extract all event names from ``daily_event_df_dict``.

            changepoints: `dict` [`str`, `any`] or None, optional
                How to model changes in trend and seasonality. A dictionary with keys corresponding to
                parameters in `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

                Allowed keys: "changepoints_dict", "seasonality_changepoints_dict", "changepoint_detector".
            autoregression: `dict` [`str`, `any`] or None, optional
                Specifies the autoregression configuration. Dictionary with the following optional key:

                ``"autoreg_dict"``: `dict` or `str` or None or a list of such values for grid search
                    If a `dict`: A dictionary with arguments for `~greykite.common.features.timeseries_lags.build_autoreg_df`.
                    That function's parameter ``value_col`` is inferred from the input of
                    current function ``self.forecast``. Other keys are:

                        ``"lag_dict"`` : `dict` or None
                        ``"agg_lag_dict"`` : `dict` or None
                        ``"series_na_fill_func"`` : callable

                    If a `str`: The string will represent a method and a dictionary will be
                    constructed using that `str`.
                    Currently only implemented method is "auto" which uses
                    `~greykite.algo.forecast.silverkite.SilverkiteForecast.__get_default_autoreg_dict`
                    to create a dictionary.
                    See more details for above parameters in
                    `~greykite.common.features.timeseries_lags.build_autoreg_df`.

            regressors: `dict` [`str`, `any`] or None, optional
                How to model the regressors.

                Allowed keys: None. (Use ``model_components.custom["extra_pred_cols"]`` to specify
                regressors.)
            lagged_regressors: `dict` [`str`, `dict`] or None, optional
                Specifies the lagged regressors configuration. Dictionary with the following optional key:

                ``"lagged_regressor_dict"``: `dict` or None or a list of such values for grid search
                    A dictionary with arguments for `~greykite.common.features.timeseries_lags.build_autoreg_df_multi`.
                    The keys of the dictionary are the target lagged regressor column names.
                    It can leverage the regressors included in ``df``.
                    The value of each key is either a `dict` or `str`.
                    If `dict`, it has the following keys:

                        ``"lag_dict"`` : `dict` or None
                        ``"agg_lag_dict"`` : `dict` or None
                        ``"series_na_fill_func"`` : callable

                    If `str`, it represents a method and a dictionary will be constructed using that `str`.
                    Currently the only implemented method is "auto" which uses ``SilverkiteForecast``'s
                    `~greykite.algo.forecast.silverkite.SilverkiteForecast.__get_default_lagged_regressor_dict`
                    to create a dictionary for each lagged regressor.
                    An example::

                        lagged_regressor_dict = {
                            "regressor1": {
                                "lag_dict": {"orders": [1, 2, 3]},
                                "agg_lag_dict": {
                                    "orders_list": [[7, 7 * 2, 7 * 3]],
                                    "interval_list": [(8, 7 * 2)]},
                                "series_na_fill_func": lambda s: s.bfill().ffill()},
                            "regressor2": "auto"}

                    Check the docstring of `~greykite.common.features.timeseries_lags.build_autoreg_df_multi`
                    for more details for each argument.
            uncertainty: `dict` [`str`, `any`] or None, optional
                How to model the uncertainty. A dictionary with keys corresponding to
                parameters in `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

                Allowed keys: ``"uncertainty_dict"``.
            custom: `dict` [`str`, `any`] or None, optional
                Custom parameters that don't fit the categories above. A dictionary with keys corresponding to
                parameters in `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

                Allowed keys:
                    ``"silverkite"``, ``"silverkite_diagnostics"``,
                    ``"origin_for_time_vars"``, ``"extra_pred_cols"``,
                    ``"drop_pred_cols"``, ``"explicit_pred_cols"``,
                    ``"fit_algorithm_dict"``, ``"min_admissible_value"``,
                    ``"max_admissible_value"``.

                .. note::

                    ``"extra_pred_cols"`` should contain the desired growth terms, regressor names, and event names.

                ``fit_algorithm_dict`` is a dictionary with ``fit_algorithm`` and ``fit_algorithm_params``
                parameters to `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`:

                        fit_algorithm_dict : `dict` or None, optional
                            How to fit the model. A dictionary with the following optional keys.

                            ``"fit_algorithm"`` : `str`, optional, default "linear"
                                The type of predictive model used in fitting.

                                See `~greykite.algo.common.ml_models.fit_model_via_design_matrix`
                                for available options and their parameters.
                            ``"fit_algorithm_params"`` : `dict` or None, optional, default None
                                Parameters passed to the requested fit_algorithm.
                                If None, uses the defaults in `~greykite.algo.common.ml_models.fit_model_via_design_matrix`.

            hyperparameter_override: `dict` [`str`, `any`] or None or `list` [`dict` [`str`, `any`] or None], optional
                After the above model components are used to create a hyperparameter grid, the result is
                updated by this dictionary, to create new keys or override existing ones.
                Allows for complete customization of the grid search.

                Keys should have format ``{named_step}__{parameter_name}`` for the named steps of the
                `sklearn.pipeline.Pipeline` returned by this function. See `sklearn.pipeline.Pipeline`.

                For example::

                    hyperparameter_override={
                        "estimator__origin_for_time_vars": 2018.0,
                        "input__response__null__impute_algorithm": "ts_interpolate",
                        "input__response__null__impute_params": {"orders": [7, 14]},
                        "input__regressors_numeric__normalize__normalize_algorithm": "RobustScaler",
                    }

                If a list of dictionaries, grid search will be done for each dictionary in the list.
                Each dictionary in the list override the defaults. This enables grid search
                over specific combinations of parameters to reduce the search space.

                    * For example, the first dictionary could define combinations of parameters for a
                      "complex" model, and the second dictionary could define combinations of parameters
                      for a "simple" model, to prevent mixed combinations of simple and complex.
                    * Or the first dictionary could grid search over fit algorithm, and the second dictionary
                      could use a single fit algorithm and grid search over seasonality.

                The result is passed as the ``param_distributions`` parameter
                to `sklearn.model_selection.RandomizedSearchCV`.
        model_template: `str`
            This class only accepts "SK".
    """
    DEFAULT_MODEL_TEMPLATE = "SK"
    """The default model template. See `~greykite.framework.templates.model_templates.ModelTemplateEnum`.
    Uses a string to avoid circular imports.
    Overrides the value from `~greykite.framework.templates.forecast_config_defaults.ForecastConfigDefaults`.
    """
    def __init__(
            self,
            estimator: BaseForecastEstimator = SilverkiteEstimator()):
        super().__init__(estimator=estimator)

    @property
    def allow_model_template_list(self):
        """SilverkiteTemplate does not allow `config.model_template` to be a list."""
        return False

    @property
    def allow_model_components_param_list(self):
        """SilverkiteTemplate does not allow `config.model_components_param` to be a list."""
        return False

    def get_regressor_cols(self):
        """Returns regressor column names.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        The intersection of ``extra_pred_cols`` from model components
        and ``self.df`` columns, excluding ``time_col`` and ``value_col``.

        Returns
        -------
        regressor_cols : `list` [`str`] or None
            See `~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        """
        extra_pred_cols = get_extra_pred_cols(model_components=self.config.model_components_param)
        if extra_pred_cols is not None:
            regressor_cols = [col for col in self.df.columns
                              if col not in [self.config.metadata_param.time_col, self.config.metadata_param.value_col]
                              and col in extra_pred_cols]
        else:
            regressor_cols = None
        return regressor_cols

    def get_lagged_regressor_info(self):
        """Returns lagged regressor column names and minimal/maximal lag order. The lag order
        can be used to check potential imputation in the computation of lags.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Returns
        -------
        lagged_regressor_info : `dict`
            A dictionary that includes the lagged regressor column names and maximal/minimal lag order
            The keys are:

                lagged_regressor_cols : `list` [`str`] or None
                    See `~greykite.framework.pipeline.pipeline.forecast_pipeline`.
                overall_min_lag_order : `int` or None
                overall_max_lag_order : `int` or None

            For example::

                self.config.model_components_param.lagged_regressors["lagged_regressor_dict"] = [
                    {"regressor1": {
                        "lag_dict": {"orders": [7]},
                        "agg_lag_dict": {
                            "orders_list": [[7, 7 * 2, 7 * 3]],
                            "interval_list": [(8, 7 * 2)]},
                        "series_na_fill_func": lambda s: s.bfill().ffill()}
                    },
                    {"regressor2": {
                        "lag_dict": {"orders": [2]},
                        "agg_lag_dict": {
                            "orders_list": [[7, 7 * 2]],
                            "interval_list": [(8, 7 * 2)]},
                        "series_na_fill_func": lambda s: s.bfill().ffill()}
                    },
                    {"regressor3": "auto"}
                ]

            Then the function returns::

                lagged_regressor_info = {
                    "lagged_regressor_cols": ["regressor1", "regressor2", "regressor3"],
                    "overall_min_lag_order": 2,
                    "overall_max_lag_order": 21
                }

            Note that "regressor3" is skipped as the "auto" option makes sure the lag order is proper.
        """
        lagged_regressor_info = {
            "lagged_regressor_cols": None,
            "overall_min_lag_order": None,
            "overall_max_lag_order": None
        }
        if (self.config is None or self.config.model_components_param is None or
                self.config.model_components_param.lagged_regressors is None):
            return lagged_regressor_info

        lag_reg_dict = self.config.model_components_param.lagged_regressors.get("lagged_regressor_dict", None)
        if lag_reg_dict is None or lag_reg_dict == [None]:
            return lagged_regressor_info

        lag_reg_dict_list = [lag_reg_dict] if isinstance(lag_reg_dict, dict) else lag_reg_dict
        lagged_regressor_cols = []
        overall_min_lag_order = np.inf
        overall_max_lag_order = -np.inf
        for d in lag_reg_dict_list:
            if isinstance(d, dict):
                lagged_regressor_cols += list(d.keys())
                # Also gets the minimal lag order for each lagged_regressor_dict.
                # Looks at each individual regressor column, "auto" is skipped because
                # "auto" always makes sure that minimal lag order is at least forecast horizon.
                for key, value in d.items():
                    if isinstance(value, dict):
                        d_tmp = {key: value}
                        lag_reg_components = build_autoreg_df_multi(value_lag_info_dict=d_tmp)
                        overall_min_lag_order = min(
                            lag_reg_components["min_order"],
                            overall_min_lag_order)
                        overall_max_lag_order = max(
                            lag_reg_components["max_order"],
                            overall_max_lag_order)
        lagged_regressor_cols = list(set(lagged_regressor_cols))

        lagged_regressor_info["lagged_regressor_cols"] = lagged_regressor_cols
        lagged_regressor_info["overall_min_lag_order"] = overall_min_lag_order
        lagged_regressor_info["overall_max_lag_order"] = overall_max_lag_order
        return lagged_regressor_info

    def get_hyperparameter_grid(self):
        """Returns hyperparameter grid.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Uses ``self.time_properties`` and ``self.config`` to generate the hyperparameter grid.

        Converts model components and time properties into
        :class:`~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator`
        hyperparameters.

        Notes
        -----
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`
        handles the train/test splits according to ``EvaluationPeriodParam``,
        so ``estimator__train_test_thresh`` and ``estimator__training_fraction`` are always None.

        ``estimator__changepoint_detector`` is always None, to prevent leaking future information
        into the past. Pass ``changepoints_dict`` with method="auto" for automatic detection.

        Returns
        -------
        hyperparameter_grid : `dict`, `list` [`dict`] or None
            See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
            The output dictionary values are lists, combined in grid search.
        """
        self.config.model_components_param = apply_default_model_components(
            model_components=self.config.model_components_param,
            time_properties=self.time_properties)

        # returns a single set of parameters for grid search
        hyperparameter_grid = {
            "estimator__silverkite": self.config.model_components_param.custom["silverkite"],
            "estimator__silverkite_diagnostics": self.config.model_components_param.custom["silverkite_diagnostics"],
            "estimator__origin_for_time_vars": self.config.model_components_param.custom["origin_for_time_vars"],
            "estimator__extra_pred_cols": self.config.model_components_param.custom["extra_pred_cols"],
            "estimator__drop_pred_cols": self.config.model_components_param.custom["drop_pred_cols"],
            "estimator__explicit_pred_cols": self.config.model_components_param.custom["explicit_pred_cols"],
            "estimator__train_test_thresh": [None],
            "estimator__training_fraction": [None],
            "estimator__fit_algorithm_dict": self.config.model_components_param.custom["fit_algorithm_dict"],
            "estimator__daily_event_df_dict": self.config.model_components_param.events["daily_event_df_dict"],
            "estimator__fs_components_df": self.config.model_components_param.seasonality["fs_components_df"],
            "estimator__autoreg_dict": self.config.model_components_param.autoregression["autoreg_dict"],
            "estimator__simulation_num": self.config.model_components_param.autoregression["simulation_num"],
            "estimator__fast_simulation": self.config.model_components_param.autoregression["fast_simulation"],
            "estimator__lagged_regressor_dict": self.config.model_components_param.lagged_regressors["lagged_regressor_dict"],
            "estimator__changepoints_dict": self.config.model_components_param.changepoints["changepoints_dict"],
            "estimator__seasonality_changepoints_dict": self.config.model_components_param.changepoints["seasonality_changepoints_dict"],
            "estimator__changepoint_detector": [None],
            "estimator__min_admissible_value": self.config.model_components_param.custom["min_admissible_value"],
            "estimator__max_admissible_value": self.config.model_components_param.custom["max_admissible_value"],
            "estimator__normalize_method": self.config.model_components_param.custom["normalize_method"],
            "estimator__regression_weight_col": self.config.model_components_param.custom["regression_weight_col"],
            "estimator__uncertainty_dict": self.config.model_components_param.uncertainty["uncertainty_dict"],
        }

        # Overwrites values by `model_components.hyperparameter_override`
        # This may produce a list of dictionaries for grid search.
        hyperparameter_grid = update_dictionaries(
            hyperparameter_grid,
            overwrite_dicts=self.config.model_components_param.hyperparameter_override)

        # Ensures all items have the proper type for
        # `sklearn.model_selection.RandomizedSearchCV`.
        # List-type hyperparameters are specified below
        # with their accepted non-list type values.
        hyperparameter_grid = dictionaries_values_to_lists(
            hyperparameter_grid,
            hyperparameters_list_type={
                "estimator__extra_pred_cols": [None]}
        )
        return hyperparameter_grid

    def apply_template_decorator(func):
        """Decorator for ``apply_template_for_pipeline_params`` function.

        Overrides the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Raises
        ------
        ValueError if config.model_template != "SK"
        """
        @functools.wraps(func)
        def process_wrapper(self, df: pd.DataFrame, config: Optional[ForecastConfig] = None):
            # sets defaults
            config = self.apply_forecast_config_defaults(config)
            # input validation
            if config.model_template != "SK":
                raise ValueError(f"SilverkiteTemplate only supports config.model_template='SK', "
                                 f"found '{config.model_template}'")
            pipeline_params = func(self, df, config)
            return pipeline_params
        return process_wrapper

    @apply_template_decorator
    def apply_template_for_pipeline_params(
            self,
            df: pd.DataFrame,
            config: Optional[ForecastConfig] = None) -> Dict:
        """Explicitly calls the method in
        `~greykite.framework.templates.base_template.BaseTemplate`
        to make use of the decorator in this class.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The time series dataframe with ``time_col`` and ``value_col`` and optional regressor columns.
        config : `~greykite.framework.templates.autogen.forecast_config.ForecastConfig`.
            The `ForecastConfig` class that includes model training parameters.

        Returns
        -------
        pipeline_parameters : `dict`
            The pipeline parameters consumable by
            `~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        """
        return super().apply_template_for_pipeline_params(df=df, config=config)

    apply_template_decorator = staticmethod(apply_template_decorator)
