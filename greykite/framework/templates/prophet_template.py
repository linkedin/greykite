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
# original author: Rachit Kumar, Albert Chen
"""Template for `greykite.sklearn.estimator.prophet_estimator`.
Takes input data and forecast config,
and returns parameters to call
:func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
"""

import dataclasses
import functools
import warnings
from typing import Dict
from typing import Optional

import pandas as pd

from greykite.common.python_utils import dictionaries_values_to_lists
from greykite.common.python_utils import update_dictionaries
from greykite.common.python_utils import update_dictionary
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.base_template import BaseTemplate
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator
from greykite.sklearn.estimator.prophet_estimator import ProphetEstimator


class ProphetTemplate(BaseTemplate):
    """A template for :class:`~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator`.

    Takes input data and optional configuration parameters
    to customize the model. Returns a set of parameters to call
    :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.

    Notes
    -----
    The attributes of a `~greykite.framework.templates.autogen.forecast_config.ForecastConfig` for
    :class:`~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator` are:

        computation_param: `ComputationParam` or None, default None
            How to compute the result. See
            :class:`~greykite.framework.templates.autogen.forecast_config.ComputationParam`.
        coverage: `float` or None, default None
            Intended coverage of the prediction bands (0.0 to 1.0)
            If None, the upper/lower predictions are not returned
            Same as coverage in ``forecast_pipeline``
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

            seasonality: `dict` [`str`, `any`] or None
                Seasonality config dictionary, with the following optional keys.

                ``"seasonality_mode"``: `str` or None or list of such values for grid search
                    Can be 'additive' (default) or 'multiplicative'.
                ``"seasonality_prior_scale"``: `float` or None or list of such values for grid search
                    Parameter modulating the strength of the seasonality model.
                    Larger values allow the model to fit larger seasonal fluctuations, smaller values dampen the seasonality.
                    Specify for individual seasonalities using add_seasonality_dict.
                ``"yearly_seasonality"``: `str` or `bool` or `int` or list of such values for grid search, default 'auto'
                    Determines the yearly seasonality
                    Can be 'auto', True, False, or a number of Fourier terms to generate.
                ``"weekly_seasonality"``: `str` or `bool` or `int` or list of such values for grid search, default 'auto'
                    Determines the weekly seasonality
                    Can be 'auto', True, False, or a number of Fourier terms to generate.
                ``"daily_seasonality"``: `str` or `bool` or `int` or list of such values for grid search, default 'auto'
                    Determines the daily seasonality
                    Can be 'auto', True, False, or a number of Fourier terms to generate.
                ``"add_seasonality_dict"``: `dict` or None or list of such values for grid search
                    dict of custom seasonality parameters to be added to the model, default=None
                    Key is the seasonality component name e.g. 'monthly'; parameters are specified via dict.
                    See :class:`~greykite.sklearn.estimator.prophet_estimator` for details.

            growth: `dict` [`str`, `any`] or None
                Specifies the growth parameter configuration.
                Dictionary with the following optional key:

                ``"growth_term"``: `str` or None or list of such values for grid search
                    How to model the growth. Valid options are "linear" and "logistic"
                    Specify a linear or logistic trend, these terms have their origin at the train start date.

            events: `dict` [`str`, `any`] or None
                Holiday/events configuration dictionary with the following optional keys:

                ``"holiday_lookup_countries"``: `list` [`str`] or "auto" or None
                    Which countries' holidays to include. Must contain all the holidays you intend to model.
                    If "auto", uses a default list of countries with a good coverage of global holidays.
                    If None or an empty list, no holidays are modeled.
                ``"holidays_prior_scale"``: `float` or None or list of such values for grid search, default 10.0
                    Modulates the strength of the holiday effect.
                ``"holiday_pre_num_days"``: `list` [`int`] or None, default 2
                    Model holiday effects for holiday_pre_num_days days before the holiday.
                    Grid search is not supported. Must be a list with one element or None.
                ``"holiday_post_num_days"``: `list` [`int`] or None, default 2
                    Model holiday effects for holiday_post_num_days days after the holiday
                    Grid search is not supported. Must be a list with one element or None.

            changepoints: `dict` [`str`, `any`] or None
                Specifies the changepoint configuration. Dictionary with the following
                optional keys:

                ``"changepoint_prior_scale"`` : `float` or None or list of such values for grid search, default 0.05
                    Parameter modulating the flexibility of the automatic changepoint selection. Large values will allow many
                    changepoints, small values will allow few changepoints.
                ``"changepoints"`` : `list` [`datetime.datetime`] or None or list of such values for grid search, default None
                    List of dates at which to include potential changepoints. If not specified,
                    potential changepoints are selected automatically.
                ``"n_changepoints"`` : `int` or None or list of such values for grid search, default 25
                    Number of potential changepoints to include. Not used if input `changepoints` is supplied.
                    If `changepoints` is not supplied, then n_changepoints potential changepoints are selected uniformly from
                    the first `changepoint_range` proportion of the history.
                ``"changepoint_range"`` : `float` or None or list of such values for grid search, default 0.8
                    Proportion of history in which trend changepoints will be estimated. Permitted values: (0,1]
                    Not used if input `changepoints` is supplied.

            regressors: `dict` [`str`, `any`] or None
                Specifies the regressors to include in the model (e.g. macro-economic factors).
                Dictionary with the following optional keys:

                ``"add_regressor_dict"`` : `dict` or None or list of such values for grid search, default None
                    Dictionary of extra regressors to be modeled.
                    See `~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator` for details.

            uncertainty: `dict` [`str`, `any`] or None
                Specifies the uncertainty configuration. A dictionary with the following optional keys:

                ``"mcmc_samples"`` : `int` or None or list of such values for grid search, default 0
                    if greater than 0, will do full Bayesian inference with the specified number of MCMC samples.
                    If 0, will do MAP estimation.
                ``"uncertainty_samples"`` : `int` or None or list of such values for grid search, default 1000
                    Number of simulated draws used to estimate
                    uncertainty intervals. Setting this value to 0 or False will disable
                    uncertainty estimation and speed up the calculation.

            hyperparameter_override: `dict` [`str`, `any`] or None or `list` [`dict` [`str`, `any`] or None]
                After the above model components are used to create a hyperparameter grid, the result is
                updated by this dictionary, to create new keys or override existing ones.
                Allows for complete customization of the grid search.

                Keys should have format ``{named_step}__{parameter_name}`` for the named steps of the
                `sklearn.pipeline.Pipeline` returned by this function. See `sklearn.pipeline.Pipeline`.

                For example::

                    hyperparameter_override={
                        "estimator__yearly_seasonality": [True, False],
                        "estimator__seasonality_prior_scale": [5.0, 15.0],
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
            autoregression: `dict` [`str`, `any`] or None
                Ignored. Prophet template does not support autoregression.
            lagged_regressors: `dict` [`str`, `any`] or None
                Ignored. Prophet template does not support lagged regressors.
            custom: `dict` [`str`, `any`] or None
                Ignored. There are no custom options.
        model_template: `str`
            This class only accepts "PROPHET".
    """
    DEFAULT_MODEL_TEMPLATE = "PROPHET"
    """The default model template. See `~greykite.framework.templates.model_templates.ModelTemplateEnum`.
    Uses a string to avoid circular imports.
    Overrides the value from `~greykite.framework.templates.forecast_config_defaults.ForecastConfigDefaults`.
    """
    HOLIDAY_LOOKUP_COUNTRIES_AUTO = (
        "UnitedStates", "UnitedKingdom", "India", "France", "China")
    """Default holiday countries to use if countries='auto'"""
    def __init__(
            self,
            estimator: Optional[BaseForecastEstimator] = None):
        try:
            global make_holidays_df
            from prophet.make_holidays import make_holidays_df
        except ModuleNotFoundError:
            raise ValueError("Module 'prophet' is not installed. Please install it manually.")
        if estimator is None:
            estimator = ProphetEstimator()
        super().__init__(estimator=estimator)

    @property
    def allow_model_template_list(self):
        """ProphetTemplate does not allow `config.model_template` to be a list."""
        return False

    @property
    def allow_model_components_param_list(self):
        """ProphetTemplate does not allow `config.model_components_param` to be a list."""
        return False

    def get_prophet_holidays(
            self,
            year_list,
            countries="auto",
            lower_window=-2,
            upper_window=2):
        """Generates holidays for Prophet model.

        Parameters
        ----------
        year_list : `list` [`int`]
            List of years for selecting the holidays across given countries.
        countries : `list` [`str`] or "auto" or None, default "auto"
            Countries for selecting holidays.

            * If "auto", uses a default list of countries with a good coverage of global holidays.
            * If a list, a list of country names.
            * If None, the function returns None.

        lower_window : `int` or None, default -2
            Negative integer. Model holiday effects for given number of days before the holiday.
        upper_window : `int` or None, default 2
            Positive integer. Model holiday effects for given number of days after the holiday.

        Returns
        -------
        holidays : `pandas.DataFrame`
            holidays dataframe to pass to Prophet's `holidays` argument.

        See Also
        --------
        `~greykite.common.features.timeseries_features.get_available_holiday_lookup_countries`
        to list available countries for modeling.

        `~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator`.
        """
        holidays = None
        if countries is None:
            countries = []
        elif countries == "auto":
            # countries with a good coverage of global holidays
            countries = self.HOLIDAY_LOOKUP_COUNTRIES_AUTO
        elif not isinstance(countries, (list, tuple)):
            raise ValueError(f"`countries` should be a list, found {countries}")

        # Suppresses the warnings such as "We only support Diwali and Holi holidays from 2010 to 2025"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for country in countries:
                country_holidays = make_holidays_df(year_list=year_list, country=country)
                country_holidays["lower_window"] = lower_window
                country_holidays["upper_window"] = upper_window
                if holidays is not None:
                    holidays = pd.concat([holidays, country_holidays], axis=0)
                else:
                    holidays = country_holidays
        if holidays is not None:
            holidays.drop_duplicates(inplace=True)
        return holidays

    def get_regressor_cols(self):
        """Returns regressor column names.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Returns
        -------
        regressor_cols : `list` [`str`] or None
            The names of regressor columns used in any hyperparameter set
            requested by ``model_components``.
            None if there are no regressors.
        """
        # ``add_regressor_dict`` is a list of dictionaries to initialize
        # ProphetEstimator.add_seasonality_dict. Each dictionary's keys are
        # the regressors used in that model
        reg_names = set()
        if self.config.model_components_param.regressors is not None:
            add_regressor_dict = self.config.model_components_param.regressors.get("add_regressor_dict", [None])
            if isinstance(add_regressor_dict, dict):
                add_regressor_dict = [add_regressor_dict]
            for reg_dict in add_regressor_dict:
                # None indicates no regressors
                if reg_dict is not None:
                    reg_names.update(reg_dict.keys())
        regressor_cols = list(reg_names) if reg_names else None
        return regressor_cols

    def apply_prophet_model_components_defaults(
            self,
            model_components=None,
            time_properties=None):
        """Sets default values for ``model_components``.

        Called by ``get_hyperparameter_grid`` after ``time_properties` is defined.
        Requires ``time_properties`` as well as ``model_components``
        so we do not simply override
        `~greykite.framework.templates.forecast_config_defaults.ForecastConfigDefaults.apply_model_components_defaults`.

        Parameters
        ----------
        model_components : :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` or None, default None
            Configuration of model growth, seasonality, events, etc.
            See the docstring of this class for details.
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

            If None, start_year is set to 2015 and end_year to 2030.

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
        if time_properties is None:
            time_properties = {
                "start_year": 2015,
                "end_year": 2030,
            }

        # seasonality
        default_seasonality = {
            "seasonality_mode": ["additive"],
            "seasonality_prior_scale": [10.0],
            "yearly_seasonality": ['auto'],
            "weekly_seasonality": ['auto'],
            "daily_seasonality": ['auto'],
            "add_seasonality_dict": [None]
        }
        # If seasonality params are not provided, uses default params. Otherwise, prefers provided params.
        # `allow_unknown_keys=False` requires `model_components.seasonality` keys to be a subset of
        # `default_seasonality` keys.
        model_components.seasonality = update_dictionary(
            default_dict=default_seasonality,
            overwrite_dict=model_components.seasonality,
            allow_unknown_keys=False)

        # growth
        default_growth = {
            "growth_term": ["linear"]
        }
        model_components.growth = update_dictionary(
            default_dict=default_growth,
            overwrite_dict=model_components.growth,
            allow_unknown_keys=False)

        # events
        default_events = {
            "holiday_lookup_countries": "auto",  # see `get_prophet_holidays` for defaults
            "holiday_pre_num_days": [2],
            "holiday_post_num_days": [2],
            "start_year": time_properties["start_year"],
            "end_year": time_properties["end_year"],
            "holidays_prior_scale": [10.0]
        }
        model_components.events = update_dictionary(
            default_dict=default_events,
            overwrite_dict=model_components.events,
            allow_unknown_keys=False)

        # Creates events dictionary for prophet estimator
        # Expands the range of holiday years by 1 year on each end, to ensure we have coverage of most relevant holidays.
        year_list = list(range(
            model_components.events["start_year"]-1,
            model_components.events["end_year"]+2))
        # Currently we support only one set of holiday_lookup_countries, holiday_pre_num_days and holiday_post_num_days.
        # Shows a warning if user supplies >1 set.
        if len(model_components.events["holiday_pre_num_days"]) > 1:
            warnings.warn(
                f"`events['holiday_pre_num_days']` list has more than 1 element. We currently support only 1 element. "
                f"Using {model_components.events['holiday_pre_num_days'][0]}.")
        if len(model_components.events["holiday_post_num_days"]) > 1:
            warnings.warn(
                f"`events['holiday_post_num_days']` list has more than 1 element. We currently support only 1 element. "
                f"Using {model_components.events['holiday_post_num_days'][0]}.")
        # If events["holiday_lookup_countries"] has multiple options, picks the first option
        if (model_components.events["holiday_lookup_countries"] is not None
                and model_components.events["holiday_lookup_countries"] != "auto"):
            if len(model_components.events["holiday_lookup_countries"]) > 1:
                # There are multiple elements
                if (any(isinstance(x, list) for x in model_components.events["holiday_lookup_countries"])
                        or None in model_components.events["holiday_lookup_countries"]
                        or "auto" in model_components.events["holiday_lookup_countries"]):
                    # Not a flat list of country names
                    warnings.warn(
                        f"`events['holiday_lookup_countries']` contains multiple options. "
                        f"We currently support only 1 option. Using {model_components.events['holiday_lookup_countries'][0]}.")
                    model_components.events["holiday_lookup_countries"] = model_components.events["holiday_lookup_countries"][0]
            elif isinstance(model_components.events["holiday_lookup_countries"][0], (list, tuple)):
                # There's only one element, and it's a list of countries
                model_components.events["holiday_lookup_countries"] = model_components.events["holiday_lookup_countries"][0]

        model_components.events = {
            "holidays_df": self.get_prophet_holidays(
                year_list=year_list,
                countries=model_components.events["holiday_lookup_countries"],
                # holiday effect is modeled from "holiday_pre_num_days" days before
                # to "holiday_post_num_days" days after the holiday
                lower_window=-model_components.events["holiday_pre_num_days"][0],  # Prophet expects a negative value for `lower_window`
                upper_window=model_components.events["holiday_post_num_days"][0]),
            "holidays_prior_scale": model_components.events["holidays_prior_scale"]
        }

        # changepoints_dict
        default_changepoints = {
            "changepoint_prior_scale": [0.05],
            "changepoints": [None],
            "n_changepoints": [25],
            "changepoint_range": [0.8]
        }
        model_components.changepoints = update_dictionary(
            default_dict=default_changepoints,
            overwrite_dict=model_components.changepoints,
            allow_unknown_keys=False)

        # uncertainty
        default_uncertainty = {
            "mcmc_samples": [0],
            "uncertainty_samples": [1000]
        }
        model_components.uncertainty = update_dictionary(
            default_dict=default_uncertainty,
            overwrite_dict=model_components.uncertainty,
            allow_unknown_keys=False)

        # regressors
        default_regressors = {
            "add_regressor_dict": [None]
        }
        model_components.regressors = update_dictionary(
            default_dict=default_regressors,
            overwrite_dict=model_components.regressors,
            allow_unknown_keys=False)

        # there are no custom parameters for Prophet

        # sets to {} if None, for each item if
        # `model_components.hyperparameter_override` is a list of dictionaries
        model_components.hyperparameter_override = update_dictionaries(
            {},
            overwrite_dicts=model_components.hyperparameter_override)

        return model_components

    def get_hyperparameter_grid(self):
        """Returns hyperparameter grid.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Uses ``self.time_properties`` and ``self.config`` to generate the hyperparameter grid.

        Converts model components and time properties into
        :class:`~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator`
        hyperparameters.

        Returns
        -------
        hyperparameter_grid : `dict` [`str`, `list` [`any`]] or None
            :class:`~greykite.sklearn.estimator.prophet_estimator.ProphetEstimator`
            hyperparameters.

            See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
            The output dictionary values are lists, combined in grid search.
        """
        self.config.model_components_param = self.apply_prophet_model_components_defaults(
            model_components=self.config.model_components_param,
            time_properties=self.time_properties)
        # Returns a set of parameters for grid search
        hyperparameter_grid = {
            "estimator__growth": self.config.model_components_param.growth["growth_term"],
            "estimator__seasonality_mode": self.config.model_components_param.seasonality["seasonality_mode"],
            "estimator__seasonality_prior_scale": self.config.model_components_param.seasonality["seasonality_prior_scale"],
            "estimator__yearly_seasonality": self.config.model_components_param.seasonality["yearly_seasonality"],
            "estimator__weekly_seasonality": self.config.model_components_param.seasonality["weekly_seasonality"],
            "estimator__daily_seasonality": self.config.model_components_param.seasonality["daily_seasonality"],
            "estimator__add_seasonality_dict": self.config.model_components_param.seasonality["add_seasonality_dict"],
            "estimator__holidays": [self.config.model_components_param.events["holidays_df"]],
            "estimator__holidays_prior_scale": self.config.model_components_param.events["holidays_prior_scale"],
            "estimator__changepoint_prior_scale": self.config.model_components_param.changepoints["changepoint_prior_scale"],
            "estimator__changepoints": self.config.model_components_param.changepoints["changepoints"],
            "estimator__n_changepoints": self.config.model_components_param.changepoints["n_changepoints"],
            "estimator__changepoint_range": self.config.model_components_param.changepoints["changepoint_range"],
            "estimator__mcmc_samples": self.config.model_components_param.uncertainty["mcmc_samples"],
            "estimator__uncertainty_samples": self.config.model_components_param.uncertainty["uncertainty_samples"],
            "estimator__add_regressor_dict": self.config.model_components_param.regressors["add_regressor_dict"]
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
            hyperparameters_list_type={"estimator__changepoints": [None]}
        )
        return hyperparameter_grid

    def apply_template_decorator(func):
        """Decorator for ``apply_template_for_pipeline_params`` function.

        Overrides the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Raises
        ------
        ValueError if config.model_template != "PROPHET"
        """
        @functools.wraps(func)
        def process_wrapper(self, df: pd.DataFrame, config: Optional[ForecastConfig] = None):
            # sets defaults
            config = self.apply_forecast_config_defaults(config)
            # input validation
            if config.model_template != "PROPHET":
                if config.model_template != "PROPHET":
                    raise ValueError(f"ProphetTemplate only supports config.model_template='PROPHET', "
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
