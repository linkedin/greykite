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
# original author: Sayan Patra

import dataclasses
import functools
from typing import Dict
from typing import Optional

import pandas as pd

from greykite.common.python_utils import dictionaries_values_to_lists
from greykite.common.python_utils import update_dictionaries
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.base_template import BaseTemplate
from greykite.sklearn.estimator.auto_arima_estimator import AutoArimaEstimator
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator


class AutoArimaTemplate(BaseTemplate):
    """A template for :class:`~greykite.sklearn.estimator.auto_arima_estimator.AutoArimaEstimator`.

    Takes input data and optional configuration parameters
    to customize the model. Returns a set of parameters to call
    :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.

    Notes
    -----
    The attributes of a `~greykite.framework.templates.autogen.forecast_config.ForecastConfig` for
    :class:`~greykite.sklearn.estimator.auto_arima_estimator.AutoArimaEstimator` are:

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
                Ignored. Pass the relevant Auto Arima arguments via custom.

            growth: `dict` [`str`, `any`] or None
                Ignored. Pass the relevant Auto Arima arguments via custom.

            events: `dict` [`str`, `any`] or None
                Ignored. Pass the relevant Auto Arima arguments via custom.

            changepoints: `dict` [`str`, `any`] or None
                Ignored. Pass the relevant Auto Arima arguments via custom.

            regressors: `dict` [`str`, `any`] or None
                Ignored. Auto Arima template currently does not support regressors.

            uncertainty: `dict` [`str`, `any`] or None
                Ignored. Pass the relevant Auto Arima arguments via custom.

            hyperparameter_override: `dict` [`str`, `any`] or None or `list` [`dict` [`str`, `any`] or None]
                After the above model components are used to create a hyperparameter grid, the result is
                updated by this dictionary, to create new keys or override existing ones.
                Allows for complete customization of the grid search.

                Keys should have format ``{named_step}__{parameter_name}`` for the named steps of the
                `sklearn.pipeline.Pipeline` returned by this function. See `sklearn.pipeline.Pipeline`.

                For example::

                    hyperparameter_override={
                        "estimator__max_p": [8, 10],
                        "estimator__information_criterion": ["bic"],
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
                Ignored. Pass the relevant Auto Arima arguments via custom.
            custom: `dict` [`str`, `any`] or None
                Any parameter in the
                :class:`~greykite.sklearn.estimator.auto_arima_estimator.AutoArimaEstimator`
                can be passed.
        model_template: `str`
            This class only accepts "AUTO_ARIMA".
    """
    DEFAULT_MODEL_TEMPLATE = "AUTO_ARIMA"

    def __init__(
            self,
            estimator: BaseForecastEstimator = AutoArimaEstimator()):
        super().__init__(estimator=estimator)

    @property
    def allow_model_template_list(self):
        """AutoArimaTemplate does not allow `config.model_template` to be a list."""
        return False

    @property
    def allow_model_components_param_list(self):
        """AutoArimaTemplate does not allow `config.model_components_param` to be a list."""
        return False

    def get_regressor_cols(self):
        """Returns regressor column names from the model components.

        Currently does not implement regressors.
        """
        return None

    def apply_auto_arima_model_components_defaults(
            self,
            model_components=None):
        """Sets default values for ``model_components``.

        Parameters
        ----------
        model_components : :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` or None, default None
            Configuration of model growth, seasonality, events, etc.
            See the docstring of this class for details.

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

        default_custom = dict(
            # pmdarima fit parameters
            start_p=2,
            d=None,
            start_q=2,
            max_p=5,
            max_d=2,
            max_q=5,
            start_P=1,
            D=None,
            start_Q=1,
            max_P=2,
            max_D=1,
            max_Q=2,
            max_order=5,
            m=1,
            seasonal=True,
            stationary=False,
            information_criterion='aic',
            alpha=0.05,
            test='kpss',
            seasonal_test='ocsb',
            stepwise=True,
            n_jobs=1,
            start_params=None,
            trend=None,
            method='lbfgs',
            maxiter=50,
            offset_test_args=None,
            seasonal_test_args=None,
            suppress_warnings=True,
            error_action='trace',
            trace=False,
            random=False,
            random_state=None,
            n_fits=10,
            out_of_sample_size=0,
            scoring='mse',
            scoring_args=None,
            with_intercept="auto",
            # pmdarima predict parameters
            return_conf_int=True,
            dynamic=False
        )
        model_components.custom = update_dictionaries(
            default_dict=default_custom,
            overwrite_dicts=model_components.custom,
            allow_unknown_keys=False
        )

        # Sets to {} if None, for each item if
        # `model_components.hyperparameter_override` is a list of dictionaries
        model_components.hyperparameter_override = update_dictionaries(
            default_dict={},
            overwrite_dicts=model_components.hyperparameter_override,
            allow_unknown_keys=True
        )

        return model_components

    def get_hyperparameter_grid(self):
        """Returns hyperparameter grid.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Uses ``self.time_properties`` and ``self.config`` to generate the hyperparameter grid.

        Converts model components into
        :class:`~greykite.sklearn.estimator.auto_arima_estimator.AutoArimaEstimator`.
        hyperparameters.

        The output dictionary values are lists, combined via grid search in
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.

        Parameters
        ----------
        model_components : :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` or None, default None
            Configuration of parameter space to search the order (p, d, q etc.) of SARIMAX model.
            See :func:`~greykite.framework.templates.auto_arima_templates.auto_arima_template` for details.

        coverage : `float` or None, default=0.95
            Intended coverage of the prediction bands (0.0 to 1.0)

        Returns
        -------
        hyperparameter_grid : `dict` [`str`, `list` [`any`]] or None
            :class:`~greykite.sklearn.estimator.auto_arima_estimator.AutoArimaEstimator`
            hyperparameters.

            See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
            The output dictionary values are lists, combined in grid search.
        """
        self.config.model_components_param = self.apply_auto_arima_model_components_defaults(
            model_components=self.config.model_components_param)
        # Returns a set of parameters for grid search
        hyperparameter_grid = {
            # Additional parameters
            "estimator__freq": self.config.metadata_param.freq,
            # pmdarima fit parameters
            "estimator__start_p": self.config.model_components_param.custom["start_p"],
            "estimator__d": self.config.model_components_param.custom["d"],
            "estimator__start_q": self.config.model_components_param.custom["start_q"],
            "estimator__max_p": self.config.model_components_param.custom["max_p"],
            "estimator__max_d": self.config.model_components_param.custom["max_d"],
            "estimator__max_q": self.config.model_components_param.custom["max_q"],
            "estimator__start_P": self.config.model_components_param.custom["start_P"],
            "estimator__D": self.config.model_components_param.custom["D"],
            "estimator__start_Q": self.config.model_components_param.custom["start_Q"],
            "estimator__max_P": self.config.model_components_param.custom["max_P"],
            "estimator__max_D": self.config.model_components_param.custom["max_D"],
            "estimator__max_Q": self.config.model_components_param.custom["max_Q"],
            "estimator__max_order": self.config.model_components_param.custom["max_order"],
            "estimator__m": self.config.model_components_param.custom["m"],
            "estimator__seasonal": self.config.model_components_param.custom["seasonal"],
            "estimator__stationary": self.config.model_components_param.custom["stationary"],
            "estimator__information_criterion": self.config.model_components_param.custom["information_criterion"],
            "estimator__alpha": self.config.model_components_param.custom["alpha"],
            "estimator__test": self.config.model_components_param.custom["test"],
            "estimator__seasonal_test": self.config.model_components_param.custom["seasonal_test"],
            "estimator__stepwise": self.config.model_components_param.custom["stepwise"],
            "estimator__n_jobs": self.config.model_components_param.custom["n_jobs"],
            "estimator__start_params": self.config.model_components_param.custom["start_params"],
            "estimator__trend": self.config.model_components_param.custom["trend"],
            "estimator__method": self.config.model_components_param.custom["method"],
            "estimator__maxiter": self.config.model_components_param.custom["maxiter"],
            "estimator__offset_test_args": self.config.model_components_param.custom["offset_test_args"],
            "estimator__seasonal_test_args": self.config.model_components_param.custom["seasonal_test_args"],
            "estimator__suppress_warnings": self.config.model_components_param.custom["suppress_warnings"],
            "estimator__error_action": self.config.model_components_param.custom["error_action"],
            "estimator__trace": self.config.model_components_param.custom["trace"],
            "estimator__random": self.config.model_components_param.custom["random"],
            "estimator__random_state": self.config.model_components_param.custom["random_state"],
            "estimator__n_fits": self.config.model_components_param.custom["n_fits"],
            "estimator__out_of_sample_size": self.config.model_components_param.custom["out_of_sample_size"],
            "estimator__scoring": self.config.model_components_param.custom["scoring"],
            "estimator__scoring_args": self.config.model_components_param.custom["scoring_args"],
            "estimator__with_intercept": self.config.model_components_param.custom["with_intercept"],
            # pmdarima predict parameters
            "estimator__return_conf_int": self.config.model_components_param.custom["return_conf_int"],
            "estimator__dynamic": self.config.model_components_param.custom["dynamic"],
        }

        # Overwrites values by `model_components.hyperparameter_override`
        # This may produce a list of dictionaries for grid search.
        hyperparameter_grid = update_dictionaries(
            hyperparameter_grid,
            overwrite_dicts=self.config.model_components_param.hyperparameter_override,
            allow_unknown_keys=False
        )

        # Ensures all items have the proper type for
        # `sklearn.model_selection.RandomizedSearchCV`.
        # List-type hyperparameters are specified below
        # with their accepted non-list type values.
        hyperparameter_grid = dictionaries_values_to_lists(
            hyperparameter_grid)

        return hyperparameter_grid

    def apply_template_decorator(func):
        """Decorator for ``apply_template_for_pipeline_params`` function.

        Overrides the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Raises
        ------
        ValueError if config.model_template != "AUTO_ARIMA"
        """
        @functools.wraps(func)
        def process_wrapper(self, df: pd.DataFrame, config: Optional[ForecastConfig] = None):
            # sets defaults
            config = self.apply_forecast_config_defaults(config)
            # input validation
            if config.model_template != "AUTO_ARIMA":
                raise ValueError(f"AutoArimaTemplate only supports config.model_template='AUTO_ARIMA', "
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
