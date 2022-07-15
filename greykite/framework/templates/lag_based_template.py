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
"""Lag based template.
Uses past observations with aggregation function as predictions.
Uses `~greykite.sklearn.estimator.lag_based_estimator.LagBasedEstimator`.
"""

import functools
from typing import Dict
from typing import Optional

import pandas as pd

from greykite.common.aggregation_function_enum import AggregationFunctionEnum
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import dictionaries_values_to_lists
from greykite.common.python_utils import update_dictionaries
from greykite.common.python_utils import update_dictionary
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.base_template import BaseTemplate
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator
from greykite.sklearn.estimator.lag_based_estimator import LagBasedEstimator
from greykite.sklearn.estimator.lag_based_estimator import LagUnitEnum


class LagBasedTemplate(BaseTemplate):
    """A template for :class: `~greykite.sklearn.estimator.lag_based_estimator.LagBasedEstimator`.
    """

    DEFAULT_MODEL_TEMPLATE = "LAG_BASED"

    def __init__(
            self,
            estimator: BaseForecastEstimator = LagBasedEstimator()):
        super().__init__(estimator=estimator)

    @property
    def allow_model_template_list(self):
        """LagBasedTemplate does not allow `config.model_template` to be a list."""
        return False

    @property
    def allow_model_components_param_list(self):
        """LagBasedTemplate does not allow `config.model_components_param` to be a list."""
        return False

    def get_regressor_cols(self):
        """Returns regressor column names from the model components.
        LagBasedTemplate does not support regressors.
        """
        return None

    def apply_lag_based_model_components_defaults(
            self,
            model_components: Optional[ModelComponentsParam] = None):
        """Fills the default values to ``model_components`` if not provided.

        Parameters
        ----------
        model_components : :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` or None, default None
            Configuration for `LagBasedTemplate`.
            Should only have values in the "custom" key.

        Returns
        -------
        model_components : :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`
            The provided ``model_components`` with default values set.
        """
        if model_components is None:
            model_components = ModelComponentsParam()
        default_params = dict(
            freq=None,
            lag_unit=LagUnitEnum.week.name,
            lags=[1],
            agg_func=AggregationFunctionEnum.mean.name,
            agg_func_params=None,
            past_df=None,
            series_na_fill_func=None
        )
        default_uncertainty = dict(
            uncertainty_dict=None
        )
        # Checks if ``model_components.custom`` has unknown keys.
        # They will be removed and a warning will be logged.
        if model_components.custom is None:
            model_components.custom = {}
        extra_keys_in_custom = [key for key in model_components.custom.keys()
                                if key not in default_params.keys()]
        if extra_keys_in_custom:
            log_message(
                message="The following keys are not recognized and ignored for `LagBasedTemplate`: "
                        f"{extra_keys_in_custom}",
                level=LoggingLevelEnum.WARNING
            )
            custom = {key: value for key, value in model_components.custom.items()
                      if key not in extra_keys_in_custom}
        else:
            custom = model_components.custom
        # Updates the defaults.
        model_components.custom = update_dictionary(
            default_params,
            overwrite_dict=custom,
            allow_unknown_keys=False)
        model_components.uncertainty = update_dictionary(
            default_uncertainty,
            overwrite_dict=model_components.uncertainty,
            allow_unknown_keys=False
        )

        # Gets ``freq`` from ``self.config.metadata`` if provided.
        if (model_components.custom["freq"] is None and self.config is not None
                and self.config.metadata_param is not None and self.config.metadata_param.freq is not None):
            model_components.custom["freq"] = self.config.metadata_param.freq

        return model_components

    def get_hyperparameter_grid(self):
        """Returns hyperparameter grid.

        Implements the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Uses ``self.config`` to generate the hyperparameter grid.

        Converts model components into
        :class:`~greykite.sklearn.estimator.lag_based_estimator.LagBasedEstimator`
        hyperparameters.

        Returns
        -------
        hyperparameter_grid : `dict`, `list` [`dict`] or None
            See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
            The output dictionary values are lists, combined in grid search.
        """
        self.config.model_components_param = self.apply_lag_based_model_components_defaults(
            model_components=self.config.model_components_param)

        # Returns a single set of parameters for grid search
        hyperparameter_grid = {
            "estimator__freq": self.config.model_components_param.custom["freq"],
            "estimator__lag_unit": self.config.model_components_param.custom["lag_unit"],
            "estimator__lags": self.config.model_components_param.custom["lags"],
            "estimator__agg_func": self.config.model_components_param.custom["agg_func"],
            "estimator__agg_func_params": self.config.model_components_param.custom["agg_func_params"],
            "estimator__uncertainty_dict": self.config.model_components_param.uncertainty["uncertainty_dict"],
            "estimator__past_df": self.config.model_components_param.custom["past_df"],
            "estimator__series_na_fill_func": self.config.model_components_param.custom["series_na_fill_func"]
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
                "estimator__lags": [None]}
        )
        return hyperparameter_grid

    def apply_template_decorator(func):
        """Decorator for ``apply_template_for_pipeline_params`` function.

        Overrides the method in `~greykite.framework.templates.base_template.BaseTemplate`.

        Raises
        ------
        ValueError if config.model_template != "LAG_BASED"
        """
        @functools.wraps(func)
        def process_wrapper(self, df: pd.DataFrame, config: Optional[ForecastConfig] = None):
            # sets defaults
            config = self.apply_forecast_config_defaults(config)
            # input validation
            if config.model_template != "LAG_BASED":
                raise ValueError(f"LagBasedTemplate only supports config.model_template='LAG_BASED', "
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
