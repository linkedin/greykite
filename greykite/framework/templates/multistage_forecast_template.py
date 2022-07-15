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


from itertools import product
from typing import Dict
from typing import List
from typing import Type

import pandas as pd

from greykite.common import constants as cst
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import unique_in_list
from greykite.common.python_utils import update_dictionaries
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.base_template import BaseTemplate
from greykite.framework.templates.multistage_forecast_template_config import MultistageForecastTemplateConfig
from greykite.framework.templates.multistage_forecast_template_config import MultistageForecastTemplateConstants
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator
from greykite.sklearn.estimator.multistage_forecast_estimator import MultistageForecastEstimator
from greykite.sklearn.estimator.multistage_forecast_estimator import MultistageForecastModelConfig


class MultistageForecastTemplate(BaseTemplate):
    """The model template for Multistage Forecast Estimator."""

    DEFAULT_MODEL_TEMPLATE = "SILVERKITE_TWO_STAGE"

    def __init__(
            self,
            constants: MultistageForecastTemplateConstants = MultistageForecastTemplateConstants,
            # The parameters here don't matter. They are set for compatibility.
            estimator: BaseForecastEstimator = MultistageForecastEstimator(
                forecast_horizon=1,
                model_configs=[]
            )):
        """The init function.

        The estimator parameters in init is just for compatibility.
        It does not affect the results.
        """
        super().__init__(estimator=estimator)
        self._constants = constants()

    @property
    def constants(self) -> MultistageForecastTemplateConstants:
        """Constants used by the template class. Includes the model templates and their default values.
        """
        return self._constants

    def __get_regressor_templates(self):
        """Gets the model templates for each sub-model.

        These templates are for ``self.get_regressor_cols`` and ``self.get_lagged_regressor_info``
        to use to extract those information from each single model.

        Returns
        -------
        templates : `list` [`~greykite.framework.templates.base_template.BaseTemplate`]
            A list of model template class instances.
        """
        if self.config.model_components_param.custom is None:
            return None
        multistage_forecast_configs = self.config.model_components_param.custom.get(
            "multistage_forecast_configs", None)
        if multistage_forecast_configs is None or multistage_forecast_configs == []:
            return []
        if isinstance(multistage_forecast_configs, MultistageForecastTemplateConfig):
            multistage_forecast_configs = [multistage_forecast_configs]
        templates = []
        for config in multistage_forecast_configs:
            template = self.__get_template_class(ForecastConfig(model_template=config.model_template))()
            template.df = self.df
            template.config = ForecastConfig(
                model_template=config.model_template,
                model_components_param=config.model_components)
            templates.append(template)
        return templates

    def get_regressor_cols(self):
        """Gets the regressor columns in the model.

        Iterates over each submodel to extract the regressor columns.

        Returns
        -------
        regressor_cols : `list` [`str`]
            A list of the regressor column names used in any of the submodels.
        """
        templates = self.__get_regressor_templates()
        regressor_cols = []
        if templates is None or templates == []:
            return []
        for template in templates:
            try:
                regressors = template.get_regressor_cols()
            except AttributeError:
                continue
            regressor_cols += regressors if regressors is not None else []
        return unique_in_list(
            array=regressor_cols,
            ignored_elements=(None,))

    def get_lagged_regressor_info(self):
        """Gets the lagged regressor info for the model

        Iterates over each submodel to extract the lagged regressor info.

        Returns
        -------
        lagged_regressor_info : `dict`
            The combined lagged regressor info from all submodels.
        """
        templates = self.__get_regressor_templates()
        lagged_regressor_info = {
            "lagged_regressor_cols": None,
            "overall_min_lag_order": None,
            "overall_max_lag_order": None
        }
        if templates is None or templates == []:
            return lagged_regressor_info
        for template in templates:
            try:
                info = template.get_lagged_regressor_info()
            except AttributeError:
                continue
            # Combines the ``lagged_regressor_info`` from each model.
            cols = info["lagged_regressor_cols"]
            min_order = info["overall_min_lag_order"]
            max_order = info["overall_max_lag_order"]
            if lagged_regressor_info["lagged_regressor_cols"] is None:
                lagged_regressor_info["lagged_regressor_cols"] = cols
            elif cols is not None:
                lagged_regressor_info["lagged_regressor_cols"] += cols
            if lagged_regressor_info["overall_min_lag_order"] is None:
                lagged_regressor_info["overall_min_lag_order"] = min_order
            elif min_order is not None:
                lagged_regressor_info["overall_min_lag_order"] = min(
                    lagged_regressor_info["overall_min_lag_order"], min_order)
            if lagged_regressor_info["overall_max_lag_order"] is None:
                lagged_regressor_info["overall_max_lag_order"] = min_order
            elif min_order is not None:
                lagged_regressor_info["overall_max_lag_order"] = max(
                    lagged_regressor_info["overall_max_lag_order"], max_order)

        return lagged_regressor_info

    def get_hyperparameter_grid(self):
        """Gets the hyperparameter grid for the Multistage Forecast Model.

        Returns
        -------
        hyperparameter_grid : `dict` [`str`, `list` [`any`]] or `list` [ `dict` [`str`, `list` [`any`]] ]
            hyperparameter_grid for grid search in
            :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
            The output dictionary values are lists, combined in grid search.
        """
        if self.config is None:
            raise ValueError(f"Forecast config must be provided, but `self.config` is `None`.")
        model_template = self.config.model_template
        model_components = self.config.model_components_param

        # Gets the model components from model template.
        default_model_components = self.__get_default_model_components(model_template)
        default_multistage_forecast_configs = default_model_components.custom.get("multistage_forecast_configs")

        # Checks if any parameter is specified in fields other than "custom".
        not_none_parameters = []
        for key, value in model_components.__dict__.items():
            if value is not None and value != {} and key not in ["custom", "uncertainty"]:
                not_none_parameters.append(key)
        if not_none_parameters:
            log_message(
                message=f"Multistage Forecast template only takes configuration through ``custom`` "
                        f"and ``uncertainty`` in ``model_components_param``. "
                        f"The following inputs are ignored \n{not_none_parameters}.",
                level=LoggingLevelEnum.WARNING
            )

        # When ``custom`` is not None, we look for the ``multistage_forecast_configs`` key.
        custom = model_components.custom
        # Gets the ``multistage_forecast_configs`` from ``default_multistage_forecast_configs`` and
        # overriden by ``custom["multistage_forecast_configs"]``.
        # If no customized configs, the default configs will be ``new_configs``.
        new_configs = self.__get_multistage_forecast_configs_override(
            custom=custom,
            model_template=model_template,
            default_multistage_forecast_configs=default_multistage_forecast_configs)

        # Converts template configs into estimator parameters.
        estimator_list, estimator_params_list = self.__get_estimators_and_params_from_template_configs(
            new_configs=new_configs
        )

        # Now the estimator parameters may contain grids, i.e., list of parameters from template classes.
        # We need to flatten them and wrap them into list of parameters for `MultistageForecastEstimator`.
        # The following function call gets the flattened estimator parameters,
        # in the format of a list of lists (different sets of parameters) of dictionaries (different stage models).
        flattened_dictionaries = self.__flatten_estimator_params_list(
            estimator_params_list=estimator_params_list
        )

        # Then we construct the `MultistageForecastEstimator` parameters.
        multistage_model_configs = []
        for grid in flattened_dictionaries:
            list_of_model_configs = []
            for i, config in enumerate(new_configs):
                # This is a single ``MultistageForecastModelConfig`` that corresponds to
                # a single stage model in a set of configuration.
                model_config = MultistageForecastModelConfig(
                    train_length=config.train_length,
                    fit_length=config.fit_length,
                    agg_func=config.agg_func,
                    agg_freq=config.agg_freq,
                    estimator=estimator_list[i],
                    estimator_params=grid[i]
                )
                # Appends this single ``MultistageForecastModelConfig`` to get all stage of models.
                list_of_model_configs.append(model_config)
            # The hyperparameter grid consists of a list of all stage of models for grid search.
            # This corresponds to different sets of configurations.
            multistage_model_configs.append(list_of_model_configs)
        # ``freq`` is the data frequency, which is from ``metadata_param``.
        freq = self.config.metadata_param.freq if self.config.metadata_param is not None else None

        # Gets the uncertainty parameter.
        uncertainty = model_components.uncertainty
        if uncertainty is not None:
            uncertainty_dict = uncertainty.get("uncertainty_dict", None)
        else:
            uncertainty_dict = None

        # Gets the hyperparameter grid.
        multistage_forecast_hyperparameter_grid = dict(
            estimator__forecast_horizon=[self.config.forecast_horizon],
            estimator__freq=[freq],
            estimator__model_configs=multistage_model_configs,
            estimator__uncertainty_dict=[uncertainty_dict]
        )
        return multistage_forecast_hyperparameter_grid

    @staticmethod
    def __get_multistage_forecast_configs_override(
            custom: Dict[str, any],
            model_template: str,
            default_multistage_forecast_configs: List[MultistageForecastTemplateConfig]):
        """Gets the overriden Multistage Forecast configs by ``custom``.

        Parameters
        ----------
        custom : `dict` [`str`, any]
            The custom dictionary in `ModelComponentsParam`.
            The only recognizable key is ``multistage_forecast_configs``,
            which takes a list of
            `~greykite.framework.templates.multistage_forecast_template_config.MultistageForecastTemplateConfig`.
        model_template : `str`
            The model template used in Multistage Forecast template.
        default_multistage_forecast_configs : `list` [
        `~greykite.framework.templates.multistage_forecast_template_config.MultistageForecastTemplateConfig`]
            The default Multistage Forecast configs from ``model_template``.

        Returns
        -------
        new_configs : `list` [
        `~greykite.framework.templates.multistage_forecast_template_config.MultistageForecastTemplateConfig`]
            The Multistage Forecast configs overriden by the ``multistage_forecast_configs`` in ``custom``.
        """
        if custom is not None:
            # Checks if any parameter is specified in "custom" other than "multistage_forecast_config".
            not_none_parameters = []
            for key, value in custom.items():
                if value is not None and value != [] and key != "multistage_forecast_configs":
                    not_none_parameters.append(key)
            if not_none_parameters:
                log_message(
                    message=f"Multistage Forecast template only takes configurations through "
                            f"``custom.multistage_forecast_configs``. The following inputs are "
                            f"ignored \n{not_none_parameters}.",
                    level=LoggingLevelEnum.WARNING
                )
            # Uses ``multistage_forecast_configs`` to override the default components if it's not None.
            multistage_forecast_configs = custom.get("multistage_forecast_configs", None)
            if ((multistage_forecast_configs is None or multistage_forecast_configs == [])
                    and model_template == "MULTISTAGE_EMPTY"):
                raise ValueError(f"``MULTISTAGE_EMPTY`` can not be used without overriding. "
                                 f"You must provide parameters in "
                                 f"``ModelComponentsParam.custom.multistage_forecast_configs``.")
            if multistage_forecast_configs is not None:
                # Wraps in a list if it's a single ``MultistageForecastTemplateConfig``.
                if isinstance(multistage_forecast_configs, MultistageForecastTemplateConfig):
                    multistage_forecast_configs = [multistage_forecast_configs]
                # Must be a list.
                if not isinstance(multistage_forecast_configs, list):
                    raise ValueError(f"The ``multistage_forecast_configs`` parameter must be a list of "
                                     f"``MultistageForecastTemplateConfig`` objects, found "
                                     f"\n{multistage_forecast_configs}.")
                # Checks the lengths of default configs and overriding configs.
                num_configs_in_default = len(default_multistage_forecast_configs)
                num_configs_in_override = len(multistage_forecast_configs)
                extra_configs = []
                if num_configs_in_default != num_configs_in_override:
                    if num_configs_in_default != 0:
                        log_message(
                            message=f"The number of configs in ``ModelComponentsParam`` ({num_configs_in_override}) "
                                    f"does not match the number of configs in the default template "
                                    f"({num_configs_in_default}). Appending extra configs to the end.",
                            level=LoggingLevelEnum.WARNING
                        )
                    # These configs are extra configs, either from the default or from overriding.
                    # No matter where they come from, they will be appended to the end.
                    extra_configs = (
                        multistage_forecast_configs[-(num_configs_in_override - num_configs_in_default):]
                        if num_configs_in_override >= num_configs_in_default
                        else default_multistage_forecast_configs[-(num_configs_in_default - num_configs_in_override):]
                    )

                # Overrides the default ``MultistageForecastTemplateConfig`` objects.
                num_to_override = min(num_configs_in_default, num_configs_in_override)
                new_configs = []
                for i in range(num_to_override):
                    default_config = default_multistage_forecast_configs[i]
                    new_config = multistage_forecast_configs[i]
                    # Overrides the original parameters.
                    keys = ["train_length", "fit_length", "agg_freq", "agg_func"]
                    for key in keys:
                        if getattr(new_config, key, None) is not None:
                            setattr(default_config, key, getattr(new_config, key))
                    # For ``model_template`` and ``model_components``,
                    # both will be overriden if the new ``model_template`` is different
                    # from the default ``model_template``. However, if both ``model_templates``
                    # are the same, the keys/values in the new ``model_components`` will be
                    # used to override the keys/values in the default ``model_components``,
                    # instead of replacing the entire default ``model_components`` with the new ``model_components``.
                    # The consideration here is that if one only specifies partial ``model_components`` and hope
                    # the rest can be kept as the default, this is the right way to do. If one hopes to
                    # use only the parameters specified in the new ``model_components`` and do not apply defaults,
                    # they should have used the ``MULTISTAGE_EMPTY`` template,
                    # and this is also the correct behavior.
                    if (new_config.model_template != default_config.model_template
                            or default_config.model_components is None):
                        for key in ["model_template", "model_components"]:
                            if getattr(new_config, key, None) is not None:
                                setattr(default_config, key, getattr(new_config, key))
                    else:
                        for key in new_config.model_components.__dict__.keys():
                            allow_unknown_keys = (key == "hyperparameter_override")
                            updated_value = update_dictionaries(
                                default_dict=getattr(default_config.model_components, key, {}) or {},
                                overwrite_dicts=getattr(new_config.model_components, key),
                                allow_unknown_keys=allow_unknown_keys)
                            setattr(default_config.model_components, key, updated_value)
                    new_configs.append(default_config)
                new_configs += extra_configs
            else:
                # If `ModelComponentsParam.custom["multistage_forecast_configs]"` is None,
                # use the default from template.
                new_configs = default_multistage_forecast_configs
        else:
            # If `ModelComponentsParam.custom` is None,
            # use the default from template.
            new_configs = default_multistage_forecast_configs
        return new_configs

    def __get_estimators_and_params_from_template_configs(
            self,
            new_configs: List[MultistageForecastTemplateConfig]):
        """Gets the estimators and estimator parameters from ``MultistageForecastTemplateConfig`` objects.

        Parameters
        ----------
        new_configs : `list` [
        `~greykite.framework.templates.multistage_forecast_template_config.MultistageForecastTemplateConfig`]
            The Multistage Forecast configs overriden by the ``multistage_forecast_configs`` in ``custom``.

        Returns
        -------
        estimators : `list` [`~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`]
            The estimator classes in each stage.
        estimator_params : `list` [`dict` [`str`, any]]
            The estimator parameters in each stage.
            These parameters are in ``hyperparameter_grid`` format and may contain nested grids.
        """
        estimator_list = []
        estimator_params_list = []
        for config in new_configs:
            template = self.__get_template_class(ForecastConfig(model_template=config.model_template))()
            estimator = template._estimator.__class__
            # It's not common that `self.config.metadata_param` is None,
            # but since `get_hyperparameter_grid` is a public method,
            # in case people call it directly, we set the value defaults.
            if self.config.metadata_param is not None and self.config.metadata_param.time_col is not None:
                time_col = self.config.metadata_param.time_col
            else:
                time_col = cst.TIME_COL
            if self.config.metadata_param is not None and self.config.metadata_param.value_col is not None:
                value_col = self.config.metadata_param.value_col
            else:
                value_col = cst.VALUE_COL
            date_format = self.config.metadata_param.date_format if self.config.metadata_param is not None else None
            # Creates a sample df for the template class to generate hyperparameter grid.
            # The ``apply_template_for_pipeline_params`` function does not use any information from ``df``
            # when generating the hyperparameter grid.
            sample_df = pd.DataFrame({
                time_col: pd.date_range(
                    end=pd.to_datetime(self.df[time_col]).max().date(),
                    periods=100,
                    freq=config.agg_freq
                ),
                value_col: 0
            })
            estimator_params_grid = template.apply_template_for_pipeline_params(
                df=sample_df,
                # Here we ignore the ``forecast_horizon`` parameter.
                # Even the wrong ``forecast_horizon`` is inferred for this model,
                # the correct ``forecast_horizon`` will be used to override in the estimator's ``fit``
                # method.
                config=ForecastConfig(
                    metadata_param=MetadataParam(
                        time_col=time_col,
                        value_col=value_col,
                        freq=config.agg_freq,
                        date_format=date_format,
                    ),
                    model_template=config.model_template,
                    model_components_param=config.model_components
                )
            )["hyperparameter_grid"]
            estimator_list.append(estimator)
            estimator_params_list.append(estimator_params_grid)
        return estimator_list, estimator_params_list

    @staticmethod
    def __flatten_estimator_params_list(
            estimator_params_list: List[Dict[str, any]]):
        """Flattens the ``estimator_params_list``.

        The ``estimator_params_list`` is from ``self.__get_estimators_and_params_from_template_configs``,
        and may contain nested grids within each parameter.
        This function flattens it into the format of list of lists of ``estimator_params``.

        For example, the original ``estimator_params_list`` is

            [{"a": [1], "b": [2, 3]}, {"c": [4, 5]}]

        It consists of 2 stages of models. Each stage of model's parameters are in a dictionary.
        The parameter values are in lists and could have multiple possible values.

        After flattening the ``estimator_params_list``, it becomes

            [[{"a": 1, "b": 2}, {"c": 4}], [{"a": 1, "b": 3}, {"c": 4}],
             [{"a": 1, "b": 2}, {"c": 5}], [{"a": 1, "b": 3}, {"c": 5}]]

        There are 2 x 2 = 4 sets of parameters, i.e., 4 sets of ``estimator_params``,
        each of which includes two dictionaries which correspond to the two stages of models.

        Parameters
        ----------
        estimator_params_list : `list` [`dict` [`str`, any]]
            The estimator parameter list in hyperparameter grids.

        Returns
        -------
        flattened_estimator_params : `list` [`list` [`dict` [`str`, any]]]
            The flattened list of lists of estimator parameter dictionaries.
        """
        # Although Python 3.7 keeps the order in dictionary from insertion,
        # to be more compatible, we use lists to ensure the keys and values are matched.
        # For example, we have
        # [{"a": [1], "b": [2, 3]}, {"c": [4, 5]}]
        keys = []
        params = []
        for index, dictionary in enumerate(estimator_params_list):
            keys.append([])
            params.append([])
            for key, value in dictionary.items():
                # ``time_properties`` are automatically inferred from the other parameters.
                if "estimator__" in key and key != "estimator__time_properties":
                    keys[index].append(key.split("__")[1])
                    params[index].append(value)
        # Here we get a list of flattened values.
        # [((1, 2), (4)), ((1, 3), (4)), ((1, 2), (5)), ((1, 3), (5))]
        # The inner product gets all cross products for the value combinations within a stage.
        # The outer product gets all cross products for the value combinations across stages.
        flattened_params = list(product(*[list(product(*param)) for param in params]))
        # Then we map the flattened parameters with their keys and flatten them.
        # [[{"a": 1, "b": 2}, {"c": 4}], [{"a": 1, "b": 3}, {"c": 4}],
        #  [{"a": 1, "b": 2}, {"c": 5}], [{"a": 1, "b": 3}, {"c": 5}]]
        flattened_dictionaries = [
            [
                {key: value for (key, value) in zip(subkeys, subvalues)}
                for subkeys, subvalues in zip(keys, single_value)
            ]
            for single_value in flattened_params
        ]
        return flattened_dictionaries

    def __get_default_model_components(
            self,
            template: str):
        """Gets the default model components from a model template name.

        Parameters
        ----------
        template : `str`
            The model template name.

        Returns
        -------
        template : `~greykite.framework.templates.base_template.BaseTemplate`
            The model template class.
        """
        try:
            template = getattr(self._constants, template)
        except (AttributeError, TypeError):
            raise ValueError(f"The template name {template} is not recognized!")
        return template

    @property
    def allow_model_template_list(self) -> bool:
        return False

    @property
    def allow_model_components_param_list(self) -> bool:
        return False

    def __get_template_class(self, config: ForecastConfig = None) -> Type[BaseTemplate]:
        """Extracts template class (e.g. `SimpleSilverkiteTemplate`) from the config.
        Currently only supports single templates in
        `~greykite.framework.templates.model_templates.ModelTemplateEnum`.

        Parameters
        ----------
        config : :class:`~greykite.framework.templates.model_templates.ForecastConfig` or None
            Config object for template class to use.
            See :class:`~greykite.framework.templates.model_templates.ForecastConfig`.

        Returns
        -------
        template_class : Type[`~greykite.framework.templates.base_template.BaseTemplate`]
            An implementation of `~greykite.framework.templates.template_interface.TemplateInterface`.
        """
        model_template_enum = self._constants.MultistageForecastModelTemplateEnum
        valid_names = list(model_template_enum.__members__.keys())
        if config.model_template not in valid_names:
            raise ValueError(
                f"Currently Multistage Forecast only supports a known string of single model template. "
                f"Model Template '{config.model_template}' is not recognized! Must be one of: {valid_names}.")
        template_class = model_template_enum[config.model_template].value
        return template_class
