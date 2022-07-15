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
"""Main entry point to create a forecast.
Generates a forecast from input data and config and stores the result.
"""
import json
from copy import deepcopy
from enum import Enum
from typing import Dict
from typing import Optional
from typing import Type

import pandas as pd

from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import unique_elements_in_list
from greykite.framework.pipeline.pipeline import ForecastResult
from greykite.framework.pipeline.pipeline import forecast_pipeline
from greykite.framework.pipeline.utils import get_basic_pipeline
from greykite.framework.templates.auto_model_template import get_auto_silverkite_model_template
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import forecast_config_from_dict
from greykite.framework.templates.forecast_config_defaults import ForecastConfigDefaults
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.pickle_utils import dump_obj
from greykite.framework.templates.pickle_utils import load_obj
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate
from greykite.framework.templates.template_interface import TemplateInterface
from greykite.sklearn.estimator.one_by_one_estimator import OneByOneEstimator


class Forecaster:
    """The main entry point to create a forecast.

    Call the :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`
    method to create a forecast. It takes a dataset and forecast configuration parameters.

    Notes
    -----
    This class can create forecasts using any of the model templates in
    `~greykite.framework.templates.model_templates.ModelTemplateEnum`.
    Model templates provide suitable default values for the available
    forecast estimators depending on the data characteristics.

    The model template is selected via the ``config.model_template``
    parameter to :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`.

    To add your own custom algorithms or template classes in our framework,
    pass ``model_template_enum`` and ``default_model_template_name``
    to the constructor.
    """
    def __init__(
            self,
            model_template_enum: Type[Enum] = ModelTemplateEnum,
            default_model_template_name: str = ModelTemplateEnum.AUTO.name):
        # Optional user input
        self.model_template_enum: Type[Enum] = model_template_enum
        """The available template names. An Enum class where names are template names, and values are of type
        `~greykite.framework.templates.model_templates.ModelTemplate`.
        """
        self.default_model_template_name: str = default_model_template_name
        """The default template name if not provided by ``config.model_template``.
        Should be a name in ``model_template_enum`` or "auto".
        Used by :py:meth:`~greykite.framework.templates.forecaster.Forecaster.__get_template_class`.
        """
        # The following are set by `self.run_forecast_config`.
        self.template_class: Optional[Type[TemplateInterface]] = None
        """Template class used. Must implement
        `~greykite.framework.templates.template_interface.TemplateInterface`
        and be one of the classes in ``self.model_template_enum``.
        Available for debugging purposes.
        Set by :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`.
        """
        self.template: Optional[TemplateInterface] = None
        """Instance of ``template_class`` used to run the forecast.
        See the docstring of the specific template class used.

            - `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`
            - `~greykite.framework.templates.silverkite_template.SilverkiteTemplate`
            - `~greykite.framework.templates.prophet_template.ProphetTemplate`
            - etc.

        Available for debugging purposes.
        Set by :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`.
        """
        self.config: Optional[ForecastConfig] = None
        """`~greykite.framework.templates.autogen.forecast_config.ForecastConfig`
        passed to the template class.
        Set by :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`.
        """
        self.pipeline_params: Optional[Dict] = None
        """Parameters used to call :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        Available for debugging purposes.
        Set by :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`.
        """
        self.forecast_result: Optional[ForecastResult] = None
        """The forecast result, returned by :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        Set by :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`.
        """

    def __get_config_with_default_model_template_and_components(self, config: Optional[ForecastConfig] = None) -> ForecastConfig:
        """Gets config with default value for `model_template` and `model_components_param` if not provided.

            - model_template : default value is ``self.default_model_template_name``.
            - model_components_param : default value is an empty ModelComponentsParam().

        Parameters
        ----------
        config : :class:`~greykite.framework.templates.model_templates.ForecastConfig` or None
            Config object for template class to use.
            See :class:`~greykite.framework.templates.model_templates.ForecastConfig`.
            If None, uses an empty ForecastConfig.

        Returns
        -------
        config : :class:`~greykite.framework.templates.model_templates.ForecastConfig`
            Input ``config`` with default ``model_template`` populated.
            If ``config.model_template`` is None, it is set to ``self.default_model_template_name``.
            If ``config.model_components_param`` is None, it is set to ``ModelComponentsParam()``.
        """
        config = deepcopy(config) if config is not None else ForecastConfig()
        # Unpacks list of a single element and sets default value if None.
        # NB: Does not call `apply_forecast_config_defaults`.
        #   Only sets `model_template` and `model_components_param`.
        #   The template class may have its own implementation of forecast config defaults.
        forecast_config_defaults = ForecastConfigDefaults()
        forecast_config_defaults.DEFAULT_MODEL_TEMPLATE = self.default_model_template_name
        config.model_template = forecast_config_defaults.apply_model_template_defaults(config.model_template)
        config.model_components_param = forecast_config_defaults.apply_model_components_defaults(config.model_components_param)
        return config

    def __get_template_class(self, config: Optional[ForecastConfig] = None) -> Type[TemplateInterface]:
        """Extracts template class (e.g. `SimpleSilverkiteTemplate`) from the config.

        Parameters
        ----------
        config : :class:`~greykite.framework.templates.model_templates.ForecastConfig` or None
            Config object for template class to use.
            See :class:`~greykite.framework.templates.model_templates.ForecastConfig`.

        Returns
        -------
        template_class : Type[`~greykite.framework.templates.template_interface.TemplateInterface`]
            An implementation of `~greykite.framework.templates.template_interface.TemplateInterface`.
        """
        config = self.__get_config_with_default_model_template_and_components(config)

        if isinstance(config.model_template, list):
            # Parses `config.model_template` to extract the template class, with validation.
            # Handles a list of model templates.
            template_classes = [self.__get_template_class(config=ForecastConfig(model_template=mt))
                                for mt in config.model_template]
            for tc in template_classes:
                if tc != template_classes[0]:
                    raise ValueError("All model templates must use the same template class. "
                                     f"Found {template_classes}")
            template_class = template_classes[0]
            if not template_class().allow_model_template_list:
                raise ValueError(f"The template class {template_class} does not allow `model_template` to be a list. "
                                 f"Pass a string instead.")
        else:
            # Handles other situations (string, data class).
            try:
                # Tries to look up in `self.model_template_enum`.
                template_class = self.model_template_enum[config.model_template].value.template_class
            except (KeyError, TypeError):
                # Template is not found in the enum.
                # NB: The logic in this clause is written for the default `self.model_template_enum`,
                #   which contains only one template class that is a subclass of SimpleSilverkiteTemplate.
                #   If a custom `self.model_template_enum` is provided it may be useful to override this logic.
                valid_names = ", ".join(self.model_template_enum.__dict__["_member_names_"])
                # Checks if template enum has a template class that supports generic naming
                #   i.e. a subclass of `SimpleSilverkiteTemplate`.
                subclass_simple_silverkite = [mte for mte in self.model_template_enum
                                              if issubclass(mte.value.template_class, SimpleSilverkiteTemplate)]
                if len(subclass_simple_silverkite) > 0:
                    try:
                        log_message(f"Model template {config.model_template} is not found in the template enum. "
                                    f"Checking if model template is suitable for `SimpleSilverkiteTemplate`.", LoggingLevelEnum.DEBUG)
                        SimpleSilverkiteTemplate().check_template_type(config.model_template)
                        possible_template_classes = unique_elements_in_list([mte.value.template_class
                                                                             for mte in subclass_simple_silverkite])
                        if len(possible_template_classes) > 1:
                            log_message(f"Multiple template classes could be used for the model "
                                        f"template {config.model_template}: {possible_template_classes}", LoggingLevelEnum.DEBUG)
                        # arbitrarily take a class that supports generic naming
                        template_class = subclass_simple_silverkite[0].value.template_class
                        log_message(f"Using template class {template_class} for the model "
                                    f"template {config.model_template}", LoggingLevelEnum.DEBUG)
                    except ValueError:
                        raise ValueError(f"Model Template '{config.model_template}' is not recognized! Must be one of: {valid_names}"
                                         " or satisfy the `SimpleSilverkiteTemplate` rules.")
                else:
                    raise ValueError(f"Model Template '{config.model_template}' is not recognized! Must be one of: {valid_names}.")

        # Validates `model_components_param` compatibility with the template
        if not template_class().allow_model_components_param_list and isinstance(config.model_components_param, list):
            raise ValueError(f"Model template {config.model_template} does not support a list of `ModelComponentsParam`.")

        return template_class

    def __apply_forecast_one_by_one_to_pipeline_parameters(self):
        """If forecast_one_by_one is activated,

            1. replaces the estimator with ``OneByOneEstimator`` in pipeline.
            2. Adds one by one estimator's parameters to ``hyperparameter_grid``.
        """
        if self.config.forecast_one_by_one not in (None, False):
            pipeline = get_basic_pipeline(
                estimator=OneByOneEstimator(
                    estimator=self.template.estimator.__class__.__name__,
                    forecast_horizon=self.config.forecast_horizon),
                score_func=self.template.score_func,
                score_func_greater_is_better=self.template.score_func_greater_is_better,
                agg_periods=self.template.config.evaluation_metric_param.agg_periods,
                agg_func=self.template.config.evaluation_metric_param.agg_func,
                relative_error_tolerance=self.template.config.evaluation_metric_param.relative_error_tolerance,
                coverage=self.template.config.coverage,
                null_model_params=self.template.config.evaluation_metric_param.null_model_params,
                regressor_cols=self.template.regressor_cols)
            self.pipeline_params["pipeline"] = pipeline
            if isinstance(self.pipeline_params["hyperparameter_grid"], list):
                for i in range(len(self.pipeline_params["hyperparameter_grid"])):
                    self.pipeline_params["hyperparameter_grid"][i]["estimator__forecast_horizon"] = [
                        self.config.forecast_horizon]
                    self.pipeline_params["hyperparameter_grid"][i]["estimator__estimator_map"] = [
                        self.config.forecast_one_by_one]
            else:
                self.pipeline_params["hyperparameter_grid"]["estimator__forecast_horizon"] = [
                    self.config.forecast_horizon]
                self.pipeline_params["hyperparameter_grid"]["estimator__estimator_map"] = [
                    self.config.forecast_one_by_one]

    def __get_model_template(
            self,
            df: pd.DataFrame,
            config: ForecastConfig) -> str:
        """Gets the default model template when "auto" is given.

        This is called after ``config`` has been filled with the default values
        and all fields are not None.

        Parameters
        ----------
        df : `pandas.DataFrame`
            Timeseries data to forecast.
            Contains columns [`time_col`, `value_col`], and optional regressor columns
            Regressor columns should include future values for prediction
        config : :class:`~greykite.framework.templates.model_templates.ForecastConfig`
            Config object for template class to use.
            Must be an instance with all fields not None.
            See :class:`~greykite.framework.templates.model_templates.ForecastConfig`.

        Returns
        -------
        model_template : `str`
            The corresponding model template.
        """
        # Gets the model template from config.
        # Model template should already be a string when this function is called,
        # which is handled by `self.__get_config_with_default_model_template_and_components`.
        model_template = config.model_template

        # Returns the model template if it's not "auto".
        if not isinstance(model_template, str) or model_template.lower() != "auto":
            return model_template

        # Handles the "auto" case.
        # Since `get_auto_silverkite_model_template` resolves "AUTO" to
        # a specific SILVERKITE template, the fallback template passed to it cannot be "AUTO".
        # We use SILVERKITE if `self.default_model_template_name` is "AUTO".
        default_template_for_auto = (self.default_model_template_name
                                     if self.default_model_template_name.lower() != "auto"
                                     else ModelTemplateEnum.SILVERKITE.name)
        model_template = get_auto_silverkite_model_template(
            df=df,
            default_model_template_name=default_template_for_auto,
            config=config
        )

        return model_template

    def apply_forecast_config(
            self,
            df: pd.DataFrame,
            config: Optional[ForecastConfig] = None) -> Dict:
        """Fetches pipeline parameters from the ``df`` and ``config``,
        but does not run the pipeline to generate a forecast.

        :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`
        calls this function and also runs the forecast pipeline.

        Available for debugging purposes to check pipeline parameters before
        running a forecast. Sets these attributes for debugging:

            - ``pipeline_params`` : the parameters passed to
              :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
            - ``template_class``, ``template`` : the template class used to generate the
              pipeline parameters.
            - ``config`` : the :class:`~greykite.framework.templates.model_templates.ForecastConfig`
              passed as input to template class, to translate into pipeline parameters.

        Provides basic validation on the compatibility of ``config.model_template``
        with ``config.model_components_param``.

        Parameters
        ----------
        df : `pandas.DataFrame`
            Timeseries data to forecast.
            Contains columns [`time_col`, `value_col`], and optional regressor columns
            Regressor columns should include future values for prediction
        config : :class:`~greykite.framework.templates.model_templates.ForecastConfig` or None
            Config object for template class to use.
            See :class:`~greykite.framework.templates.model_templates.ForecastConfig`.

        Returns
        -------
        pipeline_params : `dict` [`str`, `any`]
            Input to :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        """
        self.config = self.__get_config_with_default_model_template_and_components(config)
        self.config.model_template = self.__get_model_template(df=df, config=self.config)
        self.template_class = self.__get_template_class(self.config)
        self.template = self.template_class()
        self.pipeline_params = self.template.apply_template_for_pipeline_params(df=df, config=self.config)
        self.__apply_forecast_one_by_one_to_pipeline_parameters()
        return self.pipeline_params

    def run_forecast_config(
            self,
            df: pd.DataFrame,
            config: Optional[ForecastConfig] = None) -> ForecastResult:
        """Creates a forecast from input data and config.
        The result is also stored as ``self.forecast_result``.

        Parameters
        ----------
        df : `pandas.DataFrame`
            Timeseries data to forecast.
            Contains columns [`time_col`, `value_col`], and optional regressor columns
            Regressor columns should include future values for prediction
        config : :class:`~greykite.framework.templates.model_templates.ForecastConfig`
            Config object for template class to use.
            See :class:`~greykite.framework.templates.model_templates.ForecastConfig`.

        Returns
        -------
        forecast_result : :class:`~greykite.framework.pipeline.pipeline.ForecastResult`
            Forecast result, an object of type
            :class:`~greykite.framework.pipeline.pipeline.ForecastResult`.

            The output of :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`,
            according to the ``df`` and ``config`` configuration parameters.
        """
        pipeline_parameters = self.apply_forecast_config(
            df=df,
            config=config)
        self.forecast_result = forecast_pipeline(**pipeline_parameters)
        return self.forecast_result

    def run_forecast_json(
            self,
            df: pd.DataFrame,
            json_str: str = "{}") -> ForecastResult:
        """Calls :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`
        according to the ``json_str`` configuration parameters.

        Parameters
        ----------
        df : `pandas.DataFrame`
            Timeseries data to forecast.
            Contains columns [`time_col`, `value_col`], and optional regressor columns
            Regressor columns should include future values for prediction
        json_str : `str`
            Json string of the config object for Forecast to use.
            See :class:`~greykite.framework.templates.model_templates.ForecastConfig`.

        Returns
        -------
        forecast_result : :class:`~greykite.framework.pipeline.pipeline.ForecastResult`
            Forecast result.
            The output of :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`,
            called using the template class with specified configuration.
            See :class:`~greykite.framework.pipeline.pipeline.ForecastResult`
            for details.
        """
        config_dict = json.loads(json_str)
        config = forecast_config_from_dict(config_dict)
        self.run_forecast_config(
            df=df,
            config=config)
        return self.forecast_result

    def dump_forecast_result(
            self,
            destination_dir,
            object_name="object",
            dump_design_info=True,
            overwrite_exist_dir=False):
        """Dumps ``self.forecast_result`` to local pickle files.

        Parameters
        ----------
        destination_dir : `str`
            The pickle destination directory.
        object_name : `str`
            The stored file name.
        dump_design_info : `bool`, default True
            Whether to dump design info.
            Design info is a patsy class that includes the design matrix information.
            It takes longer to dump design info.
        overwrite_exist_dir : `bool`, default False
            What to do when ``destination_dir`` already exists.
            Removes the original directory when exists, if set to True.

        Returns
        -------
        This function writes to local files and does not return anything.
        """
        if self.forecast_result is None:
            raise ValueError("self.forecast_result is None, nothing to dump.")
        dump_obj(
            obj=self.forecast_result,
            dir_name=destination_dir,
            obj_name=object_name,
            dump_design_info=dump_design_info,
            overwrite_exist_dir=overwrite_exist_dir
        )

    def load_forecast_result(
            self,
            source_dir,
            load_design_info=True):
        """Loads ``self.forecast_result`` from local files created by ``self.dump_result``.

        Parameters
        ----------
        source_dir : `str`
            The source file directory.
        load_design_info : `bool`, default True
            Whether to load design info.
            Design info is a patsy class that includes the design matrix information.
            It takes longer to load design info.
        """
        if self.forecast_result is not None:
            raise ValueError("self.forecast_result is not None, please create a new instance.")
        self.forecast_result = load_obj(
            dir_name=source_dir,
            obj=None,
            load_design_info=load_design_info
        )
