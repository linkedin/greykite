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
"""Provides default parameter values for ``ForecastConfig``, shared across templates.
The "*_template.py" files contain defaults particular to a specific template.
"""
import dataclasses
from typing import List
from typing import Optional
from typing import Union

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.framework.constants import COMPUTATION_N_JOBS
from greykite.framework.constants import COMPUTATION_VERBOSE
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.constants import EVALUATION_PERIOD_CV_MAX_SPLITS
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam


class ForecastConfigDefaults:
    """Class that applies default values to a
    `~greykite.framework.templates.autogen.forecast_config.ForecastConfig` object.

    Provides these methods:

        - apply_metadata_defaults
        - apply_evaluation_metric_defaults
        - apply_evaluation_period_defaults
        - apply_computation_defaults
        - apply_model_components_defaults
        - apply_forecast_config_defaults

    Subclasses may override these if different defaults are desired.
    """
    DEFAULT_MODEL_TEMPLATE = "AUTO"
    """The default model template. See `~greykite.framework.templates.model_templates.ModelTemplateEnum`.
    Uses a string to avoid circular imports.
    """

    @staticmethod
    def apply_computation_defaults(computation: Optional[ComputationParam] = None) -> ComputationParam:
        """Applies the default ComputationParam values to the given object.
        If an expected attribute value is provided, the value is unchanged. Otherwise the default value for it is used.
        Other attributes are untouched.
        If the input object is None, it creates a ComputationParam object.

        Parameters
        ----------
        computation : `~greykite.framework.templates.autogen.forecast_config.ComputationParam` or None
            The ComputationParam object.

        Returns
        -------
        computation : `~greykite.framework.templates.autogen.forecast_config.ComputationParam`
            Valid ComputationParam object with the provided attribute values and the default attribute values if not.
        """
        if computation is None:
            computation = ComputationParam()
        if computation.n_jobs is None:
            computation.n_jobs = COMPUTATION_N_JOBS
        if computation.verbose is None:
            computation.verbose = COMPUTATION_VERBOSE
        return computation

    @staticmethod
    def apply_evaluation_metric_defaults(evaluation: Optional[EvaluationMetricParam] = None) -> EvaluationMetricParam:
        """Applies the default EvaluationMetricParam values to the given object.
        If an expected attribute value is provided, the value is unchanged. Otherwise the default value for it is used.
        Other attributes are untouched.
        If the input object is None, it creates a EvaluationMetricParam object.

        Parameters
        ----------
        evaluation : `~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam` or None
            The EvaluationMetricParam object.

        Returns
        -------
        evaluation : `~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam`
            Valid EvaluationMetricParam object with the provided attribute values and the default attribute values if not.
        """
        if evaluation is None:
            evaluation = EvaluationMetricParam()
        if evaluation.cv_selection_metric is None:
            # NB: subclass may want to override, if designed for a different objective (e.g. quantile loss)
            evaluation.cv_selection_metric = EvaluationMetricEnum.MeanAbsolutePercentError.name
        if evaluation.cv_report_metrics is None:
            evaluation.cv_report_metrics = CV_REPORT_METRICS_ALL
        return evaluation

    @staticmethod
    def apply_evaluation_period_defaults(evaluation: Optional[EvaluationPeriodParam] = None) -> EvaluationPeriodParam:
        """Applies the default EvaluationPeriodParam values to the given object.
        If an expected attribute value is provided, the value is unchanged. Otherwise the default value for it is used.
        Other attributes are untouched.
        If the input object is None, it creates a EvaluationPeriodParam object.

        Parameters
        ----------
        evaluation : `~greykite.framework.templates.autogen.forecast_config.EvaluationPeriodParam` or None
            The EvaluationMetricParam object.

        Returns
        -------
        evaluation : `~greykite.framework.templates.autogen.forecast_config.EvaluationPeriodParam`
            Valid EvaluationPeriodParam object with the provided attribute values and the default attribute values if not.
        """
        if evaluation is None:
            evaluation = EvaluationPeriodParam()
        if evaluation.cv_max_splits is None:
            evaluation.cv_max_splits = EVALUATION_PERIOD_CV_MAX_SPLITS
        if evaluation.cv_periods_between_train_test is None:
            evaluation.cv_periods_between_train_test = evaluation.periods_between_train_test
        if evaluation.cv_expanding_window is None:
            # NB: subclass may want to override.
            evaluation.cv_expanding_window = True   # good for long-term forecasts, or when data are limited
        return evaluation

    @staticmethod
    def apply_metadata_defaults(metadata: Optional[MetadataParam] = None) -> MetadataParam:
        """Applies the default MetadataParam values to the given object.
        If an expected attribute value is provided, the value is unchanged. Otherwise the default value for it is used.
        Other attributes are untouched.
        If the input object is None, it creates a MetadataParam object.

        Parameters
        ----------
        metadata : `~greykite.framework.templates.autogen.forecast_config.MetadataParam` or None
            The MetadataParam object.

        Returns
        -------
        metadata : `~greykite.framework.templates.autogen.forecast_config.MetadataParam`
            Valid MetadataParam object with the provided attribute values and the default attribute values if not.
        """
        if metadata is None:
            metadata = MetadataParam()
        if metadata.time_col is None:
            metadata.time_col = TIME_COL
        if metadata.value_col is None:
            metadata.value_col = VALUE_COL
        return metadata

    @staticmethod
    def apply_model_components_defaults(model_components: Optional[Union[ModelComponentsParam, List[Optional[ModelComponentsParam]]]] = None) \
            -> Union[ModelComponentsParam, List[ModelComponentsParam]]:
        """Applies the default ModelComponentsParam values to the given object.

        Converts None to a ModelComponentsParam object.
        Unpacks a list of a single element to the element itself.

        Parameters
        ----------
        model_components : `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` or None or list of such items
            The ModelComponentsParam object.

        Returns
        -------
        model_components : `~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam` or list of such items
            Valid ModelComponentsParam object with the provided attribute values and the default attribute values if not.
        """
        # Converts single element to a list
        if not isinstance(model_components, list):
            model_components = [model_components]
        # Replaces all `None` with ModelComponentsParam()
        model_components = [m if m is not None else ModelComponentsParam() for m in model_components]
        # model_components can be provided as a list or a single element.
        # A list of a single element is unpacked to that element.
        # (Some template classes like SilverkiteTemplate do not allow model_components
        # to be a list.)
        if isinstance(model_components, list) and len(model_components) == 1:
            model_components = model_components[0]
        return model_components

    def apply_model_template_defaults(self, model_template: Optional[Union[str, List[Optional[str]]]] = None) -> Union[str, List[str]]:
        """Applies the default model template to the given object.

        Unpacks a list of a single element to the element itself.
        Sets default value if None.

        Parameters
        ----------
        model_template : `str` or None or `list` [None, `str`]
            The model template name.
            See valid names in `~greykite.framework.templates.model_templates.ModelTemplateEnum`.

        Returns
        -------
        model_template : `str` or `list` [`str`]
            The model template name, with defaults value used if not provided.
        """
        # Converts single element to a list
        if not isinstance(model_template, list):
            model_template = [model_template]
        model_template = [m if m is not None else self.DEFAULT_MODEL_TEMPLATE for m in model_template]
        if isinstance(model_template, list) and len(model_template) == 1:
            # model_template can be provided as a list or a single element.
            # A list of a single element is unpacked to that element.
            # (Some template classes like SilverkiteTemplate do not allow model_template
            # to be a list.)
            model_template = model_template[0]
        return model_template

    def apply_forecast_config_defaults(self, config: Optional[ForecastConfig] = None) -> ForecastConfig:
        """Applies the default Forecast Config values to the given config.
        If an expected attribute value is provided, the value is unchanged.
        Otherwise the default value for it is used.
        Other attributes are untouched.
        If the input config is None, it creates a Forecast Config.

        Parameters
        ----------
        config : :class:`~greykite.framework.templates.autogen.forecast_config.ForecastConfig` or None
            Forecast configuration if available. See
            :class:`~greykite.framework.templates.autogen.forecast_config.ForecastConfig`.

        Returns
        -------
        config : :class:`~greykite.framework.templates.model_templates.ForecastConfig`
            A valid Forecast Config which contains the provided attribute values and the default attribute values if not.
        """
        if config is None:
            config = ForecastConfig()
        else:
            # makes a copy to avoid mutating input
            config = dataclasses.replace(config)

        config.computation_param = self.apply_computation_defaults(config.computation_param)
        config.evaluation_metric_param = self.apply_evaluation_metric_defaults(config.evaluation_metric_param)
        config.evaluation_period_param = self.apply_evaluation_period_defaults(config.evaluation_period_param)
        config.metadata_param = self.apply_metadata_defaults(config.metadata_param)
        config.model_components_param = self.apply_model_components_defaults(config.model_components_param)
        config.model_template = self.apply_model_template_defaults(config.model_template)
        return config
