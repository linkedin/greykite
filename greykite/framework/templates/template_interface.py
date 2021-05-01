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
"""Template interface. All templates must implement this interface to
work with `model_templates.py`.
"""
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional

import pandas as pd

from greykite.framework.templates.autogen.forecast_config import ForecastConfig


class TemplateInterface(ABC):
    """Defines the interface for the template class.

    Contains a single function that takes input data and forecast config,
    and returns parameters to call
    :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
    """
    @abstractmethod
    def __init__(self):
        """Attributes are the parameters and return value of
        `greykite.framework.templates.template_interface.TemplateInterface.apply_template_for_pipeline_params`.
        """
        self.df: Optional[pd.DataFrame] = None
        """Timeseries data to forecast."""
        self.config: Optional[ForecastConfig] = None
        """Forecast configuration."""
        self.pipeline_params: Optional[Dict] = None
        """Parameters (keyword arguments) to call
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        """

    @property
    @abstractmethod
    def allow_model_template_list(self):
        """Whether the template accepts a list for `config.model_template` (bool)"""

    @property
    @abstractmethod
    def allow_model_components_param_list(self):
        """Whether the template accepts a list for `config.model_components_param` (bool)"""

    @abstractmethod
    def apply_template_for_pipeline_params(self, df: pd.DataFrame, config: Optional[ForecastConfig] = None) -> Dict:
        """Converts forecast config to pipeline params.

        Takes input data and optional configuration parameters
        to customize the model. Returns a set of parameters to call
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.

        Parameters
        ----------
        df : `pandas.DataFrame`
            Timeseries data to forecast.
            See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline` for the format.
            This is the only required parameter.

            ``df`` should only contain time_col, value_col, and regressors.
            Every column in ``df`` besides time_col, value_col is assumed to be a regressor.
            Function does not allow prediction beyond the max available date with any regressor.
        config : :class:`~greykite.framework.templates.autogen.forecast_config.ForecastConfig` or None
            Forecast configuration, including metadata about ``df``,
            forecast horizon, prediction band coverage, evaluation setup,
            model components, and computation. See
            :class:`~greykite.framework.templates.autogen.forecast_config.ForecastConfig`.
            Can be initialized via::

                config = ForecastConfig(
                    model_template=model_template,
                    forecast_horizon=forecast_horizon,
                    coverage=coverage,
                    metadata_param=MetadataParam(...),
                    evaluation_metric_param=EvaluationMetricParam(...),
                    evaluation_period_param=EvaluationPeriodParam(...),
                    model_components_param=ModelComponentsParam(...),
                    computation_param=ComputationParam(...),
                )

        Returns
        -------
        forecast_parameters : `dict` [ `str`, `any` ]
            Parameters (keyword arguments) to call
            :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        """
        pass
