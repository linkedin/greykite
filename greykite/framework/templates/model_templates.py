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
"""This file applies and runs pre-packaged model templates.

A model template provides a wrapper around
:func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`
that provides suitable parameters for modeling different types of input data
(frequency, forecast horizon, etc.).

Each model template provides default parameters to call
:func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`,
most importantly:

    * ``estimator_name``,
    * ``hyperparameter_grid``,
    * ``pipeline``

These defaults can be customized by the other parameters of a
:class:`~greykite.framework.templates.model_templates.ForecastConfig`.

Thus, a model template defines a model, whereas
:func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`
takes a model through the training and validation steps.

Thus, model templates make it easy to create and tune a forecast.
Users are advised to start with the provided templates and tune the
model component parameters as necessary.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Type

from greykite.framework.templates.auto_arima_template import AutoArimaTemplate
from greykite.framework.templates.lag_based_template import LagBasedTemplate
from greykite.framework.templates.multistage_forecast_template import MultistageForecastTemplate
from greykite.framework.templates.prophet_template import ProphetTemplate
from greykite.framework.templates.silverkite_template import SilverkiteTemplate
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate
from greykite.framework.templates.template_interface import TemplateInterface


@dataclass
class ModelTemplate:
    """A model template consists of a template class,
    a description, and a name.

    This class holds the template class and description.
    The model template name is the member name in
    `greykite.framework.templates.model_templates.ModelTemplateEnum`.
    """
    template_class: Type[TemplateInterface]
    """A class that implements the template interface."""
    description: str
    """A description of the model template."""


class ModelTemplateEnum(Enum):
    """Available model templates.

    Enumerates the possible values for the ``model_template`` attribute of
    :class:`~greykite.framework.templates.model_templates.ForecastConfig`.

    The value has type `~greykite.framework.templates.model_templates.ModelTemplate` which contains:

        - the template class that recognizes the model_template. Template classes implement the
          `~greykite.framework.templates.template_interface.TemplateInterface` interface.
        - a plain-text description of what the model_template is for,

    The description should be unique across enum members. The template class
    can be shared, because a template class can recognize multiple model templates.
    For example, the same template class may use different default values for
    ``ForecastConfig.model_components_param`` depending on ``ForecastConfig.model_template``.

    Notes
    -----
    The template classes
    `~greykite.framework.templates.silverkite_template.SilverkiteTemplate`
    and `~greykite.framework.templates.prophet_template.ProphetTemplate`
    recognize only the model templates explicitly enumerated here.

    However, the `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`
    template class allows additional model templates to be specified generically.
    Any object of type `~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`
    can be used as the model_template.
    These generic model templates are valid but not enumerated here.
    """
    SILVERKITE = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model with automatic growth, seasonality, holidays, "
                    "automatic autoregression, normalization "
                    "and interactions. Best for hourly and daily frequencies."
                    "Uses `SimpleSilverkiteEstimator`.")
    """Silverkite model with automatic growth, seasonality, holidays,
    automatic autoregression, normalization
    and interactions. Best for hourly and daily frequencies.
    Uses `SimpleSilverkiteEstimator`.
    """
    SILVERKITE_DAILY_1_CONFIG_1 = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Config 1 in template ``SILVERKITE_DAILY_1``. "
                    "Compared to ``SILVERKITE``, it uses parameters "
                    "specifically tuned for daily data and 1-day forecast.")
    """Config 1 in template ``SILVERKITE_DAILY_1``.
    Compared to ``SILVERKITE``, it uses parameters
    specifically tuned for daily data and 1-day forecast.
    """
    SILVERKITE_DAILY_1_CONFIG_2 = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Config 2 in template ``SILVERKITE_DAILY_1``. "
                    "Compared to ``SILVERKITE``, it uses parameters "
                    "specifically tuned for daily data and 1-day forecast.")
    """Config 2 in template ``SILVERKITE_DAILY_1``.
    Compared to ``SILVERKITE``, it uses parameters
    specifically tuned for daily data and 1-day forecast.
    """
    SILVERKITE_DAILY_1_CONFIG_3 = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Config 3 in template ``SILVERKITE_DAILY_1``. "
                    "Compared to ``SILVERKITE``, it uses parameters "
                    "specifically tuned for daily data and 1-day forecast.")
    """Config 3 in template ``SILVERKITE_DAILY_1``.
    Compared to ``SILVERKITE``, it uses parameters
    specifically tuned for daily data and 1-day forecast.
    """
    SILVERKITE_DAILY_1 = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model specifically tuned for daily data and 1-day forecast. "
                    "Contains 3 candidate configs for grid search, "
                    "optimized the seasonality and changepoint parameters.")
    """Silverkite model specifically tuned for daily data and 1-day forecast.
    Contains 3 candidate configs for grid search,
    optimized the seasonality and changepoint parameters.
    """
    SILVERKITE_DAILY_90 = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model specifically tuned for daily data with 90 days forecast horizon. "
                    "Contains 4 hyperparameter combinations for grid search. "
                    "Uses `SimpleSilverkiteEstimator`.")
    """Silverkite model specifically tuned for daily data with 90 days forecast horizon.
    Contains 4 hyperparameter combinations for grid search.
    Uses `SimpleSilverkiteEstimator`.
    """
    SILVERKITE_WEEKLY = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model specifically tuned for weekly data. "
                    "Contains 4 hyperparameter combinations for grid search. "
                    "Uses `SimpleSilverkiteEstimator`.")
    """Silverkite model specifically tuned for weekly data.
    Contains 4 hyperparameter combinations for grid search.
    Uses `SimpleSilverkiteEstimator`.
    """
    SILVERKITE_MONTHLY = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model specifically tuned for monthly data. "
                    "Uses `SimpleSilverkiteEstimator`.")
    """Silverkite model specifically tuned for monthly data.
    Uses `SimpleSilverkiteEstimator`.
    """
    SILVERKITE_HOURLY_1 = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model specifically tuned for hourly data with 1 hour forecast horizon. "
                    "Contains 3 hyperparameter combinations for grid search. "
                    "Uses `SimpleSilverkiteEstimator`.")
    """Silverkite model specifically tuned for hourly data with 1 hour forecast horizon.
    Contains 4 hyperparameter combinations for grid search.
    Uses `SimpleSilverkiteEstimator`."""
    SILVERKITE_HOURLY_24 = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model specifically tuned for hourly data with 24 hours (1 day) forecast horizon. "
                    "Contains 4 hyperparameter combinations for grid search. "
                    "Uses `SimpleSilverkiteEstimator`.")
    """Silverkite model specifically tuned for hourly data with 24 hours (1 day) forecast horizon.
    Contains 4 hyperparameter combinations for grid search.
    Uses `SimpleSilverkiteEstimator`."""
    SILVERKITE_HOURLY_168 = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model specifically tuned for hourly data with 168 hours (1 week) forecast horizon. "
                    "Contains 4 hyperparameter combinations for grid search. "
                    "Uses `SimpleSilverkiteEstimator`.")
    """Silverkite model specifically tuned for hourly data with 168 hours (1 week) forecast horizon.
    Contains 4 hyperparameter combinations for grid search.
    Uses `SimpleSilverkiteEstimator`."""
    SILVERKITE_HOURLY_336 = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model specifically tuned for hourly data with 336 hours (2 weeks) forecast horizon. "
                    "Contains 4 hyperparameter combinations for grid search. "
                    "Uses `SimpleSilverkiteEstimator`.")
    """Silverkite model specifically tuned for hourly data with 336 hours (2 weeks) forecast horizon.
    Contains 4 hyperparameter combinations for grid search.
    Uses `SimpleSilverkiteEstimator`.
    """
    SILVERKITE_EMPTY = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Silverkite model with no component included by default. Fits only a constant intercept. "
                    "Select and customize this template to add only the terms you want. "
                    "Uses `SimpleSilverkiteEstimator`.")
    """Silverkite model with no component included by default. Fits only a constant intercept.
    Select and customize this template to add only the terms you want.
    Uses `SimpleSilverkiteEstimator`.
    """
    SK = ModelTemplate(
        template_class=SilverkiteTemplate,
        description="Silverkite model with low-level interface. For flexible model tuning "
                    "if SILVERKITE template is not flexible enough. Not for use out-of-the-box: "
                    "customization is needed for good performance. Uses `SilverkiteEstimator`.")
    """Silverkite model with low-level interface. For flexible model tuning
    if SILVERKITE template is not flexible enough. Not for use out-of-the-box:
    customization is needed for good performance. Uses `SilverkiteEstimator`.
    """
    PROPHET = ModelTemplate(
        template_class=ProphetTemplate,
        description="Prophet model with growth, seasonality, holidays, additional regressors "
                    "and prediction intervals. Uses `ProphetEstimator`.")
    """Prophet model with growth, seasonality, holidays, additional regressors
    and prediction intervals. Uses `ProphetEstimator`."""
    AUTO_ARIMA = ModelTemplate(
        template_class=AutoArimaTemplate,
        description="Auto ARIMA model with fit and prediction intervals. "
                    "Uses `AutoArimaEstimator`.")
    """ARIMA model with automatic order selection. Uses `AutoArimaEstimator`."""
    SILVERKITE_TWO_STAGE = ModelTemplate(
        template_class=MultistageForecastTemplate,
        description="MultistageForecastTemplate's default model template. A two-stage model. "
                    "The first step takes a longer history and learns the long-term effects, "
                    "while the second step takes a shorter history and learns the short-term residuals."
    )
    """Multistage forecast model's default model template. A two-stage model. "
    "The first step takes a longer history and learns the long-term effects, "
    "while the second step takes a shorter history and learns the short-term residuals.
    """
    MULTISTAGE_EMPTY = ModelTemplate(
        template_class=MultistageForecastTemplate,
        description="Empty configuration for Multistage Forecast. "
                    "All parameters will be exactly what user inputs. "
                    "Not to be used without overriding."""
    )
    """Empty configuration for Multistage Forecast.
    All parameters will be exactly what user inputs.
    Not to be used without overriding.
    """
    AUTO = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="Automatically selects the SimpleSilverkite model template that corresponds to the forecast "
                    "problem. Selection is based on data frequency, forecast horizon, and CV configuration."
    )
    """Automatically selects the SimpleSilverkite model template that corresponds to the forecast problem.
    Selection is based on data frequency, forecast horizon, and CV configuration.
    """
    LAG_BASED = ModelTemplate(
        template_class=LagBasedTemplate,
        description="Uses aggregated past observations as predictions. Examples are "
                    "past day, week-over-week, week-over-3-week median, etc."
    )
    """Uses aggregated past observations as predictions. Examples are
    past day, week-over-week, week-over-3-week median, etc.
    """
    SILVERKITE_WOW = ModelTemplate(
        template_class=MultistageForecastTemplate,
        description="The Silverkite+WOW model uses Silverkite to model yearly/quarterly/monthly seasonality, "
                    "growth and holiday effects first, then uses week over week to estimate the residuals. "
                    "The final prediction is the total of the two models. "
                    "This avoids the normal week over week (WOW) estimation's weakness in capturing "
                    "growth and holidays."
    )
    """The Silverkite+WOW model uses Silverkite to model yearly/quarterly/monthly seasonality,
    growth and holiday effects first, then uses week over week to estimate the residuals.
    The final prediction is the total of the two models.
    This avoids the normal week over week (WOW) estimation's weakness in capturing growth and holidays.
    """
