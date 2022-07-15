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
"""Automatically populates model template based on input."""


from typing import Optional

import pandas as pd

from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.time_properties import min_gap_in_seconds
from greykite.framework.pipeline.utils import get_default_time_parameters
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.forecast_config_defaults import ForecastConfigDefaults
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_DAILY_90
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_HOURLY_1
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_HOURLY_24
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_HOURLY_168
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_HOURLY_336
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_WEEKLY
from greykite.sklearn.cross_validation import RollingTimeSeriesSplit


def get_auto_silverkite_model_template(
        df: pd.DataFrame,
        default_model_template_name: str,
        config: Optional[ForecastConfig] = None):
    """Gets the most appropriate model template that fits the
    input df's frequency, forecast horizon and number of cv splits.

    We define the cv to be sufficient if both number of splits is at least 5
    and the number of evaluated points is at least 30.
    Multi-template will be used only when cv is sufficient.

    Parameter
    ---------
    df : `pandas.DataFrame`
        The input time series.
    default_model_template_name : `str`
        The default model template name.
        The default must be something other than "AUTO", since this function resolves
        "AUTO" to a specific model template.
    config : :class:`~greykite.framework.templates.model_templates.ForecastConfig` or None, default None
        Config object for template class to use.
        See :class:`~greykite.framework.templates.model_templates.ForecastConfig`.

    Returns
    -------
    model_template : `str`
        The model template name that best fits the scenario.
    """
    if default_model_template_name == "AUTO":
        raise ValueError("The `default_model_template_name` in "
                         "`get_auto_silverkite_model_template` cannot be 'AUTO'.")

    model_template = default_model_template_name
    if config is None:
        return model_template

    forecast_config_defaults = ForecastConfigDefaults()

    # Tries to get the data frequency.
    # If failed, uses the default model template.
    config = forecast_config_defaults.apply_forecast_config_defaults(config)
    metadata = config.metadata_param
    freq = metadata.freq
    if freq is None:
        freq = pd.infer_freq(
            df[metadata.time_col]
        )
    if freq is None:
        # NB: frequency inference fails if there are missing points in the input data
        log_message(
            message=f"Model template was set to 'auto', however, the data frequency "
                    f"is not given and can not be inferred. "
                    f"Using default model template '{model_template}'.",
            level=LoggingLevelEnum.INFO
        )
        return model_template

    # Gets the number of cv splits.
    # This is to decide if we want to go with multiple templates.
    evaluation_period = config.evaluation_period_param
    # Tries to infer ``forecast_horizon``, ``test_horizon``, ``cv_horizon``, etc. if not given.
    period = min_gap_in_seconds(df=df, time_col=metadata.time_col)
    default_time_params = get_default_time_parameters(
        period=period,
        num_observations=df.shape[0],
        forecast_horizon=config.forecast_horizon,
        test_horizon=evaluation_period.test_horizon,
        periods_between_train_test=evaluation_period.periods_between_train_test,
        cv_horizon=evaluation_period.cv_horizon,
        cv_min_train_periods=evaluation_period.cv_min_train_periods,
        cv_periods_between_train_test=evaluation_period.cv_periods_between_train_test)
    forecast_horizon = default_time_params.get("forecast_horizon")
    cv_horizon = default_time_params.get("cv_horizon")
    cv = RollingTimeSeriesSplit(
        forecast_horizon=cv_horizon,
        min_train_periods=default_time_params.get("cv_min_train_periods"),
        expanding_window=evaluation_period.cv_expanding_window,
        use_most_recent_splits=evaluation_period.cv_use_most_recent_splits,
        periods_between_splits=evaluation_period.cv_periods_between_splits,
        periods_between_train_test=default_time_params.get("cv_periods_between_train_test"),
        max_splits=evaluation_period.cv_max_splits)
    testing_length = default_time_params.get("test_horizon")
    testing_length += default_time_params.get("periods_between_train_test")
    if testing_length > 0:
        df_sample = df.iloc[:-testing_length].copy()
    else:
        df_sample = df.copy()
    n_splits = cv.get_n_splits(X=df_sample)
    # We define the cv to be sufficient if both number of splits is at least 5
    # and the number of evaluated points is at least 30.
    splits_sufficient = (n_splits >= 5) and (n_splits * cv_horizon >= 30) and (evaluation_period.cv_max_splits != 0)

    # Handles the frequencies separately.
    # Depending on the forecast horizon and the number of splits,
    # we choose the most appropriate model template.
    # If no close model template is available,
    # the model template remains the default.
    if freq == "H":
        if not splits_sufficient:
            # For 1 hour case, the best single model template
            # uses linear fit algorithm which has small risk of numerical issues.
            # We removed them from our auto template and use SILVERKITE in both cases.
            if forecast_horizon == 1:
                model_template = SILVERKITE_HOURLY_1[0]
            elif forecast_horizon <= 24 * 2:
                model_template = SILVERKITE_HOURLY_24[0]
            elif forecast_horizon <= 24 * 7 + 24:
                model_template = SILVERKITE_HOURLY_168[0]
            elif forecast_horizon <= 24 * 7 * 3:
                model_template = SILVERKITE_HOURLY_336[0]
        else:
            if forecast_horizon == 1:
                model_template = ModelTemplateEnum.SILVERKITE_HOURLY_1.name
            elif forecast_horizon <= 24 * 2:
                model_template = ModelTemplateEnum.SILVERKITE_HOURLY_24.name
            elif forecast_horizon <= 24 * 7 + 24:
                model_template = ModelTemplateEnum.SILVERKITE_HOURLY_168.name
            elif forecast_horizon <= 24 * 7 * 3:
                model_template = ModelTemplateEnum.SILVERKITE_HOURLY_336.name
    elif freq == "D":
        if not splits_sufficient:
            if forecast_horizon <= 7:
                model_template = ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_1.name
            elif forecast_horizon >= 90:
                model_template = SILVERKITE_DAILY_90[0]
        else:
            if forecast_horizon <= 7:
                model_template = ModelTemplateEnum.SILVERKITE_DAILY_1.name
            elif forecast_horizon >= 90:
                model_template = ModelTemplateEnum.SILVERKITE_DAILY_90.name
    elif freq in ["W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
        if not splits_sufficient:
            model_template = SILVERKITE_WEEKLY[0]
        else:
            model_template = ModelTemplateEnum.SILVERKITE_WEEKLY.name
    elif freq in ["M", "MS", "SM", "BM", "CBM", "SMS", "BMS", "CBMS"]:
        # Monthly template includes monthly data and some variants.
        # See pandas documentation
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        model_template = ModelTemplateEnum.SILVERKITE_MONTHLY.name
    log_message(
        message=f"Model template was set to 'auto'. "
                f"Automatically found most appropriate model template '{model_template}'.",
        level=LoggingLevelEnum.INFO
    )
    return model_template
