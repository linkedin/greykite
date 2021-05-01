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
"""Enums for data frequency and seasonality."""

from collections import namedtuple
from enum import Enum


class TimeEnum(Enum):
    """Time constants"""
    ONE_WEEK_IN_DAYS = 7
    ONE_MONTH_IN_DAYS = 30
    ONE_QUARTER_IN_DAYS = 90
    ONE_YEAR_IN_DAYS = 365
    # Approximate number of seconds corresponding to each period.
    # May vary for leap year, daylight savings, etc.
    ONE_MINUTE_IN_SECONDS = 60
    ONE_HOUR_IN_SECONDS = 3600
    ONE_DAY_IN_SECONDS = 24 * 3600
    ONE_WEEK_IN_SECONDS = 7 * 24 * 3600
    ONE_MONTH_IN_SECONDS = 30 * 24 * 3600
    ONE_QUARTER_IN_SECONDS = 90 * 24 * 3600
    ONE_YEAR_IN_SECONDS = 365 * 24 * 3600


class SeasonalityEnum(Enum):
    """Valid types of seasonality available to use"""
    DAILY_SEASONALITY = "DAILY_SEASONALITY"
    WEEKLY_SEASONALITY = "WEEKLY_SEASONALITY"
    MONTHLY_SEASONALITY = "MONTHLY_SEASONALITY"
    QUARTERLY_SEASONALITY = "QUARTERLY_SEASONALITY"
    YEARLY_SEASONALITY = "YEARLY_SEASONALITY"


class FrequencyEnum(Enum):
    """Valid types of Frequency available to use"""
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    QUARTER = "QUARTER"
    YEAR = "YEAR"
    MULTIYEAR = "MULTIYEAR"


class SimpleTimeFrequencyEnum(Enum):
    """Provides default properties (horizon, seasonality) for various time frequencies.
    Used to define default values based on the closest frequency to the input data.
    """
    # default_horizon: how far ahead to forecast
    # seconds_per_observation: approximation of the period of each observation (in seconds)
    # valid_seas: valid seasonalities
    Frequency = namedtuple("Frequency", "default_horizon, seconds_per_observation, valid_seas")

    # minutely
    MINUTE = Frequency(
        default_horizon=60,
        seconds_per_observation=TimeEnum.ONE_MINUTE_IN_SECONDS.value,
        valid_seas={SeasonalityEnum.DAILY_SEASONALITY.name,
                    SeasonalityEnum.WEEKLY_SEASONALITY.name,
                    SeasonalityEnum.MONTHLY_SEASONALITY.name,
                    SeasonalityEnum.QUARTERLY_SEASONALITY.name,
                    SeasonalityEnum.YEARLY_SEASONALITY.name})

    # hourly
    HOUR = Frequency(
        default_horizon=24,
        seconds_per_observation=TimeEnum.ONE_HOUR_IN_SECONDS.value,
        valid_seas={SeasonalityEnum.DAILY_SEASONALITY.name,
                    SeasonalityEnum.WEEKLY_SEASONALITY.name,
                    SeasonalityEnum.MONTHLY_SEASONALITY.name,
                    SeasonalityEnum.QUARTERLY_SEASONALITY.name,
                    SeasonalityEnum.YEARLY_SEASONALITY.name})

    # daily
    # monthly seasonality is not used for hourly/daily data to avoid over-specification
    DAY = Frequency(
        default_horizon=30,
        seconds_per_observation=TimeEnum.ONE_DAY_IN_SECONDS.value,
        valid_seas={SeasonalityEnum.WEEKLY_SEASONALITY.name,
                    SeasonalityEnum.MONTHLY_SEASONALITY.name,
                    SeasonalityEnum.QUARTERLY_SEASONALITY.name,
                    SeasonalityEnum.YEARLY_SEASONALITY.name})

    # weekly
    WEEK = Frequency(
        default_horizon=12,
        seconds_per_observation=TimeEnum.ONE_WEEK_IN_SECONDS.value,
        valid_seas={SeasonalityEnum.MONTHLY_SEASONALITY.name,
                    SeasonalityEnum.QUARTERLY_SEASONALITY.name,
                    SeasonalityEnum.YEARLY_SEASONALITY.name})

    # monthly
    MONTH = Frequency(
        default_horizon=12,
        seconds_per_observation=TimeEnum.ONE_MONTH_IN_SECONDS.value,
        valid_seas={SeasonalityEnum.QUARTERLY_SEASONALITY.name,
                    SeasonalityEnum.YEARLY_SEASONALITY.name})

    # quarterly
    QUARTER = Frequency(
        default_horizon=12,
        seconds_per_observation=TimeEnum.ONE_QUARTER_IN_SECONDS.value,
        valid_seas={SeasonalityEnum.YEARLY_SEASONALITY.name})

    # yearly
    YEAR = Frequency(
        default_horizon=2,
        seconds_per_observation=TimeEnum.ONE_YEAR_IN_SECONDS.value,
        valid_seas={})

    # longer than yearly
    MULTIYEAR = Frequency(
        default_horizon=2,
        seconds_per_observation=TimeEnum.ONE_YEAR_IN_SECONDS.value * 2,  # not used anywhere, just a placeholder
        valid_seas={})
