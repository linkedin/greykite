#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# original author: Rachit Arora

from dataclasses import dataclass
from enum import Enum
from typing import Type

from greykite.common.enums import FrequencyEnum
from greykite.common.enums import SeasonalityEnum


@dataclass
class SilverkiteFrequency:
    """Provides properties for modeling for various time frequencies in Silverkite."""

    name: FrequencyEnum
    """The name of the FrequencyEnum e.g. MINUTE, HOUR etc."""

    auto_fourier_seas: [SeasonalityEnum]
    """The recommended seasonality periods to model with fourier series"""


class SilverkiteTimeFrequencyEnum(Enum):
    """Provides properties for modeling for various time frequencies in Silverkite.
    The enum names is the time frequency, corresponding to the simple time
    frequencies in `~greykite.common.enums.SimpleTimeFrequencyEnum`.
    """

    # minutely
    MINUTE: SilverkiteFrequency = SilverkiteFrequency(
        name=FrequencyEnum.MINUTE,
        auto_fourier_seas={SeasonalityEnum.DAILY_SEASONALITY.name,
                           SeasonalityEnum.WEEKLY_SEASONALITY.name,
                           # MONTHLY_SEASONALITY is excluded from defaults
                           SeasonalityEnum.QUARTERLY_SEASONALITY.name,
                           SeasonalityEnum.YEARLY_SEASONALITY.name})

    # hourly
    HOUR: SilverkiteFrequency = SilverkiteFrequency(
        name=FrequencyEnum.HOUR,
        auto_fourier_seas={SeasonalityEnum.DAILY_SEASONALITY.name,
                           SeasonalityEnum.WEEKLY_SEASONALITY.name,
                           # MONTHLY_SEASONALITY is excluded from defaults
                           SeasonalityEnum.QUARTERLY_SEASONALITY.name,
                           SeasonalityEnum.YEARLY_SEASONALITY.name})

    # daily
    DAY: SilverkiteFrequency = SilverkiteFrequency(
        name=FrequencyEnum.DAY,
        auto_fourier_seas={SeasonalityEnum.WEEKLY_SEASONALITY.name,
                           # MONTHLY_SEASONALITY is excluded from defaults
                           SeasonalityEnum.QUARTERLY_SEASONALITY.name,
                           SeasonalityEnum.YEARLY_SEASONALITY.name})

    # weekly
    WEEK: SilverkiteFrequency = SilverkiteFrequency(
        name=FrequencyEnum.WEEK,
        auto_fourier_seas={SeasonalityEnum.MONTHLY_SEASONALITY.name,
                           SeasonalityEnum.QUARTERLY_SEASONALITY.name,
                           SeasonalityEnum.YEARLY_SEASONALITY.name})

    # monthly
    MONTH: SilverkiteFrequency = SilverkiteFrequency(
        name=FrequencyEnum.MONTH,
        auto_fourier_seas={
            # QUARTERLY_SEASONALITY and YEARLY_SEASONALITY are excluded from defaults
            # It's better to use `C(month)` as a categorical feature indicating the month
        })

    # quarterly
    QUARTER: SilverkiteFrequency = SilverkiteFrequency(
        name=FrequencyEnum.QUARTER,
        auto_fourier_seas={
            # YEARLY_SEASONALITY is excluded from defaults
            # It's better to use `C(quarter)` as a categorical feature indicating the quarter
        })

    # yearly
    YEAR: SilverkiteFrequency = SilverkiteFrequency(
        name=FrequencyEnum.YEAR,
        auto_fourier_seas={})

    # longer than yearly
    MULTIYEAR: SilverkiteFrequency = SilverkiteFrequency(
        name=FrequencyEnum.MULTIYEAR,
        auto_fourier_seas={})


class SilverkiteTimeFrequencyEnumMixin:
    """Provides a mixin for the SilverkiteTimeFrequencyEnum constants"""

    def get_silverkite_time_frequency_enum(self) -> Type[SilverkiteTimeFrequencyEnum]:
        """Return the SilverkiteTimeFrequencyEnum constants"""
        return SilverkiteTimeFrequencyEnum
