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

from greykite.common.constants import TimeFeaturesEnum
from greykite.common.enums import TimeEnum


@dataclass
class SilverkiteSeasonality:
    """Contains information to create `fs_components_df` parameter for `forecast_silverkite` for modeling seasonality."""

    name: str
    """Name of the timeseries feature (e.g. tod, tow etc.)"""

    period: float
    """Period of the fourier series"""

    order: int
    """Recommended order of the fourier series. Add interactions to increase flexibility"""

    seas_names: str
    """Additional label for the seasonality term"""

    default_min_days: int
    """Recommended minimum input data size before adding the seasonality"""


class SilverkiteSeasonalityEnum(Enum):
    """Defines default seasonalities for Silverkite estimator.
    Names should match those in SeasonalityEnum.
    The default order for various seasonalities is stored in this enum.
    """

    DAILY_SEASONALITY: SilverkiteSeasonality = SilverkiteSeasonality(
        name=TimeFeaturesEnum.tod.value,
        period=24.0,
        order=12,
        seas_names="daily",
        default_min_days=2)
    """``tod`` is 0-24 time of day (tod granularity based on input data, up to second level).
    Requires at least two full cycles to add the seasonal term (``default_min_days=2``).
    """

    WEEKLY_SEASONALITY: SilverkiteSeasonality = SilverkiteSeasonality(
        name=TimeFeaturesEnum.tow.value,
        period=7.0,
        order=4,
        seas_names="weekly",
        default_min_days=TimeEnum.ONE_WEEK_IN_DAYS.value * 2)
    """``tow`` is 0-7 time of week (tow granularity based on input data, up to second level).
    ``order=4`` for full flexibility to model daily input.
    """

    MONTHLY_SEASONALITY: SilverkiteSeasonality = SilverkiteSeasonality(
        name=TimeFeaturesEnum.tom.value,
        period=1.0,
        order=2,
        seas_names="monthly",
        default_min_days=TimeEnum.ONE_MONTH_IN_DAYS.value * 2)
    """``tom`` is 0-1 time of month (tom granularity based on input data, up to daily level)."""

    QUARTERLY_SEASONALITY: SilverkiteSeasonality = SilverkiteSeasonality(
        name=TimeFeaturesEnum.toq.value,
        period=1.0,
        order=5,
        seas_names="quarterly",
        default_min_days=TimeEnum.ONE_QUARTER_IN_DAYS.value * 2)
    """``toq`` (continuous time of quarter) with natural period.
    Each day is mapped to a value in [0.0, 1.0) based on its position in the calendar quarter:
    (Jan1-Mar31, Apr1-Jun30, Jul1-Sep30, Oct1-Dec31). The start of each quarter is 0.0.
    """

    YEARLY_SEASONALITY: SilverkiteSeasonality = SilverkiteSeasonality(
        name=TimeFeaturesEnum.ct1.value,
        period=1.0,
        order=15,
        seas_names="yearly",
        default_min_days=round(TimeEnum.ONE_YEAR_IN_DAYS.value * 1.5))
    """``ct1`` (continuous year) with natural period."""


class SilverkiteSeasonalityEnumMixin:
    """Provides a mixin for the SilverkiteSeasonalityEnum constants"""

    def get_silverkite_seasonality_enum(self) -> Type[SilverkiteSeasonalityEnum]:
        """Return the SilverkiteSeasonalityEnum constants"""
        return SilverkiteSeasonalityEnum
