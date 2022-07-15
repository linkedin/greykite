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


@dataclass
class SilverkiteComponent:
    """Defines groupby time feature, xlabel and ylabel for Silverkite Component Plots."""
    groupby_time_feature: str

    xlabel: str

    ylabel: str


class SilverkiteComponentsEnum(Enum):
    """Defines groupby time feature, xlabel and ylabel for Silverkite Component Plots."""

    DAILY_SEASONALITY: SilverkiteComponent = SilverkiteComponent(
        groupby_time_feature=TimeFeaturesEnum.tod.value,
        xlabel="Hour of day",
        ylabel="daily")

    WEEKLY_SEASONALITY: SilverkiteComponent = SilverkiteComponent(
        groupby_time_feature=TimeFeaturesEnum.tow.value,
        xlabel="Day of week",
        ylabel="weekly")

    MONTHLY_SEASONALITY: SilverkiteComponent = SilverkiteComponent(
        groupby_time_feature=TimeFeaturesEnum.tom.value,
        xlabel="Time of month",
        ylabel="monthly")

    QUARTERLY_SEASONALITY: SilverkiteComponent = SilverkiteComponent(
        groupby_time_feature=TimeFeaturesEnum.toq.value,
        xlabel="Time of quarter",
        ylabel="quarterly")

    YEARLY_SEASONALITY: SilverkiteComponent = SilverkiteComponent(
        groupby_time_feature=TimeFeaturesEnum.toy.value,
        xlabel="Time of year",
        ylabel="yearly")


class SilverkiteComponentsEnumMixin:
    """Provides a mixin for the SilverkiteComponentsEnum constants"""

    def get_silverkite_components_enum(self) -> Type[SilverkiteComponentsEnum]:
        """Return the SilverkiteComponentsEnum constants"""
        return SilverkiteComponentsEnum
