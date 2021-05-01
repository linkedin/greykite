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

from greykite.algo.forecast.silverkite.constants.silverkite_column import SilverkiteColumnMixin
from greykite.algo.forecast.silverkite.constants.silverkite_component import SilverkiteComponentsEnumMixin
from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHolidayMixin
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonalityEnumMixin
from greykite.algo.forecast.silverkite.constants.silverkite_time_frequency import SilverkiteTimeFrequencyEnumMixin


class SilverkiteConstant(
    SilverkiteColumnMixin,
    SilverkiteComponentsEnumMixin,
    SilverkiteHolidayMixin,
    SilverkiteSeasonalityEnumMixin,
    SilverkiteTimeFrequencyEnumMixin
):
    """Uses the appropriate constant mixins to provide all the constants that will be used by Silverkite."""
    pass


default_silverkite_constant = SilverkiteConstant()
