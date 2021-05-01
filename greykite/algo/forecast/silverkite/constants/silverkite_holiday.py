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
# original author: Rachit Arora

from typing import Type


class SilverkiteHoliday:
    """Holiday constants to be used by Silverkite"""

    HOLIDAY_LOOKUP_COUNTRIES_AUTO = ("UnitedStates", "UnitedKingdom", "India", "France", "China")
    """Auto setting for the countries that contain the holidays to include in the model"""

    HOLIDAYS_TO_MODEL_SEPARATELY_AUTO = (
        "New Year's Day",
        "Chinese New Year",
        "Christmas Day",
        "Independence Day",
        "Thanksgiving",
        "Labor Day",
        "Good Friday",
        "Easter Monday [England, Wales, Northern Ireland]",
        "Memorial Day",
        "Veterans Day")
    """Auto setting for the holidays to include in the model"""

    ALL_HOLIDAYS_IN_COUNTRIES = "ALL_HOLIDAYS_IN_COUNTRIES"
    """Value for `holidays_to_model_separately` to request all holidays in the lookup countries"""

    HOLIDAYS_TO_INTERACT = (
        "Christmas Day",
        "Christmas Day_minus_1",
        "Christmas Day_minus_2",
        "Christmas Day_plus_1",
        "Christmas Day_plus_2",
        "New Years Day",
        "New Years Day_minus_1",
        "New Years Day_minus_2",
        "New Years Day_plus_1",
        "New Years Day_plus_2",
        "Thanksgiving",  # always on a Thursday, so Thursday/Friday may be affected
        "Thanksgiving_plus_1",
        "Independence Day")
    """Significant holidays that may have a different daily seasonality pattern"""


class SilverkiteHolidayMixin:
    """Provides a mixin for the SilverkiteHoliday constants"""

    def get_silverkite_holiday(self) -> Type[SilverkiteHoliday]:
        """Return the SilverkiteHoliday constants"""
        return SilverkiteHoliday
