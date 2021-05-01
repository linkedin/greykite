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


class SilverkiteColumn:
    """Silverkite feature sets for sub-daily data."""

    COLS_HOUR_OF_WEEK: str = "hour_of_week"
    """Silverkite feature_sets_enabled key. constant hour of week effect"""

    COLS_WEEKEND_SEAS: str = "is_weekend:daily_seas"
    """Silverkite feature_sets_enabled key. daily seasonality interaction with is_weekend"""

    COLS_DAY_OF_WEEK_SEAS: str = "day_of_week:daily_seas"
    """Silverkite feature_sets_enabled key. daily seasonality interaction with day of week"""

    COLS_TREND_DAILY_SEAS: str = "trend:is_weekend:daily_seas"
    """Silverkite feature_sets_enabled key. allow daily seasonality to change over time, depending on is_weekend"""

    COLS_EVENT_SEAS: str = "event:daily_seas"
    """Silverkite feature_sets_enabled key. allow sub-daily event effects"""

    # silverkite feature sets for sub-weekly data
    COLS_EVENT_WEEKEND_SEAS: str = "event:is_weekend:daily_seas"
    """Silverkite feature_sets_enabled key. allow sub-daily event effect to interact with is_weekend"""

    COLS_DAY_OF_WEEK: str = "day_of_week"
    """Silverkite feature_sets_enabled key. constant day of week effect"""

    COLS_TREND_WEEKEND: str = "trend:is_weekend"
    """Silverkite feature_sets_enabled key. allow trend (growth, changepoints) to interact with is_weekend"""

    COLS_TREND_DAY_OF_WEEK: str = "trend:day_of_week"
    """Silverkite feature_sets_enabled key. allow trend to interact with day of week"""

    COLS_TREND_WEEKLY_SEAS: str = "trend:weekly_seas"
    """Silverkite feature_sets_enabled key. allow weekly seasonality to change over time"""


class SilverkiteColumnMixin:
    """Provides a mixin for the SilverkiteColumn constants"""

    def get_silverkite_column(self) -> Type[SilverkiteColumn]:
        """Return the SilverkiteColumn constants"""
        return SilverkiteColumn
