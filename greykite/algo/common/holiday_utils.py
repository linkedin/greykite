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
# original author: Yi Su
"""Constants and utility functions for `HolidayInferrer` and `HolidayGrouper`."""

import datetime


HOLIDAY_NAME_COL = "country_holiday"
"""Holiday name column used in `HolidayInferrer`.
This comes from the default output of `holidays_ext.get_holidays.get_holiday_df`.
"""

HOLIDAY_DATE_COL = "ts"
"""Holiday date column used in `HolidayInferrer`.
This comes from the default output of `holidays_ext.get_holidays.get_holiday_df`.
"""

HOLIDAY_IMPACT_DICT = {
    "Halloween": (1, 1),  # 10/31.
    "New Year's Day": (3, 3),  # 1/1.
}
"""An example. Number of pre/post days that a holiday has impact on.
For example, if Halloween has neighbor (1, 1), Halloween_minus_1
and Halloween_plus_1 will be generated as two additional events.
If not specified, (0, 0) will be used.
"""


def get_dow_grouped_suffix(date: datetime.datetime) -> str:
    """Utility function to generate a suffix given an input ``date``.

    Parameters
    ----------
    date : `datetime.datetime`
        Input timestamp.

    Returns
    -------
    suffix : `str`
        The suffix string starting with "_".
    """
    if date.day_name() == "Saturday":
        return "_Sat"
    elif date.day_name() == "Sunday":
        return "_Sun"
    else:
        return "_WD"


def get_weekday_weekend_suffix(date: datetime.datetime) -> str:
    """Utility function to generate a suffix given an input ``date``.

    Parameters
    ----------
    date : `datetime.datetime`
        Input timestamp.

    Returns
    -------
    suffix : `str`
        The suffix string starting with "_".
    """
    if date.day_name() in ("Saturday", "Sunday"):
        return "_WE"
    else:
        return "_WD"
