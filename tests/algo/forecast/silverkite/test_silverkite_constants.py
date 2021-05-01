from typing import Type

from overrides import overrides

from greykite.algo.forecast.silverkite.constants.silverkite_column import SilverkiteColumn
from greykite.algo.forecast.silverkite.constants.silverkite_component import SilverkiteComponentsEnum
from greykite.algo.forecast.silverkite.constants.silverkite_constant import SilverkiteConstant
from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonalityEnum
from greykite.algo.forecast.silverkite.constants.silverkite_time_frequency import SilverkiteTimeFrequencyEnum
from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.algo.forecast.silverkite.forecast_simple_silverkite import SimpleSilverkiteForecast
from greykite.common.enums import FrequencyEnum
from greykite.common.enums import SeasonalityEnum


def test_silverkite_seasonality_enum():
    """Tests SilverkiteSeasonalityEnum accessors"""
    assert SilverkiteSeasonalityEnum.DAILY_SEASONALITY.value.name == "tod"
    assert SilverkiteSeasonalityEnum.DAILY_SEASONALITY.value.period == 24.0
    assert SilverkiteSeasonalityEnum.DAILY_SEASONALITY.value.order == 12
    assert SilverkiteSeasonalityEnum.DAILY_SEASONALITY.value.seas_names == "daily"
    assert SilverkiteSeasonalityEnum.DAILY_SEASONALITY.value.default_min_days == 2

    assert SilverkiteSeasonalityEnum.WEEKLY_SEASONALITY.value.name == "tow"
    assert SilverkiteSeasonalityEnum.MONTHLY_SEASONALITY.value.name == "tom"
    assert SilverkiteSeasonalityEnum.QUARTERLY_SEASONALITY.value.name == "toq"
    assert SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value.seas_names == "yearly"
    assert SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value.default_min_days == 548

    for name in SilverkiteSeasonalityEnum.__dict__["_member_names_"]:
        assert name == "Seasonality" or name in SeasonalityEnum.__dict__["_member_names_"]


def test_silverkite_time_frequency_enum():
    """Tests SilverkiteTimeFrequencyEnum accessors"""
    assert SilverkiteTimeFrequencyEnum.MINUTE.value.auto_fourier_seas == {
        SeasonalityEnum.DAILY_SEASONALITY.name,
        SeasonalityEnum.WEEKLY_SEASONALITY.name,
        SeasonalityEnum.QUARTERLY_SEASONALITY.name,
        SeasonalityEnum.YEARLY_SEASONALITY.name}
    assert SilverkiteTimeFrequencyEnum.DAY.value.auto_fourier_seas == {
        SeasonalityEnum.WEEKLY_SEASONALITY.name,
        SeasonalityEnum.QUARTERLY_SEASONALITY.name,
        SeasonalityEnum.YEARLY_SEASONALITY.name}
    assert SilverkiteTimeFrequencyEnum.QUARTER.value.auto_fourier_seas == {}


def test_silverkite_components_enum():
    """Tests SilverkiteSeasonalityEnum accessors"""
    assert SilverkiteComponentsEnum.DAILY_SEASONALITY.value.groupby_time_feature == "tod"
    assert SilverkiteComponentsEnum.DAILY_SEASONALITY.value.xlabel == "Hour of day"
    assert SilverkiteComponentsEnum.DAILY_SEASONALITY.value.ylabel == "daily"

    assert SilverkiteComponentsEnum.WEEKLY_SEASONALITY.value.groupby_time_feature == "tow"
    assert SilverkiteComponentsEnum.MONTHLY_SEASONALITY.value.groupby_time_feature == "tom"
    assert SilverkiteComponentsEnum.QUARTERLY_SEASONALITY.value.groupby_time_feature == "toq"
    assert SilverkiteComponentsEnum.YEARLY_SEASONALITY.value.groupby_time_feature == "toy"

    for component in SilverkiteComponentsEnum.__dict__["_member_names_"]:
        assert component == "Component" or component in SeasonalityEnum.__dict__["_member_names_"]


def test_silverkite_constants():
    silverkite = SilverkiteForecast()
    assert silverkite._silverkite_seasonality_enum is SilverkiteSeasonalityEnum


def test_simple_silverkite_constants():
    silverkite = SimpleSilverkiteForecast()

    assert silverkite._silverkite_column is SilverkiteColumn
    assert silverkite._silverkite_column.COLS_HOUR_OF_WEEK == "hour_of_week"
    assert silverkite._silverkite_column.COLS_WEEKEND_SEAS == "is_weekend:daily_seas"

    assert silverkite._silverkite_holiday is SilverkiteHoliday
    assert silverkite._silverkite_holiday.ALL_HOLIDAYS_IN_COUNTRIES == "ALL_HOLIDAYS_IN_COUNTRIES"
    assert silverkite._silverkite_holiday.HOLIDAY_LOOKUP_COUNTRIES_AUTO == (
        "UnitedStates", "UnitedKingdom", "India", "France", "China")

    assert silverkite._silverkite_seasonality_enum is SilverkiteSeasonalityEnum
    assert silverkite._silverkite_seasonality_enum.DAILY_SEASONALITY.value.name == "tod"
    assert silverkite._silverkite_seasonality_enum.WEEKLY_SEASONALITY.value.name == "tow"

    assert silverkite._silverkite_time_frequency_enum is SilverkiteTimeFrequencyEnum
    assert silverkite._silverkite_time_frequency_enum.MINUTE.value.name == FrequencyEnum.MINUTE
    assert silverkite._silverkite_time_frequency_enum.HOUR.value.name == FrequencyEnum.HOUR


def test_override_silverkite_constant():
    class OverrideSilverkiteColumn(SilverkiteColumn):
        COLS_HOUR_OF_WEEK = "override_" + SilverkiteColumn.COLS_HOUR_OF_WEEK
        COLS_WEEKEND_SEAS = "override_" + SilverkiteColumn.COLS_WEEKEND_SEAS

    class OverrideSilverkiteHoliday(SilverkiteHoliday):
        ALL_HOLIDAYS_IN_COUNTRIES = "override_" + SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES
        HOLIDAY_LOOKUP_COUNTRIES_AUTO = ("Override")

    class OverrideSilverkiteConstant(SilverkiteConstant):
        @overrides
        def get_silverkite_column(self) -> Type[OverrideSilverkiteColumn]:
            return OverrideSilverkiteColumn

        @overrides
        def get_silverkite_holiday(self) -> Type[OverrideSilverkiteHoliday]:
            return OverrideSilverkiteHoliday

    silverkite = SimpleSilverkiteForecast(constants=OverrideSilverkiteConstant())

    assert silverkite._silverkite_column is OverrideSilverkiteColumn
    assert silverkite._silverkite_column.COLS_HOUR_OF_WEEK == "override_hour_of_week"
    assert silverkite._silverkite_column.COLS_WEEKEND_SEAS == "override_is_weekend:daily_seas"

    assert silverkite._silverkite_holiday is OverrideSilverkiteHoliday
    assert silverkite._silverkite_holiday.ALL_HOLIDAYS_IN_COUNTRIES == "override_ALL_HOLIDAYS_IN_COUNTRIES"
    assert silverkite._silverkite_holiday.HOLIDAY_LOOKUP_COUNTRIES_AUTO == ("Override")
