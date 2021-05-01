from greykite.common.enums import SeasonalityEnum
from greykite.common.enums import SimpleTimeFrequencyEnum
from greykite.common.enums import TimeEnum


def test_time_enum():
    """Tests TimeEnum accessors"""
    assert TimeEnum.ONE_MONTH_IN_DAYS.value == 30
    assert TimeEnum.ONE_YEAR_IN_DAYS.value == 365
    assert TimeEnum.ONE_YEAR_IN_SECONDS.value == 31536000


def test_seasonality_enum():
    """Tests SeasonalityEnum accessors"""
    assert SeasonalityEnum.DAILY_SEASONALITY.value == "DAILY_SEASONALITY"
    assert SeasonalityEnum.WEEKLY_SEASONALITY.value == "WEEKLY_SEASONALITY"
    assert SeasonalityEnum.MONTHLY_SEASONALITY.value == "MONTHLY_SEASONALITY"
    assert SeasonalityEnum.QUARTERLY_SEASONALITY.value == "QUARTERLY_SEASONALITY"
    assert SeasonalityEnum.YEARLY_SEASONALITY.value == "YEARLY_SEASONALITY"


def test_simple_time_frequency_enum():
    """Tests SimpleTimeFrequencyEnum accessors"""
    assert SimpleTimeFrequencyEnum.MINUTE.value.default_horizon == 60
    assert SimpleTimeFrequencyEnum.MINUTE.value.seconds_per_observation == 60
    assert SimpleTimeFrequencyEnum.MINUTE.value.valid_seas == {
        SeasonalityEnum.DAILY_SEASONALITY.name,
        SeasonalityEnum.WEEKLY_SEASONALITY.name,
        SeasonalityEnum.MONTHLY_SEASONALITY.name,
        SeasonalityEnum.QUARTERLY_SEASONALITY.name,
        SeasonalityEnum.YEARLY_SEASONALITY.name}
    assert SimpleTimeFrequencyEnum.MONTH.value.valid_seas == {
        SeasonalityEnum.QUARTERLY_SEASONALITY.name,
        SeasonalityEnum.YEARLY_SEASONALITY.name}

    assert SimpleTimeFrequencyEnum.HOUR.value.default_horizon == 24
    assert SimpleTimeFrequencyEnum.DAY.value.default_horizon == 30
    assert SimpleTimeFrequencyEnum.WEEK.value.default_horizon == 12
    assert SimpleTimeFrequencyEnum.MONTH.value.default_horizon == 12
    assert SimpleTimeFrequencyEnum.QUARTER.value.default_horizon == 12
    assert SimpleTimeFrequencyEnum.YEAR.value.default_horizon == 2
    assert SimpleTimeFrequencyEnum.MULTIYEAR.value.default_horizon == 2
