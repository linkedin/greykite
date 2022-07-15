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
# original author: Kaixu Yang
"""Automatically infers significant holidays."""

import datetime
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from holidays_ext.get_holidays import get_holiday_df
from plotly import graph_objs as go
from plotly.subplots import make_subplots

from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import EVENT_INDICATOR
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message


HOLIDAY_POSITIVE_GROUP_NAME = "Holiday_positive_group"
HOLIDAY_NEGATIVE_GROUP_NAME = "Holiday_negative_group"
INFERRED_GROUPED_POSITIVE_HOLIDAYS_KEY = "together_holidays_positive"
INFERRED_GROUPED_NEGATIVE_HOLIDAYS_KEY = "together_holidays_negative"
INFERRED_INDEPENDENT_HOLIDAYS_KEY = "independent_holidays"


class HolidayInferrer:
    """Implements methods to automatically infer holiday effects.

    The class works for daily and sub-daily data.
    Sub-daily data is aggregated into daily data.
    It pulls holiday candidates from `pypi:holidays-ext`,
    and adds a pre-specified number of days before/after the holiday candidates
    as the whole holiday candidates pool.
    Every day in the candidate pool is compared with a pre-defined baseline imputed from surrounding days
    (e.g. the average of -7 and +7 days)
    and a score is generated to indicate deviation.
    The score is averaged if a holiday has multiple occurrences through the timeseries period.
    The holidays are ranked according to the magnitudes of the scores.
    Holidays are classified into:

        - model independently
        - model together
        - do not model

    according to their score magnitudes.
    For example, if the sum of the absolute scores is 1000,
    and the threshold for independent holidays is 0.8,
    the method keeps adding holidays to the independent modeling list
    from the largest magnitude until the sum reaches 1000 x 0.8 = 800.
    Then it continues to count the together modeling list.

    Attributes
    ----------
    baseline_offsets : `list` [`int`] or None
        The offsets in days to calculate baselines.
    post_search_days : `int` or None
        The number of days after each holiday to be counted as candidates.
    pre_search_days : `int` or None
        The number of days before each holiday to be counted as candidates.
    independent_holiday_thres : `float` or None
        A certain proportion of the total holiday effects that are allocated for holidays
        that are modeled independently. For example, 0.8 means the holidays that contribute
        to the first 80% of the holiday effects are modeled independently.
    together_holiday_thres : `float` or None
        A certain proportion of the total holiday effects that are allocated for holidays
        that are modeled together. For example, if ``independent_holiday_thres`` is 0.8 and
        ``together_holiday_thres`` is 0.9, then after the first 80% of the holiday effects
        are counted, the rest starts to be allocated for the holidays that are modeled together
        until the cum sum exceeds 0.9.
    extra_years : `int`, default 2
        Extra years after ``self.year_end`` to pull holidays in ``self.country_holiday_df``.
        This can be used to cover the forecast periods.
    df : `pandas.DataFrame` or None
        The timeseries after daily aggregation.
    time_col : `str` or None
        The column name for timestamps in ``df``.
    value_col : `str` or None
        The column name for values in ``df``.
    year_start : `int` or None
        The year of the first timeseries observation in ``df``.
    year_end : `int` or None
        The year of the last timeseries observation in ``df``.
    ts : `set` [`datetime`] or None
        The existing timestamps in ``df`` for fast look up.
    country_holiday_df : `pandas.DataFrame` or None
        The holidays between ``year_start`` and ``year_end``.
        This is the output from `pypi:holidays-ext`.
        Duplicates are dropped.
        Observed holidays are merged.
    holidays : `list` [`str`] or None
        A list of holidays in ``country_holiday_df``.
    score_result : `dict` [`str`, `list` [`float`]] or None
        The scores from comparing holidays and their baselines.
        The keys are holidays.
        The values are a list of the scores for each occurrence.
    score_result_avg : `dict` [`str`, `float`] or None
        The scores from ``score_result`` where the values are averaged.
    result : `dict` [`str`, any]
        The output of the model. Includes:

            - "scores": `dict` [`str`, `list` [`float`]]
                The ``score_result`` from ``self._get_scores_for_holidays``.
            - "country_holiday_df": `pandas.DataFrame`
                The ``country_holiday_df`` from ``pypi:holidays_ext``.
            - "independent_holidays": `list` [`tuple` [`str`, `str`]]
                The holidays to be modeled independently. Each item is in (country, holiday) format.
            - "together_holidays_positive": `list` [`tuple` [`str`, `str`]]
                The holidays with positive effects to be modeled together. Each item is in (country, holiday) format.
            - "together_holidays_negative": `list` [`tuple` [`str`, `str`]]
                The holidays with negative effects to be modeled together. Each item is in (country, holiday) format.
            - "fig": `plotly.graph_objs.Figure`
                The visualization if activated.
    """

    def __init__(self):
        # Parameters
        self.baseline_offsets: Optional[List[int]] = None
        self.post_search_days: Optional[int] = None
        self.pre_search_days: Optional[int] = None
        self.independent_holiday_thres: Optional[float] = None
        self.together_holiday_thres: Optional[float] = None
        self.extra_years: Optional[int] = None
        # Data set info
        self.df: Optional[pd.DataFrame] = None
        self.time_col: Optional[str] = None
        self.value_col: Optional[str] = None
        self.year_start: Optional[int] = None
        self.year_end: Optional[int] = None
        self.ts: Optional[Set[datetime.date]] = None
        # Derived results
        self.country_holiday_df: Optional[pd.DataFrame] = None
        self.holidays: Optional[List[str]] = None
        self.score_result: Optional[Dict[str, List[float]]] = None
        self.score_result_avg: Optional[Dict[str, float]] = None
        self.result: Optional[dict] = None

    def infer_holidays(
            self,
            df: pd.DataFrame,
            time_col: str = TIME_COL,
            value_col: str = VALUE_COL,
            countries: List[str] = ("US",),
            pre_search_days: int = 2,
            post_search_days: int = 2,
            baseline_offsets: List[int] = (-7, 7),
            plot: bool = False,
            independent_holiday_thres: float = 0.8,
            together_holiday_thres: float = 0.99,
            extra_years: int = 2) -> Optional[Dict[str, any]]:
        """Infers significant holidays and holiday configurations.

        The class works for daily and sub-daily data.
        Sub-daily data is aggregated into daily data.
        It pulls holiday candidates from `pypi:holidays-ext`,
        and adds a pre-specified number of days before/after the holiday candidates
        as the whole holiday candidates pool.
        Every day in the candidate pool is compared with a pre-defined baseline imputed from surrounding days
        (e.g. the average of -7 and +7 days)
        and a score is generated to indicate deviation.
        The score is averaged if a holiday has multiple occurrences through the timeseries period.
        The holidays are ranked according to the magnitudes of the scores.
        Holidays are classified into:

            - model independently
            - model together
            - do not model

        according to their score magnitudes.
        For example, if the sum of the absolute scores is 1000,
        and the threshold for independent holidays is 0.8,
        the method keeps adding holidays to the independent modeling list
        from the largest magnitude until the sum reaches 1000 x 0.8 = 800.
        Then it continues to count the together modeling list.

        Parameters
        ----------
        df : `pd.DataFrame`
            The input timeseries.
        time_col : `str`, default `TIME_COL`
            The column name for timestamps in ``df``.
        value_col : `str`, default `VALUE_COL`
            The column name for values in ``df``.
        countries : `list` [`str`], default ("UnitedStates",)
            A list of countries to look up holiday candidates.
            Available countries can be listed with
            ``holidays_ext.get_holidays.get_available_holiday_lookup_countries()``.
            Two-character country names are preferred.
        pre_search_days : `int`, default 2
            The number of days to include as holidays candidates before each holiday.
        post_search_days : `int`, default 2
            The number of days to include as holidays candidates after each holiday.
        baseline_offsets : `list` [`int`], default (-7, 7)
            The offsets in days as a baseline to compare with each holiday.
        plot : `bool`, default False
            Whether to generate visualization.
        independent_holiday_thres : `float`, default 0.8
            A certain proportion of the total holiday effects that are allocated for holidays
            that are modeled independently. For example, 0.8 means the holidays that contribute
            to the first 80% of the holiday effects are modeled independently.
        together_holiday_thres : `float`, default 0.99
            A certain proportion of the total holiday effects that are allocated for holidays
            that are modeled together. For example, if ``independent_holiday_thres`` is 0.8 and
            ``together_holiday_thres`` is 0.9, then after the first 80% of the holiday effects
            are counted, the rest starts to be allocated for the holidays that are modeled together
            until the cum sum exceeds 0.9.
        extra_years : `int`, default 2
            Extra years after ``self.year_end`` to pull holidays in ``self.country_holiday_df``.
            This can be used to cover the forecast periods.

        Returns
        -------
        result : `dict` [`str`, any] or None
            A dictionary with the following keys:

                - "scores": `dict` [`str`, `list` [`float`]]
                    The ``score_result`` from ``self._get_scores_for_holidays``.
                - "country_holiday_df": `pandas.DataFrame`
                    The ``country_holiday_df`` from ``pypi:holidays_ext``.
                - "independent_holidays": `list` [`tuple` [`str`, `str`]]
                    The holidays to be modeled independently. Each item is in (country, holiday) format.
                - "together_holidays_positive": `list` [`tuple` [`str`, `str`]]
                    The holidays with positive effects to be modeled together.
                    Each item is in (country, holiday) format.
                - "together_holidays_negative": `list` [`tuple` [`str`, `str`]]
                    The holidays with negative effects to be modeled together.
                    Each item is in (country, holiday) format.
                - "fig": `plotly.graph_objs.Figure`
                    The visualization if activated.

        """
        # Sets model parameters.
        self.baseline_offsets = baseline_offsets
        if post_search_days < 0 or pre_search_days < 0:
            raise ValueError("Both 'post_search_days' and 'pre_search_days' must be non-negative integers.")
        self.post_search_days = post_search_days
        self.pre_search_days = pre_search_days
        if not 0 <= independent_holiday_thres <= together_holiday_thres <= 1:
            raise ValueError("Both 'independent_holiday_thres' and 'together_holiday_thres' must be between "
                             "0 and 1 (inclusive).")
        self.independent_holiday_thres = independent_holiday_thres
        self.together_holiday_thres = together_holiday_thres
        if extra_years < 1:
            # At least 1 year for completeness.
            raise ValueError("The parameter 'extra_years' must be a positive integer.")
        self.extra_years = extra_years

        # Pre-processes data.
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        min_increment = min((df[time_col] - df[time_col].shift(1)).dropna())
        # Holidays is not activated for frequencies greater than daily.
        if min_increment > timedelta(days=1):
            log_message(
                message="Data frequency is greater than daily, "
                        "holiday inferring is skipped.",
                level=LoggingLevelEnum.INFO
            )
            return None

        # Holidays are daily events.
        # If data frequency is sub-daily,
        # we aggregate into daily.
        if min_increment < timedelta(days=1):
            df = df.resample("D", on=time_col).sum().reset_index(drop=False)
        df[time_col] = df[time_col].dt.date

        # From now on, data is in daily frequency.
        # Sets data attributes.
        self.year_start = df[time_col].min().year
        self.year_end = df[time_col].max().year
        self.ts = set(df[time_col])
        self.df = df
        self.time_col = time_col
        self.value_col = value_col

        # Gets holiday candidates.
        self.country_holiday_df, self.holidays = self._get_candidate_holidays(countries=countries)
        # Gets scores for holidays.
        self.score_result = self._get_scores_for_holidays()
        # Gets the average scores over multiple occurrences for each holiday.
        self.score_result_avg = self._get_averaged_scores()
        # Gets significant holidays.
        self.result = self._infer_holidays()
        # Makes plots if needed.
        if plot:
            self.result["fig"] = self._plot()
        else:
            self.result["fig"] = None

        return self.result

    def _infer_holidays(self) -> Dict[str, any]:
        """When the scores are computed,
        calculates the contributions and classifies holidays into:

            - model independently
            - model together
            - do not model

        Returns
        -------
        result : `dict` [`str`, any]
            A dictionary with the following keys:

                - "scores": `dict` [`str`, `list` [`float`]]
                    The ``score_result`` from ``self._get_scores_for_holidays``.
                - "country_holiday_df": `pandas.DataFrame`
                    The ``country_holiday_df`` from ``pypi:holidays_ext``.
                - "independent_holidays": `list` [`tuple` [`str`, `str`]]
                    The holidays to be modeled independently. Each item is in (country, holiday) format.
                - "together_holidays_positive": `list` [`tuple` [`str`, `str`]]
                    The holidays with positive effects to be modeled together.
                    Each item is in (country, holiday) format.
                - "together_holidays_negative": `list` [`tuple` [`str`, `str`]]
                    The holidays with negative effects to be modeled together.
                    Each item is in (country, holiday) format.
        """
        independent_holidays, together_holidays_positive, together_holidays_negative = self._get_significant_holidays()
        return {
            "scores": self.score_result,
            "country_holiday_df": self.country_holiday_df,
            INFERRED_INDEPENDENT_HOLIDAYS_KEY: independent_holidays,
            INFERRED_GROUPED_POSITIVE_HOLIDAYS_KEY: together_holidays_positive,
            INFERRED_GROUPED_NEGATIVE_HOLIDAYS_KEY: together_holidays_negative
        }

    def _get_candidate_holidays(
            self,
            countries: List[str]) -> (pd.DataFrame, List[str]):
        """Gets the candidate holidays from a list of countries.
        Uses `pypi:holidays-ext`.
        Duplicates are dropped.
        Observed holidays are renamed to original holidays
        and corresponding original holidays in the same years are removed.

        Parameters
        ----------
        countries : `list` [`str`]
            A list of countries to look up candidate holidays.

        Returns
        -------
        result : `tuple`
            Includes:

                country_holiday_df : `pandas.DataFrame`
                    The holidays between ``year_start`` and ``year_end``.
                    This is the output from `pypi:holidays-ext`.
                    Duplicates are dropped.
                    Observed holidays are merged.
                holidays : `list` [`str`]
                    A list of holidays in ``country_holiday_df``.
                    The holidays are in the format of "{country_name}_{holiday_name}".

        """
        country_holiday_df = get_holiday_df(
            country_list=countries,
            years=list(range(self.year_start, self.year_end + self.extra_years))
        )
        # Drops duplications.
        country_holiday_df.drop_duplicates(keep="first", subset=["ts"], inplace=True)

        # Handles observed holidays.
        # If observed holiday and original holiday are both listed in the same year,
        # the observed holiday will be renamed to the original holiday
        # and the original holiday in the same year will be removed.

        # Sub-df that contains observed holidays only.
        observed_df = country_holiday_df[country_holiday_df["holiday"].str[-10:] == "(Observed)"]
        # Row indices to rename.
        rows_to_rename = observed_df.index.tolist()
        # Date-holiday tuple to remove.
        # ":-11" truncates the " (Observed)" suffix.
        # This is used to identify rows to remove.
        date_holiday_to_remove = [(row[1]["ts"], row[1]["holiday"][:-11]) for row in observed_df.iterrows()]
        # Row indices to remove.
        # For each (date, holiday) tuple, look up the match in ``country_holiday_df`` and record the row indices.
        # The match happens when the holiday name matches and the time diff is at most 3 days.
        rows_to_remove = [idx for date, holiday in date_holiday_to_remove
                          for idx in country_holiday_df[
                              (abs((pd.DatetimeIndex(country_holiday_df["ts"]) - date).days) <= 3) &
                              (country_holiday_df["holiday"] == holiday)].index.tolist()]
        # Renames and removes.
        country_holiday_df.loc[rows_to_rename, "holiday"] = country_holiday_df.loc[
            rows_to_rename, "holiday"].str[:-11]
        country_holiday_df.loc[rows_to_rename, "country_holiday"] = country_holiday_df.loc[
            rows_to_rename, "country_holiday"].str[:-11]
        country_holiday_df.drop(
            rows_to_remove,
            axis=0,
            inplace=True
        )
        country_holiday_df.reset_index(drop=True, inplace=True)

        holidays = country_holiday_df["country_holiday"].unique().tolist()
        return country_holiday_df, holidays

    @staticmethod
    def _transform_country_holidays(
            country_holidays: List[Union[str, Tuple[str, str]]]) -> List[Union[Tuple[str, str], str]]:
        """Decouples a list of {country}_{holiday} names into a list of (country, holiday) tuple
        or the other way around, depending on the input type.

        Parameters
        ----------
        country_holidays : `list` [`str` or `tuple` [`str`, `str`]]
            One of:

                - A list of country-holiday strings of the format {country}_{holiday}.
                  The country part is not expected to have "_".
                - A list of (country, holiday) tuples.


        Returns
        -------
        country_holiday_list : `list` [`tuple` [`str`, `str`] or `str`]
            A list of (country, holiday) tuples or a list of {country}_{holiday} strings,
            depending on the input type.
        """
        country_holiday_list = []
        for country_holiday in country_holidays:
            if isinstance(country_holiday, str):
                split = country_holiday.split("_")
                country = split[0]
                holiday = "_".join(split[1:])
                country_holiday_list.append((country, holiday))
            elif isinstance(country_holiday, tuple) and len(country_holiday) == 2:
                country_holiday_item = f"{country_holiday[0]}_{country_holiday[1]}"
                country_holiday_list.append(country_holiday_item)
            else:
                raise ValueError("Every item in ``country_holidays`` must be a string or a length-2 tuple.")
        return country_holiday_list

    def _get_score_for_dates(
            self,
            event_dates: List[pd.Timestamp]) -> List[float]:
        """Gets the score for each day in ``event_dates``.
        The score is defined as the observation on the day minus the baseline,
        which is the average of the ``self.baseline_offsets`` offset observations.

        Parameters
        ----------
        event_dates : `list` [`pandas.Timestamp`]
            The timestamps for a single event.

        Returns
        -------
        scores : `list` [`float`]
            The scores for a list of occurrences of an event.
        """
        scores = []
        for date in event_dates:
            # Calculates the dates for baseline.
            baseline_dates = []
            for offset in self.baseline_offsets:
                new_date = date + timedelta(days=offset)
                counter = 1
                # If a baseline date falls on another holiday, it is moving further.
                # But the total iterations cannot exceed 3.
                while new_date in event_dates and counter < 3:
                    counter += 1
                    new_date += timedelta(days=offset)
                baseline_dates.append(new_date)
            # Calculates the average of the baseline observations.
            baseline = self.df[self.df[self.time_col].isin(baseline_dates)][self.value_col].mean()
            # Calculates the score for the current occurrence.
            score = self.df[self.df[self.time_col] == date][self.value_col].values[0] - baseline
            scores.append(score)
        return scores

    def _get_scores_for_holidays(self) -> Dict[str, List[float]]:
        """Calculates the scores for a list of events, each with multiple occurrences.

        Returns
        -------
        result : `dict` [`str`, `list` [`float`]]
            A dictionary with keys being the holiday names and values
            being the scores for all occurrences of the holiday.
        """
        result = {}
        for holiday in self.holidays:
            # Gets all occurrences of the holiday
            holiday_dates = self.country_holiday_df[
                self.country_holiday_df["country_holiday"] == holiday]["ts"].tolist()
            # Iterates over pre/post days to get the scores
            for i in range(-self.pre_search_days, self.post_search_days + 1):
                event_dates = [(date + timedelta(days=1) * i).date() for date in holiday_dates]
                event_dates = [date for date in event_dates if date in self.ts]
                score = self._get_score_for_dates(
                    event_dates=event_dates,
                )
                result[f"{holiday}_{'{0:+}'.format(i)}"] = score  # format is with +/- signs
        return result

    def _get_averaged_scores(self) -> Dict[str, float]:
        """Calculates the average score for each event date.

        Returns
        -------
        result : `dict` [`str`, `float`]
            A dictionary with keys being the holiday names and values
            being the average scores.
        """
        result = {}
        for holiday, score in self.score_result.items():
            result[holiday] = np.nanmean(score)
        return result

    def _get_significant_holidays(self) -> (List[str], List[str], List[str]):
        """Classifies holidays into model independently, model together
        and do not model according to their scores.

        Returns
        -------
        result : `tuple`
            A result tuple including:

                - "independent_holidays": `list` [`tuple` [`str`, `str`]]
                    The holidays to be modeled independently. Each item is in (country, holiday) format.
                - "together_holidays_positive": `list` [`tuple` [`str`, `str`]]
                    The holidays with positive effects to be modeled together.
                    Each item is in (country, holiday) format.
                - "together_holidays_negative": `list` [`tuple` [`str`, `str`]]
                    The holidays with negative effects to be modeled together.
                    Each item is in (country, holiday) format.

        """
        # Calculates the total holiday deviations.
        total_changes = np.nansum(np.abs(list(self.score_result_avg.values())))
        independent_holiday_thres = self.independent_holiday_thres * total_changes
        together_holiday_thres = self.together_holiday_thres * total_changes
        # Sorts the holidays by their magnitudes.
        ranked_effects = sorted(self.score_result_avg.items(), key=lambda x: abs(x[1]), reverse=True)
        # Iterates over the sorted holidays until it reaches the thresholds.
        cum_effect = 0  # cumulative holiday deviations so far
        idx = 0  # index for the current holiday
        independent_holidays = []  # stores holidays to be modeled independently
        together_holidays_positive = []  # stores holidays with positive effects to be modeled together
        together_holidays_negative = []  # stores holidays with negative effects to be modeled together
        # Starts adding independent holidays until threshold
        while cum_effect < independent_holiday_thres and idx < len(ranked_effects):
            if np.isfinite(ranked_effects[idx][1]):
                independent_holidays.append(ranked_effects[idx][0])
                cum_effect += abs(ranked_effects[idx][1])
            idx += 1
        # Starts adding together holidays until threshold
        while cum_effect < together_holiday_thres and idx < len(ranked_effects):
            if np.isfinite(ranked_effects[idx][1]):
                if ranked_effects[idx][1] > 0:
                    together_holidays_positive.append(ranked_effects[idx][0])
                elif ranked_effects[idx][1] < 0:
                    together_holidays_negative.append(ranked_effects[idx][0])
                cum_effect += abs(ranked_effects[idx][1])
            idx += 1
        return (self._transform_country_holidays(independent_holidays),
                self._transform_country_holidays(together_holidays_positive),
                self._transform_country_holidays(together_holidays_negative))

    def _plot(self) -> go.Figure:
        """Makes a plot that includes the following two subplots:

            - Bar chart for holiday effects grouped by holidays ordered by their holiday effects.
            - Bar chart for holiday effects and their classifications
              ranked by their effects.

        Returns
        -------
        fig : `plotly.graph_objs`
            The figure object.
        """
        # Makes the plot.
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                "Inferred holiday effects grouped by holiday",
                "Inferred holiday effects grouped by effects"
            ],
            vertical_spacing=0.4
        )
        # Adds the subplot: holiday effects grouped by holidays.
        # Gets all holidays and their scores.
        holidays = []
        scores = []
        for holiday, score in self.score_result_avg.items():
            holidays.append(holiday)
            scores.append(score)
        # Removes the pre/post numbers of days from the end of the holiday names.
        # This is used to make the plot grouped by holidays.
        holidays_without_plus_minus = list(set(["_".join(holiday.split("_")[:-1]) for holiday in holidays]))
        # Sorts holidays according to their effects.
        holidays_without_plus_minus = sorted(
            holidays_without_plus_minus,
            key=lambda x: abs(self.score_result_avg[f"{x}_+0"]),
            reverse=True)
        # Iterates over each holiday + i day to plot the bars.
        for i in range(-self.pre_search_days, self.post_search_days + 1):
            if i == 0:
                name = "holiday"
            elif abs(i) == 1:
                name = f"holiday {'{0:+}'.format(i)} day"
            else:
                name = f"holiday {'{0:+}'.format(i)} days"
            # Gets the list of holiday names with the current +/- day.
            holidays_with_plus_minus = [key + f"_{'{0:+}'.format(i)}" for key in holidays_without_plus_minus]
            # Gets the corresponding scores for the current +/- day.
            current_values = [scores[idx] for idx in [
                holidays.index(holiday) for holiday in holidays_with_plus_minus]]
            # Adds to the plot.
            fig.add_trace(
                go.Bar(
                    # Truncates the text for better view.
                    x=[holiday[:30] for holiday in holidays_without_plus_minus],
                    y=current_values,
                    name=name,
                    legendgroup=1
                ),
                row=1,
                col=1
            )
        # Adds the subplot: holiday effects grouped by effects.
        # Sorts holidays by their effect magnitude.
        ranked_holidays, ranked_scores = list(zip(
            *sorted(self.score_result_avg.items(), key=lambda x: abs(x[1]), reverse=True)))
        # Adds to the plot.
        fig.add_trace(
            go.Bar(
                # Truncates the text for better view.
                x=["_".join(holiday.split("_")[:-1])[:30] + holiday.split("_")[-1] for holiday in ranked_holidays],
                y=ranked_scores,
                legendgroup=2,
                name="holidays"
            ),
            row=2,
            col=1
        )
        # Adds vertical regions to indicate the classification of the holidays.
        start = -0.5  # start of bar chart x axis
        independent_holidays_end = start + len(self.result[INFERRED_INDEPENDENT_HOLIDAYS_KEY])
        together_holiday_end = (independent_holidays_end + len(self.result[INFERRED_GROUPED_POSITIVE_HOLIDAYS_KEY])
                                + len(self.result[INFERRED_GROUPED_NEGATIVE_HOLIDAYS_KEY]))
        end = start + len(holidays)
        fig.add_vrect(
            x0=start,
            x1=independent_holidays_end,
            annotation_text="model independently",
            annotation_position="top left",
            opacity=0.15,
            fillcolor="green",
            line_width=0,
            row=2,
            col=1
        )
        fig.add_vrect(
            x0=independent_holidays_end,
            x1=together_holiday_end,
            annotation_text="model together",
            annotation_position="top left",
            opacity=0.15,
            fillcolor="purple",
            line_width=0,
            row=2,
            col=1
        )
        fig.add_vrect(
            x0=together_holiday_end,
            x1=end,
            annotation_text="do not model",
            annotation_position="top left",
            opacity=0.15,
            fillcolor="yellow",
            line_width=0,
            row=2,
            col=1
        )
        fig.add_vline(
            x=independent_holidays_end,
            line=dict(color="black"),
            line_width=1,
            row=2,
            col=1
        )
        fig.add_vline(
            x=together_holiday_end,
            line=dict(color="black"),
            line_width=1,
            row=2,
            col=1
        )
        # Adjusts layouts.
        fig.update_layout(
            height=1000,
            title="Inferred holiday effects",
            legend_tracegroupgap=360,
        )
        fig.update_xaxes(
            tickangle=90,
            title="Holidays",
            row=1,
            col=1
        )
        fig.update_yaxes(
            title="Effect",
            row=1,
            col=1
        )
        fig.update_xaxes(
            tickangle=90,
            title="Holidays",
            row=2,
            col=1
        )
        fig.update_yaxes(
            title="Effect",
            row=2,
            col=1
        )
        return fig

    def _get_event_df_for_single_event(
            self,
            holiday: Tuple[str, str],
            country_holiday_df: pd.DataFrame) -> pd.DataFrame:
        """Gets the event df for a single holiday.
        An event df has the format:

        pd.DataFrame({
            "date": ["2020-09-01", "2021-09-01"],
            "event_name": "is_event"
        })

        Parameters
        ----------
        holiday : `tuple` [`str`, `str`]
            A tuple of length 2.
            The first element is the country name.
            The second element has the format of f"{holiday}_{x}",
            where "x" is a signed integer acting as a neighboring operator.
            For example, ("US", "Christmas Day_+1") means the day after
            every US's Christmas Day.
            This is consistent with the output from ``self.infer_holidays``.
        country_holiday_df : `pandas.DataFrame`
            The dataframe that contains the country/holiday/dates information
            for holidays. Must cover the periods need in training/forecasting
            for all holidays.
            This has the same format as ``self.country_holiday_df``.

        Returns
        -------
        event_df : `pandas.DataFrame`
            The event df for a single holiday in the format of

            pd.DataFrame({
                "date": ["2020-12-24", "2021-12-24"],
                "event_name": "US_Christmas Day_minus_1"
            })
        """
        # Splits holiday into country name, holiday name and neighboring offset days.
        country = holiday[0]
        holiday_split = holiday[1].split("_")
        holiday_name = "_".join(holiday_split[:-1])
        neighboring_offset = int(holiday_split[-1])
        # Gets holiday dates from ``country_holiday_df``.
        holiday_dates = country_holiday_df[
            (country_holiday_df["country"] == country) &
            (country_holiday_df["holiday"] == holiday_name)]["ts"].tolist()
        holiday_dates = [date + timedelta(days=neighboring_offset) for date in holiday_dates]
        # Constructs the event df.
        # The holiday name matches the column names
        # constructed from `SimpleSilverkiteForecast`'s holiday generating functions.
        if neighboring_offset < 0:
            holiday_name_adj = f"{country}_{holiday_name}_minus_{abs(neighboring_offset)}"
        elif neighboring_offset == 0:
            holiday_name_adj = f"{country}_{holiday_name}"
        else:
            holiday_name_adj = f"{country}_{holiday_name}_plus_{neighboring_offset}"
        holiday_name_adj = holiday_name_adj.replace("'", "")  # Single quote conflicts patsy formula.
        event_df = pd.DataFrame({
            EVENT_DF_DATE_COL: holiday_dates,
            EVENT_DF_LABEL_COL: holiday_name_adj
        })

        return event_df

    def generate_daily_event_dict(
            self,
            country_holiday_df: Optional[pd.DataFrame] = None,
            holiday_result: Optional[Dict[str, List[Tuple[str, str]]]] = None) -> Dict[str, pd.DataFrame]:
        """Generates daily event dict for all holidays inferred.
        The daily event dict will contain:

            - Single events for every holiday or holiday neighboring day
              that is to be modeled independently.
            - A single event for all holiday or holiday neighboring days
              with positive effects that are modeled together.
            - A single event for all holiday or holiday neighboring days
              with negative effects that are modeled together.

        Parameters
        ----------
        country_holiday_df : `pandas.DataFrame` or None, default None
            The dataframe that contains the country/holiday/dates information
            for holidays. Must cover the periods need in training/forecasting
            for all holidays.
            This has the same format as ``self.country_holiday_df``.
            If None, it pulls from ``self.country_holiday_df``.
        holiday_result : `dict` [`str`, `list` [`tuple` [`str`, `str`]]] or None, default None
            A dictionary with the following keys:

                - INFERRED_INDEPENDENT_HOLIDAYS_KEY
                - INFERRED_GROUPED_POSITIVE_HOLIDAYS_KEY
                - INFERRED_GROUPED_NEGATIVE_HOLIDAYS_KEY

            Each key's value is a list of length-2 tuples of the format (country, holiday).
            This format is the output of ``self.infer_holidays``.
            If None, it pulls from ``self.result``.

        Returns
        -------
        daily_event_dict : `dict`
            The daily event dict that is consumable by
            `~greykite.algo.forecast.silverkite.forecast_simple_silverkite.SimpleSilverkiteForecast` or
            `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast`.
            The keys are the event names.
            The values are dataframes with the event dates.
        """
        daily_event_dict = {}
        # Gets default parameters.
        if country_holiday_df is None:
            country_holiday_df = self.country_holiday_df
        if holiday_result is None:
            holiday_result = self.result
        if country_holiday_df is None or holiday_result is None:
            raise ValueError("Both 'country_holiday_df' and 'holidays' must be given. "
                             "Alternatively, you can run 'infer_holidays' first and "
                             "they will be pulled automatically.")

        # Gets independent holidays.
        independent_holidays = holiday_result.get(INFERRED_INDEPENDENT_HOLIDAYS_KEY, [])
        for holiday in independent_holidays:
            event_df = self._get_event_df_for_single_event(
                holiday=holiday,
                country_holiday_df=country_holiday_df
            )
            if event_df.shape[0] > 0:
                event_name = event_df[EVENT_DF_LABEL_COL].iloc[0]
                daily_event_dict[event_name] = event_df

        # Gets positive together holidays.
        together_holidays_positive = holiday_result.get(INFERRED_GROUPED_POSITIVE_HOLIDAYS_KEY, [])
        event_df = pd.DataFrame()
        for holiday in together_holidays_positive:
            event_df_temp = self._get_event_df_for_single_event(
                holiday=holiday,
                country_holiday_df=country_holiday_df
            )
            event_df = pd.concat([event_df, event_df_temp], axis=0)
        if event_df.shape[0] > 0:
            event_df[EVENT_DF_LABEL_COL] = EVENT_INDICATOR
            daily_event_dict[HOLIDAY_POSITIVE_GROUP_NAME] = event_df.drop_duplicates(subset=[EVENT_DF_DATE_COL]).reset_index(drop=True)

        # Gets negative together holidays.
        together_holidays_negative = holiday_result.get(INFERRED_GROUPED_NEGATIVE_HOLIDAYS_KEY, [])
        event_df = pd.DataFrame()
        for holiday in together_holidays_negative:
            event_df_temp = self._get_event_df_for_single_event(
                holiday=holiday,
                country_holiday_df=country_holiday_df
            )
            event_df = pd.concat([event_df, event_df_temp], axis=0)
        if event_df.shape[0] > 0:
            event_df[EVENT_DF_LABEL_COL] = EVENT_INDICATOR
            daily_event_dict[HOLIDAY_NEGATIVE_GROUP_NAME] = event_df.drop_duplicates(subset=[EVENT_DF_DATE_COL]).reset_index(drop=True)

        return daily_event_dict
