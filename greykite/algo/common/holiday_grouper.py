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
"""Automatically scores and groups holidays of similar effects."""

from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

from greykite.algo.common.holiday_inferrer import HolidayInferrer
from greykite.algo.common.holiday_utils import HOLIDAY_DATE_COL
from greykite.algo.common.holiday_utils import HOLIDAY_NAME_COL
from greykite.algo.common.holiday_utils import get_dow_grouped_suffix
from greykite.algo.common.holiday_utils import get_weekday_weekend_suffix
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.viz.timeseries_plotting import plot_multivariate


class HolidayGrouper:
    """This module estimates the impact of holidays and their neighboring days
    given a raw holiday dataframe ``holiday_df``, and a time series containing
    the observed values to construct the baselines.
    It groups events with similar effects to several groups using kernel density estimation (KDE)
    and generates the grouped events in a dictionary of dataframes that is recognizable by
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast`.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Input time series that contains ``time_col`` and ``value_col``.
        The values will be used to construct baselines to estimate the holiday impact.
    time_col : `str`
        Name of the time column in ``df``.
    value_col : `str`
        Name of the value column in ``df``.
    holiday_df : `pandas.DataFrame`
        Input holiday dataframe that contains the dates and names of the holidays.
    holiday_date_col : `str`
        Name of the holiday date column in ``holiday_df``.
    holiday_name_col : `str`
        Name of the holiday name column in ``holiday_df``.
    holiday_impact_pre_num_days: `int`, default 0
        Default number of days before the holiday that will be modeled for holiday effect if the given holiday
        is not specified in ``holiday_impact_dict``.
    holiday_impact_post_num_days: `int`, default 0
        Default number of days after the holiday that will be modeled for holiday effect if the given holiday
        is not specified in ``holiday_impact_dict``.
    holiday_impact_dict : `Dict` [`str`, Any] or None, default None
        A dictionary containing the neighboring impacting days of a certain holiday. This overrides the
        default ``pre_num`` and ``post_num`` for each holiday specified here.
        The key is the name of the holiday matching those in the provided ``holiday_df``.
        The value is a tuple of two values indicating the number of neighboring days
        before and after the holiday. For example, a valid dictionary may look like:

            .. code-block:: python

                holiday_impact_dict = {
                    "Christmas Day": [3, 3],
                    "Memorial Day": [0, 0]
                }

    get_suffix_func : Callable or `str` or None, default "wd_we"
        A function that generates a suffix (usually a time feature e.g. "_WD" for weekday,
        "_WE" for weekend) given an input date.
        This can be used to estimate the interaction between floating holidays
        and on which day they are getting observed.
        We currently support two defaults:

            - "wd_we" to generate suffixes based on whether the day falls on weekday or weekend.
            - "dow_grouped" to generate three categories: ["_WD", "_Sat", "_Sun"].

        If None, no suffix is added.

    Attributes
    ----------
    expanded_holiday_df : `pandas.DataFrame`
        An expansion of ``holiday_df`` after adding the neighboring dates provided in
        ``holiday_impact_dict`` and the suffix generated by ``get_suffix_func``.
        For example, if ``"Christmas Day": [3, 3]`` and "wd_we" are used, events
        such as "Christmas Day_WD_plus_1_WE" or "Christmas Day_WD_minus_3_WD" will be generated
        for a Christmas that falls on Friday.
    baseline_offsets : `Tuple`[`int`] or None
        The offsets in days to calculate baselines for a given holiday.
        By default, the same days of the week before and after are used.
    use_relative_score : `bool` or None
        Whether to use relative or absolute score when estimating the holiday impact.
    clustering_method : `str` or None
        Clustering method used to group the holidays.
        Since we are doing 1-D clustering, current supported methods include
        (1) "kde" for kernel density estimation, and (2) "kmeans" for k-means clustering.
    bandwidth : `float` or None
        The bandwidth used in the kernel density estimation.
        Higher bandwidth results in less clusters.
        If None, it is automatically inferred with the ``bandwidth_multiplier`` factor.
    bandwidth_multiplier : `float` or None
        Multiplier to be multiplied to the kernel density estimation's default parameter calculated from
        `here<https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator>_`.
        This multiplier has been found useful in adjusting the default bandwidth parameter in many cases.
        Only used when ``bandwidth`` is not specified.
    kde : `KernelDensity` or None
        The `KernelDensity` object if ``clustering_method == "kde"``.
    n_clusters : `int` or None
        Number of clusters in the k-means algorithm.
    kmeans : `KMeans` or None
        The `KMeans` object if ``clustering_method == "kmeans"``.
    include_diagnostics : `bool` or None
        Whether to include ``kmeans_diagnostics`` and ``kmeans_plot`` in the output ``result_dict``.
    result_dict : `Dict`[`str`, Any] or None
        A dictionary that stores the scores and clustering results, with the following keys.

            - "holiday_inferrer": the `~greykite.algo.common.holiday_inferrer.HolidayInferrer`
                instance used for calculating the scores.
            - "score_result_original": a dictionary with keys being the names of all holiday events
                after expansion (i.e. the keys in ``expanded_holiday_df``), values being a list of scores
                of all dates corresponding to this event.
            - "score_result_avg_original": a dictionary with the same key as in
                ``result_dict["score_result_original"]``.
                But the values are the average scores of each event across all occurrences.
            - "score_result": same as ``result_dict["score_result_original"]``, but after removing
                holidays with inconsistent / negligible scores.
            - "score_result_avg": same as ``result_dict["score_result_original"]``, but after removing
                holidays with inconsistent / negligible scores.
            - "daily_event_df_dict_with_score": a dictionary of dataframes.
                Key is the group name ``"holiday_group_{k}"``.
                Value is a dataframe of all holiday events in this group, containing 4 columns:
                "date" (``EVENT_DF_DATE_COL``), "event_name" (``EVENT_DF_LABEL_COL``), "original_name", "avg_score".
            - "daily_event_df_dict": a dictionary of dataframes that is ready to use in `SilverkiteForecast`.
                Contains 2 keys: ``EVENT_DF_DATE_COL`` and ``EVENT_DF_LABEL_COL``.
            - "kde_cutoffs": a list of `float`, the cutoffs returned by the kernel density clustering.
            - "kde_res": a dataframe that contains "score" and "density" from the kernel density estimation.
            - "kde_plot": a plot of the kernel density estimation.
            - "kmeans_diagnostics": a dataframe containing metrics for different number of clusters.
                Columns are:

                    - "k": number of clusters;
                    - "wsse": within-cluster sum of squared error (lower is better);
                    - "sil_score": Silhouette coefficient, a value between [-1, 1] that describes
                        the separation of clusters (higher is better).

                Only generated when ``include_diagnostics`` is True. See `group_holidays` for details.
            - "kmeans_plot": a plot visualizing how the diagnostic metrics change over K.
                Only generated when ``include_diagnostics`` is True. See `group_holidays` for details.
    """
    def __init__(
            self,
            df: pd.DataFrame,
            time_col: str,
            value_col: str,
            holiday_df: pd.DataFrame,
            holiday_date_col: str,
            holiday_name_col: str,
            holiday_impact_pre_num_days: int = 0,
            holiday_impact_post_num_days: int = 0,
            holiday_impact_dict: Optional[Dict[str, Tuple[int, int]]] = None,
            get_suffix_func: Optional[Union[Callable, str]] = "wd_we"):
        self.df = df.copy()
        self.time_col = time_col
        self.value_col = value_col
        self.holiday_df = holiday_df.copy()
        self.holiday_date_col = holiday_date_col
        self.holiday_name_col = holiday_name_col
        self.holiday_impact_pre_num_days = holiday_impact_pre_num_days
        self.holiday_impact_post_num_days = holiday_impact_post_num_days
        if holiday_impact_dict is None:
            holiday_impact_dict = {}
        self.holiday_impact_dict = holiday_impact_dict.copy()
        self.get_suffix_func = get_suffix_func

        # Derived attributes.
        # Casts time columns to `datetime`.
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        self.holiday_df[holiday_date_col] = pd.to_datetime(self.holiday_df[holiday_date_col])
        # Creates `HOLIDAY_DATE_COL` and `HOLIDAY_NAME_COL` (if not exists) to be recognized by `HolidayInferrer`.
        self.holiday_df[HOLIDAY_DATE_COL] = self.holiday_df[self.holiday_date_col]
        self.holiday_df[HOLIDAY_NAME_COL] = self.holiday_df[self.holiday_name_col]

        # Other attributes that are not needed for initialization.
        self.baseline_offsets: Optional[Tuple[int, int]] = None
        self.use_relative_score: Optional[bool] = None
        self.clustering_method: Optional[str] = None
        self.bandwidth: Optional[float] = None
        self.bandwidth_multiplier: Optional[float] = None
        self.kde: Optional[KernelDensity] = None
        self.n_clusters: Optional[int] = None
        self.kmeans: Optional[KMeans] = None
        self.include_diagnostics: Optional[bool] = None
        # Result dictionary will be populated after the scoring and grouping functions are run.
        self.result_dict: Optional[Dict[str, Any]] = None

        # Expands `holiday_df` to include neighboring days
        # and suffixes (e.g. "_WD" for weekdays, "_Sat" for Saturdays).
        self.expanded_holiday_df = self.expand_holiday_df_with_suffix(
            holiday_df=self.holiday_df,
            holiday_date_col=HOLIDAY_DATE_COL,
            holiday_name_col=HOLIDAY_NAME_COL,
            holiday_impact_pre_num_days=self.holiday_impact_pre_num_days,
            holiday_impact_post_num_days=self.holiday_impact_post_num_days,
            holiday_impact_dict=self.holiday_impact_dict,
            get_suffix_func=self.get_suffix_func
        )

    def group_holidays(
            self,
            baseline_offsets: Tuple[int, int] = (-7, 7),
            use_relative_score: bool = True,
            min_n_days: int = 1,
            min_same_sign_ratio: float = 0.66,
            min_abs_avg_score: float = 0.03,
            clustering_method: str = "kmeans",
            bandwidth: Optional[float] = None,
            bandwidth_multiplier: Optional[float] = 0.2,
            n_clusters: Optional[int] = 5,
            include_diagnostics: bool = False) -> None:
        """Estimates the impact of holidays and their neighboring days and
        groups events with similar effects to several groups using kernel density estimation (KDE).
        Then generates the grouped events and stores the results in ``self.result_dict``.

        Parameters
        ----------
        baseline_offsets : `Tuple`[`int`], default (-7, 7)
            The offsets in days to calculate baselines for a given holiday.
            By default, the same days of the week before and after are used.
        use_relative_score : `bool`, default True
            Whether to use relative or absolute score when estimating the holiday impact.
        min_n_days : `int`, default 1
            Minimal number of occurrences for a holiday event to be kept before grouping.
        min_same_sign_ratio : `float`, default 0.66
            Threshold of the ratio of the same-sign scores for an event's occurrences.
            For example, if an event has two occurrences, they both need to have positive or negative
            scores for the ratio to achieve 0.66.
            Similarly, if an event has 3 occurrences, at least 2 of them must have the same directional impact.
            This parameter is intended to rule out holidays that have indefinite effects.
        min_abs_avg_score : `float`, default 0.03
            The minimal average score of an event (across all its occurrences) to be kept
            before grouping.
            When ``use_relative_score = True``, 0.03 means the effect must be greater than 3%.
        clustering_method : `str`, default "kmeans"
            Clustering method used to group the holidays.
            Since we are doing 1-D clustering, current supported methods include
            (1) "kde" for kernel density estimation, and (2) "kmeans" for k-means clustering.
        bandwidth : `float` or None, default None
            The bandwidth used in the kernel density estimation.
            Higher bandwidth results in less clusters.
            If None, it is automatically inferred with the ``bandwidth_multiplier`` factor.
            Only used when ``clustering_method == "kde"``.
        bandwidth_multiplier : `float` or None, default 0.2
            Multiplier to be multiplied to the kernel density estimation's default parameter calculated from
            `here<https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator>_`.
            This multiplier has been found useful in adjusting the default bandwidth parameter in many cases.
            Only used when ``bandwidth`` is not specified and ``clustering_method == "kde"``.
        n_clusters : `int` or None, default 5
            Number of clusters in the k-means algorithm.
            Only used when ``clustering_method == "kmeans"``.
        include_diagnostics : `bool`, default False
            Whether to include ``kmeans_diagnostics`` and ``kmeans_plot`` in the output ``result_dict``.

        Returns
        -------
            Saves the results in the ``result_dict`` attribute.
        """
        # Parameters should have already been set during initialization.
        # If new values are provided, they will override the original values.
        self.baseline_offsets = baseline_offsets
        self.use_relative_score = use_relative_score
        self.clustering_method = clustering_method

        # Runs baselines to get scores on holiday events.
        self.result_dict = self.get_holiday_scores(
            baseline_offsets=baseline_offsets,
            use_relative_score=use_relative_score,
            min_n_days=min_n_days,
            min_same_sign_ratio=min_same_sign_ratio,
            min_abs_avg_score=min_abs_avg_score
        )
        # Extracts results and prepares data for kernel density estimation.
        score_result_avg = self.result_dict["score_result_avg"]
        scores_df_original = (
            pd.DataFrame(score_result_avg, index=["avg_score"])
            .transpose()
            .reset_index(drop=False)
            .rename(columns={"index": "event_name"})
        )
        # In rare cases some holiday events may fall on exactly the same day, hence the same score.
        # We drop duplicated average scores before clustering, for example,
        # Halloween (10/31) plus 1 always has the same scores as All Saints Day (11/1).
        scores_df = scores_df_original.drop_duplicates("avg_score").sort_values("avg_score").reset_index(drop=True)
        scores_x = np.array(scores_df["avg_score"]).reshape(-1, 1)

        # The following parameters are set to None unless the clustering method is called to populate them.
        kde_res = None
        kde_plot = None
        kde_cutoffs = None
        kmeans_diagnostics = None
        kmeans_plot = None

        if clustering_method.lower() == "kde":
            if bandwidth is None and bandwidth_multiplier is None:
                raise ValueError(f"At least one of `bandwidth` or `bandwidth_multiplier` must be provided!")
            if bandwidth is None:
                # Automatically infers the best `bandwidth`.
                std = scores_x.std()
                iqr = np.percentile(scores_x, 75) - np.percentile(scores_x, 25)
                sigma = min(std, iqr / 1.34)
                bandwidth = 0.9 * sigma * (len(scores_x) ** (-1 / 5)) * bandwidth_multiplier

            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(scores_x)
            y = np.exp(kde.score_samples(scores_x))
            kde_res = pd.DataFrame({"score": scores_df["avg_score"], "density": y.tolist()})
            kde_res = kde_res.sort_values("score").reset_index(drop=True)
            kde_plot = plot_multivariate(
                df=kde_res,
                x_col="score",
                title="Kernel density of the holiday scores",
                xlabel="Holiday impact",
                ylabel="Kernel density"
            )

            # Performs holiday clustering based on the kernel densities.
            scores = kde_res["score"].to_list()
            densities = kde_res["density"].to_list()
            # Find the cutoffs such that scores <= each cutoff are grouped together.
            kde_cutoffs = []
            for i, x in enumerate(densities):
                if 0 < i < len(densities) - 1 and x < densities[i - 1] and x < densities[i + 1]:
                    kde_cutoffs.append(scores[i])

            # The group around 0 may contain mixed signs, hence we manually add 0 as a cutoff.
            # This might introduce an extra group with no scores - we will remove it later.
            # Note that there are no scores smaller than `min_abs_avg_score`.
            kde_cutoffs = sorted(kde_cutoffs + [0])

            # Constructs `daily_event_df_dict` for all groups.
            daily_event_df_dict_raw = {
                f"holiday_group_{i}": pd.DataFrame({
                    EVENT_DF_DATE_COL: [],
                    EVENT_DF_LABEL_COL: [],
                    "original_name": [],
                    "avg_score": []
                }) for i in range(len(kde_cutoffs) + 1)}

            for key, value in score_result_avg.items():
                # Gets group assignment.
                for i, cutoff in enumerate(sorted(kde_cutoffs)):
                    if value <= cutoff:
                        break
                else:  # If `value` > the largest cutoff, assigns it to the last group.
                    i += 1
                # Now, `i` is the group this holiday belongs to.

                # Gets the dates for each event.
                # Since holiday inferrer automatically adds "+0" to all events, here we remove it.
                key_without_plus_minus = "_".join(key.split("_")[:-1])
                idx = self.expanded_holiday_df[HOLIDAY_NAME_COL] == key_without_plus_minus
                dates = self.expanded_holiday_df.loc[idx, HOLIDAY_DATE_COL]

                # Creates `event_df`.
                group_name = f"holiday_group_{i}"
                event_df = pd.DataFrame({
                    EVENT_DF_DATE_COL: dates,
                    EVENT_DF_LABEL_COL: group_name,
                    "original_name": key_without_plus_minus,
                    "avg_score": value
                })
                daily_event_df_dict_raw[group_name] = pd.concat(
                    [daily_event_df_dict_raw[group_name], event_df],
                    ignore_index=True)
            # Removes potential empty groups.
            daily_event_df_dict = {}
            new_idx = 0
            for i in range(len(daily_event_df_dict_raw)):
                event_df = daily_event_df_dict_raw[f"holiday_group_{i}"]
                if len(event_df) > 0:
                    daily_event_df_dict[f"holiday_group_{new_idx}"] = (
                        event_df
                        .sort_values(by=["avg_score", EVENT_DF_DATE_COL])
                        .reset_index(drop=True)
                    )
                    new_idx += 1
            del daily_event_df_dict_raw
            # Overrides the attributes in the end.
            self.bandwidth = bandwidth
            self.bandwidth_multiplier = bandwidth_multiplier
            self.kde = kde

        elif clustering_method.lower() == "kmeans":
            # Runs K-means ++ to generate group assignments.
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="k-means++").fit(scores_x)
            # Predicts on the original `scores_df` since two different events may fall on the
            # same dates where the score was calculated, but they may have different dates in the future.
            predicted_labels = kmeans.predict(np.array(scores_df_original["avg_score"]).reshape(-1, 1))
            scores_df_with_label = scores_df_original.copy()
            scores_df_with_label["labels"] = list(predicted_labels)
            # Since `predicted_labels` is not ordered, we sort the groups according to the average score.
            group_rank_df = scores_df_with_label.groupby(by="labels")["avg_score"].agg(np.nanmean).reset_index()
            group_rank_df["group_id"] = (
                group_rank_df["avg_score"]
                .rank(method="dense", ascending=True)
                .astype(int)
            ) - 1  # Group indices start from 0.
            # Merges back to the original scores dataframe.
            scores_df_with_label = scores_df_with_label.merge(
                group_rank_df[["labels", "group_id"]],
                on="labels", how="left")

            daily_event_df_dict = {
                f"holiday_group_{i}": pd.DataFrame({
                    EVENT_DF_DATE_COL: [],
                    EVENT_DF_LABEL_COL: [],
                    "original_name": [],
                    "avg_score": []
                }) for i in range(n_clusters)}
            for key, value in score_result_avg.items():
                # Gets group assignment.
                group_id = scores_df_with_label.loc[scores_df_with_label["event_name"] == key, "group_id"].values[0]
                group_name = f"holiday_group_{group_id}"

                # Gets the dates for each event.
                # Since holiday inferrer automatically adds "+0" to all events, here we remove it.
                key_without_plus_minus = "_".join(key.split("_")[:-1])
                idx = self.expanded_holiday_df[HOLIDAY_NAME_COL] == key_without_plus_minus
                dates = self.expanded_holiday_df.loc[idx, HOLIDAY_DATE_COL]

                # Creates `event_df`.
                event_df = pd.DataFrame({
                    EVENT_DF_DATE_COL: dates,
                    EVENT_DF_LABEL_COL: group_name,
                    "original_name": key_without_plus_minus,
                    "avg_score": value
                })
                daily_event_df_dict[group_name] = pd.concat([daily_event_df_dict[group_name], event_df])
            # Sorts the output dataframes.
            for i in range(len(daily_event_df_dict)):
                event_df = daily_event_df_dict[f"holiday_group_{i}"]
                daily_event_df_dict[f"holiday_group_{i}"] = (
                    event_df
                    .sort_values(by=["avg_score", EVENT_DF_DATE_COL])
                    .reset_index(drop=True)
                )

            # Generates diagnostics for K means to help choose the optimal K (`n_clusters`).
            if include_diagnostics:
                kmeans_diagnostics = {"k": [], "wsse": [], "sil_score": []}
                for candidate_k in range(2, min(len(scores_df) // 2, 20) + 1):
                    tmp_model = KMeans(n_clusters=candidate_k).fit(scores_x)
                    # Gets Silhouette score.
                    kmeans_diagnostics["k"].append(candidate_k)
                    kmeans_diagnostics["sil_score"].append(silhouette_score(
                        X=scores_x,
                        labels=tmp_model.labels_,
                        metric="euclidean"
                    ))
                    # Gets within-cluster sum of squared errors.
                    centroids = tmp_model.cluster_centers_
                    pred_clusters = tmp_model.predict(scores_x)
                    curr_sse = 0
                    for i in range(len(scores_x)):
                        curr_center = centroids[pred_clusters[i]]
                        curr_sse += (scores_x[i, 0] - curr_center[0]) ** 2
                    kmeans_diagnostics["wsse"].append(curr_sse)
                kmeans_diagnostics = pd.DataFrame(kmeans_diagnostics)
                kmeans_plot = plot_multivariate(
                    df=kmeans_diagnostics,
                    x_col="k",
                    xlabel="n_clusters",
                    title="K-means diagnostics<br>"
                          "(1) Within-cluster SSE: lower is better<br>"
                          "(2) Silhouette scores: higher is better"
                )

            # Overrides the attributes in the end.
            self.n_clusters = n_clusters
            self.kmeans = kmeans
            self.include_diagnostics = include_diagnostics
        else:
            raise NotImplementedError(f"`clustering_method` {clustering_method} is not supported! "
                                      f"Must be one of \"kde\" (kernel density estimation) or "
                                      f"\"kmeans\" (k-means).")

        # Cleans up and removes duplicate dates.
        new_daily_event_df_dict = {
            key: (df[[EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL]]
                  .drop_duplicates(EVENT_DF_DATE_COL)
                  .reset_index(drop=True)) for key, df in daily_event_df_dict.items()
        }
        self.result_dict.update({
            "daily_event_df_dict_with_score": daily_event_df_dict,
            "daily_event_df_dict": new_daily_event_df_dict,
            "kde_cutoffs": kde_cutoffs,
            "kde_res": kde_res,
            "kde_plot": kde_plot,
            "kmeans_diagnostics": kmeans_diagnostics,
            "kmeans_plot": kmeans_plot
        })

    def get_holiday_scores(
            self,
            baseline_offsets: Tuple[int, int] = (-7, 7),
            use_relative_score: bool = True,
            min_n_days: int = 1,
            min_same_sign_ratio: float = 0.66,
            min_abs_avg_score: float = 0.05) -> Dict[str, Any]:
        """Computes the score of all holiday events and their neighboring days
        in ``self.expanded_holiday_df``, by comparing their observed values with a baseline
        value that is an average of the values on the days specified in ``baseline_offsets``.
        If a baseline date falls on another holiday, the algorithm looks for the next
        value with the same step size as the given offset, up to 3 extra iterations.
        Please see more details in
        `~greykite.algo.common.holiday_inferrer.HolidayInferrer._get_scores_for_holidays`.
        An additional pruning step is done to remove holidays with inconsistent / negligible scores.
        Both the results before and after the pruning are returned.

        Parameters
        ----------
        baseline_offsets : `Tuple`[`int`], default (-7, 7)
            The offsets in days to calculate baselines for a given holiday.
            By default, the same days of the week before and after are used.
        use_relative_score : `bool`, default True
            Whether to use relative or absolute score when estimating the holiday impact.
        min_n_days : `int`, default 1
            Minimal number of occurrences for a holiday event to be kept before grouping.
        min_same_sign_ratio : `float`, default 0.66
            Threshold of the ratio of the same-sign scores for an event's occurrences.
            For example, if an event has two occurrences, they both need to have positive or negative
            scores for the ratio to achieve 0.66.
            Similarly, if an event has 3 occurrences, at least 2 of them must have the same directional impact.
            This parameter is intended to rule out holidays that have indefinite effects.
        min_abs_avg_score : `float`, default 0.05
            The minimal average score of an event (across all its occurrences) to be kept
            before grouping.
            When ``use_relative_score = True``, 0.05 means the effect must be greater than 5%.

        Returns
        -------
        result_dict : `Dict` [`str`, Any]
            A dictionary containing the scoring results.
            In particular the following keys are set: "holiday_inferrer", "score_result_original",
            "score_result_avg_original", "score_result", and "score_result_avg".
            Please refer to the docstring of the ``self.result_dict`` attribute of `HolidayGrouper`.
        """
        # Initializes `HolidayInferrer` and sets the parameters.
        hi = HolidayInferrer()
        hi.df = self.df.copy()
        # In `HolidayInferrer._get_scores_for_holidays`, `time_col` must be of `datetime.date` or `str` type.
        hi.df[self.time_col] = pd.to_datetime(hi.df[self.time_col]).dt.date
        hi.ts = set(hi.df[self.time_col])
        hi.time_col = self.time_col
        hi.value_col = self.value_col
        hi.baseline_offsets = baseline_offsets
        hi.use_relative_score = use_relative_score
        hi.pre_search_days = 0
        hi.post_search_days = 0
        hi.country_holiday_df = self.expanded_holiday_df.copy()
        hi.all_holiday_dates = self.expanded_holiday_df[HOLIDAY_DATE_COL].tolist()
        hi.holidays = self.expanded_holiday_df[HOLIDAY_NAME_COL].unique().tolist()

        # Gets the scores for each single date in `self.expanded_holiday_df`.
        hi.score_result = hi._get_scores_for_holidays()
        # Gets the average scores over multiple occurrences for each holiday in `self.expanded_holiday_df`.
        hi.score_result_avg = hi._get_averaged_scores()

        # Prunes holiday that has too few datapoints or inconsistent / negligible scores.
        pruned_result = self._prune_holiday_by_score(
            score_result=hi.score_result,
            score_result_avg=hi.score_result_avg,
            min_n_days=min_n_days,
            min_same_sign_ratio=min_same_sign_ratio,
            min_abs_avg_score=min_abs_avg_score
        )

        # Returns result both before and after pruning.
        self.result_dict = {
            "holiday_inferrer": hi,
            "score_result_original": hi.score_result,
            "score_result_avg_original": hi.score_result_avg,
            "score_result": pruned_result["score_result"],
            "score_result_avg": pruned_result["score_result_avg"]
        }
        return self.result_dict

    def check_scores(
            self,
            holiday_name_pattern: str,
            show_pruned: bool = True) -> None:
        """Spot checks the score of certain holidays containing pattern ``holiday_name_pattern``.
        Prints out the dates, individual day scores of all occurrences,
        and the average scores of all matching holiday events.
        Note that it only checks the keys in ``self.expanded_holiday_df``,
        and it assumes `get_holiday_scores` is already run.

        Parameters
        ----------
        holiday_name_pattern : `str`
            Any substring of the holiday event names (``self.expanded_holiday_df[self.holiday_name_col]``).
        show_pruned : `bool`, default True
            Whether to show pruned holidays along with the remaining holidays.

        Returns
        -------
        Prints out the dates, individual day scores of all occurrences,
        and the average scores of all matching holiday events.
        """
        result_dict = self.result_dict
        if result_dict is None:
            return

        if show_pruned:
            score_result = result_dict["score_result_original"]
            score_result_avg = result_dict["score_result_avg_original"]
        else:
            score_result = result_dict["score_result"]
            score_result_avg = result_dict["score_result_avg"]
        res_dict = {}
        for key, value in score_result_avg.items():
            if holiday_name_pattern in key:
                # `HolidayInferrer` automatically adds "+" and "-" to the end, we remove them.
                key_without_plus_minus = "_".join(key.split("_")[:-1])
                dates = self.expanded_holiday_df.loc[
                    self.expanded_holiday_df[HOLIDAY_NAME_COL] == key_without_plus_minus,  # Uses exact matching.
                    HOLIDAY_DATE_COL
                ]
                dates = dates.dt.strftime("%Y-%m-%d").to_list()
                dates = [date for date in dates if date in self.df[self.time_col].dt.strftime("%Y-%m-%d").tolist()]
                # Prints out the date and impact of each day.
                impacts = score_result[key]
                print(f"{key_without_plus_minus}:\n"
                      f"Dates: {dates}\n"
                      f"Scores: {impacts}\n")
                # Extracts average score.
                res_dict[key_without_plus_minus] = value
        print("Average impact:")
        display(res_dict)

    def check_holiday_group(
            self,
            holiday_name_pattern: str = "",
            holiday_groups: Optional[Union[List[int], int]] = None) -> None:
        """Prints out the holiday groups that contain holidays matching ``holiday_name_pattern`` and their scores.
        The searching is limited to the given ``holiday_groups``.
        Note that it assumes `group_holidays` has already been run.

        Parameters
        ----------
        holiday_name_pattern : `str`
            Any substring of the holiday event names (``self.expanded_holiday_df[self.holiday_name_col]``).
        holiday_groups : `List`[`int`] or `int`, default None
            The indices of holiday groups that the searching is limited in.
            If None, all groups are available to search.

        Returns
        -------
        Prints out all qualifying holiday groups and their scores.
        """
        result_dict = self.result_dict
        if result_dict is None or "daily_event_df_dict_with_score" not in result_dict.keys():
            raise Exception(f"Method `group_holidays` must be run before using the `check_holiday_group` method.")

        daily_event_df_dict_with_score = result_dict["daily_event_df_dict_with_score"]
        if holiday_groups is None:
            holiday_groups = list(range(len(daily_event_df_dict_with_score)))
        if isinstance(holiday_groups, int):
            holiday_groups = [holiday_groups]

        is_found = False
        for group_id in holiday_groups:
            group_name = f"holiday_group_{group_id}"
            event_df = daily_event_df_dict_with_score.get(group_name)
            if event_df is not None and event_df["original_name"].str.contains(holiday_name_pattern).sum() > 0:
                is_found = True
                print(f"`{group_name}` contains events matching the provided pattern.\n"
                      f"This group includes {event_df['original_name'].nunique()} distinct events.\n")
                with pd.option_context("display.max_rows", None):
                    display(event_df)

        if not is_found:
            print(f"No matching records found given pattern {holiday_name_pattern.__repr__()} "
                  f"and holiday groups {holiday_groups}.")

    def _prune_holiday_by_score(
            self,
            score_result: Dict[str, List[float]],
            score_result_avg: Dict[str, float],
            min_n_days: int = 1,
            min_same_sign_ratio: float = 0.66,
            min_abs_avg_score: float = 0.05) -> Dict[str, Any]:
        """Removes events that have too few datapoints or inconsistent / negligible scores
        given ``score_result`` and ``score_result_avg``.

        Parameters
        ----------
        score_result : `Dict`[`str`, `List`[`float`]]
            A dictionary with keys being the names of all holiday events,
            values being a list of scores of all dates corresponding to this event.
        score_result_avg : `Dict`[`str`, `float`]
            A dictionary with the same key as in ``result_dict["score_result_original"]``.
            But the values are the average scores of each event across all occurrences.
        min_n_days : `int`, default 1
            Minimal number of occurrences for a holiday event to be kept before grouping.
        min_same_sign_ratio : `float`, default 0.66
            Threshold of the ratio of the same-sign scores for an event's occurrences.
            For example, if an event has two occurrences, they both need to have positive or negative
            scores for the ratio to achieve 0.66.
            Similarly, if an event has 3 occurrences, at least 2 of them must have the same directional impact.
            This parameter is intended to rule out holidays that have indefinite effects.
        min_abs_avg_score : `float`, default 0.05
            The minimal average score of an event (across all its occurrences) to be kept
            before grouping.
            When ``use_relative_score = True``, 0.05 means the effect must be greater than 5%.

        Returns
        -------
        result : `Dict`[`str`, Any]
            A dictionary with two keys: "score_result", "score_result_avg", values being
            the same dictionary as the input ``score_result``, ``score_result_avg``,
            but only with the remaining events after pruning.
        """
        res_score = {}
        res_score_avg = {}
        for key, value in score_result.items():
            # `key` is the name of the event.
            # `value` is a list of scores, we need to check the following.
            # First removes NAs before the following filtering.
            value_non_na = [val for val in value if not np.isnan(val)]
            # (1) It has minimum length `min_n_days`.
            if len(value_non_na) < min_n_days:
                continue

            # (2) The ratio of same-sign scores is at least `min_same_sign_ratio`.
            signs = [(score > 0) * 1 for score in value_non_na]
            n_pos, n_neg = sum(signs), len(signs) - sum(signs)
            if max(n_pos, n_neg) < min_same_sign_ratio * (n_pos + n_neg):
                continue

            # (3) The average score needs to meet `min_abs_avg_score` to be included.
            if abs(score_result_avg[key]) < min_abs_avg_score:
                continue

            # (4) The average score is not NaN.
            if np.isnan(score_result_avg[key]):
                continue

            res_score[key] = value
            res_score_avg[key] = score_result_avg[key]
        log_message(
            message=f"Holidays before pruning: {len(score_result)}; after pruning: {len(res_score)}.",
            level=LoggingLevelEnum.INFO
        )
        return {
            "score_result": res_score,
            "score_result_avg": res_score_avg
        }

    @staticmethod
    def expand_holiday_df_with_suffix(
            holiday_df: pd.DataFrame,
            holiday_date_col: str,
            holiday_name_col: str,
            holiday_impact_pre_num_days: int = 0,
            holiday_impact_post_num_days: int = 0,
            holiday_impact_dict: Optional[Dict[str, Tuple[int, int]]] = None,
            get_suffix_func: Optional[Union[Callable, str]] = "wd_we") -> pd.DataFrame:
        """Expands an input holiday dataframe ``holiday_df`` to include the neighboring days
        specified in ``holiday_impact_dict`` or through ``holiday_impact_pre_num_days`` and
        `holiday_impact_post_num_days`.
        Also adds suffixes generated by ``get_suffix_func`` to better model the effects
        of events falling on different days of week.

        Parameters
        ----------
        holiday_df : `pandas.DataFrame`
            Input holiday dataframe that contains the dates and names of the holidays.
        holiday_date_col : `str`
            Name of the holiday date column in ``holiday_df``.
        holiday_name_col : `str`
            Name of the holiday name column in ``holiday_df``.
        holiday_impact_pre_num_days: `int`, default 0
            Default number of days before the holiday that will be modeled for holiday effect if the given holiday
            is not specified in ``holiday_impact_dict``.
        holiday_impact_post_num_days: `int`, default 0
            Default number of days after the holiday that will be modeled for holiday effect if the given holiday
            is not specified in ``holiday_impact_dict``.
        holiday_impact_dict : `Dict` [`str`, Any] or None, default None
            A dictionary containing the neighboring impacting days of a certain holiday. This overrides the
            default ``pre_num`` and ``post_num`` for each holiday specified here.
            The key is the name of the holiday matching those in the provided ``holiday_df``.
            The value is a tuple of two values indicating the number of neighboring days
            before and after the holiday. For example, a valid dictionary may look like:

                .. code-block:: python

                    holiday_impact_dict = {
                        "Christmas Day": [3, 3],
                        "Memorial Day": [0, 0]
                    }

        get_suffix_func : Callable or `str` or None, default "wd_we"
            A function that generates a suffix (usually a time feature e.g. "_WD" for weekday,
            "_WE" for weekend) given an input date.
            This can be used to estimate the interaction between floating holidays
            and on which day they are getting observed.
            We currently support two defaults:

                - "wd_we" to generate suffixes based on whether the day falls on weekday or weekend.
                - "dow_grouped" to generate three categories: ["_WD", "_Sat", "_Sun"].

            If None, no suffix is added.

        Returns
        -------
        expanded_holiday_df : `pandas.DataFrame`
            An expansion of ``holiday_df`` after adding the neighboring dates provided in
            ``holiday_impact_dict`` and the suffix generated by ``get_suffix_func``.
            For example, if ``"Christmas Day": [3, 3]`` and "wd_we" are used, events
            such as "Christmas Day_WD_plus_1_WE" or "Christmas Day_WD_minus_3_WD" will be generated
            for a Christmas that falls on Friday.
        """
        error_message = f"`get_suffix_func` {get_suffix_func.__repr__()} is not supported! " \
                        f"Only supports None, Callable, \"dow_grouped\" or \"wd_we\"."
        if get_suffix_func is None:
            def get_suffix_func(x): return ""
        elif get_suffix_func == "wd_we":
            get_suffix_func = get_weekday_weekend_suffix
        elif get_suffix_func == "dow_grouped":
            get_suffix_func = get_dow_grouped_suffix
        elif isinstance(get_suffix_func, Callable):
            get_suffix_func = get_suffix_func
        else:
            raise NotImplementedError(error_message)

        if holiday_impact_dict is None:
            holiday_impact_dict = {}

        expanded_holiday_df = pd.DataFrame()
        for _, row in holiday_df.iterrows():
            # Handles different holidays differently.
            if row[holiday_name_col] in holiday_impact_dict.keys():
                pre_search_days, post_search_days = holiday_impact_dict[row[holiday_name_col]]
            else:
                pre_search_days, post_search_days = holiday_impact_pre_num_days, holiday_impact_post_num_days

            for i in range(-pre_search_days, post_search_days + 1):
                original_dow_flag = get_suffix_func(row[holiday_date_col])
                new_ts = (row[holiday_date_col] + timedelta(days=1) * i)
                new_dow_flag = get_suffix_func(new_ts)
                if i < 0:
                    suffix = f"{original_dow_flag}_minus_{-i}{new_dow_flag}"
                elif i > 0:
                    suffix = f"{original_dow_flag}_plus_{i}{new_dow_flag}"
                else:
                    suffix = f"{original_dow_flag}"
                new_row = {
                    holiday_date_col: new_ts,
                    holiday_name_col: f"{row[holiday_name_col]}{suffix}",
                }
                expanded_holiday_df = pd.concat([
                    expanded_holiday_df,
                    pd.DataFrame.from_dict({k: [v] for k, v in new_row.items()})
                ], ignore_index=True)

        return expanded_holiday_df.sort_values(holiday_date_col).reset_index(drop=True)
