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
# original author: Albert Chen
"""Functions to summarize the output of
`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
"""

import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from greykite.common.constants import FRACTION_OUTSIDE_TOLERANCE
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.python_utils import assert_equal
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.pipeline.utils import get_score_func_with_aggregation


def get_ranks_and_splits(
        grid_search,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        greater_is_better=False,
        combine_splits=True,
        decimals=None,
        warn_metric=True):
    """Extracts CV results from ``grid_search`` for the specified score function.
    Returns the correct ranks on the test set and a tuple of the scores across splits,
    for both test set and train set (if available).

    Notes
    -----
    While ``cv_results`` contains keys with the ranks, these ranks are inverted
    if lower values are better and the ``scoring`` function was initialized
    with ``greater_is_better=True`` to report metrics with their original sign.

    This function always returns the correct ranks, accounting for metric direction.

    Parameters
    ----------
    grid_search : `~sklearn.model_selection.RandomizedSearchCV`
        Grid search output (fitted RandomizedSearchCV object).
    score_func : `str` or callable, default ``EvaluationMetricEnum.MeanAbsolutePercentError.name``
        Score function to get the ranks for.
        If a callable, takes arrays ``y_true``, ``y_pred`` and returns a float.
        If a string, must be either a
        `~greykite.common.evaluation.EvaluationMetricEnum` member name
        or `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.

        Should be the same as what was passed to
        :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`,
        or `~greykite.framework.pipeline.pipeline.forecast_pipeline`,
        or `~greykite.framework.pipeline.utils.get_hyperparameter_searcher`.
    greater_is_better : `bool` or None, default False
        True if ``score_func`` is a score function, meaning higher is better,
        and False if it is a loss function, meaning lower is better.
        Must be provided if ``score_func`` is a callable (custom function).
        Ignored if ``score_func`` is a string, because the direction is known.

        Used in this function to rank values in the proper direction.

        Should be the same as what was passed to
        :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`,
        or `~greykite.framework.pipeline.pipeline.forecast_pipeline`,
        or `~greykite.framework.pipeline.utils.get_hyperparameter_searcher`.
    combine_splits : `bool`, default True
        Whether to report split scores as a tuple in a single column.
        If True, a single column is returned for all the splits
        of a given metric and train/test set.
        For example, "split_train_score" would contain the values
        (split1_train_score, split2_train_score, split3_train_score)
        as as tuple.
        If False, they are reported in their original columns.
    decimals : `int` or None, default None
        Number of decimal places to round to.
        If decimals is negative, it specifies the number of
        positions to the left of the decimal point.
        If None, does not round.
    warn_metric : `bool`, default True
        Whether to issue a warning if the requested metric is
        not found in the CV results.

    Returns
    -------
    ranks_and_splits : `dict`
        Ranks and split scores.
        Dictionary with the following keys:

            ``"short_name"`` : `int`
                Canonical short name for the ``score_func``.
            ``"ranks"`` : `numpy.array`
                Ranks of the test scores for the ``score_func``,
                where 1 is the best.
            ``"split_train"`` : `list` [`list` [`float`]]
                Train split scores. Outer list corresponds to the
                parameter setting; inner list contains the
                scores for that parameter setting across all splits.
            ``"split_test"`` : `list` [`list` [`float`]]
                Test split scores. Outer list corresponds to the
                parameter setting; inner list contains the
                scores for that parameter setting across all splits.
    """
    cv_results = grid_search.cv_results_
    _, greater_is_better, short_name = get_score_func_with_aggregation(
        score_func=score_func,  # string or callable
        greater_is_better=greater_is_better,
        # Dummy value, doesn't matter because we ignore the returned `score_func`
        relative_error_tolerance=0.01)

    # Warns if the metric is not available
    if f"mean_test_{short_name}" not in cv_results:
        if warn_metric:
            warnings.warn(f"Metric '{short_name}' is not available in the CV results.")
        return {
            "short_name": short_name,
            "ranks": None,
            "split_train": None,
            "split_test": None}

    # Computes the ranks, using the same tiebreaking method as in sklearn.
    scores = cv_results[f"mean_test_{short_name}"].copy()
    if greater_is_better:
        scores *= -1  # `rankdata` function ranks lowest values first
    ranks = np.asarray(rankdata(scores, method='min'), dtype=np.int32)

    # Computes split score columns.
    train_scores = None
    test_scores = None

    def round_as_list(split_scores, decimals=None):
        """Rounds `split_scores` to the specified
        `decimals` and returns the result as a list.

        Parameters
        ----------
        split_scores : `numpy.array`
             Split scores.
        decimals : `int` or None, default None
            Number of decimal places to round to.
            If decimals is negative, it specifies the number of
            positions to the left of the decimal point.
            If None, does not round.
        Returns
        -------
        split_scores_list : `list` [`float`]
            ``split_scores``, rounded according
            to ``decimals`` and returned as a list.
        """
        if decimals is not None:
            split_scores = split_scores.round(decimals)
        return split_scores.tolist()

    if combine_splits:
        # Each sublist contains the scores for split i
        # across all parameter settings.
        test_scores = [
            round_as_list(
                cv_results[f"split{i}_test_{short_name}"],
                decimals=decimals)
            for i in range(grid_search.n_splits_)]
        # Makes each sublist contain the scores for a particular
        # parameter setting across all splits.
        test_scores = list(zip(*test_scores))

        # Train scores
        if grid_search.return_train_score:
            train_scores = [
                round_as_list(
                    cv_results[f"split{i}_train_{short_name}"],
                    decimals=decimals)
                for i in range(grid_search.n_splits_)]
            train_scores = list(zip(*train_scores))

    ranks_and_splits = {
        "short_name": short_name,
        "ranks": ranks,
        "split_train": train_scores,
        "split_test": test_scores}
    return ranks_and_splits


def summarize_grid_search_results(
        grid_search,
        only_changing_params=True,
        combine_splits=True,
        decimals=None,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        score_func_greater_is_better=False,
        cv_report_metrics=CV_REPORT_METRICS_ALL,
        column_order=None):
    """Summarizes CV results for each grid search parameter combination.

    While ``grid_search.cv_results_`` could be imported into
    a `pandas.DataFrame` without this function, the following conveniences
    are provided:

        - returns the correct ranks based on each metric's greater_is_better direction.
        - summarizes the hyperparameter space, only showing the parameters that change
        - combines split scores into a tuple to save table width
        - rounds the values to specified decimals
        - orders columns by type (test score, train score, metric, etc.)

    Parameters
    ----------
    grid_search : `~sklearn.model_selection.RandomizedSearchCV`
        Grid search output (fitted RandomizedSearchCV object).
    only_changing_params : `bool`, default True
        If True, only show parameters with multiple values in
        the hyperparameter_grid.
    combine_splits : `bool`, default True
        Whether to report split scores as a tuple in a single column.

            - If True, adds a column for the test splits scores for each
              requested metric. Adds a column with train split scores if those
              are available.

              For example, "split_train_score" would contain the values
              (split1_train_score, split2_train_score, split3_train_score)
              as as tuple.
            - If False, this summary column is not added.

        The original split columns are available either way.
    decimals : `int` or None, default None
        Number of decimal places to round to.
        If decimals is negative, it specifies the number of
        positions to the left of the decimal point.
        If None, does not round.
    score_func : `str` or callable, default ``EvaluationMetricEnum.MeanAbsolutePercentError.name``
        Score function used to select optimal model in CV.
        If a callable, takes arrays ``y_true``, ``y_pred`` and returns a float.
        If a string, must be either a
        `~greykite.common.evaluation.EvaluationMetricEnum` member name
        or `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.

        Used in this function to fix the ``"rank_test_score"`` column if
        ``score_func_greater_is_better=False``.

        Should be the same as what was passed to
        :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`,
        or `~greykite.framework.pipeline.pipeline.forecast_pipeline`,
        or `~greykite.framework.pipeline.utils.get_hyperparameter_searcher`.
    score_func_greater_is_better : `bool`, default False
        True if ``score_func`` is a score function, meaning higher is better,
        and False if it is a loss function, meaning lower is better.
        Must be provided if ``score_func`` is a callable (custom function).
        Ignored if ``score_func`` is a string, because the direction is known.

        Used in this function to fix the ``"rank_test_score"`` column if
        ``score_func_greater_is_better=False``.

        Should be the same as what was passed to
        :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`,
        or `~greykite.framework.pipeline.pipeline.forecast_pipeline`,
        or `~greykite.framework.pipeline.utils.get_hyperparameter_searcher`.
    cv_report_metrics : `~greykite.framework.constants.CV_REPORT_METRICS_ALL`, or `list` [`str`], or None, default `~greykite.common.constants.CV_REPORT_METRICS_ALL`  # noqa: E501
        Additional metrics to show in the summary, besides the one specified by ``score_func``.

        If a metric is specified but not available, a warning will be given.

        Should be the same as what was passed to
        :py:meth:`~greykite.framework.templates.forecaster.Forecaster.run_forecast_config`,
        or `~greykite.framework.pipeline.pipeline.forecast_pipeline`,
        or `~greykite.framework.pipeline.utils.get_hyperparameter_searcher`,
        or a subset of computed metric to show.

        If a list of strings, valid strings are
        `greykite.common.evaluation.EvaluationMetricEnum` member names
        and `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.
    column_order : `list` [`str`] or None, default None
        How to order the columns.
        A list of regex to order column names, in greedy fashion. Column names matching
        the first item are placed first. Among remaining items, those matching the second
        items are placed next, etc.
        Use "*" as the last element to select all available columns, if desired.
        If None, uses default ordering::

            column_order = ["rank_test", "mean_test", "split_test", "mean_train",
                            "params", "param", "split_train", "time", ".*"]

    Notes
    -----
    Metrics are named in ``grid_search.cv_results_`` according to the ``scoring``
    parameter passed to `~sklearn.model_selection.RandomizedSearchCV`.

    ``"score"`` is the default used by sklearn for single metric
    evaluation.

    If a dictionary is provided to ``scoring``, as is the case through
    templates, then the metrics are named by its keys, and the
    metric used for selection is defined by ``refit``. The keys
    are derived from ``score_func`` and ``cv_report_metrics``
    in `~greykite.framework.pipeline.utils.get_scoring_and_refit`.

        - The key for ``score_func`` if it is a callable is
          `~greykite.common.constants.CUSTOM_SCORE_FUNC_NAME`.
        - The key for ``EvaluationMetricEnum`` member name is the short name
          from ``.get_metric_name()``.
        - The key for `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`
          is `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE_NAME`.

    Returns
    -------
    cv_results : `pandas.DataFrame`
        A summary of cross-validation results in tabular format.
        Each row corresponds to a set of parameters used in the grid search.

        The columns have the following format, where name is the canonical short
        name for the metric.

            ``"rank_test_{name}"`` : `int`
                The params ranked by mean_test_score (1 is best).
            ``"mean_test_{name}"`` : `float`
                Average test score.
            ``"split_test_{name}"`` : `list` [`float`]
                Test score on each split. [split 0, split 1, ...]
            ``"std_test_{name}"`` : `float`
                Standard deviation of test scores.
            ``"mean_train_{name}"`` : `float`
                Average train score.
            ``"split_train_{name}"`` : `list` [`float`]
                Train score on each split. [split 0, split 1, ...]
            ``"std_train_{name}"`` : `float`
                Standard deviation of train scores.
            ``"mean_fit_time"`` : `float`
                Average time to fit each CV split (in seconds)
            ``"std_fit_time"`` : `float`
                Std of time to fit each CV split (in seconds)
            ``"mean_score_time"`` : `float`
                Average time to score each CV split (in seconds)
            ``"std_score_time"`` : `float`
                Std of time to score each CV split (in seconds)
            ``"params"`` : `dict`
                The parameters used. If ``only_changing==True``,
                only shows the parameters which are not identical
                across all CV splits.
            ``"param_{pipeline__param__name}"`` : Any
                The value of pipeline parameter `pipeline__param__name`
                for each row.

    """
    if column_order is None:
        column_order = ["rank_test", "mean_test", "split_test", "mean_train", "params", "param", "split_train", "time", ".*"]

    cv_results = grid_search.cv_results_.copy()

    # Overwrites the params
    selected_params = []
    if only_changing_params:
        # Removes keys that don't vary
        keep_params = set()
        seen_params = {}
        for params in cv_results['params']:
            for k, v in params.items():
                if k in seen_params:
                    try:
                        assert_equal(v, seen_params[k])
                    except AssertionError:
                        # the values are different
                        keep_params.add(k)
                else:
                    seen_params[k] = v

        for params in cv_results['params']:
            explore_params = [(k, v) for k, v in params.items() if k in keep_params]
            selected_params.append(explore_params)
        cv_results['params'] = selected_params

    # Overwrites the ranks and computes combined split score columns
    # for the requested metrics.
    metric_list = [(score_func, score_func_greater_is_better, True)]
    if cv_report_metrics == CV_REPORT_METRICS_ALL:
        cv_report_metrics = EvaluationMetricEnum.__dict__["_member_names_"].copy()
        # Computes `FRACTION_OUTSIDE_TOLERANCE` if `relative_error_tolerance` is specified
        cv_report_metrics.append(FRACTION_OUTSIDE_TOLERANCE)
        metric_list += [(metric, None, False) for metric in cv_report_metrics]
    elif cv_report_metrics is not None:
        # greater_is_better is derived from the metric name
        metric_list += [(metric, None, True) for metric in cv_report_metrics]

    keep_metrics = set()
    for metric, greater_is_better, warn_metric in metric_list:
        ranks_and_splits = get_ranks_and_splits(
            grid_search=grid_search,
            score_func=metric,
            greater_is_better=greater_is_better,
            combine_splits=combine_splits,
            decimals=decimals,
            warn_metric=warn_metric)
        short_name = ranks_and_splits["short_name"]
        if ranks_and_splits["ranks"] is not None:
            cv_results[f"rank_test_{short_name}"] = ranks_and_splits["ranks"]
        if ranks_and_splits["split_train"] is not None:
            cv_results[f"split_train_{short_name}"] = ranks_and_splits["split_train"]
        if ranks_and_splits["split_test"] is not None:
            cv_results[f"split_test_{short_name}"] = ranks_and_splits["split_test"]
        keep_metrics.add(short_name)

    # Creates DataFrame and orders the columns.
    # Dictionary keys are unordered, but appears to follow insertion order.
    cv_results_df = pd.DataFrame(cv_results)
    available_cols = list(cv_results_df.columns)
    # Removes metrics not selected
    all_metrics = set(col.replace("mean_test_", "") for col in cv_results.keys()
                      if re.search("mean_test_", col))
    remove_metrics = all_metrics - keep_metrics
    remove_regex = "|".join(remove_metrics)
    if remove_regex:
        available_cols = [col for col in available_cols
                          if not re.search(remove_regex, col)]
    # Orders the columns
    ordered_cols = []
    for regex in column_order:
        selected_cols = [col for col in available_cols
                         if col not in ordered_cols and re.search(regex, col)]
        ordered_cols += selected_cols
    cv_results_df = cv_results_df[ordered_cols]

    if decimals is not None:
        cv_results_df = cv_results_df.round(decimals)

    return cv_results_df
