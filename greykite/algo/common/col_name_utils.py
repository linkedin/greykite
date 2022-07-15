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
# original author: Kaixu Yang
"""Utilities for silverkite feature column names."""

import re

from greykite.common import constants as cst


INTERCEPT = "Intercept"
"""The name of the intercept term, to show in model summary."""


def simplify_event(name):
    """Simplifies an event predictor name.

    Eliminates the levels and unnecessary characters in an event predictor's name.
    If the original name is
    "C(Q('events_Chinese New Year_minus_1'), levels=['', 'event'])[T.event]",
    it will be simplified to "events_Chinese New Year-1".
    If ``cst.EVENT_PREFIX`` is not in the name, the original name will be returned.

    Parameters
    ----------
    name : `str`
        The predictor name.

    Returns
    -------
    simplified_name : `str`
        The simplified name.
    """
    if cst.EVENT_PREFIX in name:
        result = re.search(fr"\('({cst.EVENT_PREFIX}.*)'\),", name)
        if result is not None:
            name = result.group(1)
            name = name.replace("_minus_", "-")
            name = name.replace("_plus_", "+")
    return name


def simplify_time_features(name):
    """Simplifies a time feature predictor name.

    Eliminates the levels and unnecessary characters in a time feature predictor's name.
    If the original name is like
    "C(Q('str_dow'), levels=['1-Mon', '2-Tue', '3-Wed', '4-Thu', '5-Fri', '6-Sat', '7-Sun'])[T.2-Tue]",
    it will be simplified to "str_dow_2-Tue".
    If the original name is like "str_dow[T.7-Sun]", it will be simplified to "str_dow_7-Sun".
    If the original name is like "toy", it will be kept as it is.
    If any ``cst.TimeFeaturesEnum`` is not in the name, the original name will be returned.

    Parameters
    ----------
    name : `str`
        The predictor name.

    Returns
    -------
    simplified_name : `str`
        The simplified name.
    """
    if any([x in name for x in cst.TimeFeaturesEnum.__dict__["_member_names_"]]):
        result = re.search(r"(.*)\[T.(.*)\]", name)
        if result is not None:
            name = result.group(1)
            level = "_" + result.group(2)
            if "levels" in name:
                name_result = re.search(r"\('(.*)'\),", name)
                if name_result is not None:
                    name = name_result.group(1)
        else:
            level = ""
        if name == cst.TimeFeaturesEnum.is_weekend.value:  # is_weekend only has two levels
            level = ""
        name = name + level
    return name


def simplify_changepoints(name):
    """Simplifies a changepoint predictor name.

    Changes "changepoint" to "cp" to shorten the changepoint features' names.
    If ``cst.CHANGEPOINT_COL_PREFIX`` is not in the name, the original name will be returned.

    Parameters
    ----------
    name : `str`
        The predictor name.

    Returns
    -------
    simplified_name : `str`
        The simplified name.
    """
    name = name.replace(cst.CHANGEPOINT_COL_PREFIX, cst.CHANGEPOINT_COL_PREFIX_SHORT)
    return name


def simplify_name(name):
    """Simplifies the predictor names to make them shorter.


    Parameters
    ----------
    name : `str`
        The predictor name. Should be a simple term with no interaction.

    Returns
    -------
    simplified_name : `str`
        The simplified name.
    """
    name = simplify_event(name)
    name = simplify_time_features(name)
    name = simplify_changepoints(name)
    return name


def simplify_pred_cols(pred_cols):
    """Simplifies predictor names in a list.

    Parameters
    ----------
    pred_cols : `list` [ `str` ]
        A list of predictor names to be simplified.
        Names in ``pred_cols`` could contain interactions.

    Returns
    -------
    new_pred_cols : `list` [ `str` ]
        A simplified list of predictor names by applying element-wisely
        `~greykite.algo.common.col_name_utils.simplify_name`
    """
    pred_cols_extract = [[col] if ":" not in col else col.split(":") for col in pred_cols]
    pred_cols_extract = [[simplify_name(name) for name in col] for col in pred_cols_extract]
    new_pred_cols = [col[0] if len(col) == 1 else ":".join(col) for col in pred_cols_extract]
    return new_pred_cols


def add_category_cols(coef_summary, pred_category):
    """Adds indicators columns to coefficient summary df for categaries.

    Parameters
    ----------
    coef_summary : `pandas.DataFrame`
        The coefficient summary df.
        This is typically generated by
        `~greykite.algo.common.model_summary_utils.add_model_coef_df`
        or
        `~greykite.algo.common.model_summary_utils.add_model_coef_df`.
    pred_category : `dict`
        The predictor category dictionary by
        `~greykite.algo.common.col_name_utils.create_pred_category`.

    Returns
    -------
    coef_summary_with_new_columns : `pandas.DataFrame
        New df with the following columns added:

            "is_intercept" : 0 or 1
                Intercept or not.
            "is_time_feature" : 0 or 1
                Time features or not.
                Time features belong to `~greykite.common.constants.TimeFeaturesEnum`.
            "is_event" : 0 or 1
                Event features or not.
                Event features have `~greykite.common.constants.EVENT_PREFIX`.
            "is_trend" : 0 or 1
                Trend features or not.
                Trend features have `~greykite.common.constants.CHANGEPOINT_COL_PREFIX` or "cp\\d".
            "is_seasonality" : 0 or 1
                Seasonality feature or not.
                Seasonality features have `~greykite.common.constants.SEASONALITY_REGEX`.
            "is_lag" : 0 or 1
                Lagged features or not.
                Lagged features have "lag_".
            "is_regressor" : 0 or 1
                Extra features provided by users.
                They are provided through ``extra_pred_cols`` in the fit function.
            "is_interaction" : 0 or 1
                Interaction feature or not.
                Interaction features have ":".
    """
    coef_summary = coef_summary.copy()
    pred_cols = coef_summary["Pred_col"].tolist()
    coef_summary["is_intercept"] = [1 if col in pred_category["intercept"]
                                    else 0 for col in pred_cols]
    coef_summary["is_time_feature"] = [1 if col in pred_category["time_features"]
                                       else 0 for col in pred_cols]
    coef_summary["is_event"] = [1 if col in pred_category["event_features"]
                                else 0 for col in pred_cols]
    coef_summary["is_trend"] = [1 if col in pred_category["trend_features"]
                                else 0 for col in pred_cols]
    coef_summary["is_seasonality"] = [1 if col in pred_category["seasonality_features"]
                                      else 0 for col in pred_cols]
    coef_summary["is_lag"] = [1 if col in pred_category["lag_features"]
                              else 0 for col in pred_cols]
    coef_summary["is_regressor"] = [1 if col in pred_category["regressor_features"]
                                    else 0 for col in pred_cols]
    coef_summary["is_interaction"] = [1 if col in pred_category["interaction_features"]
                                      else 0 for col in pred_cols]
    return coef_summary


def create_pred_category(pred_cols, extra_pred_cols, df_cols):
    """Creates a dictionary of predictor categories.

    The keys are categories, and the values are the corresponding
    predictor names. For detail, see
    `~greykite.sklearn.estimator.base_silverkite_estimator.BaseSilverkiteEstimator.pred_category`

    Parameters
    ----------
    pred_cols : `list` [ `str` ]
        A full list of predictor names used in the model, including extra predictor names.
    extra_pred_cols : `list` [ `str` ]
        A list of extra predictors what are manually provided for the estimator class.
        In ``SilverkiteEstimator``, this is the ``extra_pred_cols``.
        In ``SimpleSilverkiteEstimator``, this is the combination of ``regressor_cols``
        and ``extra_pred_cols``.
    df_cols : `list` [ `str` ]
        The extra columns that are present in the input df.
        Columns that are in ``df_cols`` are not considered for categories other than regressors.
        Columns are considered as regressors only if they are in ``df_cols``.

    Returns
    -------
    pred_category : `dict`
        A dictionary of categories and predictors. For details, see
        `~greykite.sklearn.estimator.base_silverkite_estimator.BaseSilverkiteEstimator.pred_category`
    """

    if extra_pred_cols is None:
        extra_pred_cols = []
    extra_pred_cols = list(set(extra_pred_cols))  # might have duplicates
    # SimpleSilverkiteEstimator and SilverkiteEstimator have different ways to specify
    # regressors and lagged regressors. The ``extra_pred_cols`` in this function
    # should have a super set of such columns.
    # Regressor columns are defined as columns that are in
    # ``df_cols`` and that are present either in ``extra_pred_cols`` directly
    # or via interaction terms.
    # Because all columns in ``regressor_cols`` are present in ``df_cols``,
    # these columns are not categorized to any other category.
    regressor_cols = [
        col for col in df_cols
        if col in [c for term in extra_pred_cols for c in term.split(":")]
    ]
    pred_category = {
        "intercept": [col for col in pred_cols if re.search(INTERCEPT, col)],
        # Time feature names could be included in seasonality features.
        # We do not want to include pure seasonality features.
        # If a term does not include interaction, it need to include
        # time feature name but not seasonality regex.
        # This keeps "ct1" and excludes "cos1_ct1_yearly"
        # If a term includes interaction, then at least one of the
        # two sub-terms need to satisfy the condition above.
        # This keeps "ct1:is_weekend" and "ct1:cos1_ct1_yearly"
        # and excludes "is_weekend:cos1_ct1_yearly".
        "time_features": [col for col in pred_cols
                          if ((re.search(":", col) is None
                               and
                               re.search("|".join(cst.TimeFeaturesEnum.__dict__["_member_names_"]), col)
                               and
                               re.search(cst.SEASONALITY_REGEX, col) is None)
                              or
                              (re.search(":", col)
                               and
                               any([re.search("|".join(cst.TimeFeaturesEnum.__dict__["_member_names_"]), subcol)
                                    and
                                    re.search(cst.SEASONALITY_REGEX, subcol) is None
                                    for subcol in col.split(":")])))
                          and col not in regressor_cols
                          ],
        "event_features": [col for col in pred_cols if cst.EVENT_PREFIX in col
                           and col not in regressor_cols],
        # the same logic as time features for trend features
        "trend_features": [col for col in pred_cols
                           if ((re.search(":", col) is None
                                and
                                re.search(cst.TREND_REGEX, col)
                                and
                                re.search(cst.SEASONALITY_REGEX, col) is None)
                               or
                               (re.search(":", col)
                                and
                                any([re.search(cst.TREND_REGEX, subcol)
                                     and
                                     re.search(cst.SEASONALITY_REGEX, subcol) is None
                                     for subcol in col.split(":")])))
                           and col not in regressor_cols
                           ],
        "seasonality_features": [col for col in pred_cols if re.search(cst.SEASONALITY_REGEX, col)
                                 and col not in regressor_cols],
        "lag_features": [col for col in pred_cols if re.search(cst.LAG_REGEX, col)
                         and col not in regressor_cols],
        # only the supplied extra_pred_col that are also in pred_cols
        "regressor_features": [col for col in pred_cols if col in regressor_cols
                               or any([x in regressor_cols for x in col.split(":")])],
        "interaction_features": [col for col in pred_cols if re.search(":", col)]
    }
    return pred_category


def filter_coef_summary(
        coef_summary,
        pred_category,
        is_intercept=None,
        is_time_feature=None,
        is_event=None,
        is_trend=None,
        is_seasonality=None,
        is_lag=None,
        is_regressor=None,
        is_interaction=None):
    """Gets the coefficient summary df after applying the given filters.

    Set any of the parameters to `bool` to enable filtering.

        - Any argument set to True will be aggregated with logical operator "or", i.e.
          a category will be displayed when set to True.
        - Any argument set to False will be aggregated with logical operator "and", i.e.
          a category will not be displayed when set to False (even it has interaction with
          a category that is set to True).
        - Any argument set to None will be ignored unless all arguments are None.
        - ``is_interaction`` is used to exclude interaction terms by setting it
          to False. It is not used when the value is True and other argument is not None.
          The design here is to use ``is_interaction`` as a second pass filter.

    Parameters
    ----------
    coef_summary : `pandas.DataFrame`
        The coefficient summary df.
    pred_category : `dict`
        The predictor category dictionary by
        `~greykite.algo.common.col_name_utils.create_pred_category`.
    is_intercept : `bool` or `None`, default `None`
        Intercept or not.
    is_time_feature : `bool` or `None`, default `None`
        Time features or not.
        Time features belong to `~greykite.common.constants.TimeFeaturesEnum`.
    is_event : `bool` or `None`, default `None`
        Event features or not.
        Event features have `~greykite.common.constants.EVENT_PREFIX`.
    is_trend : `bool` or `None`, default `None`
        Trend features or not.
        Trend features have `~greykite.common.constants.CHANGEPOINT_COL_PREFIX`.
    is_seasonality : `bool` or `None`, default `None`
        Seasonality feature or not.
        Seasonality features have `~greykite.common.constants.SEASONALITY_REGEX`.
    is_lag : `bool` or `None`, default `None`
        Lagged features or not.
        Lagged features have "lag_".
    is_regressor : `bool` or `None`, default `None`
        User supplied regressor features or not.
        They are provided with the `extra_pred_cols`.
    is_interaction : `bool` or `None`, default `None`
        Interaction feature or not.
        Interaction features have ":".

    Returns
    -------
    filtered_coef_summary : `pandas.DataFrame`
        The filtered coefficient summary df filtered by the given conditions.
    """
    coef_summary_categorized = add_category_cols(coef_summary, pred_category)
    coef_summary_categorized["Pred_col"] = simplify_pred_cols(coef_summary_categorized["Pred_col"])
    # And True values will be aggregated with "or".
    # Any False values will be aggregated with "and".
    # This aligns with human sense.
    conditions_true = []
    conditions_false = []
    for k, v in locals().items():
        if "is_" in k and k != "is_interaction":
            if v is not None:
                if v:
                    conditions_true.append(f"{k} == {int(v)}")
                else:
                    conditions_false.append(f"{k} == {int(v)}")
    query_true = f"{' or '.join(conditions_true)}"
    query_false = f"{' and '.join(conditions_false)}"
    if query_true and query_false:
        query = f"({query_true}) and ({query_false})"
    elif query_true:
        query = f"({query_true})"
    elif query_false:
        query = f"({query_false})"
    else:
        query = ""
    # Deal with is_interaction
    if query:
        if is_interaction is False:
            # When is_interaction is False, remove interaction terms.
            # By default, the filters include interaction terms.
            query = f"({query}) and (is_interaction == {int(is_interaction)})"
        new_df = coef_summary_categorized.query(query)
    else:
        # In there is no query, is_interaction is used filter rows.
        if is_interaction is not None:
            query = f"is_interaction == {int(is_interaction)}"
            new_df = coef_summary_categorized.query(query)
        else:
            new_df = coef_summary_categorized
    return new_df.reset_index(drop=True)
