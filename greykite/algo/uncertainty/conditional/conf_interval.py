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
# original author: Reza Hosseini, Sayan Patra
"""Calculates uncertainty intervals from the conditional
empirical distribution of the residual.
"""

import warnings

import numpy as np
import pandas as pd

from greykite.algo.uncertainty.conditional.dataframe_utils import limit_tuple_col
from greykite.algo.uncertainty.conditional.dataframe_utils import offset_tuple_col
from greykite.algo.uncertainty.conditional.estimate_distribution import estimate_empirical_distribution
from greykite.algo.uncertainty.conditional.normal_quantiles import normal_quantiles_df
from greykite.common.constants import ERR_STD_COL
from greykite.common.constants import QUANTILE_SUMMARY_COL


def conf_interval(
        df,
        distribution_col,
        offset_col=None,
        conditional_cols=None,
        quantiles=(0.005, 0.025, 0.975, 0.995),
        quantile_estimation_method="normal_fit",
        sample_size_thresh=5,
        small_sample_size_method="std_quantiles",
        small_sample_size_quantile=0.95,
        min_admissible_value=None,
        max_admissible_value=None):
    """A function to calculate confidence intervals (CI) for values given
    in ``distribution_col``. We allow for calculating as many quantiles as
    needed (specified by ``quantiles``) as opposed to only two quantiles
    representing a typical CI.

    Two methods are available for quantiles calculation for
    each slice of data (given in ``conditional_cols``).
         - "normal_fit" : CI is calculated using quantiles of a normal
         distribution fit.
        - "ecdf" : CI is calculated using quantiles of empirical cumulative
        distribution function.

    ``offset_col`` is used in the prediction phase to shift the calculated quantiles
    appropriately.

    Parameters
    ----------
    df : `pandas.Dataframe`
        The dataframe with the following columns:

            - distribution_col,
            - conditional_cols (optional),
            - offset_col (optional column)

    distribution_col : `str`
        The column containing the values for the variable for which confidence
        interval is needed.
    offset_col : `str` or None, default None
        The column containing the values by which the computed quantiles for
        ``distribution_col`` are shifted. Only used during prediction phase.
        If None, quantiles are not shifted.
    conditional_cols : `list` [`str`] or None, default None
        These columns are used to slice the data first then calculate quantiles
        for each slice.
    quantiles : `list` [`float`], default (0.005, 0.025, 0.975, 0.995)
        The quantiles calculated for each slice.
        These quantiles can be then used to construct the desired CIs.
        The default values [0.005, 0.025, 0.0975, 0.995] can be used to construct
        99 and 95 percent CIs.
    quantile_estimation_method : `str`, default "normal_fit"
        There are two options implemented for the quantile estimation method
        (conditional on slice):

            - "normal_fit": Uses the standard deviation of the values in each
            slice to compute normal distribution quantiles.
            - "ecdf": Uses the empirical cumulative distribution function
            to calculate sample quantiles.

    sample_size_thresh : `int`, default 5
        The minimum sample size for each slice where we allow for using the conditional
        distribution (conditioned on the "conditional_cols" argument).
        If sample size for that slice is smaller than this,
        we use the fallback method.
    small_sample_size_method : `str`, default "std_quantiles"
        The method to use for slices with small sample size

            - "std_quantile" method is implemented and it looks at the response
              std for each slice with
              sample size >= "sample_size_thresh"
              and takes the row which has  its std being closest
              to "small_sample_size_quantile" quantile.
              It assigns that row to act as fall-back for calculating conf
              intervals.

    small_sample_size_quantile : `float`, default 0.95
        Quantile to calculate for small sample size.
    min_admissible_value : Union[float, double, int], default None
        This is the lowest admissible value for the obtained ci limits
        and any value below this will be mapped back to this value.
    max_admissible_value : Union[float, double, int], default None
        This is the highest admissible value for the obtained ci limits
        and any higher value will be mapped back to this value.

    Returns
    -------
    uncertainty_model : `dict`
        Dictionary with following items (main component is the ``predict`` function).

            - "ecdf_df" : `pandas.DataFrame`
                ecdf_df generated by "estimate_empirical_distribution"
            - "ecdf_df_overall" : `pandas.DataFrame`
                ecdf_df_overall generated by "estimate_empirical_distribution"
            - "ecdf_df_fallback" : `pandas.DataFrame`
                ecdf_df_fallback, a fall back data to get the CI quantiles
                when the sample size for that slice is small or that slice
                is unobserved in that case.

                    - if small_sample_size_method = "std_quantiles",
                      we use std quantiles to pick a slice which has a std close
                      to that quantile and fall-back to that slice.
                    - otherwise we fallback to "ecdf_overall"
            - "distribution_col" : `str`
                Input ``distribution_col``
            - "offset_col": `str`
                Input ``offset_col``
            - "quantiles" : `list` [`float`]
                Input ``quantiles``
            - "min_admissible_value": `float`
                Input ``min_admissible_value``
            - "max_admissible_value": `float`
                Input ``max_admissible_value``
            - "conditional_cols": `list` [`str`]
                Input ``conditional_cols``
            - "std_col": `str`
                The column name with standard deviations.
            - "quantile_summary_col": `str`
                The column name with computed quantiles.
            - "fall_back_for_all": `bool`
                Indicates if fallback method should be used for the
                whole dataset.
    """
    std_col = f"{distribution_col}_std"
    sample_size_col = f"{distribution_col}_count"

    model_dict = estimate_empirical_distribution(
        df=df,
        distribution_col=distribution_col,
        quantile_grid_size=None,
        quantiles=quantiles,
        conditional_cols=conditional_cols,
        remove_conditional_mean=True
    )
    ecdf_df = model_dict["ecdf_df"]
    ecdf_df_overall = model_dict["ecdf_df_overall"]

    # Two methods are implemented: ecdf; normal_fit.
    if quantile_estimation_method == "ecdf":
        quantile_summary_col = f"{distribution_col}_ecdf_quantile_summary"
    elif quantile_estimation_method == "normal_fit":
        quantile_summary_col = f"{distribution_col}_normal_quantile_summary"
        ecdf_df = normal_quantiles_df(
            df=ecdf_df,
            std_col=std_col,
            mean_col=None,
            fixed_mean=0.0,
            quantiles=quantiles,
            quantile_summary_col=quantile_summary_col
        )
        ecdf_df_fallback = normal_quantiles_df(
            df=ecdf_df_overall,
            std_col=std_col,
            mean_col=None,
            fixed_mean=0.0,
            quantiles=quantiles,
            quantile_summary_col=quantile_summary_col
        )
    else:
        raise NotImplementedError(
            f"CI calculation method {quantile_estimation_method} is not either of: normal_fit; ecdf")

    # handling slices with small sample size
    # if a method is provided via the argument "small_sample_size_method" then it is used here
    # the idea is to take a relatively high volatility
    # when the new point does not have enough (as specified by "sample_size_thresh")
    # similar points in the past
    fall_back_for_all = False
    if small_sample_size_method == "std_quantiles":
        ecdf_df_large_ss = ecdf_df.loc[ecdf_df[sample_size_col] >= sample_size_thresh].reset_index(drop=True)
        assert set(ecdf_df_large_ss.columns).intersection(["std_quantile", "std_quantile_diff"]) == set(), (
            "column names: std_quantile, std_quantile_diff should not appear in ecdf_df")
        if len(ecdf_df_large_ss) == 0:
            warnings.warn("No slice had sufficient sample size. We fall back to the overall distribution.")
            # If ``ecdf_df_large_ss`` is empty it means we do not have any sufficient
            # samples for any slices.
            # Therefore we have to fall back in all cases and we set ``ecdf_df``
            # to ``ecdf_df_fall_back``
            ecdf_df = ecdf_df_fallback
            fall_back_for_all = True
        else:
            ecdf_df_large_ss["std_quantile"] = np.argsort(ecdf_df_large_ss[std_col]) / ecdf_df_large_ss.shape[0]
            # Calculates the distance between "std_quantile" column values and ``small_sample_size_quantile``
            ecdf_df_large_ss["std_quantile_diff"] = abs(ecdf_df_large_ss["std_quantile"] - small_sample_size_quantile)
            # Chooses the row with closes value in "std_quantile" column to ``small_sample_size_quantile``
            # Note the resulting dataframe below ``ecdf_df_fallback`` will have one row
            ecdf_df_fallback = ecdf_df_large_ss.loc[[ecdf_df_large_ss["std_quantile_diff"].idxmin()]]
            del ecdf_df_fallback["std_quantile"]
            del ecdf_df_fallback["std_quantile_diff"]
            del ecdf_df_large_ss["std_quantile"]
            del ecdf_df_large_ss["std_quantile_diff"]
            # we re-assign ecdf_df by removing the combinations with small sample size
            # this is done so that in predict phase those values are not populated from
            # small sample sizes and use ``ecdf_fallback``
            ecdf_df = ecdf_df_large_ss
    elif small_sample_size_method is not None:
        raise NotImplementedError(
            f"small_sample_size_method {small_sample_size_method} is not implemented.")

    return {
        "ecdf_df": ecdf_df,
        "ecdf_df_overall": ecdf_df_overall,
        "ecdf_df_fallback": ecdf_df_fallback,
        "distribution_col": distribution_col,
        "offset_col": offset_col,
        "quantiles": quantiles,
        "min_admissible_value": min_admissible_value,
        "max_admissible_value": max_admissible_value,
        "conditional_cols": conditional_cols,
        "std_col": std_col,
        "quantile_summary_col": quantile_summary_col,
        "fall_back_for_all": fall_back_for_all}


def predict_ci(
        new_df,
        ci_model):
    """Predicts the quantiles of the ``offset_col`` (defined in ``ci_model``) in ``new_df``.

    Parameters
    ----------
    new_df : `pd.Dataframe`
        Prediction dataframe which minimally includes ``offset_col``
        and ``conditional_cols`` defined in the ``ci_model``.
    ci_model : `dict`
        Returned CI model from ``conf_interval``.

    Returns
    -------
    pred_df : `pd.Dataframe`
        A dataframe which includes ``new_df`` and new columns containing
        the quantiles.
    """

    ecdf_df = ci_model["ecdf_df"]
    ecdf_df_fallback = ci_model["ecdf_df_fallback"]
    offset_col = ci_model["offset_col"]
    min_admissible_value = ci_model["min_admissible_value"]
    max_admissible_value = ci_model["max_admissible_value"]
    conditional_cols = ci_model["conditional_cols"]
    std_col = ci_model["std_col"]
    quantile_summary_col = ci_model["quantile_summary_col"]
    fall_back_for_all = ci_model["fall_back_for_all"]

    # Copies ``pred_df`` so that input df to predict is not altered
    pred_df = new_df.reset_index(drop=True)
    pred_df["temporary_overall_dummy"] = 0
    ecdf_df_fallback_dummy = ecdf_df_fallback.copy()
    ecdf_df_fallback_dummy["temporary_overall_dummy"] = 0
    pred_df_fallback = pd.merge(
        pred_df,
        ecdf_df_fallback_dummy,
        on=["temporary_overall_dummy"],
        how="left")
    del pred_df["temporary_overall_dummy"]
    del pred_df_fallback["temporary_overall_dummy"]

    if (conditional_cols is None) or (conditional_cols == []) or fall_back_for_all:
        pred_df_conditional = pred_df_fallback.copy()
    else:
        pred_df_conditional = pd.merge(
            pred_df,
            ecdf_df,
            on=conditional_cols,
            how="left")

    # When we have missing in the grouped case (which can happen if a level
    # in ``match_cols`` didn't appear in train dataset)
    # we fall back to the overall case
    for col in [quantile_summary_col, std_col]:
        na_index = pred_df_conditional[col].isnull()
        pred_df_conditional.loc[na_index, col] = (
            pred_df_fallback.loc[na_index, col])

    # offsetting the values in ``distribution_col`` by ``offset_col``
    if offset_col is None:
        pred_df_conditional[QUANTILE_SUMMARY_COL] = pred_df_conditional[quantile_summary_col]
    else:
        pred_df_conditional[QUANTILE_SUMMARY_COL] = offset_tuple_col(
            df=pred_df_conditional,
            offset_col=offset_col,
            tuple_col=quantile_summary_col)

    pred_df_conditional = limit_tuple_col(
        df=pred_df_conditional,
        tuple_col=QUANTILE_SUMMARY_COL,
        lower=min_admissible_value,
        upper=max_admissible_value)

    # Only returning needed cols
    returned_cols = [QUANTILE_SUMMARY_COL, std_col]
    if conditional_cols is not None:
        returned_cols = conditional_cols + returned_cols

    pred_df[returned_cols] = pred_df_conditional[returned_cols]
    pred_df.rename(columns={
        std_col: ERR_STD_COL
    }, inplace=True)

    return pred_df
