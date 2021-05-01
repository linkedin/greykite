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
# original author: Reza Hosseini
"""Creates a forecast using similar time points in the past."""
import warnings

import numpy as np
import pandas as pd

from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.features.timeseries_features import convert_date_to_continuous_time
from greykite.common.time_properties import describe_timeseries


def forecast_similarity_based(
        df,
        time_col,
        value_cols,
        agg_method,
        agg_func=None,
        grid_size_str=None,
        match_cols=[],
        origin_for_time_vars=None,
        recent_k=1):
    """Fits a basic forecast model which is aggregate based. As an example for an hourly time series we
    can assign the value of the most recent three weeks at the same time of the week as forecast.
    This works for multiple responses passed as a list in "value_cols".
    Also we do not require the timestamps to be regular and for example dataframe can have missing
    timestamps.

    :param df: the dataframe which includes the training data, the value_cols and match_cols if needed
    :param time_col: the column of the dataframe which includes the timestamps of the series
    :param value_cols: the response columns for which forecast is desired
    :param agg_method: a string which specifies the aggregation method. Options are
        "mean": the mean of the values at timestamps which match with desired time on match_cols
        "median": the median of the values at timestamps which match with desired time on match_cols
        "min": the min of the values at timestamps which match with desired time on match_cols
        "max": the max of the values at timestamps which match with desired time on match_cols
        "most_recent": the mean of "recent_k" (given in last argument of the function)
        values matching with desired time on match_cols
    :param agg_func: the aggregation function needed for aggregating w.r.t "match_cols".
        This is needed if agg_method is not available.
    :param grid_size_str: the expected time increment. If not provided it is inferred.
    :param match_cols: the variables used for grouping in aggregation
    :param origin_for_time_vars: the time origin for continuous time variables
    :param recent_k: the number of most recent timestamps to consider for aggregation

    :return: A dictionary consisting of a "model" and a "predict" function.
        The "model" object is simply a dictionary of two items (dataframes).
        First item is an aggregated dataframe of "value_cols" w.r.t "match_cols": "pred_df"
        Second item is an aggregated dataframe across all data: "pred_df_overall".
        This is useful if a level in match_cols didn't appear in train dataset,
        therefore in that case we fall back to overall prediction
        The "predict" is a function which performs prediction for new data
        The "predict_n" is a function which predicts the future for any given number of steps specified
    """
    # This is only needed for predict_n function.
    # But it makes sense to do this only once for computational efficiency
    timeseries_info = describe_timeseries(df, time_col)
    if grid_size_str is None:
        grid_size = timeseries_info["median_delta"]
        grid_size_str = str(int(grid_size / np.timedelta64(1, "s"))) + "s"
    max_timestamp = timeseries_info["max_timestamp"]

    # a dictionary of methods and their corresponding aggregation function
    agg_method_func_dict = {
        "mean": (lambda x: np.mean(x)),
        "median": (lambda x: np.median(x)),
        "min": (lambda x: np.min(x)),
        "max": (lambda x: np.max(x)),
        "most_recent": (lambda x: np.mean(x[-recent_k:]))
    }

    if agg_func is None:
        if agg_method not in agg_method_func_dict.keys():
            raise Exception(
                "The aggregation method you specified is not implemented." +
                "These are the available methods: mean, median, min, max, most_recent")
        agg_func = agg_method_func_dict[agg_method]

    # sets default origin so that "ct1" feature from "build_time_features_df" starts at 0 on training start date
    if origin_for_time_vars is None:
        origin_for_time_vars = convert_date_to_continuous_time(df[time_col][0])

    # we calculate time features here which might appear in match_cols and not available in df by default
    def add_time_features(df):
        time_df = build_time_features_df(dt=df[time_col], conti_year_origin=origin_for_time_vars)
        for col in match_cols:
            if col not in df.columns:
                df[col] = time_df[col].values
        return df

    df = add_time_features(df=df)

    # for value_col in value_cols:
    # aggregate w.r.t. given columns in match_cols
    agg_dict = {value_col: agg_func for value_col in value_cols}
    # we create a coarse prediction by simply aggregating every value column globally
    # this is useful for cases where no matching data is available for a given timestamp to be forecasted
    pred_df_overall = df.groupby([True]*len(df), as_index=False).agg(agg_dict)
    if len(match_cols) == 0:
        pred_df = pred_df_overall
    else:
        pred_df = df.groupby(match_cols, as_index=False).agg(agg_dict)

    model = {
        "pred_df": pred_df,
        "pred_df_overall": pred_df_overall}

    def predict(new_df, new_external_regressor_df=None):
        """Predicts for new dataframe (new_df) using the fitted model.
        :param new_df: a dataframe of new data which must include the time_col and match_cols
        :param new_external_regressor_df: a regressor dataframe if needed
        :return: new_df is augmented with predictions for value_cols and returned
        """
        new_df = new_df.copy(deep=True)
        new_df = add_time_features(df=new_df)
        if new_external_regressor_df is not None:
            new_df = pd.concat([new_df, new_external_regressor_df])

        # if the response columns appear in the new_df columns, we take them out to prevent issues in aggregation
        for col in value_cols:
            if col in new_df.columns:
                warnings.warn(f"{col} is a response column and appeared in new_df. Hence it was removed.")
                del new_df[col]

        new_df["temporary_overall_dummy"] = 0
        pred_df_overall["temporary_overall_dummy"] = 0

        new_df_grouped = pd.merge(new_df, pred_df, on=match_cols, how="left")
        new_df_overall = pd.merge(new_df, pred_df_overall, on=["temporary_overall_dummy"], how="left")

        # when we have missing in the grouped case (which can happen if a level in match_cols didn't appear in train dataset)
        # we fall back to the overall case
        for col in value_cols:
            new_df_grouped.loc[new_df_grouped[col].isnull(), col] = new_df_overall.loc[new_df_grouped[col].isnull(), col]

        del new_df_grouped["temporary_overall_dummy"]

        return new_df_grouped

    def predict_n(fut_time_num, new_external_regressor_df=None):
        """This is the forecast function which can be used to forecast.
        It accepts extra predictors if needed in the form of a dataframe: new_external_regressor_df.
        :param fut_time_num: number of needed future values
        :param new_external_regressor_df: extra predictors if available
        """
        # we create the future time grid
        date_list = pd.date_range(
            start=max_timestamp + pd.Timedelta(grid_size_str),
            periods=fut_time_num,
            freq=grid_size_str).tolist()

        fut_df = pd.DataFrame({time_col: date_list})
        return predict(fut_df, new_external_regressor_df=new_external_regressor_df)

    return {"model": model, "predict": predict, "predict_n": predict_n}
