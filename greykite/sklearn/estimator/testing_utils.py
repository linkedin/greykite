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
"""Functions for testing silverkite estimators."""

import datetime

import pandas as pd

from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import cols_interact
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import generate_holiday_events
from greykite.common.constants import TimeFeaturesEnum
from greykite.common.features.timeseries_features import convert_date_to_continuous_time


def params_components():
    """Parameters for ``forecast_silverkite``"""
    autoreg_dict = {
        "lag_dict": {"orders": [7]},
        "agg_lag_dict": {
            "orders_list": [[7, 7*2, 7*3]],
            "interval_list": [(7, 7*2)]},
        "series_na_fill_func": lambda s: s.bfill().ffill()}

    uncertainty_dict = {
        "uncertainty_method": "simple_conditional_residuals",
        "params": {
            "conditional_cols": [TimeFeaturesEnum.dow.value],
            "quantiles": [0.025, 0.975],
            "quantile_estimation_method": "normal_fit",
            "sample_size_thresh": 5,
            "small_sample_size_method": "std_quantiles",
            "small_sample_size_quantile": 0.98}}

    # generate holidays
    countries = ["US", "India"]
    holidays_to_model_separately = [
        "New Year's Day",
        "Christmas Day",
        "Independence Day",
        "Thanksgiving",
        "Labor Day",
        "Memorial Day",
        "Veterans Day"]
    event_df_dict = generate_holiday_events(
        countries=countries,
        holidays_to_model_separately=holidays_to_model_separately,
        year_start=2015,
        year_end=2025,
        pre_num=2,
        post_num=2)
    # constant event effect at daily level
    event_cols = [f"Q('events_{key}')" for key in event_df_dict.keys()]
    interaction_cols = cols_interact(
        static_col=TimeFeaturesEnum.is_weekend.value,
        fs_name=TimeFeaturesEnum.tow.value,
        fs_order=4,
        fs_seas_name="weekly")
    extra_pred_cols = [TimeFeaturesEnum.ct_sqrt.value,
                       TimeFeaturesEnum.dow_hr.value,
                       TimeFeaturesEnum.ct1.value,
                       f"{TimeFeaturesEnum.ct1.value}:{TimeFeaturesEnum.tod.value}",
                       "regressor1",
                       "regressor2"] + event_cols + interaction_cols

    # seasonality terms
    fs_components_df = pd.DataFrame({
        "name": [
            TimeFeaturesEnum.tod.value,
            TimeFeaturesEnum.tow.value,
            TimeFeaturesEnum.ct1.value],
        "period": [24.0, 7.0, 1.0],
        "order": [12, 4, 5],
        "seas_names": ["daily", "weekly", "yearly"]})

    # changepoints
    changepoints_dict = dict(
        method="custom",
        dates=["2018-01-01", "2019-01-02-16", "2019-01-03", "2019-02-01"],
        continuous_time_col=TimeFeaturesEnum.ct2.value)

    return {
        "coverage": 0.95,
        "origin_for_time_vars": convert_date_to_continuous_time(datetime.datetime(2018, 1, 3)),
        "extra_pred_cols": extra_pred_cols,
        "train_test_thresh": None,
        "training_fraction": None,
        "fit_algorithm": "ridge",
        "daily_event_df_dict": event_df_dict,
        "changepoints_dict": changepoints_dict,
        "fs_components_df": fs_components_df,
        "autoreg_dict": autoreg_dict,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "uncertainty_dict": uncertainty_dict
    }
