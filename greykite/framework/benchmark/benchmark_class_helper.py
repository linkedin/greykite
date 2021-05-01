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
# original author: Sayan Patra
"""Helper functions to framework.benchmark model templates."""

import timeit
from typing import Dict

import pandas as pd
from tqdm.autonotebook import tqdm

from greykite.common.constants import TIME_COL
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.framework.pipeline.pipeline import forecast_pipeline
from greykite.sklearn.cross_validation import RollingTimeSeriesSplit


def forecast_pipeline_rolling_evaluation(
        pipeline_params: Dict,
        tscv: RollingTimeSeriesSplit):
    """Runs ``forecast_pipeline`` on a rolling window basis.

    Parameters
    ----------
    pipeline_params : `Dict`
        A dictionary containing the input to the
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
    tscv : `~greykite.sklearn.cross_validation.RollingTimeSeriesSplit`
        Cross-validation object that determines the rolling window evaluation.
        See :class:`~greykite.sklearn.cross_validation.RollingTimeSeriesSplit` for details.

    Returns
    -------
    rolling_evaluation : `dict`
        Stores benchmarking results for each split, e.g.
        split_0 contains result for first split, split_1 contains result for second split and so on.
        Number of splits is determined by the input parameters.
        Every split is a dictionary with keys "runtime_sec" and "pipeline_result".
    """
    if pipeline_params["forecast_horizon"] != tscv.forecast_horizon:
        raise ValueError("Forecast horizon in 'pipeline_params' does not match that of the 'tscv'.")

    if pipeline_params["periods_between_train_test"] != tscv.periods_between_train_test:
        raise ValueError("'periods_between_train_test' in 'pipeline_params' does not match that of the 'tscv'.")

    df = pipeline_params["df"]
    time_col = pipeline_params.get("time_col", TIME_COL)
    date_format = pipeline_params.get("date_format")
    # Disables backtest. For rolling evaluation we know the actual values in forecast period.
    # So out of sample performance can be calculated using pipeline_result.forecast
    pipeline_params["test_horizon"] = 0

    rolling_evaluation = {}
    with tqdm(list(tscv.split(X=df)), ncols=800, leave=True) as progress_bar:
        for (split_num, (train, test)) in enumerate(progress_bar):
            # Description will be displayed on the left of progress bar
            progress_bar.set_description(f"Split '{split_num}' ")
            train_end_date = pd.to_datetime(
                df.iloc[train[-1]][time_col],
                format=date_format,
                infer_datetime_format=True)
            pipeline_params["train_end_date"] = train_end_date

            start_time = timeit.default_timer()
            pipeline_result = forecast_pipeline(**pipeline_params)
            runtime = timeit.default_timer() - start_time

            pipeline_output = dict(
                runtime_sec=round(runtime, 3),
                pipeline_result=pipeline_result)
            rolling_evaluation[f"split_{split_num}"] = pipeline_output

            log_message(f"Completed evaluation for split {split_num}.", LoggingLevelEnum.DEBUG)

    return rolling_evaluation
