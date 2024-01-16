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
# original author: Sayan Patra, Reza Hosseini


import numpy as np
import pandas as pd

from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils_anomalies import contaminate_df_with_anomalies
from greykite.common.viz.timeseries_annotate import plot_lines_markers
from greykite.detection.detector.ad_utils import get_anomaly_df


def generate_anomaly_data(
        freq,
        periods,
        anomaly_block_list,
        noise_std=10.0,
        intercept=500,
        delta_range_lower=0,
        delta_range_upper=0.2):
    """Generates dataset for anomaly detection unit tests.

    Parameters
    ----------
    freq: `str`
        Frequency of the dataset.
    periods: `int`
        Number of periods to generate.
    anomaly_block_list: `List`[`List`[`int`]]
        List of blocks of indices to insert anomalies in.
    noise_std: `float` or None, default 10
        Standard deviation of gaussian noise.
    intercept: `float` or None, default 500
        Intercept of the data generation model.
    delta_range_lower: `float` or None, default 0
        Lower boundary of the interval to choose delta from.
    delta_range_upper: `float` or None, default 0.2
        Upper boundary of the interval to choose delta from.

    Returns
    -------
    data: `dict`
        A dictionary with two keys.
            - "df": `pd.DataFrame`
                Dataset containing anomalies.
            - "anomaly_df": `pd.DataFrame`
                Dataframe with anomaly information.
    """
    df = generate_df_for_tests(
        freq=freq,
        periods=periods,
        noise_std=noise_std,
        intercept=intercept,
        train_start_date="2020-01-01",
        train_frac=0.99,
        seed=123
    )["df"]
    # Introduces anomalies
    df = contaminate_df_with_anomalies(
        df,
        anomaly_block_list=anomaly_block_list,
        delta_range_lower=delta_range_lower,
        delta_range_upper=delta_range_upper,
        value_col=VALUE_COL,
        min_admissible_value=None,
        max_admissible_value=None
    )
    anomaly_df = get_anomaly_df(df=df, anomaly_col=ANOMALY_COL)
    df = df.drop(columns=[VALUE_COL, ANOMALY_COL]).rename(
        columns={"contaminated_y": VALUE_COL}
    )

    return {
        "df": df,
        "anomaly_df": anomaly_df
    }


def generate_anomaly_data_daily():
    """Generates daily data to be used for end-to-end AD testing."""
    anomaly_block_list = [
        np.arange(100, 105),
        np.arange(200, 210),
        np.arange(310, 315)
    ]
    res = generate_anomaly_data(
        freq="D",
        periods=30*14,
        anomaly_block_list=anomaly_block_list
    )

    return res


def generate_anomaly_data_hourly():
    """Generates hourly data to be used for end-to-end AD testing."""
    anomaly_block_list = [
        np.arange(1000, 1050),
        np.arange(5000, 5100),
        np.arange(8000, 8050)
    ]
    res = generate_anomaly_data(
        freq="H",
        periods=24*400,
        anomaly_block_list=anomaly_block_list
    )
    # Introduces missing data
    res["df"][VALUE_COL].iloc[150:160] = np.nan
    # Changes the datetype format of time column
    res["df"][TIME_COL] = res["df"][TIME_COL].dt.strftime("%Y-%m-%d-%H")
    # Renames `df` columns
    res["df"] = res["df"].rename(
        columns={
            TIME_COL: "time",
            VALUE_COL: "value"
        }
    )

    return res


def generate_anomaly_data_weekly():
    """Generates weekly data to be used for end-to-end AD testing."""
    anomaly_block_list = [
        np.arange(50, 55),
        np.arange(110, 120)
    ]
    res = generate_anomaly_data(
        freq="W-MON",
        periods=200,
        anomaly_block_list=anomaly_block_list
    )

    return res


def sim_anomalous_data_and_forecasts(
        sample_size,
        anomaly_num,
        anomaly_magnitude=20,
        seed=None):
    """This function construct normal data and injects anomalies into the data
        (returned in ``"df"`` item of the result).
        It also creates two forecasts (given in ``"forecast_dfs"``) for the same
        data of varying quality in terms of accuracy.
        The first forecast is constructed to be more accurate.
        The fuction also returns train and test versions of these data which are
        constructed to be the first and second half of the generated data respectively.

    Parameters
    ----------
    sample_size : `int`
        The total sample size of the data generated
    anomaly_num : `int`
        The number of anomalies injected into the data
    anomaly_magnitude : `float`, default 20
        The magnitude of the anomalies injected, which is the mean of the normal
        distribution used to add the anomalies.
    seed : `int` or None, default None
        The seed used in randomization. If None, no seed is set.


    Returns
    -------
    result : `dict`
    A dictionary consisting of these items

        - ``"df"`` : `pandas.DataFrame`
            A dataframe which includes a time series with columns
            - "ts" : to denote time, ranging from 0 to ``sample_size``
            - "y" : the value of the series
            - "is_anomaly" : boolean to denote is the point is an anomaly
        - ``"forecast_dfs"`` : `list` [`pandas.DataFrame`]
            A list of two dataframes which are to be considered to come from
            a predictive model.
            These are constructed simply by adding noise to observations.
            The first forecast is more accurate by construction (less noisy).
            below.
        - ``"df_train"`` : `pandas.DataFrame`
            First half of ``"df"`` defined above, which is usually to be used in
            training step.
        - ``"forecast_dfs_train"`` : `list` [`pandas.DataFrame`]
            First half of the forecast data given in ``"forecast_dfs"`` defined
            above.
        - ``"df_test"`` : `pandas.DataFrame`
            Second half of ``"df"`` defined above, which is usually used in testing
            step.
        - ``"forecast_dfs_test"`` : `list` [`pandas.DataFrame`]
            Second half of the forecast data given in ``"forecast_dfs"`` defined
            above.
        - ``"fig"``: `plotly.graph_objects.Figure`
            A plotly interactive figure, to compare the observed data with the two
            predictions constructed in this function.

    """

    np.random.seed(seed=seed)
    y = np.arange(0, sample_size, dtype=float)
    is_anomaly = [False] * sample_size
    y_pred0 = y + np.random.normal(
        loc=0.0,
        scale=1.0,
        size=sample_size)

    y_pred1 = y + np.random.normal(
        loc=0.0,
        scale=10.0,
        size=sample_size)

    anomaly_idx = np.random.choice(
        np.arange(0, sample_size, dtype=int),
        size=anomaly_num,
        replace=False)

    anomaly_idx.sort()

    for idx in anomaly_idx:
        # Randomly introduces positive or negative anomalies
        p = np.random.uniform(low=0.0, high=1.0)
        if p > 0.5:
            y[idx] += np.random.normal(
                loc=anomaly_magnitude,
                scale=1.0)
        else:
            y[idx] += -np.random.normal(
                loc=anomaly_magnitude,
                scale=1.0)

        is_anomaly[idx] = True

    ts = range(sample_size)
    df = pd.DataFrame({
        "ts": ts,
        "y": y,
        "is_anomaly": is_anomaly})

    df0 = pd.DataFrame({
        "ts": ts,
        "y_pred": y_pred0})

    df1 = pd.DataFrame({
        "ts": ts,
        "y_pred": y_pred1})

    df_all = pd.DataFrame({
        "ts": ts,
        "y": y,
        "y_pred0": y_pred0,
        "y_pred1": y_pred1})

    forecast_dfs = {}
    forecast_dfs[0] = df0
    forecast_dfs[1] = df1

    fig = plot_lines_markers(
        df=df_all,
        x_col="ts",
        line_cols=["y", "y_pred0", "y_pred1"])

    train_size = int(sample_size / 2)
    df_train = df[:train_size]
    forecast_dfs_train = {
        k: df[:train_size] for k, df in forecast_dfs.items()}

    df_test = df[train_size:]
    forecast_dfs_test = {
        k: df[train_size:] for k, df in forecast_dfs.items()}

    return {
        "df": df,
        "forecast_dfs": forecast_dfs,
        "df_train": df_train,
        "forecast_dfs_train": forecast_dfs_train,
        "df_test": df_test,
        "forecast_dfs_test": forecast_dfs_test,
        "fig": fig}
