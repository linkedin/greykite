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
# original authors: Apratim Dey, Reza Hosseini, Sayan Patra
"""Utility functions for simulating anomalies."""

import datetime

import numpy as np
import pandas as pd

from greykite.common.testing_utils import generate_df_for_tests


def generate_anomaly_blocks(
        timeseries_length,
        block_number,
        mean_block_size=5):
    """Returns blocks of indices to insert anomalies into.

    :param timeseries_length: int
        length of the time series into which we want to inject anomalies
    :param block_number: int
        initial number of blocks; may change in the output
    :param mean_block_size: float
        initial average number of indices per block; may change in the output

    :return: Dict[list, any]
        contains a list of blocks of indices, number of blocks and a list of size of blocks
    """
    anomaly_start = np.random.choice(timeseries_length-1, block_number, replace=False)
    anomaly_start = np.sort(anomaly_start)
    interval_length = np.random.poisson(lam=mean_block_size, size=block_number)
    anomaly_blocks = []
    for i in range(len(anomaly_start)):
        block_lower = anomaly_start[i]
        block_upper = min(timeseries_length, anomaly_start[i]+interval_length[i]+1)
        anomaly_blocks.append(list(range(block_lower, block_upper)))
    anomaly_block_list = []
    anomaly_block_list.append(anomaly_blocks[0])
    for j in range(1, len(anomaly_blocks)):
        if anomaly_blocks[j][0] <= anomaly_block_list[-1][-1] + 1:
            anomaly_block_list[-1] = list(np.sort(np.array((list(set(anomaly_block_list[-1] + anomaly_blocks[j]))))))
        else:
            anomaly_block_list.append(anomaly_blocks[j])

    return {"anomaly_block_list": anomaly_block_list,
            "block_number": len(anomaly_block_list),
            "block_size": [len(anomaly_block_list[x]) for x in range(len(anomaly_block_list))]}


def contaminate_df_with_anomalies(
        df,
        anomaly_block_list,
        delta_range_lower,
        delta_range_upper,
        value_col="y",
        min_admissible_value=None,
        max_admissible_value=None):
    """Contaminate a dataframe with anomalies. If original value is y, the anomalous value is (1 +/- delta)y,
    the + or - chosen randomly.

    :param df: pd.DataFrame
        dataframe with values in column named "y"
    :param anomaly_block_list: list
        list of blocks of indices to insert anomalies in
    :param delta_range_lower: float
        lower boundary of the interval to choose delta from
    :param delta_range_upper: float
        upper boundary of the interval to choose delta from
    :param value_col: str
        The value columns which is to be contaminated
    :param min_admissible_value: Optional[float]
        minimum admissible value in df["y"]
    :param max_admissible_value: Optional[float]
        maximum admissible value in df["y"]

    :return: pd.DataFrame
        contains the dataframe df with two columns appended:
            "contaminated_y": values from column "y" changed to have outliers in the blocks given by anomaly_block_list
            "is_anomaly": 0 for clean point, 1 for outlier
    """
    y = np.array(df[value_col])
    is_anomaly = np.zeros(df.shape[0], dtype=float)
    for i in range(len(anomaly_block_list)):
        index_set = anomaly_block_list[i]
        # generate a random sign: either 1 or -1
        s = 2*np.random.binomial(1, 0.5, 1) - 1
        for j in index_set:
            multiplier = 1 + (s*np.random.uniform(delta_range_lower, delta_range_upper, 1))
            y[j] = y[j]*multiplier
            if min_admissible_value is not None:
                y[j] = max(min_admissible_value, y[j])
            if max_admissible_value is not None:
                y[j] = min(max_admissible_value, y[j])
            is_anomaly[j] = 1
    df["contaminated_y"] = y
    df["is_anomaly"] = is_anomaly
    return df


def calc_quantiles_simulated_df(
        sim_df_func,
        quantiles=[0.25, 0.75],
        simulation_num=50,
        **params):

    """Calculates quantiles corresponding to probs by simulating time series with specified parameters
    :param sim_df_func: callable
        a function which simulates a dataframe
    :param quantiles: List[float]
        list of probabilities to compute quantiles at
    :param simulation_num: int
        number of simulations to calculate quantiles
    :param: **params
        parameters of ``sim_df_func``

    :return: pd.DataFrame
        contains quantiles corresponding to probs computed at each time point
    """

    df = [sim_df_func(**params)["y"] for x in range(simulation_num)]
    df = pd.DataFrame(df)
    quantiles_df = np.transpose(df.quantile(quantiles, 0))

    return quantiles_df


def generate_df_with_anomalies_sim_based(
        freq,
        periods,
        block_number,
        mean_block_size,
        train_start_date=datetime.datetime(2018, 7, 1),
        train_end_date=None,
        train_frac=0.8,
        conti_year_origin=None,
        noise_std=2.0,
        remove_extra_cols=True,
        autoreg_coefs=None,
        fs_coefs=[-1, 3, 4],
        growth_coef=3.0,
        growth_pow=1.1,
        intercept=10.0,
        quantiles=[0.25, 0.75],
        simulation_num=50,
        filter_coef=1.5,
        anomaly_coef=3):
    """Generates a time series data frame by simulation and estimates quantiles
        by simulation and annotates the former with outliers
    :param freq: str
        pd.date_range freq parameter, e.g. H or D
    :param periods: int
        number of periods to generate
    :param block_number: int
        initial number of blocks; may change in the output
    :param mean_block_size: float
        initial average number of indices per block; may change in the output
    :param train_start_date: datetime.datetime
        train start date
    :param train_end_date: Optional[datetime.datetime]
        train end date
    :param train_frac: Optional[float]
        fraction of data to use for training
        only used if train_end_date isn't provided
    :param noise_std: float
        standard deviation of gaussian noise
    :param conti_year_origin: float
        the time origin for continuous time variables
    :param remove_extra_cols: bool
        whether to remove extra columns besides TIME_COL, VALUE_COL
    :param autoreg_coefs: Optional[List[int]]
        The coefficients for the autoregressive terms.
        If provided the generated series denoted mathematically by Y(t) will be
        converted as follows:
        Y(t) -> Y(t) + c1 Y(t-1) + c2 Y(t-2) + c3 Y(t-3) + ...
        where autoreg_coefs = [c1, c2, c3, ...]
        In this fashion, the obtained series will have autoregressive
        properties not explained by seasonality and growth.
    :param fs_coefs: List[float]
        The fourier series coefficients used.
    :param growth_coef: float
        Multiplier for growth
    :param growth_pow: float
        Power for growth, as function of continuous time
    :param intercept: float
        Constant term added to Y(t)
    :param quantiles: List[float]
        list of probabilities to compute quantiles at
    :param simulation_num: int
        number of simulations to calculate quantiles
    :param filter_coef: float
        threshold coefficient to detect anomalies while labeling
    :param anomaly_coef: float
        coefficient of iqr while creating anomalies

    :return: Dict containing
        anomaly block list;
        df with four columns appended:
            "contaminated_y": values from column "y" changed to have outliers in the blocks given by anomaly_block_list
            "is_anomaly": 0 for clean point, 1 for outlier
            "lower": lower bound used for outlier filtering
            "upper": upper bound used for outlier filtering;
        df containing quantiles calculated through simulation
    """
    res = generate_anomaly_blocks(
        timeseries_length=periods,
        block_number=block_number,
        mean_block_size=mean_block_size)

    anomaly_block_list = res["anomaly_block_list"]

    def sim_df_func():
        return generate_df_for_tests(
            freq,
            periods,
            train_start_date,
            train_end_date,
            train_frac,
            conti_year_origin,
            noise_std,
            remove_extra_cols,
            autoreg_coefs,
            fs_coefs,
            growth_coef,
            growth_pow,
            intercept,
            seed=None)["df"]

    df = sim_df_func()
    y = np.array(df["y"])

    quantiles_df = calc_quantiles_simulated_df(
        sim_df_func=sim_df_func,
        quantiles=quantiles,
        simulation_num=simulation_num)

    iqr = quantiles_df[quantiles[-1]] - quantiles_df[quantiles[0]]
    lower = quantiles_df[quantiles[0]] - filter_coef*iqr
    upper = quantiles_df[quantiles[-1]] + filter_coef*iqr
    outlier_indices_upper_crossing = (y > upper)
    outlier_indices_lower_crossing = (y < lower)

    y[outlier_indices_upper_crossing] = np.array(quantiles_df[quantiles[-1]])[outlier_indices_upper_crossing]
    y[outlier_indices_lower_crossing] = np.array(quantiles_df[quantiles[0]])[outlier_indices_lower_crossing]

    is_anomaly = np.zeros(df.shape[0], dtype=float)

    for i in range(len(anomaly_block_list)):
        index_set = anomaly_block_list[i]
        s = 2*np.random.binomial(1, 0.5, 1)-1
        for j in index_set:
            y[j] = y[j] + (s*anomaly_coef*iqr[j])

    outlier_indices_upper_crossing = (y > upper)
    outlier_indices_lower_crossing = (y < lower)

    df["contaminated_y"] = y
    is_anomaly[outlier_indices_lower_crossing] = 1
    is_anomaly[outlier_indices_upper_crossing] = 1
    df["is_anomaly"] = is_anomaly
    df["lower"] = lower
    df["upper"] = upper

    return {
        "anomaly_block_list": anomaly_block_list,
        "df": df,
        "quantiles_df": quantiles_df}
