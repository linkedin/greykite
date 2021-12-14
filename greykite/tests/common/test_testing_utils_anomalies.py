import datetime

import numpy as np
import pandas as pd

from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils_anomalies import calc_quantiles_simulated_df
from greykite.common.testing_utils_anomalies import contaminate_df_with_anomalies
from greykite.common.testing_utils_anomalies import generate_anomaly_blocks
from greykite.common.testing_utils_anomalies import generate_df_with_anomalies_sim_based


def test_generate_anomaly_blocks():
    res = generate_anomaly_blocks(
            timeseries_length=100,
            block_number=5,
            mean_block_size=5)
    assert len(res["anomaly_block_list"]) == res["block_number"]
    assert res["anomaly_block_list"][-1][-1] <= 100


def test_contaminate_df_with_anomalies():
    # data size
    n = 2000
    res = generate_df_for_tests(
            freq="1D",
            periods=n,
            train_start_date=datetime.datetime(2018, 7, 1))
    df = res["df"]

    res = generate_anomaly_blocks(
            timeseries_length=n,
            block_number=20,
            mean_block_size=5)

    anomaly_block_list = res["anomaly_block_list"]

    df = contaminate_df_with_anomalies(
        df=df,
        anomaly_block_list=anomaly_block_list,
        delta_range_lower=5,
        delta_range_upper=6,
        value_col="y",
        min_admissible_value=None,
        max_admissible_value=None
    )
    assert df.shape == (2000, 4)
    assert list(df.columns) == ["ts", "y", "contaminated_y", "is_anomaly"]
    assert not df.isna().any().any()


def test_calc_quantiles_simulated_df():
    def sim_df_func():
        return pd.DataFrame({"y": np.random.uniform(3, 5, 100)})

    quantiles_df = calc_quantiles_simulated_df(
        sim_df_func=sim_df_func,
        quantiles=[0.25, 0.75],
        simulation_num=50)

    assert quantiles_df.shape == (100, 2)

    def sim_df_func(x):
        return pd.DataFrame({"y": np.random.uniform(x, 5, 100)})

    quantiles_df = calc_quantiles_simulated_df(
        sim_df_func=sim_df_func,
        quantiles=[0.25, 0.75],
        simulation_num=50,
        x=1)

    assert quantiles_df.shape == (100, 2)


def test_generate_df_with_anomalies_sim_based():
    res = generate_df_with_anomalies_sim_based(
        freq="5min",
        periods=24*12*10,
        block_number=10,
        mean_block_size=5)
    df = res["df"]
    assert df.shape == (24*12*10, 6)
    assert not df.isna().any().any()
    quantiles_df = res["quantiles_df"]
    assert quantiles_df.shape == (24*12*10, 2)
    assert not quantiles_df.isna().any().any()
