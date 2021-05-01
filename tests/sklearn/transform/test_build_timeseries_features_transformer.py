from datetime import datetime

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from greykite.common.constants import VALUE_COL
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.sklearn.transform.build_timeseries_features_transformer import BuildTimeseriesFeaturesTransformer


def test_BuildTimeseriesFeaturesTransformer_1():
    """Checks if the transformer class returns same output as build_time_features_df"""
    date_list = pd.date_range(
        start=datetime(2019, 1, 1),
        periods=100,
        freq="H").tolist()
    df = pd.DataFrame({"ts": date_list})

    timeseries_transform = BuildTimeseriesFeaturesTransformer(time_col="ts")
    result = timeseries_transform.fit_transform(df)
    features_ts = build_time_features_df(dt=df["ts"], conti_year_origin=2019)
    expected = pd.concat([df, features_ts], axis=1)
    assert result.equals(expected)


def test_BuildTimeseriesFeaturesTransformer_2():
    """Manually checks if the transformer class returns are correct"""
    df = pd.DataFrame({
        "time": [datetime(2018, 1, 1, 1, 0, 1),
                 datetime(2018, 1, 2, 2, 0, 2),
                 datetime(2018, 1, 3, 4, 0, 10),  # intentionally out of order
                 datetime(2018, 1, 4, 10, 0, 4)],
        VALUE_COL: [1, 2, 3, 4]
    })

    timeseries_transform = BuildTimeseriesFeaturesTransformer(time_col="time")

    with pytest.raises(NotFittedError, match="This instance is not fitted yet"):
        timeseries_transform.transform(df)

    result = timeseries_transform.fit_transform(df)
    assert result["year"].equals(pd.Series([2018, 2018, 2018, 2018]))
    assert result["month"].equals(pd.Series([1, 1, 1, 1]))
    assert result["dom"].equals(pd.Series([1, 2, 3, 4]))
    assert result["hour"].equals(pd.Series([1, 2, 4, 10]))
