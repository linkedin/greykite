import numpy as np

from greykite.common.aggregation_function_enum import AggregationFunctionEnum


def test_aggregate_function_enum():
    """Tests the functions in ``AggregationFunctionEnum``."""
    array = np.array([1, 2, 6])
    assert AggregationFunctionEnum.mean.value(array) == 3
    assert AggregationFunctionEnum.median.value(array) == 2
    assert AggregationFunctionEnum.nanmean.value(array) == 3
    assert AggregationFunctionEnum.maximum.value(array) == 6
    assert AggregationFunctionEnum.minimum.value(array) == 1
    assert AggregationFunctionEnum.weighted_average.value(array) == 3
