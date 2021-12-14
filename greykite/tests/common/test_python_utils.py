import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pytest
import scipy

from greykite.common import constants as cst
from greykite.common.evaluation import ElementwiseEvaluationMetricEnum
from greykite.common.python_utils import apply_func_to_columns
from greykite.common.python_utils import assert_equal
from greykite.common.python_utils import dictionaries_values_to_lists
from greykite.common.python_utils import dictionary_values_to_lists
from greykite.common.python_utils import flatten_list
from greykite.common.python_utils import get_integer
from greykite.common.python_utils import get_pattern_cols
from greykite.common.python_utils import group_strs_with_regex_patterns
from greykite.common.python_utils import ignore_warnings
from greykite.common.python_utils import mutable_field
from greykite.common.python_utils import reorder_columns
from greykite.common.python_utils import unique_dict_in_list
from greykite.common.python_utils import unique_elements_in_list
from greykite.common.python_utils import unique_in_list
from greykite.common.python_utils import update_dictionaries
from greykite.common.python_utils import update_dictionary


def test_update_dictionary():
    """Tests for update_dictionary"""
    # no conflicts, takes union
    default_dict = {"a": 1, "d": 4}
    overwrite_dict = {"b": 2, "c": 3}
    merged_dict = update_dictionary(
        default_dict,
        overwrite_dict=overwrite_dict)
    assert merged_dict == {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4}

    # overwrite takes precedence in conflicts
    default_dict = {"a": 1, "d": 4}
    overwrite_dict = {"a": 2, "c": 3}
    merged_dict = update_dictionary(
        default_dict,
        overwrite_dict=overwrite_dict)
    assert merged_dict == {
        "a": 2,
        "c": 3,
        "d": 4}

    # overwrite can be None or {}
    default_dict = {"a": 1, "b": 4}
    merged_dict = update_dictionary(
        default_dict,
        overwrite_dict=None)
    assert merged_dict == default_dict
    merged_dict = update_dictionary(
        default_dict,
        overwrite_dict={})
    assert merged_dict == default_dict

    default_dict = {"a": 1, "b": 4, "c": 5}
    overwrite_dict = {"a": 1, "b": 4}
    update_dictionary(
        default_dict,
        overwrite_dict=overwrite_dict,
        allow_unknown_keys=False)
    # overwrite cannot be a strict superset of default
    with pytest.raises(ValueError, match=r"Unexpected key\(s\) found"):
        overwrite_dict = {"a": 1, "b": 4, "d": 1}
        update_dictionary(
            default_dict,
            overwrite_dict=overwrite_dict,
            allow_unknown_keys=False)


def test_update_dictionaries():
    """Tests for update_dictionaries"""
    # overwrite takes precedence in conflicts
    default_dict = {"a": 1, "d": 4}
    overwrite_dict = {"a": 2, "c": 3}
    merged_dict = update_dictionaries(
        default_dict,
        overwrite_dicts=overwrite_dict)
    assert merged_dict == {
        "a": 2,
        "c": 3,
        "d": 4}

    # overwrite can be None or {}
    default_dict = {"a": 1, "b": 4}
    merged_dict = update_dictionaries(
        default_dict,
        overwrite_dicts=None)
    assert merged_dict == default_dict
    merged_dict = update_dictionaries(
        default_dict,
        overwrite_dicts={})
    assert merged_dict == default_dict

    default_dict = {"a": 1, "b": 4, "c": 5}
    overwrite_dict = {"a": 1, "b": 4}
    update_dictionaries(
        default_dict,
        overwrite_dicts=overwrite_dict,
        allow_unknown_keys=False)
    # overwrite cannot be a strict superset of default
    with pytest.raises(ValueError, match=r"Unexpected key\(s\) found"):
        overwrite_dict = {"a": 1, "b": 4, "d": 1}
        update_dictionaries(
            default_dict,
            overwrite_dicts=overwrite_dict,
            allow_unknown_keys=False)

    # can provide a list of dictionaries
    default_dict = {"a": 1, "d": 4}
    overwrite_dict = [
        {"a": 1, "b": 4},
        None,
        {},
        {"a": 1, "b": 4, "d": 1}
    ]
    merged_dict = update_dictionaries(
        default_dict,
        overwrite_dicts=overwrite_dict,
        allow_unknown_keys=True)
    assert merged_dict == [
        {"a": 1, "b": 4, "d": 4},
        default_dict,
        default_dict,
        {"a": 1, "b": 4, "d": 1}
    ]

    # ValueError is raised when a list of dictionaries is passed.
    default_dict = {"a": 1, "b": 4, "c": 5}
    overwrite_dict = [{"a": 1, "b": 4, "d": 1}]
    with pytest.raises(ValueError, match=r"Unexpected key\(s\) found"):
        update_dictionaries(
            default_dict,
            overwrite_dicts=overwrite_dict,
            allow_unknown_keys=False)


def test_unique_elements_in_list():
    """Tests unique_elements_in_list"""
    items = []
    assert unique_elements_in_list(items) == []

    items = ["a"]
    assert unique_elements_in_list(items) == ["a"]

    items = ["a", "b", "c", "b", "a"]
    assert unique_elements_in_list(items) == ["a", "b", "c"]

    items = ["c", "a", "e", "b", "d", "a"]
    assert unique_elements_in_list(items) == ["c", "a", "e", "b", "d"]


def test_unique_dict_in_list():
    items = []
    assert unique_dict_in_list(items) == []
    items = [
        dict(a=1, b="a", c=None, d=[1, 2, 3], e={"a": [1, 2, 3], "b": "s"}),
        dict(a=1, b="a", c=None, d=[1, 2, 3], e={"a": [1, 2, 3], "b": "d"}),
        dict(a=1, b="a", c=None, d=[1, 2, 3], e={"a": [1, 2, 3], "b": "s"})
    ]
    result = unique_dict_in_list(items)
    assert result == [
        dict(a=1, b="a", c=None, d=[1, 2, 3], e={"a": [1, 2, 3], "b": "s"}),
        dict(a=1, b="a", c=None, d=[1, 2, 3], e={"a": [1, 2, 3], "b": "d"}),
    ]


def test_get_pattern_cols():
    cols = ["train_MSE", "test_MSE"]

    pattern_cols = get_pattern_cols(cols=cols, pos_pattern=None, neg_pattern=None)
    assert_equal(pattern_cols, [])

    pattern_cols = get_pattern_cols(cols=cols, pos_pattern="train", neg_pattern=None)
    assert_equal(pattern_cols, ["train_MSE"])

    pattern_cols = get_pattern_cols(cols=cols, pos_pattern=None, neg_pattern="train")
    assert_equal(pattern_cols, [])

    pattern_cols = get_pattern_cols(cols=cols, pos_pattern=".*", neg_pattern="train")
    assert_equal(pattern_cols, ["test_MSE"])

    pattern_cols = get_pattern_cols(cols=cols, pos_pattern="train", neg_pattern="train")
    assert_equal(pattern_cols, [])

    # `feature_cols` is a real example from the sklearn/ module,
    # generated by the following code:
    #
    # from greykite.sklearn.estimator.silverkite_estimator import \
    #     SilverkiteEstimator
    # from greykite.sklearn.estimator.testing_utils import \
    #     params_components
    # from greykite.common.testing_utils import daily_data_reg
    # params_daily = params_components()
    # # removing daily seasonality terms
    # params_daily["fs_components_df"] = pd.DataFrame({
    #     "name": ["tow", "ct1"],
    #     "period": [7.0, 1.0],
    #     "order": [4, 5],
    #     "seas_names": ["weekly", "yearly"]})
    # params_daily.pop("fit_algorithm", None)
    # params_daily.pop("fit_algorithm_params", None)
    # model = SilverkiteEstimator(**params_daily)
    # # Test get_silverkite_components function (part of fit method)
    # train_df = daily_data_reg().get("train_df").copy()
    # model.fit(train_df)
    # feature_cols = model.feature_cols
    feature_cols = ["Intercept", "dow_hr[T.2_00]", "dow_hr[T.3_00]", "dow_hr[T.4_00]",
                    "dow_hr[T.5_00]", "dow_hr[T.6_00]", "dow_hr[T.7_00]",
                    "Q('events_New Years Day')[T.event]",
                    "Q('events_Christmas Day')[T.event]",
                    "Q('events_Independence Day')[T.event]",
                    "Q('events_Thanksgiving')[T.event]", "Q('events_Labor Day')[T.event]",
                    "Q('events_Memorial Day')[T.event]",
                    "Q('events_Veterans Day')[T.event]", "Q('events_Other')[T.event]",
                    "Q('events_New Years Day_minus_1')[T.event]",
                    "Q('events_New Years Day_minus_2')[T.event]",
                    "Q('events_New Years Day_plus_1')[T.event]",
                    "Q('events_New Years Day_plus_2')[T.event]",
                    "Q('events_Christmas Day_minus_1')[T.event]",
                    "Q('events_Christmas Day_minus_2')[T.event]",
                    "Q('events_Christmas Day_plus_1')[T.event]",
                    "Q('events_Christmas Day_plus_2')[T.event]",
                    "Q('events_Independence Day_minus_1')[T.event]",
                    "Q('events_Independence Day_minus_2')[T.event]",
                    "Q('events_Independence Day_plus_1')[T.event]",
                    "Q('events_Independence Day_plus_2')[T.event]",
                    "Q('events_Thanksgiving_minus_1')[T.event]",
                    "Q('events_Thanksgiving_minus_2')[T.event]",
                    "Q('events_Thanksgiving_plus_1')[T.event]",
                    "Q('events_Thanksgiving_plus_2')[T.event]",
                    "Q('events_Labor Day_minus_1')[T.event]",
                    "Q('events_Labor Day_minus_2')[T.event]",
                    "Q('events_Labor Day_plus_1')[T.event]",
                    "Q('events_Labor Day_plus_2')[T.event]",
                    "Q('events_Memorial Day_minus_1')[T.event]",
                    "Q('events_Memorial Day_minus_2')[T.event]",
                    "Q('events_Memorial Day_plus_1')[T.event]",
                    "Q('events_Memorial Day_plus_2')[T.event]",
                    "Q('events_Veterans Day_minus_1')[T.event]",
                    "Q('events_Veterans Day_minus_2')[T.event]",
                    "Q('events_Veterans Day_plus_1')[T.event]",
                    "Q('events_Veterans Day_plus_2')[T.event]",
                    "Q('events_Other_minus_1')[T.event]",
                    "Q('events_Other_minus_2')[T.event]",
                    "Q('events_Other_plus_1')[T.event]",
                    "Q('events_Other_plus_2')[T.event]", "ct_sqrt", "ct1", "ct1:tod",
                    "regressor1", "regressor2", "sin1_tow_weekly",
                    "is_weekend[T.True]:sin1_tow_weekly", "cos1_tow_weekly",
                    "is_weekend[T.True]:cos1_tow_weekly", "sin2_tow_weekly",
                    "is_weekend[T.True]:sin2_tow_weekly", "cos2_tow_weekly",
                    "is_weekend[T.True]:cos2_tow_weekly", "sin3_tow_weekly",
                    "is_weekend[T.True]:sin3_tow_weekly", "cos3_tow_weekly",
                    "is_weekend[T.True]:cos3_tow_weekly", "sin4_tow_weekly",
                    "is_weekend[T.True]:sin4_tow_weekly", "sin1_ct1_yearly",
                    "cos1_ct1_yearly", "sin2_ct1_yearly", "cos2_ct1_yearly",
                    "sin3_ct1_yearly", "cos3_ct1_yearly", "sin4_ct1_yearly",
                    "cos4_ct1_yearly", "sin5_ct1_yearly", "cos5_ct1_yearly",
                    "changepoint0_2018_01_01_00", "changepoint1_2019_01_02_16",
                    "changepoint2_2019_01_03_00", "y_lag7", "y_avglag_7_14_21",
                    "y_avglag_7_to_14"]

    trend_cols = get_pattern_cols(feature_cols, cst.TREND_REGEX, cst.SEASONALITY_REGEX)
    expected_trend_cols = [
        "ct_sqrt",
        "ct1",
        "ct1:tod",
        "changepoint0_2018_01_01_00",
        "changepoint1_2019_01_02_16",
        "changepoint2_2019_01_03_00"]
    assert trend_cols == expected_trend_cols

    seasonality_cols = get_pattern_cols(feature_cols, cst.SEASONALITY_REGEX)
    expected_seasonality_cols = [
        "sin1_tow_weekly",
        "is_weekend[T.True]:sin1_tow_weekly",
        "cos1_tow_weekly",
        "is_weekend[T.True]:cos1_tow_weekly",
        "sin2_tow_weekly",
        "is_weekend[T.True]:sin2_tow_weekly",
        "cos2_tow_weekly",
        "is_weekend[T.True]:cos2_tow_weekly",
        "sin3_tow_weekly",
        "is_weekend[T.True]:sin3_tow_weekly",
        "cos3_tow_weekly",
        "is_weekend[T.True]:cos3_tow_weekly",
        "sin4_tow_weekly",
        "is_weekend[T.True]:sin4_tow_weekly",
        "sin1_ct1_yearly",
        "cos1_ct1_yearly",
        "sin2_ct1_yearly",
        "cos2_ct1_yearly",
        "sin3_ct1_yearly",
        "cos3_ct1_yearly",
        "sin4_ct1_yearly",
        "cos4_ct1_yearly",
        "sin5_ct1_yearly",
        "cos5_ct1_yearly"]
    assert seasonality_cols == expected_seasonality_cols

    event_cols = get_pattern_cols(feature_cols, pos_pattern=cst.EVENT_REGEX)
    expected_event_cols = [
        "Q('events_New Years Day')[T.event]",
        "Q('events_Christmas Day')[T.event]",
        "Q('events_Independence Day')[T.event]",
        "Q('events_Thanksgiving')[T.event]",
        "Q('events_Labor Day')[T.event]",
        "Q('events_Memorial Day')[T.event]",
        "Q('events_Veterans Day')[T.event]",
        "Q('events_Other')[T.event]",
        "Q('events_New Years Day_minus_1')[T.event]",
        "Q('events_New Years Day_minus_2')[T.event]",
        "Q('events_New Years Day_plus_1')[T.event]",
        "Q('events_New Years Day_plus_2')[T.event]",
        "Q('events_Christmas Day_minus_1')[T.event]",
        "Q('events_Christmas Day_minus_2')[T.event]",
        "Q('events_Christmas Day_plus_1')[T.event]",
        "Q('events_Christmas Day_plus_2')[T.event]",
        "Q('events_Independence Day_minus_1')[T.event]",
        "Q('events_Independence Day_minus_2')[T.event]",
        "Q('events_Independence Day_plus_1')[T.event]",
        "Q('events_Independence Day_plus_2')[T.event]",
        "Q('events_Thanksgiving_minus_1')[T.event]",
        "Q('events_Thanksgiving_minus_2')[T.event]",
        "Q('events_Thanksgiving_plus_1')[T.event]",
        "Q('events_Thanksgiving_plus_2')[T.event]",
        "Q('events_Labor Day_minus_1')[T.event]",
        "Q('events_Labor Day_minus_2')[T.event]",
        "Q('events_Labor Day_plus_1')[T.event]",
        "Q('events_Labor Day_plus_2')[T.event]",
        "Q('events_Memorial Day_minus_1')[T.event]",
        "Q('events_Memorial Day_minus_2')[T.event]",
        "Q('events_Memorial Day_plus_1')[T.event]",
        "Q('events_Memorial Day_plus_2')[T.event]",
        "Q('events_Veterans Day_minus_1')[T.event]",
        "Q('events_Veterans Day_minus_2')[T.event]",
        "Q('events_Veterans Day_plus_1')[T.event]",
        "Q('events_Veterans Day_plus_2')[T.event]",
        "Q('events_Other_minus_1')[T.event]",
        "Q('events_Other_minus_2')[T.event]",
        "Q('events_Other_plus_1')[T.event]",
        "Q('events_Other_plus_2')[T.event]"]
    assert event_cols == expected_event_cols


def test_assert_equal_basic():
    """Tests assert_equal"""
    # basic types
    assert_equal(None, None)
    assert_equal(True, True)
    assert_equal(False, False)
    assert_equal("string", "string")
    assert_equal(1, 1)
    assert_equal(1.234, 1.234)
    assert_equal(3.0, 4.0, rel=0.25)
    assert_equal(4.0, 3.0, rel=0.25)
    with pytest.raises(AssertionError):
        assert_equal(3.0, 4.0, rel=0.24)
    assert_equal([1.234], [1.234])
    assert_equal(
        ["str", 0, 1.234, 2.345],
        ["str", 0, 1.234, 2.345])

    # series
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert_equal(s1, s2)

    # pandas Dataframe, kwargs
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
    assert_equal(df1, df1)
    assert_equal(df1, df2, check_dtype=False)

    # index
    ind1 = pd.Index([1.0, 2.0, 3.0, 4.0, 5.0])
    ind2 = pd.Index([1.0, 2.0, 3.0, 4.0, 5.0])
    assert_equal(ind1, ind2)

    # np array, `rel`
    n1 = np.array([1, 2, 3, 4])
    n2 = np.array([1, 2, 3, 4])
    assert_equal(n1, n2)

    n1 = np.array([1, 2, 3, 4.124])
    n2 = np.array([1, 2, 3, 4.123])
    assert_equal(n1, n2, rel=1e-3)

    # dictionary
    assert_equal(
        {"df": df1},
        {"df": df2},
        check_dtype=False)

    assert_equal(
        actual={
            "a": {
                "a_a": {
                    "a_a_a": df1
                },
                "a_b": s1
            },
            "b": "string",
            "c": 1
        },
        expected={
            "a": {
                "a_a": {
                    "a_a_a": df2
                },
                "a_b": s2
            },
            "b": "string",
            "c": 1
        },
        check_dtype=False)

    # list
    assert_equal(
        [1, 2, 3],
        [1, 2, 3])

    assert_equal(
        [1, 2, 3, 4.12345],
        [1, 2, 3, 4.1233],
        rel=1e-2)

    assert_equal(
        [3, 2, 1],
        [1, 2, 3],
        ignore_list_order=True)

    assert_equal(
        [s1, df1, "df"],
        [s2, df1, "df"])

    assert_equal(
        [s1, df1, "df"],
        [s2, df2, "df"],
        check_dtype=False)

    # negative cases
    with pytest.raises(
            AssertionError,
            match=r"Actual does not match expected\. Actual: 1\. Expected: 2\."):
        assert_equal([1], [2])

    with pytest.raises(
            AssertionError,
            match=r"Attribute \"dtype\" are different"):
        assert_equal(df1, df2)

    with pytest.raises(
            AssertionError,
            match=r"Actual does not match expected\. Actual: 4\.123\. Expected: 4\.124\."):
        assert_equal(
            [1, 2, 3, 4.123],
            [1, 2, 3, 4.124])

    with pytest.raises(
            AssertionError,
            match=r"Not equal to tolerance rtol=1e-05, atol=0"):
        assert_equal(
            np.array([1, 2, 3, 4.12345]),
            np.array([1, 2, 3, 4.123]))

    with pytest.raises(
            AssertionError,
            match=r"Not equal to tolerance rtol=1e-07, atol=0"):
        assert_equal(
            np.array([1, 2, 3, 4.123]),
            np.array([1, 2, 3, 4.120]),
            rel=1e-7)

    assert_equal(
        np.array([1, 2, 3, 4.123]),
        np.array([1, 2, 3, 4.120]),
        rel=1e-2)

    with pytest.raises(
            AssertionError,
            match=r"Actual should be numeric, found None"):
        assert_equal(
            actual=None,
            expected=1)

    with pytest.raises(
            AssertionError,
            match=r"Actual should be None, found 1"):
        assert_equal(
            actual=1,
            expected=None)

    with pytest.raises(
            AssertionError,
            match=r"Actual should be a pandas Series"):
        assert_equal(n1, s1)

    with pytest.raises(
            AssertionError,
            match=r"Actual should be a numpy array"):
        assert_equal(s1, n1)

    # properly identifies location in dict
    with pytest.raises(
            AssertionError,
            match=r"Error at dictionary location: dict\['a'\]\['a'\]\['a'\].\n"
                  r"Actual should be a pandas DataFrame, found df1\."):
        assert_equal(
            actual={
                "a": {
                    "a": {
                        "a": "df1"  # mismatch
                    },
                    "b": s1
                },
                "b": "string",
                "c": 1
            },
            expected={
                "a": {
                    "a": {
                        "a": df2
                    },
                    "b": s2
                },
                "b": "string",
                "c": 1
            })

    with pytest.raises(
            AssertionError,
            match=r"Error at dictionary location: dict\['d'\].\n"
                  r"Actual should be a pandas DataFrame, found df1\."):
        assert_equal(
            actual=[1, {
                "a": {
                    "a": {
                        "a": df1
                    },
                    "b": s1
                },
                "b": "string",
                "c": 1,
                "d": "df1"  # mismatch
            }],
            expected=[1, {
                "a": {
                    "a": {
                        "a": df1
                    },
                    "b": s2
                },
                "b": "string",
                "c": 1,
                "d": df1
            }])


def test_assert_equal_ignore_keys():
    """Tests assert_equal ignore_keys parameter"""
    actual = {
        "a": {
            "a": {
                "a": 1
            },
            "b": 2,
            "c": "3"  # matches expected
        },
        "b": 3,
        "c": [{
            "a": 4
        }]
    }
    expected = {
        "a": {
            "a": {
                "a": "1"
            },
            "b": "2",
            "c": "3"

        },
        "b": "3",
        "c": [{
            "a": "4"
        }]
    }

    # ignores keys at the first level
    assert_equal(
        actual,
        expected,
        ignore_keys={"a": None, "b": None, "c": None}
    )
    # ignores keys at the second level.
    # the value can be anything that's not a dict
    assert_equal(
        actual,
        expected,
        ignore_keys={"a": {"a": False, "b": True}, "b": None, "c": "not_a_dict"}
    )
    # ignores keys at the third level
    assert_equal(
        actual,
        expected,
        ignore_keys={"a": {"a": {"a": None}, "b": None}, "b": None, "c": None}
    )
    with pytest.raises(AssertionError, match=r"Error at dictionary location: dict\['c'\]\['a'\]"):
        # "a" is ignored, but ["c"]["a"] is not (keys are not confused)
        # the error message on ["c"]["a"] is unconventional, since dict["c"] is a list
        assert_equal(
            actual,
            expected,
            ignore_keys={"a": {"a": {"a": None}, "b": None}, "b": None}
        )

    with pytest.raises(AssertionError, match=r"Error at dictionary location: dict\['a'\]\['a'\]\['a'\]"):
        # ["a"]["a"]["b"] is ignored, but not ["a"]["a"]["a"]
        assert_equal(
            actual,
            expected,
            ignore_keys={"a": {"a": {"b": None}, "b": None}, "b": None, "c": None}
        )

    with pytest.raises(AssertionError, match=r"Error at dictionary location: dict\['a'\]\['b'\]"):
        # ["a"]["a"]["a"] is ignored, but not ["a"]["b"]
        assert_equal(
            actual,
            expected,
            ignore_keys={"a": {"a": {"a": None}}, "b": None, "c": None}
        )

    with pytest.raises(AssertionError, match=r"Error at dictionary location: dict\['c'\]\['a'\]"):
        with pytest.warns(Warning) as record:
            # can't ignore keys within in a list (because there are no keys to ignore)
            assert_equal(
                actual,
                expected,
                ignore_keys={"a": {"a": {"a": None}, "b": None}, "b": None, "c": {"a": None}}
            )
            assert r"At dictionary location: dict['c']. `ignore_keys` is {'a': None}, " \
                   r"but found a list. No keys will be ignored" in record[0].message.args[0]

    with pytest.raises(AssertionError, match=r"Actual should be a list or tuple"):
        # what is ignored is based on expected, not actual
        assert_equal(
            actual,
            [expected],
            ignore_keys={"a": {"a": {"a": None}, "b": None}, "b": None}
        )


def test_dictionary_values_to_lists():
    """Tests dictionary_values_to_lists"""
    exponential_distribution = scipy.stats.expon(scale=.1)
    hyperparameter_grid = {
        "param1": [],
        "param2": [None],
        "param3": ["value1", "value2"],
        "param4": [[1, 2], [3, 4]],
        "param5": [1],
        "param6": [[1], 2, [3]],
        "param7": [[1], None, [3]],
        "param8": (1, 2, 3),
        "param9": None,
        "param10": [None, ["US", "UK"]],
        "param11": [None, "auto", "special_value"],
        "param12": [None, "auto", ["US", "UK"]],
        "param13": 1.5,
        "param14": exponential_distribution,
        "param15": {"k": "v"},
        "param16": np.array([1, 2, 3]),
        "param17": pd.DataFrame([1, 2, 3]),
    }
    original_grid = hyperparameter_grid.copy()

    # 1) None for `hyperparameters_list_type`
    result = dictionary_values_to_lists(hyperparameter_grid)
    expected_grid = {
        "param1": [],
        "param2": [None],
        "param3": ["value1", "value2"],
        "param4": [[1, 2], [3, 4]],
        "param5": [1],
        "param6": [[1], 2, [3]],
        "param7": [[1], None, [3]],
        "param8": (1, 2, 3),
        "param9": [None],
        "param10": [None, ["US", "UK"]],
        "param11": [None, "auto", "special_value"],
        "param12": [None, "auto", ["US", "UK"]],
        "param13": [1.5],
        "param14": exponential_distribution,
        "param15": [{"k": "v"}],
        "param16": [np.array([1, 2, 3])],
        "param17": [pd.DataFrame([1, 2, 3])]
    }
    assert_equal(result, expected_grid)
    # Original dictionary is not modified
    assert_equal(hyperparameter_grid, original_grid)

    # 2) Set for `hyperparameters_list_type` (param1 to param12)
    hyperparameters_list_type = set({
        f"param{i+1}" for i in range(12)
    })
    result = dictionary_values_to_lists(
        hyperparameter_grid,
        hyperparameters_list_type=hyperparameters_list_type
    )
    # Param 4, 9, 11 are already in the proper format
    expected_grid = {
        "param1": [[]],
        "param2": [None],
        "param3": [["value1", "value2"]],
        "param4": [[1, 2], [3, 4]],
        "param5": [[1]],
        "param6": [[[1], 2, [3]]],
        "param7": [[1], None, [3]],
        "param8": [(1, 2, 3)],
        "param9": [None],
        "param10": [None, ["US", "UK"]],
        "param11": [[None, "auto", "special_value"]],  # converted to list
        "param12": [[None, "auto", ["US", "UK"]]],  # converted to list
        "param13": [1.5],
        "param14": exponential_distribution,
        "param15": [{"k": "v"}],
        "param16": [np.array([1, 2, 3])],
        "param17": [pd.DataFrame([1, 2, 3])]
    }
    assert_equal(result, expected_grid)
    # Original dictionary is not modified
    assert_equal(hyperparameter_grid, original_grid)

    # 3) Dict for `hyperparameters_list_type`
    hyperparameters_list_type = {
        k: [None] for k in hyperparameters_list_type
    }
    hyperparameters_list_type["param11"] = [None, "auto", "special_value"]
    hyperparameters_list_type["param12"] = [None, "auto", "special_value"]
    result = dictionary_values_to_lists(
        hyperparameter_grid,
        hyperparameters_list_type=hyperparameters_list_type
    )
    expected_grid = {
        "param1": [[]],
        "param2": [None],
        "param3": [["value1", "value2"]],
        "param4": [[1, 2], [3, 4]],
        "param5": [[1]],
        "param6": [[[1], 2, [3]]],
        "param7": [[1], None, [3]],
        "param8": [(1, 2, 3)],
        "param9": [None],
        "param10": [None, ["US", "UK"]],
        "param11": [None, "auto", "special_value"],
        "param12": [None, "auto", ["US", "UK"]],
        "param13": [1.5],
        "param14": exponential_distribution,
        "param15": [{"k": "v"}],
        "param16": [np.array([1, 2, 3])],
        "param17": [pd.DataFrame([1, 2, 3])]
    }
    assert_equal(result, expected_grid)
    # Original dictionary is not modified
    assert_equal(hyperparameter_grid, original_grid)

    # Checks exception
    with pytest.raises(
            ValueError,
            match=r"The value for param1 must be a list, tuple, or one of \[None\], found {'k': 'v'}."):
        hyperparameter_grid = {"param1": {"k": "v"}}
        dictionary_values_to_lists(
            hyperparameter_grid,
            hyperparameters_list_type={"param1"})


def test_dictionaries_values_to_lists():
    """Tests dictionaries_values_to_lists"""
    hyperparameter_grid = {
        "param1": [],
        "param2": [],
        "param3": [None],
        "param4": [None],
    }
    original_grid = hyperparameter_grid.copy()
    result = dictionaries_values_to_lists(
        hyperparameter_grid,
        hyperparameters_list_type={"param2", "param4"})
    expected_grid = {
        "param1": [],
        "param2": [[]],
        "param3": [None],
        "param4": [None],
    }
    assert_equal(result, expected_grid)
    assert_equal(hyperparameter_grid, original_grid)

    hyperparameter_grids = [
        hyperparameter_grid,
        hyperparameter_grid
    ]
    original_grid = hyperparameter_grids.copy()
    result = dictionaries_values_to_lists(
        hyperparameter_grids,
        hyperparameters_list_type={"param2", "param4"})
    assert_equal(result, [expected_grid, expected_grid])
    assert_equal(hyperparameter_grids, original_grid)


def test_unique_in_list():
    """Tests unique in list"""
    assert unique_in_list(None) is None
    assert unique_in_list([]) is None
    assert unique_in_list([0, 1, 2]) == [0, 1, 2]
    assert unique_in_list([0, 1, 2, [3]]) == [0, 1, 2, 3]
    assert unique_in_list(
        [0, 1, 2, [3]],
        ignored_elements=(1, 3)) == [0, 2]
    assert unique_in_list(
        [0, 0, [None], [[None]], [[[None]]]],
        ignored_elements=(None,)) == [0]
    assert unique_in_list(
        [0, 0, [None], [[None]]],
        ignored_elements=(0, None,)) is None


def test_flatten_list():
    """Tests flatten_list"""
    assert flatten_list([[]]) == []
    assert flatten_list([[0, 1, 2, 3]]) == [0, 1, 2, 3]
    assert flatten_list([[0], [1], [2], [3]]) == [0, 1, 2, 3]
    assert flatten_list([[0, 1], [2, 3]]) == [0, 1, 2, 3]


def test_reorder_columns():
    """Tests reorder_columns"""
    df = pd.DataFrame(np.random.randn(3, 4), columns=list("abcd"))
    reordered_df = reorder_columns(df, order_dict=None)
    assert_equal(df, reordered_df)

    order_dict = {
        "a": 3,
        "b": -1,
        "c": 5,
        "d": 2}
    reordered_df = reorder_columns(df, order_dict=order_dict)
    assert_equal(df[["b", "d", "a", "c"]], reordered_df)


def test_apply_func_to_columns():
    """Tests apply_func_to_columns function"""
    # cols can be index values
    row = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    fn = apply_func_to_columns(
        row_func=ElementwiseEvaluationMetricEnum.Residual.get_metric_func(),
        cols=["a", "b"])
    assert fn(row) == row["a"] - row["b"]

    fn = apply_func_to_columns(
        row_func=ElementwiseEvaluationMetricEnum.Residual.get_metric_func(),
        cols=["c", "b"])
    assert fn(row) == row["c"] - row["b"]

    # cols can be to dict keys
    row = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert fn(row) == row["c"] - row["b"]

    # cols can be list indices
    row = [4.0, 8.0, 5.0]
    fn = apply_func_to_columns(
        row_func=ElementwiseEvaluationMetricEnum.Residual.get_metric_func(),
        cols=[2, 1])
    assert fn(row) == row[2] - row[1]


def test_get_integer():
    """Tests get_integer function"""
    with pytest.warns(Warning) as record:
        assert get_integer(None, "val", min_value=10, default_value=20) == 20
        assert get_integer(11, "val", min_value=10, default_value=20) == 11
        assert get_integer(10.5, "val", min_value=10, default_value=20) == 10
        assert "val converted to integer 10 from 10.5" in record[0].message.args[0]

    with pytest.raises(ValueError, match="val must be an integer"):
        get_integer("q", "val")

    with pytest.raises(ValueError, match="val must be >= 1"):
        get_integer(0, "val", min_value=1)

    with pytest.raises(ValueError, match="val must be >= 1"):
        get_integer(None, "val", min_value=1, default_value=0)


def test_mutable_field():
    @dataclass
    class D:
        x: List = mutable_field([1, 2, 3])
    assert D().x is not D().x
    assert D().x == [1, 2, 3]


def test_ignore_warnings():
    """Tests ignore_warnings"""
    # warnings suppressed
    @ignore_warnings(FutureWarning)
    def func(a, b, c=1):
        warnings.warn("warning message", FutureWarning)
        return f"{a} {b} {c}"

    with pytest.warns(None):
        assert func(a=1, b=2) == "1 2 1"

    # warnings not suppressed
    @ignore_warnings(ImportWarning)
    def func2(a, b, c=1):
        warnings.warn("warning message", FutureWarning)
        return f"{a} {b} {c}"

    with pytest.warns(FutureWarning) as record:
        assert func2(a=1, b=2, c=3) == "1 2 3"
        assert "warning message" in record[0].message.args[0]


def test_group_strs_with_regex_patterns():
    """Tests ``group_strs_with_regex_patterns``."""
    # Example 1
    strings = ["sd", "sd1", "rr", "urr", "sd2", "uu"]
    regex_patterns = ["sd2", "sd", "urr", "rr"]
    result = group_strs_with_regex_patterns(
        strings=strings,
        regex_patterns=regex_patterns)

    assert result == {
        "str_groups": [["sd2"], ["sd", "sd1"], ["urr"], ["rr"]],
        "remainder": ["uu"]}

    # Example 2
    strings = ["sd", "sd1", "rr", "urr", "sd2", "uu", "11", "12"]
    # First pattern extracts strings which only include digits
    regex_patterns = ["[0-9]+", "sd", "urr", "rr"]
    result = group_strs_with_regex_patterns(
        strings=strings,
        regex_patterns=regex_patterns)

    assert result == {
        "str_groups": [["11", "12"], ["sd", "sd1", "sd2"], ["urr"], ["rr"]],
        "remainder": ["uu"]}

    # Example 3
    strings = ["sd", "sd1", "rr", "urr", "sd2", "uu", "11", "12"]
    # First regex extracts strings which include digits
    regex_patterns = [r".*\d", "sd", "urr", "rr"]
    result = group_strs_with_regex_patterns(
        strings=strings,
        regex_patterns=regex_patterns)

    assert result == {
        "str_groups": [["sd1", "sd2", "11", "12"], ["sd"], ["urr"], ["rr"]],
        "remainder": ["uu"]}

    # Example 4
    strings = ["sd", "sd1", "rr", "urr", "sd2", "sd22", "uu", "11", "12"]
    # First regex looks for an exact match with "sd2", therefore
    # "sd22" is not going to be in the first group
    regex_patterns = [r"^sd2$", r"^sd\d+$"]

    result = group_strs_with_regex_patterns(
        strings=strings,
        regex_patterns=regex_patterns)

    assert result == {
        "str_groups": [["sd2"], ["sd1", "sd22"]],
        "remainder": ["sd", "rr", "urr", "uu", "11", "12"]}
