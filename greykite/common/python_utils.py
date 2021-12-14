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
# original author: Albert Chen, Sayan Patra, Rachit Kumar, Reza Hosseini
"""Common python utility functions."""
import copy
import dataclasses
import functools
import math
import re
import warnings
from dataclasses import field

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.testing import assert_index_equal
from pandas.testing import assert_series_equal


def update_dictionary(default_dict, overwrite_dict=None, allow_unknown_keys=True):
    """Adds default key-value pairs to items in ``overwrite_dict``.

    Merges the items in ``default_dict`` and ``overwrite_dict``,
    preferring ``overwrite_dict`` if there are conflicts.

    Parameters
    ----------
    default_dict: `dict`
        Dictionary of default values.
    overwrite_dict: `dict` or None, optional, default None
        User-provided dictionary that overrides the defaults.
    allow_unknown_keys: `bool`, optional, default True
        If false, raises an error if ``overwrite_dict`` contains a key that is
        not in ``default_dict``.

    Raises
    ------
    ValueError
        if ``allow_unknown_keys`` is False and ``overwrite_dict``
        has keys that are not in ``default_dict``.

    Returns
    -------
    updated_dict : `dict`
        Updated dictionary.
        Returns ``overwrite_dicts``, with default values added
        based on ``default_dict``.
    """
    if overwrite_dict is None:
        overwrite_dict = {}

    if not allow_unknown_keys:
        extra_keys = overwrite_dict.keys() - default_dict.keys()
        if extra_keys:
            raise ValueError(f"Unexpected key(s) found: {extra_keys}. "
                             f"The valid keys are: {default_dict.keys()}")

    return dict(default_dict, **overwrite_dict)


def update_dictionaries(default_dict, overwrite_dicts=None, allow_unknown_keys=True):
    """Adds default key-value pairs to items in ``overwrite_dicts``.

    Merges the items in ``default_dict`` and ``overwrite_dicts``,
    preferring ``overwrite_dict`` if there are conflicts.

    If ``overwrite_dicts`` is a list of dictionaries, the merge is
    applied to each dictionary in the list.

    Parameters
    ----------
    default_dict: `dict`
        Dictionary of default values.
    overwrite_dicts: `dict` or None or `list` [`dict` or None], optional, default None
        User-provided dictionary that overrides the defaults,
        or a list of such dictionaries.
    allow_unknown_keys: `bool`, optional, default True
        If false, raises an error if ``overwrite_dicts`` contains a key that is
        not in ``default_dict``.

    Returns
    -------
    updated_dict : `dict` or `list` [`dict`]
        Updated dictionary of list of dictionaries.
        Returns ``overwrite_dicts``, with default values added
        to each dictionary based on ``default_dict``.
    """
    if isinstance(overwrite_dicts, (list, tuple)):
        updated_dict = [
            update_dictionary(
                default_dict,
                overwrite_dict=item,
                allow_unknown_keys=allow_unknown_keys)
            for item in overwrite_dicts]
    else:
        updated_dict = update_dictionary(
            default_dict,
            overwrite_dict=overwrite_dicts,
            allow_unknown_keys=allow_unknown_keys)
    return updated_dict


def unique_elements_in_list(array):
    """Returns the unique elements in the input list,
        preserving the original order.

    Parameters
    ----------
    array: `List` [`any`]
        List of elements.

    Returns
    -------
    unique_array : `List` [`any`]
        Unique elements in `array`, preserving the order of first appearance.
    """
    seen = set()
    result = []
    for item in array:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def unique_dict_in_list(array):
    """Returns the unique dictionaries in the input list,
    preserving the original order. Replaces ``unique_elements_in_list``
    because `dict` is not hashable.

    Parameters
    ----------
    array: `List` [`dict`]
        List of dictionaries.

    Returns
    -------
    unique_array : `List` [`dict`]
        Unique dictionaries in `array`, preserving the order of first appearance.
    """
    if not array:
        return array
    result = []
    # Brute force.
    # Avoids comparing json dumped ordered dictionary.
    # The reason is that when dictionary contains list/dict unhashable items, set does not work.
    for item in array:
        if not any([item == element for element in result]):
            result.append(item)
    return result


def get_pattern_cols(cols, pos_pattern=None, neg_pattern=None):
    """Get columns names from a list that matches ``pos_pattern``,
    but does not match ``neg_pattern``.

    If a column name matches both ``pos_pattern`` and ``neg_pattern``, it is excluded.

    Parameters
    ----------
    cols : `List` ['str']
        Usually column names of a DataFrame.
    pos_pattern : regular expression
        If column name matches this pattern, it is included in the output.
    neg_pattern : regular expression
        If column name matches this pattern, it is excluded from the output.

    Returns
    -------
    pattern_cols : `List` ['str']
        List of column names that match the pattern.
    """
    if pos_pattern is None:
        pos_pattern_cols = []
    else:
        pos_regex = re.compile(pos_pattern)
        pos_pattern_cols = [col for col in cols if pos_regex.findall(col)]
    if neg_pattern is None:
        neg_pattern_cols = []
    else:
        neg_regex = re.compile(neg_pattern)
        neg_pattern_cols = [col for col in cols if neg_regex.findall(col)]

    pattern_cols = [col for col in cols if
                    col in pos_pattern_cols and col not in neg_pattern_cols]

    return pattern_cols


def assert_equal(
        actual,
        expected,
        ignore_list_order=False,
        rel=1e-5,
        dict_path="",
        ignore_keys=None,
        **kwargs):
    """Generic equality function that raises an ``AssertionError`` if the objects are not equal.

    Notes
    -----
    Works with pandas.DataFrame, pandas.Series, numpy.ndarray, str, int, float,
    bool, None, or a dictionary or list of such items, with arbitrary nesting
    of dictionaries and lists.

    Does not check equivalence of functions, or work with nested numpy arrays.

    Parameters
    ----------
    actual : `pandas.DataFrame`, `pandas.Series`, `numpy.array`, `str`, `int`,
     `float`, `bool`, `None`, or a dictionary or list of such items
        Actual value.
    expected : `pandas.DataFrame`, `pandas.Series`, `numpy.array`, `str`, `int`,
     `float`, `bool`, `None`, or a dictionary or list of such items
        Expected value to compare against.
    ignore_list_order : `bool`, optional, default False
        If True, lists are considered equal if they contain the same elements. This option is valid
        only if the list can be sorted (all elements can be compared to each other).
        If False, lists are considered equal if they contain the same elements in the same order.
    rel : `float`, optional, default 1e-5
        To check int and float, passed to ``rel`` argument of `pytest.approx`.
        To check numpy arrays, passed to ``rtol`` argument of
        `numpy.testing.assert_allclose`.
        To check pandas dataframe, series, and index, passed to ``rtol`` argument of
        `pandas.testing.assert_frame_equal`, `pandas.testing.assert_series_equal`,
        `pandas.testing.assert_index_equal`.
    dict_path : `str`, optional, default ""
        Location within nested dictionary of the original call to this function.
        User should not set this parameter.
    ignore_keys : `dict`, optional, default None
        Keys to ignore in equality comparison. This only applies if
        `expected` is a dictionary.
        Does not compare the values of these keys. However,
        still returns false if the key is not present.

        Can be a nested dictionary. Terminal keys are those whose values should
        not be compared.


        If the expected value is an nested dictionary, this dictionary can
        also be nested, with the same structure.
        For example, if expected:
        expected = {
            "k1": {
                "k1": 1,
                "k2": [1, 2, 3],
            }
            "k2": {
                "k1": "abc"
            }
        }
        Then the following ``ignore_keys`` will ignore
        dict["k1"]["k1"] and dict["k2"]["k1"] in the comparison.
        ignore_keys = {
            "k1": {
                "k1": False  # The value can be anything, the keys determine what's ignored
            },
            "k2": {
                "k1": "skip"  # The value can be anything, the keys determine what's ignored
            }
        }
        This ``ignore_keys`` will ignore
        dict["k1"] and dict["k2"]["k1"] in the comparison. "k1" is
        ignored entirely because its value is not a dictionary.
        ignore_keys = {
            "k1": None  # The value can be anything, the keys determine what's ignored
            "k2": {
                "k1": None
            }
        }
    kwargs : keyword args, optional
        Keyword args to pass to `pandas.util.testing.assert_frame_equal`,
        `pandas.util.testing.assert_series_equal`.

    Raises
    ------
    AssertionError
        If actual does not match expected.
    """
    # a message to add to all error messages
    location = f"dictionary location: {dict_path}"
    message = "" if dict_path == "" else f"Error at {location}.\n"

    if expected is None:
        if actual is not None:
            raise AssertionError(f"{message}Actual should be None, found {actual}.")
    elif isinstance(expected, pd.DataFrame):
        if not isinstance(actual, pd.DataFrame):
            raise AssertionError(f"{message}Actual should be a pandas DataFrame, found {actual}.")
        # leverages pandas assert function and add `message` to the error
        try:
            assert_frame_equal(
                actual,
                expected,
                rtol=rel,
                **kwargs)
        except AssertionError as e:
            import sys
            raise type(e)(f"{e}{message}").with_traceback(sys.exc_info()[2])
    elif isinstance(expected, pd.Series):
        if not isinstance(actual, pd.Series):
            raise AssertionError(f"{message}Actual should be a pandas Series, found {actual}.")
        try:
            assert_series_equal(
                actual,
                expected,
                rtol=rel,
                **kwargs)
        except AssertionError as e:
            import sys
            raise type(e)(f"{message}{e}").with_traceback(sys.exc_info()[2])
    elif isinstance(expected, pd.Index):
        if not isinstance(actual, pd.Index):
            raise AssertionError(f"{message}Actual should be a pandas Index, found {actual}.")
        try:
            assert_index_equal(
                actual,
                expected,
                rtol=rel,
                **kwargs)
        except AssertionError as e:
            import sys
            raise type(e)(f"{message}{e}").with_traceback(sys.exc_info()[2])
    elif isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            raise AssertionError(f"{message}Actual should be a numpy array, found {actual}.")
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rel,
            err_msg=message)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            raise AssertionError(f"{message}Actual should be a list or tuple, found {actual}.")
        if not len(actual) == len(expected):
            raise AssertionError(f"{message}Lists have different length. "
                                 f"Actual: {actual}. Expected: {expected}.")
        if ignore_list_order:
            # order doesn't matter
            actual = sorted(actual)
            expected = sorted(expected)
        # element-wise comparison
        if ignore_keys is not None:
            warnings.warn(f"At {location}. `ignore_keys` is {ignore_keys}, but found a list. "
                          f"No keys will be ignored")
        for (item_actual, item_expected) in zip(actual, expected):
            assert_equal(
                item_actual,
                item_expected,
                ignore_list_order=ignore_list_order,
                rel=rel,
                dict_path=dict_path,
                ignore_keys=None,
                **kwargs)
    elif isinstance(expected, dict):
        # dictionaries are equal if their keys and values are equal
        if not isinstance(actual, dict):
            raise AssertionError(f"{message}Actual should be a dict, found {actual}.")
        # checks the keys
        if not actual.keys() == expected.keys():
            raise AssertionError(f"{message}Dict keys do not match. "
                                 f"Actual: {actual.keys()}. Expected: {expected.keys()}.")
        # check the next level of nesting, if not ignored by `ignore_keys`
        for k, expected_item in expected.items():
            if ignore_keys is not None and k in ignore_keys.keys():
                if isinstance(ignore_keys[k], dict):
                    # specific keys within the value are ignored
                    new_ignore_keys = ignore_keys[k]
                else:
                    # the entire value is ignored
                    continue
            else:
                # the key is not ignored, so its value should be fully compared
                new_ignore_keys = None

            # appends the key to the path
            new_path = f"{dict_path}['{k}']" if dict_path != "" else f"dict['{k}']"
            assert_equal(
                actual[k],
                expected_item,
                ignore_list_order=ignore_list_order,
                rel=rel,
                dict_path=new_path,
                ignore_keys=new_ignore_keys,
                **kwargs)
    elif isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            raise AssertionError(f"{message}Actual should be numeric, found {actual}.")
        if not math.isclose(actual, expected, rel_tol=rel, abs_tol=0.0):
            raise AssertionError(f"{message}Actual does not match expected. "
                                 f"Actual: {actual}. Expected: {expected}.")
    else:
        if actual != expected:
            raise AssertionError(f"{message}Actual does not match expected. "
                                 f"Actual: {actual}. Expected: {expected}.")


def dictionary_values_to_lists(hyperparameter_dict, hyperparameters_list_type=None):
    """Given a dictionary, returns a copy whose values are
     either lists, distributions with a ``rvs`` method, or None.

    The output is suitable for hyperparameter grid search.

    Does this by converting values that do not conform into
    singleton elements inside a list.

    Parameters
    ----------
    hyperparameter_dict : `dict` [`str`, `any`]
        Dictionary of hyperparameters.

    hyperparameters_list_type : `set` [`str`] or `dict` [`str`, `list`] or None, optional, default None
        Hyperparameters that must be a `list` or other recognized value.
        e.g. ``regressor_cols`` is `list` or None, ``holiday_lookup_countries`` is `list` or "auto"
        or None.

        Thus, a flat list must become nested. E.g. ["US", "UK"] must be converted to [["US", "UK"]].

        Specifically, the values in ``hyperparameter_dict`` must be of type
        `list` [`list` or other accepted value].

            * If a set, other accepted value = [None]
            * If a dict, other accepted value is specified by the key's value, a
              list of valid options.

              For example, to allow `list` or "auto" or None, use
              ``hyperparameters_list_type={"regressor_cols": [None, "auto"]}``.

              For example, ``hyperparameters_list_type={"regressor_cols": [None]}`` is equivalent to
              ``hyperparameters_list_type={"regressor_cols"}`` using the set specification.

    Notes
    -----
    These values are unchanged:

        * [None]
        * ["value1", "value2"]
        * [[1, 2], [3, 4]]
        * [[1], None, [3]]
        * scipy.stats.expon(scale=.1)

    These values are put in a list:

        * None
        * 1
        * np.array([1, 2, 3])
        * {"k": "v"}

    These values are put in a list if their key is
    in ``hyperparameters_list_type`` and the other
    acceptable value is [None], otherwise unchanged:
        * []
        * [1, 2, 3]
        * (1, 2, 3)
        * ["value1", "value2"]

    Raises
    ------
    ValueError
        If a key is in ``hyperparameters_list_type`` but
        its value is not a list, tuple, or None.

    Returns
    -------
    hyperparameter_grid : `dict` [`str`, `list` [`any`] or distribution with ``rvs`` method]
        A dictionary suitable to pass as ``param_distributions`` to
        `sklearn.model_selection.RandomizedSearchCV` for grid search.

        As explained in `sklearn.model_selection.RandomizedSearchCV`:
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
    """
    hyperparameter_grid = hyperparameter_dict.copy()
    if hyperparameters_list_type is None:
        hyperparameters_list_type = {}

    for param, value in hyperparameter_dict.items():
        # Distribution is allowed
        is_distribution = hasattr(value, "rvs")
        is_list = isinstance(value, (list, tuple))

        if param in hyperparameters_list_type:
            if isinstance(hyperparameters_list_type, dict):
                recognized_values = hyperparameters_list_type.get(param, [None])
            else:
                recognized_values = [None]

            if is_list:
                # A list is provided
                if (len(value) == 0 or
                        not all([isinstance(list_item, (list, tuple))
                                 or list_item in recognized_values
                                 for list_item in value])):
                    # Not all its values are acceptable for the hyperparameter,
                    # therefore enclose the value in a list.
                    hyperparameter_grid[param] = [value]
            elif value in recognized_values:
                # Not a list, but the value is acceptable
                hyperparameter_grid[param] = [value]
            else:
                raise ValueError(
                    f"The value for {param} must be a list, tuple, or one of {recognized_values}, "
                    f"found {value}.")

        else:
            # Any list or distribution is allowed.
            # Violating elements are enclosed the value in a list
            if not is_list and not is_distribution:
                hyperparameter_grid[param] = [value]

    return hyperparameter_grid


def dictionaries_values_to_lists(hyperparameter_dicts, hyperparameters_list_type=None):
    """Calls `~greykite.common.utils.python_utils.dictionary_values_to_lists`
    on the provided dictionary or on each item in a list of dictionaries.

    ``dictionary_values_to_lists`` returns a copy whose values are
     either lists, distributions with a ``rvs`` method, or None.

    Parameters
    ----------
    hyperparameter_dicts : `dict` [`str`, `any`] or `list` [`dict` [`str`, `any`]]
        Dictionary of hyperparameters, or list of such dictionaries

    hyperparameters_list_type : `set` [`str`] or `dict` [`str`, `list`] or None, optional, default None
        Hyperparameters that must be a `list` or other recognized value.
        e.g. ``regressor_cols`` is `list` or None, ``holiday_lookup_countries`` is `list` or "auto"
        or None.

        Thus, a flat list must become nested. E.g. ["US", "UK"] must be converted to [["US", "UK"]].

        Specifically, the values in ``hyperparameter_dict`` must be of type
        `list` [`list` or other accepted value].

            * If a set, other accepted value = [None]
            * If a dict, other accepted value is specified by the key's value, a
              list of valid options.

              For example, to allow `list` or "auto" or None, use
              ``hyperparameters_list_type={"regressor_cols": [None, "auto"]}``.

              For example, ``hyperparameters_list_type={"regressor_cols": [None]}`` is equivalent to
              ``hyperparameters_list_type={"regressor_cols"}`` using the set specification.

    Returns
    -------
    hyperparameter_grid : `dict` [`str`, `list` [`any`], or distribution with ``rvs`` method] or `list` [`dict]
        A dictionary or list of dictionaries suitable to pass as
        ``param_distributions`` to `sklearn.model_selection.RandomizedSearchCV`
        for grid search.

        As explained in `sklearn.model_selection.RandomizedSearchCV`:
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    See Also
    --------
    `~greykite.common.utils.python_utils.dictionary_values_to_lists`
    """
    if isinstance(hyperparameter_dicts, (list, tuple)):
        hyperparameter_grids = [
            dictionary_values_to_lists(
                item,
                hyperparameters_list_type=hyperparameters_list_type)
            for item in hyperparameter_dicts]
    else:
        hyperparameter_grids = dictionary_values_to_lists(
            hyperparameter_dicts,
            hyperparameters_list_type=hyperparameters_list_type)
    return hyperparameter_grids


def unique_in_list(array, ignored_elements=()):
    """Returns unique elements in ``array``, removing
    all levels of nesting if found.

    Parameters
    ----------
    array : `list` [`any`] or None
        List of items, with arbitrary level of nesting.
    ignored_elements : `tuple`, default ()
        Elements not to include in the output

    Returns
    -------
    unique_elements : `list`
        Unique elements in array, ignoring up to
        `level` levels of nesting.
        Elements that are `None` are removed from the output.
    """
    unique_elements = set()
    if array is not None:
        for item in array:
            if isinstance(item, (list, tuple)):
                unique_in_item = unique_in_list(item, ignored_elements=ignored_elements)
                unique_in_item = set(unique_in_item) if unique_in_item is not None else {}
                unique_elements.update(unique_in_item)
            elif item not in ignored_elements:
                unique_elements.add(item)
    return list(unique_elements) if unique_elements else None


def flatten_list(array):
    """Flattens an array by removing 1 level of nesting.

    Parameters
    ----------
    array : `list` [`list`]
        List of lists.

    Returns
    -------
    flat_arr : `list`
        Removes one level of nesting from the array.
        [[4], [3, 2], [1, [0]]] becomes [4, 3, 2, 1, [0]].
    """
    return [item for sublist in array for item in sublist]


def reorder_columns(df, order_dict=None):
    """Orders columns according to ``order_dict``.

    Can be used to order columns according to hierarchical
    constraints. Consider the tree where a parent is the sum
    of its children. Let a node's label be its BFS traversal order,
    with the root as 0. Use ``order_dict`` to map column names
    to these node labels, to get the dataframe in BFS traversal order,
    matching the structure of the tree.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Input data frame.
    order_dict : `dict` [`str`, `float`] or None
        How to order the columns.
        The key is the column name, the value is its position.
        Columns are returned in ascending order by value from left to right.
        Only column specified by ``order_dict`` are included in the output.
        If None, returns the original ``df``.

    Returns
    -------
    reordered_df : `pandas.DataFrame`
        ``df`` with the selected columns reordered.
    """
    if order_dict is not None:
        order_tuples = list(order_dict.items())
        order_tuples = sorted(order_tuples, key=lambda x: x[1])
        order_names = [x[0] for x in order_tuples]
        df = df[order_names]
    return df


def apply_func_to_columns(row_func, cols):
    """Returns a function that applies ``row_func`` to
    the selected ``cols``. Helper function for
    `~greykite.framework.output.univariate_forecast.UnivariateForecast.autocomplete_map_func_dict`.

    Parameters
    ----------
    row_func : callable
        A function.
    cols : `list` [`str` or `int`]
        Names of the columns (or dictionary keys, list indices)
        to pass to ``row_func``.

    Returns
    -------
    new_func : callable
        Takes ``row`` and returns the result of ``row_func``
        applied to the selected values ``row[col]``.
    """
    def new_func(row):
        return row_func(*[row[col] for col in cols])
    return new_func


def get_integer(val=None, name="value", min_value=0, default_value=0):
    """Returns integer value from input, with basic validation

    Parameters
    ----------
    val : `float` or None, default None
        Value to convert to integer.
    name : `str`, default "value"
        What the value represents.
    min_value : `float`, default 0
        Minimum allowed value.
    default_value : `float` , default 0
        Value to be used if ``val`` is None.

    Returns
    -------
    val : `int`
        Value parsed as an integer.
    """
    if val is None:
        val = default_value
    try:
        orig = val
        val = int(val)
    except ValueError:
        raise ValueError(f"{name} must be an integer")
    else:
        if val != orig:
            warnings.warn(f"{name} converted to integer {val} from {orig}")

    if not val >= min_value:
        raise ValueError(f"{name} must be >= {min_value}")

    return val


def mutable_field(mutable_default_value) -> dataclasses.field:
    """Can be used to set the default value in a dataclass
    to a mutable value.

    Provides a factory function that returns a copy of the provided argument.

    Parameters
    ----------
    mutable_default_value : Any
        The default value to use for the field.

    Returns
    -------
    field : `dataclasses.field`
        Set the default value to this value.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from typing import List
    >>> from greykite.common.python_utils import mutable_field
    >>> @dataclass
    >>> class D:
    >>>     x: List = mutable_field([1, 2, 3])
    >>>
    >>> assert D().x is not D().x
    >>> assert D().x == [1, 2, 3]
    """
    return field(default_factory=lambda: copy.deepcopy(mutable_default_value))


def ignore_warnings(category):
    """Returns a decorator to ignore all warnings
    in the specified category.

    Parameters
    ----------
    category : class
        Any warning that is a subclass of this category is ignored.

    Returns
    -------
    decorator_ignore : function
        A decorator that ignores all warnings in the category.
    """
    def decorator_ignore(fn):
        @functools.wraps(fn)
        def fn_ignore(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category)
                return fn(*args, **kwargs)
        return fn_ignore
    return decorator_ignore


def group_strs_with_regex_patterns(
        strings,
        regex_patterns):
    """Given a list/tuple of strings (``strings``), it partitions it into various groups (sub-lists) as
    specified in a set of patterns given in ``regex_patterns``. Note that the order
    of patterns matter as patterns will be consumed sequentially and if two patterns
    overlap, the one appearing first will get the string assigned to its group.
    Also note that the result will be a partition (``str_groups``) without overlap
    and any remaining element not satisfying any pattern will be returned in ``remainder``.

    Parameters
    ----------
    strings : `list` [`str`] or `tuple` [`str`]
        A list/tuple of strings which is to be partitioned into various groups
    regex_patterns : `list` [`str`]
        A list of strings each being a regex.

    Returns
    -------
    result : `dict`
        A dictionary with following items:

        - "str_groups": `list` [`list` [`str`]]
            A list of list of strings each corresponding to the patterns given in
            ``regex_patterns``. Note that some of these groups might be empty
            lists if that pattern is not satisfied by any regex pattern, or a
            regex pattern appearing before has already consumed all such strings.
        -"remainder": `list` [`str`]
            The remaining elements in ``strings`` which do not satisfy any of the
            patterns given in ``regex_patterns``. This list can be empty.

    """

    strings_list = list(strings)
    str_groups = []

    for regex_pattern in regex_patterns:
        group = [x for x in strings_list if bool(re.match(regex_pattern, x))]
        str_groups.append(group)
        strings_list = [x for x in strings_list if x not in group]

    return {"str_groups": str_groups, "remainder": strings_list}
