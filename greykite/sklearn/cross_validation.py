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
# original author: Albert Chen
"""Cross validator for time series cross validation,
compatible with sklearn.
"""

import math
import random
import warnings

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import get_integer


class RollingTimeSeriesSplit(BaseCrossValidator):
    """Flexible splitter for time-series cross validation and rolling window evaluation.
    Suitable for use in GridSearchCV.

    Attributes
    ----------
    min_splits : int
        Guaranteed min number of splits. This is always set to 1. If provided configuration results in 0 splits,
        the cross validator will yield a default split.

    __starting_test_index : int
        Test end index of the first CV split. Actual offset = __starting_test_index + _get_offset(X), for a particular
        dataset X.
        Cross validator ensures the last test split contains the last observation in X.

    Examples
    --------
    >>> from greykite.sklearn.cross_validation import RollingTimeSeriesSplit
    >>> X = np.random.rand(20, 4)
    >>> tscv = RollingTimeSeriesSplit(forecast_horizon=3, max_splits=4)
    >>> tscv.get_n_splits(X=X)
    4
    >>> for train, test in tscv.split(X=X):
    ...     print(train, test)
    [2 3 4 5 6 7] [ 8  9 10]
    [ 5  6  7  8  9 10] [11 12 13]
    [ 8  9 10 11 12 13] [14 15 16]
    [11 12 13 14 15 16] [17 18 19]
    >>> X = np.random.rand(20, 4)
    >>> tscv = RollingTimeSeriesSplit(forecast_horizon=2,
    ...                               min_train_periods=4,
    ...                               expanding_window=True,
    ...                               periods_between_splits=4,
    ...                               periods_between_train_test=2,
    ...                               max_splits=None)
    >>> tscv.get_n_splits(X=X)
    4
    >>> for train, test in tscv.split(X=X):
    ...     print(train, test)
    [0 1 2 3] [6 7]
    [0 1 2 3 4 5 6 7] [10 11]
    [ 0  1  2  3  4  5  6  7  8  9 10 11] [14 15]
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15] [18 19]
    >>> X = np.random.rand(5, 4)  # default split if there is not enough data
    >>> for train, test in tscv.split(X=X):
    ...     print(train, test)
    [0 1 2 3] [4]
    """
    def __init__(
            self,
            forecast_horizon,
            min_train_periods=None,
            expanding_window=False,
            use_most_recent_splits=False,
            periods_between_splits=None,
            periods_between_train_test=0,
            max_splits=3):
        """Initializes attributes of RollingTimeSeriesSplit

        Parameters
        ----------
        forecast_horizon : `int`
            How many periods in each CV test set

        min_train_periods : `int` or None, optional
            Minimum number of periods for training.
            If ``expanding_window`` is False, every training period has this size.

        expanding_window : `bool`, default False
            If True, training window for each CV split is fixed to the first available date.
            Otherwise, train start date is sliding, determined by ``min_train_periods``.

        use_most_recent_splits: `bool`, default False
            If True, splits from the end of the dataset are used.
            Else a sampling strategy is applied. Check
            `~greykite.sklearn.cross_validation.RollingTimeSeriesSplit._sample_splits`
            for details.

        periods_between_splits : `int` or None
            Number of periods to slide the test window

        periods_between_train_test : `int`
            Number of periods gap between train and test within a CV split

        max_splits : `int` or None
            Maximum number of CV splits. Given the above configuration, samples up to max_splits train/test splits,
            preferring splits toward the end of available data. If None, uses all splits.
        """
        super().__init__()
        self.forecast_horizon = get_integer(forecast_horizon, name="forecast_horizon", min_value=1)

        # by default, use at least twice the forecast horizon for training
        self.min_train_periods = get_integer(min_train_periods, name="min_train_periods",
                                             min_value=1, default_value=2 * self.forecast_horizon)

        # by default, use fixed size training window
        self.expanding_window = expanding_window

        # by default, does not force most recent splits
        self.use_most_recent_splits = use_most_recent_splits

        # by default, use non-overlapping test sets
        self.periods_between_splits = get_integer(periods_between_splits, name="periods_between_splits",
                                                  min_value=1, default_value=self.forecast_horizon)

        # by default, use test set immediately following train set
        self.periods_between_train_test = get_integer(periods_between_train_test, name="periods_between_train_test",
                                                      min_value=0, default_value=0)

        if self.min_train_periods < 2 * self.forecast_horizon:
            warnings.warn(f"`min_train_periods` is too small for your `forecast_horizon`. Should be at least"
                          f" {forecast_horizon*2}=2*`forecast_horizon`.")

        self.max_splits = max_splits
        self.min_splits = 1  # CV ensures there is always at least one split
        # test end index for the first CV split, before applying offset to ensure last data point in X is used
        self.__starting_test_index = (self.forecast_horizon
                                      + self.min_train_periods
                                      + self.periods_between_train_test
                                      - 1)

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test CV folds according to rolling
          window time series cross validation

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Must have `shape` method.

        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning problems. Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Always ignored, exists for compatibility.

        Yields
        ------
        train : `numpy.array`
            The training set indices for that split.

        test : `numpy.array`
            The testing set indices for that split.
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)

        n_splits_without_capping = self.get_n_splits_without_capping(X=X)
        n_splits = self.get_n_splits(X=X)
        if n_splits_without_capping == 0:
            warnings.warn("There are no CV splits under the requested settings. Decrease `forecast_horizon` and/or"
                          " `min_train_periods`. Using default 90/10 CV split")
        elif n_splits == 1:
            warnings.warn("There is only one CV split")
        elif n_splits >= 10:
            warnings.warn(f"There is a high number of CV splits ({n_splits}). If training is slow, increase "
                          f"`periods_between_splits` or `min_train_periods`, or decrease `max_splits`")

        log_message(f"There are {n_splits} CV splits.", LoggingLevelEnum.INFO)

        if n_splits_without_capping == 0:  # uses default split
            default_split_ratio = 0.9
            train_samples = int(round(num_samples * default_split_ratio))
            yield indices[:train_samples], indices[train_samples:]
        else:  # determines which splits to keep so that up to max_splits are returned
            splits_to_keep = self._sample_splits(n_splits_without_capping)

            last_index = num_samples - 1
            test_end_index = self.__starting_test_index + self._get_offset(X=X)
            current_split_index = 0
            while test_end_index <= last_index:
                test_start_index = test_end_index - self.forecast_horizon + 1
                train_end_index = test_start_index - self.periods_between_train_test - 1
                train_start_index = 0 if self.expanding_window else train_end_index - self.min_train_periods + 1
                assert train_start_index >= 0  # guaranteed by n_splits > 0

                if current_split_index in splits_to_keep:
                    log_message(f"CV split: Train {train_start_index} to {train_end_index}. "
                                f"Test {test_start_index} to {test_end_index}.", LoggingLevelEnum.DEBUG)
                    yield indices[train_start_index:train_end_index + 1], indices[test_start_index:test_end_index + 1]

                test_end_index += self.periods_between_splits
                current_split_index += 1

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations yielded by the cross-validator

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to split

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splitting iterations yielded by the cross-validator.
        """
        num_splits = self.get_n_splits_without_capping(X=X)
        if self.max_splits is not None and num_splits > self.max_splits:
            num_splits = self.max_splits  # num_splits is set to max limit
        if num_splits == 0:
            num_splits = self.min_splits  # not enough observations to create split, uses default
        return num_splits

    def get_n_splits_without_capping(self, X=None):
        """Returns the number of splitting iterations in the cross-validator as configured, ignoring
            self.max_splits and self.min_splits

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to split

        Returns
        -------
        n_splits : int
            The number of splitting iterations in the cross-validator as configured, ignoring
            self.max_splits and self.min_splits
        """
        last_index = X.shape[0] - 1
        starting_index = self.__starting_test_index + self._get_offset(X=X)
        if starting_index > last_index:
            return 0
        return math.ceil((last_index - starting_index + 1) / self.periods_between_splits)

    def _get_offset(self, X=None):
        """Returns an offset to add to test set indices when creating CV splits
        CV splits are shifted so that the last test observation is the last point in X.
        This shift does not affect the total number of splits.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to split

        Returns
        -------
        offset : int
            The number of observations to ignore at the beginning of X when creating CV splits
        """
        last_index = X.shape[0] - 1
        starting_index = self.__starting_test_index
        if starting_index > last_index:
            return 0
        return (last_index - starting_index) % self.periods_between_splits

    def _sample_splits(self, num_splits, seed=48912):
        """Samples up to ``max_splits`` items from list(range(`num_splits`)).

        If ``use_most_recent_splits`` is True, highest split indices up to ``max_splits``
        are retained. Otherwise, the following sampling scheme is implemented:

            - takes the last 2 splits
            - samples from the rest uniformly at random

        Parameters
        ----------
        num_splits : `int`
            Number of splits before sampling.

        seed : `int`
            Seed for random sampling.

        Returns
        -------
        n_splits : `list`
            Indices of splits to keep (subset of `list(range(num_splits))`).
        """
        split_indices = list(range(num_splits))
        if self.max_splits is not None and num_splits > self.max_splits:
            if self.use_most_recent_splits:
                # keep indices from the end up to max_splits
                keep_split_indices = split_indices[-self.max_splits:]
            else:
                # applies sampling scheme to take up to max_splits
                keep_split_indices = []
                if self.max_splits > 0:  # first takes the last split
                    keep_split_indices.append(split_indices[-1])
                if self.max_splits > 1:  # then takes the second to last split
                    keep_split_indices.append(split_indices[-2])
                if self.max_splits > 2:  # then randomly samples the remaining splits
                    random.seed(seed)
                    keep_split_indices += random.sample(split_indices[:-2], self.max_splits - 2)
            split_indices = keep_split_indices
        return split_indices

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Class directly implements `split` instead of providing this function"""
        raise NotImplementedError
