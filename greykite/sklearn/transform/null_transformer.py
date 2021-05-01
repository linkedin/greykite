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

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError

from greykite.common.features.timeseries_impute import impute_with_lags_multi
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import update_dictionary


DEFAULT_PARAMS = {
    "interpolate": dict(
        method="linear",
        limit_direction="both",
        axis=0),  # fills column-by-column
    "ts_interpolate": dict(
        orders=[7, 14, 21],
        agg_func=np.mean,
        iter_num=5)
}


class NullTransformer(BaseEstimator, TransformerMixin):
    """Imputes nulls in time series data.

    This transform is stateless in the sense that ``transform`` output
    does not depend on the data passed to ``fit``. The dataset passed to
    ``transform`` is used to impute itself.

    Parameters
    ----------
    max_frac : `float`, default 0.10
        issues warning if fraction of nulls is above this value
    impute_algorithm  : `str` or None, default "interpolate"
        Which imputation algorithm to use.
        Valid options are:

            - "interpolate" : `pandas.DataFrame.interpolate`
            - "ts_interpolate" : `~greykite.common.features.timeseries_impute.impute_with_lags_multi`.

        If None, this transformer is a no-op. No null imputation is done.

    impute_params : `dict` or None, default None
        Params to pass to the imputation algorithm.
        See `pandas.DataFrame.interpolate` and
        `~greykite.common.features.timeseries_impute.impute_with_lags_multi`
        for their respective options.

        For pandas "interpolate", the "ffill", "pad", "bfill", "backfill" methods
        are not allowed to avoid confusion with the fill axis parameter. Use "linear"
        with ``axis=0`` instead, with direction controlled by ``limit_direction``.

        If None, uses the defaults provided in this class.
    impute_all : `bool`, default True
        Whether to impute all values. If True, NaNs are not allowed in the
        transformed result. Ignored if ``impute_algorithm`` is None.

        The transform specified by ``impute_algorithm`` and
        ``impute_params`` may leave NaNs in the dataset. For example,
        if it fills in the forward direction but the first value in a
        column is NaN.

        A first pass is taken with the impute algorithm specified.
        A second pass is taken with the "interpolate" algorithm (method="linear",
        limit_direction="both") to fill in remaining NaNs.

    Attributes
    ----------
    null_frac : `int`
        The fraction data points that are null
    _is_fitted : `bool`
        Whether the transformer is fitted.
    missing_info : `dict`
        Information about the missing data.
        Set by ``transform`` if ``impute_algorithm = "ts_interpolate"``.
    """
    def __init__(
            self,
            max_frac=0.10,
            impute_algorithm=None,
            impute_params=None,
            impute_all=True):
        # sets params without modification to ensure get_params() works in grid search
        self.max_frac = max_frac
        self.impute_algorithm = impute_algorithm
        self.impute_params = impute_params
        self.impute_all = impute_all

        self.null_frac = None
        self._is_fitted = None
        self.missing_info = None
        if (self.impute_algorithm == "interpolate"
                and self.impute_params is not None
                and self.impute_params.get("method") in ["ffill", "pad", "bfill", "backfill"]):
            # These four methods treat "axis=0" as rows, "axis=1" as columns,
            # contrary to the pandas documentation. Avoid them to prevent misuse.
            raise ValueError(
                f"method '{self.impute_params['method']}' is not allowed. "
                f"Use method='linear' with `limit_direction` instead")

    def fit(self, X, y=None):
        """Updates `self.impute_params`.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Training input data. e.g. each column is a timeseries.
            Columns are expected to be numeric.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        assert isinstance(X, pd.DataFrame)
        self._is_fitted = True
        # sets default parameters
        if self.impute_algorithm is not None:
            default_params = DEFAULT_PARAMS.get(self.impute_algorithm, {})
            self.impute_params = update_dictionary(default_params, overwrite_dict=self.impute_params)
        return self

    def transform(self, X):
        """Imputes missing values in input time series.

        Checks the % of data points that are null, and provides warning if
        it exceeds ``self.max_frac``.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Data to transform. e.g. each column is a timeseries.
            Columns are expected to be numeric.

        Returns
        -------
        X_imputed : `pandas.DataFrame`
            A copy of the data frame with original values and missing values imputed
        """
        if self._is_fitted is None:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments "
                "before calling 'transform'.")
        assert isinstance(X, pd.DataFrame)

        self.null_frac = X.isna().mean()  # fraction of NaNs in each column
        if np.any(self.null_frac > self.max_frac):
            warnings.warn(f"Input data has many null values. Missing {self.null_frac.max():.2%} of one input.",
                          RuntimeWarning)
        if any(self.null_frac > 0.0):
            log_message(f"Missing data detected: {self.null_frac.mean():.2%} of all input values "
                        f"are null. (If future external regressor(s) are used, some missing values in "
                        f"`value_col` are expected.)",
                        LoggingLevelEnum.INFO)

        if self.impute_algorithm is not None:
            if self.impute_algorithm == "interpolate":
                # Uses `pandas.DataFrame.interpolate`
                X_imputed = X.interpolate(**self.impute_params)
            elif self.impute_algorithm == "ts_interpolate":
                # Uses `impute_with_lags_multi`
                impute_info = impute_with_lags_multi(df=X, **self.impute_params)
                X_imputed = impute_info["df"]
                self.missing_info = impute_info["missing_info"]
            else:
                raise ValueError(f"`impute_algorithm` '{self.impute_algorithm}' is not recognized."
                                 f"Must be one of 'ts_interpolate', 'interpolate'")

            if self.impute_all:
                # A second pass is taken to make sure there are no NaNs.
                X_imputed = X_imputed.interpolate(**DEFAULT_PARAMS["interpolate"])
        else:
            # no-op
            X_imputed = X.copy()
        return X_imputed
