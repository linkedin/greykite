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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError

from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message


class ZscoreOutlierTransformer(BaseEstimator, TransformerMixin):
    """Replaces outliers in data with NaN.
    Outliers are determined by z-score cutoff. Columns are handled independently.

    Parameters
    ----------
    z_cutoff : `float` or None, default None
        z-score cutoff to define outliers. If None, this transformer is a no-op.
    use_fit_baseline : `bool`, default False
        If True, the z-scores are calculated using the mean and standard
        deviation of the dataset passed to ``fit``.

        If False, the transformer is stateless. z-scores are calculated
        for the dataset passed to ``transform``, regardless of ``fit``.

    Attributes
    ----------
    mean : `pandas.Series`
        Mean of each column. NaNs are ignored.
    std : `pandas.Series`
        Standard deviation of each column. NaNs are ignored.
    _is_fitted : `bool`
        Whether the transformer is fitted.
    """
    def __init__(self, z_cutoff=None, use_fit_baseline=False):
        # sets params without modification to ensure get_params() works in grid search
        self.z_cutoff = z_cutoff
        self.use_fit_baseline = use_fit_baseline

        self.mean = None
        self.std = None
        self._is_fitted = None

    def fit(self, X, y=None):
        """Computes the column mean and standard deviation,
        stored as ``mean`` and ``std`` attributes.

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
        if self.z_cutoff is not None and self.use_fit_baseline:
            self.mean = X.mean()
            self.std = X.std()
        return self

    def transform(self, X):
        """Replaces outliers with NaN.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Data to transform. e.g. each column is a timeseries.
            Columns are expected to be numeric.

        Returns
        -------
        X_outlier : `pandas.DataFrame`
            A copy of the data frame with original values and outliers replaced with NaN.
        """
        assert isinstance(X, pd.DataFrame)
        result = X.copy()
        if self.z_cutoff is not None:
            if self.use_fit_baseline:
                if self._is_fitted is None:
                    raise NotFittedError(
                        "This instance is not fitted yet. Call 'fit' with appropriate arguments "
                        "before calling 'transform'.")
                mean = self.mean
                std = self.std
            else:
                mean = X.mean()
                std = X.std()
            outlier_indices = np.abs(X - mean) > std * self.z_cutoff
            if np.any(outlier_indices):
                total_na = outlier_indices.sum().sum()
                log_message(f"Detected {total_na} outlier(s).", LoggingLevelEnum.INFO)
            result = result.mask(outlier_indices)
        return result
