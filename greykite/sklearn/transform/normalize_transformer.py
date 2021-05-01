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

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


NORMALIZE_ALGORITHMS = {
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "Normalizer": Normalizer,
    "QuantileTransformer": QuantileTransformer,
    "PowerTransformer": PowerTransformer,
}


class NormalizeTransformer(BaseEstimator, TransformerMixin):
    """Normalizes time series data.

    Parameters
    ----------
    normalize_algorithm : `str` or None, default None
        Which algorithm to use. Valid options are:

            - "MinMaxScaler" : `sklearn.preprocessing.MinMaxScaler`,
            - "MaxAbsScaler" : `sklearn.preprocessing.MaxAbsScaler`,
            - "StandardScaler" : `sklearn.preprocessing.StandardScaler`,
            - "RobustScaler" : `sklearn.preprocessing.RobustScaler`,
            - "Normalizer" : `sklearn.preprocessing.Normalizer`,
            - "QuantileTransformer" : `sklearn.preprocessing.QuantileTransformer`,
            - "PowerTransformer" : `sklearn.preprocessing.PowerTransformer`,

        If None, this transformer is a no-op. No normalization is done.

    normalize_params : `dict` or None, default None
        Params to initialize the normalization scaler/transformer.

    Attributes
    ----------
    scaler : `class`
        sklearn class used for normalization
    _is_fitted : `bool`
        Whether the transformer is fitted.
    """

    def __init__(
            self,
            normalize_algorithm=None,
            normalize_params=None):
        # sets params without modification to ensure get_params() works in grid search
        self.normalize_algorithm = normalize_algorithm
        self.normalize_params = normalize_params

        self.scaler = None
        self._is_fitted = None

    def fit(self, X, y=None):
        """Fits the normalization transform.

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
        if self.normalize_algorithm is not None:
            if self.normalize_algorithm not in NORMALIZE_ALGORITHMS.keys():
                raise ValueError(
                    f"`normalize_algorithm` '{self.normalize_algorithm}' is not recognized. "
                    f"Must be one of {NORMALIZE_ALGORITHMS.keys()}")
            if self.normalize_params is None:
                self.normalize_params = {}
            self.scaler = NORMALIZE_ALGORITHMS[self.normalize_algorithm](**self.normalize_params)
            self.scaler.fit(X=X)
        return self

    def transform(self, X):
        """Normalizes data using the specified scaling method.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Data to transform. e.g. each column is a timeseries.
            Columns are expected to be numeric.

        Returns
        -------
        X_normalized : `pandas.DataFrame`
            A normalized copy of the data frame.
        """
        if self._is_fitted is None:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments "
                "before calling 'transform'.")
        assert isinstance(X, pd.DataFrame)
        if self.scaler:
            transformed = self.scaler.transform(X)
            X_normalized = pd.DataFrame(transformed, index=X.index, columns=X.columns)
        else:
            X_normalized = X.copy()
        return X_normalized
