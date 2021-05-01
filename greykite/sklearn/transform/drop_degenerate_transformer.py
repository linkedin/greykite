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

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError


class DropDegenerateTransformer(BaseEstimator, TransformerMixin):
    """Removes degenerate (constant) columns.

    Parameters
    ----------
    drop_degenerate : `bool`, default False
        Whether to drop degenerate columns.

    Attributes
    ----------
    drop_cols : `list` [`str`] or None
        Degenerate columns to drop
    keep_cols : `list` [`str`] or None
        Columns to keep
    """

    def __init__(self, drop_degenerate=False):
        # sets params without modification to ensure get_params() works in grid search
        self.drop_degenerate = drop_degenerate

        self.drop_cols = None
        self.keep_cols = None

    def fit(self, X, y=None):
        """Identifies the degenerate columns, and sets ``self.keep_cols``
        and ``self.drop_cols``.

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
        if self.drop_degenerate:
            self.keep_cols = list(X.loc[:, (X != X.iloc[0]).any()].columns)
            self.drop_cols = [col for col in X.columns if col not in self.keep_cols]
        else:
            self.keep_cols = list(X.columns)
            self.drop_cols = []

        if self.drop_cols:
            warnings.warn(f"Columns {self.drop_cols} are degenerate (constant value), "
                          f"and will not be used in the forecast.", RuntimeWarning)
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
        X_subset : `pandas.DataFrame`
            Selected columns of X. Keeps columns that were not
            degenerate on the training data.
        """
        if self.keep_cols is None:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments "
                "before calling 'transform'.")
        return X[self.keep_cols]
