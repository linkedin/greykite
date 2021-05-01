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

from greykite.common.constants import TIME_COL
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.features.timeseries_features import convert_date_to_continuous_time


class BuildTimeseriesFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Calculates time series features (e.g. year, month, hour etc.) of the input time series

    Parameters
    ----------
    time_col : string, default=TIME_COL
        issues warning if fraction of nulls is above this value

    Attributes
    ----------
    origin_for_time_vars : float (e.g. 2019.2)
        sets default origin so that "ct1" feature from `build_time_features_df` starts at 0 on start date of fitted data
    """
    def __init__(self, time_col: str = TIME_COL):
        self.time_col = time_col
        self.origin_for_time_vars = None

    def fit(self, X, y=None):
        """Sets the time origin for input time series"""
        assert isinstance(X, pd.DataFrame)
        dt = X[self.time_col]
        self.origin_for_time_vars = convert_date_to_continuous_time(dt[0])
        return self

    def transform(self, X):
        """ Calculates time series features of the input time series

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        A copy of the data frame with original time points and calculated features
        """
        if self.origin_for_time_vars is None:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments "
                "before calling 'transform'.")
        assert isinstance(X, pd.DataFrame)
        dt = X[self.time_col]
        features_ts = build_time_features_df(dt, conti_year_origin=self.origin_for_time_vars)
        output = pd.concat([dt, features_ts], axis=1)
        return output
