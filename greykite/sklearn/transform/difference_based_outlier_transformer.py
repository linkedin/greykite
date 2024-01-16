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
# original author: Yi-Wei Liu

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError

from greykite.common.features.outlier import IMPLEMENTED_DIFF_METHODS
from greykite.common.features.outlier import TukeyOutlierDetector
from greykite.common.features.outlier import ZScoreOutlierDetector
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message


class DifferenceBasedOutlierTransformer(BaseEstimator, TransformerMixin):
    """Replaces outliers in data with NaN.
    Outliers are determined by anomaly scores computed by the differences or ratios between
    the observed values and their baseline values. Cutoffs of the scores are derived through
    z-score or tukey coeficient methods. Columns are handled independently. If the baseline
    is not specified, the observed values will be used as the anomaly scores, together with
    the z-score method, the algorithm degenerates to the `ZscoreOutlierTransformer`.

    Parameters
    ----------
    method : `str`, default "z_score"
        Method used to determine the outliers. Must be either "z_score" or "tukey".
        - When `method = "z_score"`, outliers are defined as those y_{t}'s with
        absolute z-scores of the anomaly scores larger than ``z_cutoff``.
        - When `method = "tukey"`, outliers are defined as those y_{t}'s with anomaly
        scores larger than `Q3 + tukey_cutoff * IQR`, or smaller than `Q1 - tukey_cutoff * IQR`.
        Here Q1, Q3, and IQR are the first-quartile, third-quartile, and inter-quartile
        range of the anomaly scores, respectively.
    score_type : `str`, default "difference"
        Formula with respect to the baseline values to compute anomaly scores.
        Must be either "difference" or "ratio".
        Given a time series y_{t} and its baseline values b_{t}, the anomaly scores for
            - "difference" is: y_{t} - b_{t}
            - "ratio" is: (y_{t} / b_{t}) - 1
    params : `dict` [`str`, any] or None, default None
        A dictionary with seven keys:

            - "diff_method": `DiffMethod` or None, default None
              An object of the `DiffMethod` class, describing `name` and `param` of "diff_method".
              See `~greykite.common.features.outlier.DiffMethod`.
            - "agg_func": `numpy.functions` or None, default None
              The function to compute baseline values with arguments specified in "lag_orders".
              If None, the anomaly scores are the actual values of the time series y_{t}.
            - "lag_orders": `list [`int`]` or None, default None
              Values in the observed data used to compute the baseline. For example, if
              `lag_orders = [-7, -1, 1, 7]` and `agg_func = numpy.nanmean`, the baseline
              value for y_{t} is the average of (y_{t-7}, y_{t-1}, y_{t+1}, y_{t+7}).
              If None, the anomaly scores are the actual values of the time series y_{t}.
            - "trim_percent": `float` or None, default None
              Trimming percentage on anomaly scores for calculating the thresholds.
              This removes `trim_percent` of anomaly scores in symmetric fashion from both ends and
              then calculates the quantities needed (e.g., mean, standard deviation, quartiles).
              For example, in `method = "z_score"`, this will remove extreme values of anomaly scores
              to calculate the mean and variance for computing the z-scores of anomaly scores.
            - "z_cutoff": `float` or None, default None
              The cutoff on the z-scores of anomaly scores to determine outliers.
              Effective only when `method = "z_score"`. If None, no outliers are removed.
            - "tukey_cutoff": `float` or None, default None
              The tukey coefficient for anomaly scores to determine outliers.
              Effective only when `method = "tukey"`. If None, no outliers are removed.
            - "max_outlier_percent": `float` or None, default None
              Maximum percentage of outliers to be removed. Range from 0 to 100.
              When specified, for example `max_outlier_percent = 5`, the maximum portion of outliers
              to be removed is 5% of the total number of data. If the original outliers detected
              are less than 5%, the result is unaffected; if original outliers are more than 5%,
              then only the top 5% outliers with the most extreme anomaly scores will be removed.

    Attributes
    ----------
    score : `pandas.DataFrame`
        Anomaly scores for each value in the input data. The anomaly scores can be computed
        with the function `fit` in the class.
    _is_fitted : `bool`
        Whether the transformer is fitted.
    """
    def __init__(
            self,
            method: str = "z_score",
            score_type: str = "difference",
            params: dict | None = None,):
        self.score_type = score_type
        self.method = method
        self.params = params
        self.score = None
        self._is_fitted = False

    def fit(self, X, y=None):
        """Computes the column-wise anomaly scores, stored as ``score`` attribute.

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
        # Gets variables and from `params` dictionary because __init__ is only run in initialization.
        if self.params is not None:
            self.diff_method = self.params.get("diff_method")
            self.agg_func = self.params.get("agg_func")
            self.lag_orders = self.params.get("lag_orders")
            self.trim_percent = self.params.get("trim_percent")
            self.z_cutoff = self.params.get("z_cutoff")
            self.tukey_cutoff = self.params.get("tukey_cutoff")
            self.max_outlier_percent = self.params.get("max_outlier_percent")
        else:
            self.diff_method = None
            self.agg_func = None
            self.lag_orders = None
            self.trim_percent = None
            self.z_cutoff = None
            self.tukey_cutoff = None
            self.max_outlier_percent = None
        # Checks if the input variables are valid.
        if self.score_type not in ["difference", "ratio"]:
            raise NotImplementedError(
                f"{self.score_type} is an invalid 'score_type': "
                "must be either 'difference' or 'ratio'.")
        if self.method not in ["z_score", "tukey"]:
            raise NotImplementedError(
                f"{self.method} is an invalid 'method': "
                "must be either 'z_score' or 'tukey'.")
        self._is_fitted = True
        # If no threshold specified, does nothing.
        if self.method == "z_score" and self.z_cutoff is None:
            return self
        if self.method == "tukey" and self.tukey_cutoff is None:
            return self
        # If the name of `diff_method` is in the available list, uses the `diff_method` in transform.
        # Otherwise, sets `self.diff_method` to None.
        if self.diff_method is not None and self.diff_method.name in IMPLEMENTED_DIFF_METHODS:
            self.score = X
            return self
        else:
            self.diff_method = None
        if self.agg_func is not None and self.lag_orders is not None:
            # Computes the baseline values.
            lag_orders_list = []
            for lag_order in self.lag_orders:
                lag_orders_list += [X.shift(-lag_order)]
            baseline = pd.DataFrame(self.agg_func(lag_orders_list, axis=0))
            baseline.columns = X.columns
            baseline.index = X.index
            if self.score_type == "difference":
                self.score = X - baseline
            elif self.score_type == "ratio":
                self.score = (X / baseline) - 1
        else:
            self.score = X
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
        X_outliers_removed : `pandas.DataFrame`
            A copy of the data frame with original values and outliers replaced with NaN.
        """
        if self._is_fitted is False:
            raise NotFittedError(
                "This instance is not fitted yet. Call `fit` with appropriate arguments "
                "before calling `transform`.")
        result = X.copy()
        if self.score is None:
            return result
        if self.method == "z_score":
            detector = ZScoreOutlierDetector(
                z_score_cutoff=self.z_cutoff,
                trim_percent=self.trim_percent,
                diff_method=self.diff_method)
        elif self.method == "tukey":
            detector = TukeyOutlierDetector(
                tukey_cutoff=self.tukey_cutoff,
                iqr_lower=0.25,
                iqr_upper=0.75,
                trim_percent=self.trim_percent,
                diff_method=self.diff_method)
        # Creates a dataframe to store the outlier scores / indices for each column in `score`.
        outlier_scores = pd.DataFrame(0, index=self.score.index, columns=self.score.columns)
        outlier_indices = pd.DataFrame(0, index=self.score.index, columns=self.score.columns)
        for col_name in self.score.columns:
            detector.fit(self.score[col_name])
            outlier_scores[col_name] = detector.fitted.scores
            outlier_indices[col_name] = np.array(detector.fitted.is_outlier)
        # Checks for each column if the outliers are more than `max_outlier_percent`.
        if self.max_outlier_percent is not None:
            for col_name in outlier_indices.columns:
                if outlier_indices[col_name].mean() > (self.max_outlier_percent / 100):
                    upper_cutoff = outlier_scores[col_name].quantile(1 - ((self.max_outlier_percent / 100) / 2))
                    lower_cutoff = outlier_scores[col_name].quantile((self.max_outlier_percent / 100) / 2)
                    outlier_indices[col_name] = (outlier_scores[col_name] > upper_cutoff) | (outlier_scores[col_name] < lower_cutoff)

        if np.any(outlier_indices):
            total_na = outlier_indices.sum().sum()
            log_message(f"Detected {total_na} outlier(s).", LoggingLevelEnum.INFO)
        result = result.mask(outlier_indices)

        return result
