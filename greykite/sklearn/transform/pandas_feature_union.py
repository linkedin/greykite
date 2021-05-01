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
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import _fit_transform_one
from sklearn.pipeline import _transform_one
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed


class PandasFeatureUnion(FeatureUnion):
    """Concatenates results of multiple transformer objects.
    Transformers are expected to have pd.DataFrame as input and output

    Modified from sklearn.pipeline.FeatureUnion

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer,
    or removed by setting to 'drop' or ``None``.

    Read more in the :ref:`User Guide <feature_union>`.

    Parameters
    ----------
    transformer_list : list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.
        .. versionchanged:: 0.22
           Deprecated `None` as a transformer in favor of 'drop'.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None
    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.
    """
    def __init__(self, transformer_list, n_jobs=None,
                 transformer_weights=None, verbose=False):
        super().__init__(
            transformer_list=transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose)

    def fit_transform(self, X, y=None, **fit_params):
        """Fits all transformers, transforms the data and concatenates results.

        Modified from `sklearn.pipeline.FeatureUnion`.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Input data to be transformed.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : `pandas.Dataframe`, shape (n_samples, sum_n_components)
            column-wise concatenation of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)

        if not results:
            # All transformers are None
            return pd.DataFrame(np.zeros((X.shape[0], 0)))
        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)
        Xs = pd.concat(Xs, axis=1)
        return Xs

    def transform(self, X):
        """Transforms X separately by each transformer, concatenates results.

        Modified from `sklearn.pipeline.FeatureUnion`

        Parameters
        ----------
        X : `pandas.DataFrame`
            Input data to be transformed.

        Returns
        -------
        X_t : `pandas.DataFrame`, shape (n_samples, sum_n_components)
            column-wise concatenation of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return pd.DataFrame(np.zeros((X.shape[0], 0)))
        Xs = pd.concat(Xs, axis=1)
        return Xs
