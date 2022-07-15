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
# original author: Kaixu Yang
"""Defines the base uncertainty model class. All uncertainty models should inherit this class."""

from abc import abstractmethod
from typing import Dict
from typing import Optional

import pandas as pd

from greykite.common import constants as cst
from greykite.sklearn.uncertainty.exceptions import UncertaintyError


class BaseUncertaintyModel:
    """The base uncertainty model.

    Attributes
    ----------
    uncertainty_dict : `dict` [`str`, any]
        The uncertainty model specification. It should have the following keys:

                "uncertainty_method": a string that is in
                    `~greykite.sklearn.uncertainty.uncertainty_methods.UncertaintyMethodEnum`.
                "params": a dictionary that includes any additional parameters needed by the uncertainty method.

    uncertainty_method : `str` or None
        The name of the uncertainty model.
        Must be in `~greykite.sklearn.uncertainty.uncertainty_methods.UncertaintyMethodEnum`.
        This parameter is a required parameter by all subclasses.
    params : `dict` [`str`, any] or None
        The parameters to be fed into the uncertainty model.
        This class tries to populate ``value_col`` if ``cst.VALUE_COL`` is found in the training df,
        because ``value_col`` is a common parameter used by many subclasses.
    train_df : `pandas.DataFrame` or None
        The data used to fit the uncertainty model.
    uncertainty_model : any or None
        The uncertainty model.
    pred_df : `pandas.DataFrame`
        The prediction result df.
    """

    UNCERTAINTY_METHOD = None
    REQUIRED_PARAMS = []
    DEFAULT_COVERAGE = 0.95

    def __init__(
            self,
            uncertainty_dict: Dict[str, any],
            **kwargs):
        self.uncertainty_dict = uncertainty_dict
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set by ``fit`` method.
        self.uncertainty_method: Optional[str] = None
        self.params: Optional[dict] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.uncertainty_model: Optional[any] = None

        # Set by ``predict`` method.
        self.pred_df: Optional[pd.DataFrame] = None

    @abstractmethod
    def _check_input(self):
        """Checks that necessary input are provided in ``self.uncertainty_dict`` and ``self.train_df``.
        To be called after setting ``self.train_df`` in ``self.fit``.
        Every subclass need to override this method to check their own inputs.
        Do not raise errors other than
        `~greykite.sklearn.uncertainty.exceptions.UncertaintyError`,
        since this type of error will be catched and won't fail the whole pipeline.
        """
        if self.uncertainty_dict is None:
            self.uncertainty_dict = {}

        # Checks the uncertainty method matches.
        self.uncertainty_method = self.uncertainty_dict.get("uncertainty_method")
        if self.uncertainty_method != self.UNCERTAINTY_METHOD:
            raise UncertaintyError(
                f"The uncertainty method {self.uncertainty_method} is not as expected {self.UNCERTAINTY_METHOD}."
            )

        # Gets the parameters.
        self.params = self.uncertainty_dict.get("params", {})

        # Populates ``value_col`` since it's a commonly used param.
        if (self.params.get("value_col", None) is None and self.train_df is not None
                and cst.VALUE_COL in self.train_df.columns):
            self.params["value_col"] = cst.VALUE_COL

        # Checks all required parameters are given.
        for required_param in self.REQUIRED_PARAMS:
            if required_param not in self.params:
                raise UncertaintyError(
                    f"The parameter {required_param} is required but not found in "
                    f"`uncertainty['params']` {self.params}. "
                    f"The required parameters are {self.REQUIRED_PARAMS}."
                )

    def fit(
            self,
            train_df: pd.DataFrame):
        """Fits the uncertainty model.

        Parameters
        ----------
        train_df : `pandas.DataFrame`
            The training data.
        """
        self.train_df = train_df.copy()

    def predict(
            self,
            fut_df: pd.DataFrame):
        """Predicts the uncertainty columns for ``fut_df``.

        Parameters
        ----------
        fut_df : `pandas.DataFrame`
            The data used for prediction.
        """
        pass
