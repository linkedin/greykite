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
"""Defines the uncertainty methods."""

from dataclasses import dataclass
from enum import Enum
from typing import Type

from greykite.sklearn.uncertainty.base_uncertainty_model import BaseUncertaintyModel
from greykite.sklearn.uncertainty.quantile_regression_uncertainty_model import QuantileRegressionUncertaintyModel
from greykite.sklearn.uncertainty.simple_conditional_residuals_model import SimpleConditionalResidualsModel


@dataclass
class UncertaintyMethod:
    """The data class to store uncertainty models.

    Attributes
    ----------
    model_class : `~greykite.sklearn.uncertainty.base_uncertainty_model.BaseUncertaintyModel`
        The model class type.
    description : `str`
        A string introduction of the model.
    """
    model_class: Type[BaseUncertaintyModel]
    description: str


class UncertaintyMethodEnum(Enum):
    """The Enum to store uncertainty methods."""
    simple_conditional_residuals = UncertaintyMethod(
        model_class=SimpleConditionalResidualsModel,
        description="A simple uncertainty method based on conditional residuals."
    )
    """A simple uncertainty method based on conditional residuals."""
    quantile_regression = UncertaintyMethod(
        model_class=QuantileRegressionUncertaintyModel,
        description="A quantile regression based uncertainty model. "
                    "Supports fitting on both original values and residuals."
    )
    """A quantile regression based uncertainty model. Supports fitting on both original values and residuals."""
