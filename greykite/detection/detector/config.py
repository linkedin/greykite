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
# original author: Sayan Patra

import json
from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Optional

from greykite.common.python_utils import assert_equal
from greykite.framework.templates.autogen.forecast_config import from_bool
from greykite.framework.templates.autogen.forecast_config import from_float
from greykite.framework.templates.autogen.forecast_config import from_int
from greykite.framework.templates.autogen.forecast_config import from_list_float
from greykite.framework.templates.autogen.forecast_config import from_list_list_str
from greykite.framework.templates.autogen.forecast_config import from_none
from greykite.framework.templates.autogen.forecast_config import from_str
from greykite.framework.templates.autogen.forecast_config import from_union


F1 = "F1"
"""This constant means F1 in anomaly detection evaluation. This is used in `objective` field of `ADConfig`."""
RECALL = "RECALL"
"""This constant means Recall in anomaly detection evaluation. This is used in `objective` field of `ADConfig`."""
PRECISION = "PRECISION"
"""This constant means Precision in anomaly detection evaluation. This is used in `objective` field of `ADConfig`."""


@dataclass
class ADConfig:
    """Config for providing parameters to the Anomaly Detection library."""
    volatility_features_list: Optional[List[List[str]]] = None
    """Set of volatility features used to optimize anomaly detection performance."""
    coverage_grid: Optional[List[float]] = None
    """A set of coverage values to optimize anomaly detection performance.
    Optimum coverage is chosen among this list."""
    ape_grid: Optional[List[float]] = None
    """A set of absolute percentage error (APE) threshold values to optimize anomaly detection performance."""
    sape_grid: Optional[List[float]] = None
    """A set of symmetric absolute percentage error (SAPE) threshold values to optimize anomaly detection performance."""
    min_admissible_value: Optional[float] = None
    """Lowest admissible value for the obtained confidence intervals."""
    max_admissible_value: Optional[float] = None
    """Highest admissible value for the obtained confidence intervals."""
    objective: Optional[str] = None
    """The main objective for optimization. It can be either of: F1, PRECISION or RECALL."""
    target_precision: Optional[float] = None
    """Minimum precision to achieve during AD optimization in a labeled data."""
    target_recall: Optional[float] = None
    """Minimum recall to achieve during AD optimization in a labeled data."""
    soft_window_size: Optional[int] = None
    """Window size for soft precision, recall and f1 in a labeled data."""
    target_anomaly_percent: Optional[float] = None
    """Desired anomaly percent during AD optimization of an unlabeled data (0-100 scale)."""
    variance_scaling: Optional[bool] = None
    """The variance scaling method in ridge / linear regression takes into account
    (1) the degrees of freedom of the model; (2) the standard error from the coefficients,
    hence will provide more accurate variance estimate / prediction intervals."""

    @staticmethod
    def from_dict(obj: Any) -> 'ADConfig':
        """Converts a dictionary to the corresponding instance of the `ADConfig` class.
        Raises ValueError if the input is not a dictionary.
        """
        if not isinstance(obj, dict):
            raise ValueError(f"The input ({obj}) is not a dictionary.")
        volatility_features_list = from_union([from_list_list_str, from_none], obj.get("volatility_features_list"))
        coverage_grid = from_union([from_list_float, from_none], obj.get("coverage_grid"))
        ape_grid = from_union([from_list_float, from_none], obj.get("ape_grid"))
        sape_grid = from_union([from_list_float, from_none], obj.get("sape_grid"))
        min_admissible_value = from_union([from_float, from_none], obj.get("min_admissible_value"))
        max_admissible_value = from_union([from_float, from_none], obj.get("max_admissible_value"))
        objective = from_union([from_str, from_none], obj.get("objective"))
        target_precision = from_union([from_float, from_none], obj.get("target_precision"))
        target_recall = from_union([from_float, from_none], obj.get("target_recall"))
        soft_window_size = from_union([from_int, from_none], obj.get("soft_window_size"))
        target_anomaly_percent = from_union([from_float, from_none], obj.get("target_anomaly_percent"))
        variance_scaling = from_union([from_bool, from_none], obj.get("variance_scaling"))

        return ADConfig(
            volatility_features_list=volatility_features_list,
            coverage_grid=coverage_grid,
            ape_grid=ape_grid,
            sape_grid=sape_grid,
            min_admissible_value=min_admissible_value,
            max_admissible_value=max_admissible_value,
            objective=objective,
            target_precision=target_precision,
            target_recall=target_recall,
            soft_window_size=soft_window_size,
            target_anomaly_percent=target_anomaly_percent,
            variance_scaling=variance_scaling
        )

    def to_dict(self) -> dict:
        """Converts an instance of the `ADConfig` class to its dictionary format."""
        result = dict()
        result["volatility_features_list"] = from_union(
            [from_list_list_str, from_none],
            self.volatility_features_list)
        result["coverage_grid"] = from_union(
            [from_list_float, from_none],
            self.coverage_grid)
        result["ape_grid"] = from_union(
            [from_list_float, from_none],
            self.ape_grid)
        result["sape_grid"] = from_union(
            [from_list_float, from_none],
            self.sape_grid)
        result["min_admissible_value"] = from_union(
            [from_float, from_none],
            self.min_admissible_value)
        result["max_admissible_value"] = from_union(
            [from_float, from_none],
            self.max_admissible_value)
        result["objective"] = from_union(
            [from_str, from_none],
            self.objective)
        result["target_precision"] = from_union(
            [from_float, from_none],
            self.target_precision)
        result["target_recall"] = from_union(
            [from_float, from_none],
            self.target_recall)
        result["soft_window_size"] = from_union(
            [from_int, from_none],
            self.soft_window_size)
        result["target_anomaly_percent"] = from_union(
            [from_float, from_none],
            self.target_anomaly_percent)
        result["variance_scaling"] = from_union(
            [from_bool, from_none],
            self.variance_scaling)

        return result

    @staticmethod
    def from_json(obj: Any) -> 'ADConfig':
        """Converts a json string to the corresponding instance of the `ADConfig` class.
        Raises ValueError if the input is not a json string.
        """
        try:
            ad_dict = json.loads(obj)
        except Exception:
            raise ValueError(f"The input ({obj}) is not a json string.")

        return ADConfig.from_dict(ad_dict)

    def to_json(self) -> str:
        """Converts an instance of the `ADConfig` class to its json string format."""
        ad_dict = self.to_dict()
        return json.dumps(ad_dict)


def assert_equal_ad_config(
        ad_config_1: ADConfig,
        ad_config_2: ADConfig):
    """Asserts equality between two instances of `ADConfig`.
    Raises a ValueError in case of a parameter mismatch.

    Parameters
    ----------
    ad_config_1: `ADConfig`
        First instance of the
        :class:`~greykite.detection.detector.config.ADConfig` for comparing.
    ad_config_2: `ADConfig`
        Second instance of the
        :class:`~greykite.detection.detector.config.ADConfig` for comparing.

    Raises
    -------
    AssertionError
        If `ADConfig`s do not match, else returns None.
    """
    if not isinstance(ad_config_1, ADConfig):
        raise ValueError(f"The input ({ad_config_1}) is not a member of 'ADConfig' class.")
    if not isinstance(ad_config_2, ADConfig):
        raise ValueError(f"The input ({ad_config_2}) is not a member of 'ADConfig' class.")

    assert_equal(ad_config_1.to_dict(), ad_config_2.to_dict())
