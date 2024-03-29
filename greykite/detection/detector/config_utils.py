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

"""This file contains anomaly detection configs and corresponding json strings
to be used for testing.
"""

from greykite.detection.detector.config import ADConfig


AD_CONFIG_JSON_DEFAULT = dict(
    ad_config=ADConfig(),
    ad_json="{}"
)

AD_CONFIG_PARTIAL = ADConfig(
    volatility_features_list=[],
    max_admissible_value=1000,
    target_recall=None
)

AD_CONFIG_JSON_PARTIAL = dict(
    ad_config=AD_CONFIG_PARTIAL,
    ad_json=AD_CONFIG_PARTIAL.to_json()
)

AD_CONFIG_COMPLETE = ADConfig(
    volatility_features_list=[["dow", "is_event"], ["dow_hr"]],
    coverage_grid=[0.9, 0.95, 0.99],
    min_admissible_value=0,
    max_admissible_value=1000,
    target_precision=0.5,
    target_recall=0.8,
    soft_window_size=3,
    target_anomaly_percent=2.0,
    variance_scaling=False
)

AD_CONFIG_JSON_COMPLETE = dict(
    ad_config=AD_CONFIG_COMPLETE,
    ad_json=AD_CONFIG_COMPLETE.to_json()
)
