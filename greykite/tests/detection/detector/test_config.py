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

import pytest

from greykite.detection.detector.config import ADConfig
from greykite.detection.detector.config import assert_equal_ad_config
from greykite.detection.detector.config_utils import AD_CONFIG_JSON_COMPLETE
from greykite.detection.detector.config_utils import AD_CONFIG_JSON_DEFAULT
from greykite.detection.detector.config_utils import AD_CONFIG_JSON_PARTIAL


def test_ad_config_init():
    """Tests init of the `ADConfig`."""
    ad_config = ADConfig()
    assert ad_config.volatility_features_list is None
    assert ad_config.coverage_grid is None
    assert ad_config.min_admissible_value is None
    assert ad_config.max_admissible_value is None
    assert ad_config.objective is None
    assert ad_config.target_precision is None
    assert ad_config.target_recall is None
    assert ad_config.soft_window_size is None
    assert ad_config.target_anomaly_percent is None
    assert ad_config.variance_scaling is None


def test_ad_config_none():
    """Tests `ADConfig` initialization when input is None."""
    ad_config_none = ADConfig(None)
    ad_config = ADConfig()
    assert_equal_ad_config(ad_config_none, ad_config)


def test_ad_config_from_dict():
    """Tests `from_dict` method of `ADConfig`."""
    for param in [AD_CONFIG_JSON_DEFAULT, AD_CONFIG_JSON_PARTIAL, AD_CONFIG_JSON_COMPLETE]:
        ad_config = param.get("ad_config")
        ad_json = param.get("ad_json")
        ad_dict = json.loads(ad_json)
        translated_ad_config = ADConfig.from_dict(ad_dict)
        assert_equal_ad_config(ad_config, translated_ad_config)

    # Raises error when input is not a dictionary
    with pytest.raises(ValueError, match="not a dictionary"):
        ADConfig.from_dict(5)


def test_ad_config_from_json():
    """Tests `from_json` method of `ADConfig`."""
    for param in [AD_CONFIG_JSON_DEFAULT, AD_CONFIG_JSON_PARTIAL, AD_CONFIG_JSON_COMPLETE]:
        ad_config = param.get("ad_config")
        ad_json = param.get("ad_json")
        translated_ad_config = ADConfig.from_json(ad_json)
        assert_equal_ad_config(ad_config, translated_ad_config)

    # Raises error when input is not a string
    with pytest.raises(ValueError, match="is not a json string."):
        ADConfig.from_json(5)

    # Raises error when input is not a json string
    with pytest.raises(ValueError, match="not a json string"):
        json_str = "This is not a json str"
        ADConfig.from_json(json_str)


def test_assert_equal_ad_config():
    """Tests `assert_equal_ad_config`."""
    ad_config_default = AD_CONFIG_JSON_DEFAULT.get("ad_config")
    assert_equal_ad_config(ad_config_default, ad_config_default)

    ad_config_complete = AD_CONFIG_JSON_COMPLETE.get("ad_config")
    assert_equal_ad_config(ad_config_complete, ad_config_complete)

    # Raises error when the `ADConfig`s do not match
    with pytest.raises(AssertionError, match="Error at dictionary location"):
        assert_equal_ad_config(ad_config_default, ad_config_complete)

    # Raises error when one of the input is not an `ADConfig`
    with pytest.raises(ValueError, match="is not a member of 'ADConfig' class."):
        assert_equal_ad_config(ad_config_default, 4)
