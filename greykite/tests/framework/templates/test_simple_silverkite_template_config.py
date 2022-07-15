"""See also ``test_simple_silverkite_template.py``"""

from greykite.framework.templates.simple_silverkite_template_config import COMMON_MODELCOMPONENTPARAM_PARAMETERS
from greykite.framework.templates.simple_silverkite_template_config import MULTI_TEMPLATES
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_ALGO
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_AR
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_CP
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_DAILY_1_CONFIG_1
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_DAILY_1_CONFIG_2
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_DAILY_1_CONFIG_3
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_DSI
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_EMPTY
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_FEASET
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_FREQ
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_GR
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_HOL
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_MONTHLY
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_SEAS
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_WSI
from greykite.framework.templates.simple_silverkite_template_config import VALID_FREQ
from greykite.framework.templates.simple_silverkite_template_config import SimpleSilverkiteTemplateConstants
from greykite.framework.templates.simple_silverkite_template_config import SimpleSilverkiteTemplateOptions


def test_simple_silverkite_template_string_name():
    name = SimpleSilverkiteTemplateOptions()
    assert name.freq == SILVERKITE_FREQ.DEFAULT
    assert name.seas == SILVERKITE_SEAS.DEFAULT
    assert name.cp == SILVERKITE_CP.DEFAULT
    assert name.gr == SILVERKITE_GR.DEFAULT
    assert name.algo == SILVERKITE_ALGO.DEFAULT
    assert name.hol == SILVERKITE_HOL.DEFAULT
    assert name.feaset == SILVERKITE_FEASET.DEFAULT
    assert name.ar == SILVERKITE_AR.DEFAULT
    assert name.dsi == SILVERKITE_DSI.DEFAULT
    assert name.wsi == SILVERKITE_WSI.DEFAULT
    name = SimpleSilverkiteTemplateOptions(
        freq=SILVERKITE_FREQ.HOURLY,
        gr=SILVERKITE_GR.NONE,
        feaset=SILVERKITE_FEASET.OFF
    )
    assert name.freq == SILVERKITE_FREQ.HOURLY
    assert name.seas == SILVERKITE_SEAS.DEFAULT
    assert name.cp == SILVERKITE_CP.DEFAULT
    assert name.gr == SILVERKITE_GR.NONE
    assert name.algo == SILVERKITE_ALGO.DEFAULT
    assert name.hol == SILVERKITE_HOL.DEFAULT
    assert name.feaset == SILVERKITE_FEASET.OFF
    assert name.ar == SILVERKITE_AR.DEFAULT
    assert name.dsi == SILVERKITE_DSI.DEFAULT
    assert name.wsi == SILVERKITE_WSI.DEFAULT


def test_valid_freq():
    assert set(VALID_FREQ) == {"HOURLY", "DAILY", "WEEKLY"}


def test_simple_silvekite_template_constants():
    """Tests `simple_silvekite_template_constants`"""
    constants = SimpleSilverkiteTemplateConstants()
    constants_two = SimpleSilverkiteTemplateConstants()  # the mutable fields are not the same as in `constants`

    assert constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS is not constants_two.COMMON_MODELCOMPONENTPARAM_PARAMETERS
    assert constants.COMMON_MODELCOMPONENTPARAM_PARAMETERS == COMMON_MODELCOMPONENTPARAM_PARAMETERS

    assert constants.MULTI_TEMPLATES is not constants_two.MULTI_TEMPLATES
    assert constants.MULTI_TEMPLATES == MULTI_TEMPLATES

    assert constants.SILVERKITE == SILVERKITE
    assert constants.SILVERKITE_MONTHLY == SILVERKITE_MONTHLY
    assert constants.SILVERKITE_DAILY_1_CONFIG_1 == SILVERKITE_DAILY_1_CONFIG_1
    assert constants.SILVERKITE_DAILY_1_CONFIG_2 == SILVERKITE_DAILY_1_CONFIG_2
    assert constants.SILVERKITE_DAILY_1_CONFIG_3 == SILVERKITE_DAILY_1_CONFIG_3
    assert constants.SILVERKITE_COMPONENT_KEYWORDS.AR.value == SILVERKITE_AR
    assert constants.SILVERKITE_EMPTY == SILVERKITE_EMPTY

    assert constants.VALID_FREQ is not constants_two.VALID_FREQ
    assert constants.VALID_FREQ == VALID_FREQ

    assert constants.SimpleSilverkiteTemplateOptions == SimpleSilverkiteTemplateOptions
