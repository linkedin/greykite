from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.prophet_template import ProphetTemplate
from greykite.framework.templates.silverkite_template import SilverkiteTemplate
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate


def test_model_template_enum():
    """Tests ModelTemplateEnum accessors"""
    assert ModelTemplateEnum.SILVERKITE.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model with automatic growth, seasonality, holidays," in ModelTemplateEnum.SILVERKITE.value.description

    assert ModelTemplateEnum.SILVERKITE_DAILY_90.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for daily" in ModelTemplateEnum.SILVERKITE_DAILY_90.value.description

    assert ModelTemplateEnum.SILVERKITE_WEEKLY.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for weekly" in ModelTemplateEnum.SILVERKITE_WEEKLY.value.description

    assert ModelTemplateEnum.SILVERKITE_EMPTY.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model with no component included by default" in ModelTemplateEnum.SILVERKITE_EMPTY.value.description

    assert ModelTemplateEnum.SK.value.template_class == SilverkiteTemplate
    assert "Silverkite model with low-level interface" in ModelTemplateEnum.SK.value.description

    assert ModelTemplateEnum.PROPHET.value.template_class == ProphetTemplate
    assert "Prophet model" in ModelTemplateEnum.PROPHET.value.description
