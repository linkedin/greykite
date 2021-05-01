from typing import Dict

import pandas

from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.template_interface import TemplateInterface


class MyTemplate(TemplateInterface):
    def __init__(self):
        super().__init__()

    @property
    def allow_model_template_list(self):
        return False

    @property
    def allow_model_components_param_list(self):
        return False

    def apply_template_for_pipeline_params(self, df: pandas.DataFrame, config: ForecastConfig) -> Dict:
        assert config is not None
        return {"value": df.shape[0]}


def test_template_interface():
    mt = MyTemplate()
    assert mt.df is None
    assert mt.config is None
    assert mt.pipeline_params is None
    assert mt.allow_model_template_list is False
    assert mt.allow_model_components_param_list is False

    df = pandas.DataFrame({"a": [1, 2, 3]})
    assert mt.apply_template_for_pipeline_params(
        df=df,
        config=ForecastConfig()) == {"value": df.shape[0]}
