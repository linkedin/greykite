from greykite.sklearn.uncertainty.simple_conditional_residuals_model import SimpleConditionalResidualsModel
from greykite.sklearn.uncertainty.uncertainty_methods import UncertaintyMethodEnum


def test_uncertainty_method_enum():
    assert UncertaintyMethodEnum.simple_conditional_residuals.value.model_class == SimpleConditionalResidualsModel
    assert ("A simple uncertainty method based on conditional residuals."
            in UncertaintyMethodEnum.simple_conditional_residuals.value.description)
