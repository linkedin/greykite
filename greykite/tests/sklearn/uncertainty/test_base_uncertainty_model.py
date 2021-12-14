import pandas as pd
import pytest

from greykite.sklearn.uncertainty.base_uncertainty_model import BaseUncertaintyModel
from greykite.sklearn.uncertainty.uncertainty_methods import UncertaintyMethodEnum


@pytest.fixture
def uncertainty_dict():
    uncertainty_dict = dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params=dict()
    )
    return uncertainty_dict


def test_init(uncertainty_dict):
    model = BaseUncertaintyModel(
        uncertainty_dict=uncertainty_dict,
        a=1,
        b="2"
    )
    assert model .uncertainty_dict == uncertainty_dict
    assert model.a == 1
    assert model.b == "2"
    assert model.uncertainty_method is None
    assert model.params is None
    assert model.train_df is None
    assert model.uncertainty_model is None
    assert model.pred_df is None


def test_check_input():
    model = BaseUncertaintyModel(uncertainty_dict=None)
    model._check_input()
    assert model.uncertainty_dict == {}


def test_fit():
    model = BaseUncertaintyModel(
        uncertainty_dict={}
    )
    model.fit(train_df=pd.DataFrame({}))
    assert model.train_df is not None
