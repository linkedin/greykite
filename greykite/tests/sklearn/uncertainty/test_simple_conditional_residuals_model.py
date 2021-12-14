from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from greykite.common.constants import LOGGER_NAME
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.sklearn.uncertainty.exceptions import UncertaintyError
from greykite.sklearn.uncertainty.simple_conditional_residuals_model import SimpleConditionalResidualsModel
from greykite.sklearn.uncertainty.uncertainty_methods import UncertaintyMethodEnum


@pytest.fixture
def df():
    length = 100
    df = pd.DataFrame({
        TIME_COL: pd.date_range("2020-01-01", freq="D", periods=length),
        VALUE_COL: np.arange(length),
        PREDICTED_COL: np.arange(length) + np.random.randn(length) * length / 10
    })
    return df


@pytest.fixture
def uncertainty_dict():
    uncertainty_dict = dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params=dict()
    )
    return uncertainty_dict


def test_init(uncertainty_dict):
    model = SimpleConditionalResidualsModel(
        uncertainty_dict=uncertainty_dict,
        coverage=0.95,
        time_col=TIME_COL
    )
    assert model.coverage == 0.95
    assert model.time_col == TIME_COL
    assert model.uncertainty_dict == uncertainty_dict

    assert model.value_col is None
    assert model.residual_col is None
    assert model.conditional_cols is None


def test_check_input(df, uncertainty_dict):
    # Wrong uncertainty method.
    with pytest.raises(
            UncertaintyError,
            match="The uncertainty method "):
        uncertainty_dict1 = deepcopy(uncertainty_dict)
        uncertainty_dict1["uncertainty_method"] = "some_uncertainty_method"
        model = SimpleConditionalResidualsModel(
            uncertainty_dict=uncertainty_dict1,
            coverage=0.95,
            time_col=TIME_COL
        )
        model.train_df = df
        model._check_input()
    # ``value_col`` not in ``params``.
    with pytest.raises(
            UncertaintyError,
            match="The parameter value_col is required but not found in "):
        model = SimpleConditionalResidualsModel(
            uncertainty_dict=uncertainty_dict,
            coverage=0.95,
            time_col=TIME_COL
        )
        model.train_df = df[[TIME_COL]]
        model._check_input()

    # ``value_col`` is not a string.
    with pytest.raises(
            UncertaintyError,
            match="`value_col` has to be a string, but found "):
        uncertainty_dict1 = deepcopy(uncertainty_dict)
        uncertainty_dict1["params"]["value_col"] = 1
        model = SimpleConditionalResidualsModel(
            uncertainty_dict=uncertainty_dict1,
            coverage=0.95,
            time_col=TIME_COL
        )
        model.train_df = df
        model._check_input()

    # ``value_col`` not in ``train_df``.
    with pytest.raises(
            UncertaintyError,
            match="`value_col` z not found in `train_df`."):
        uncertainty_dict1 = deepcopy(uncertainty_dict)
        uncertainty_dict1["params"]["value_col"] = "z"
        model = SimpleConditionalResidualsModel(
            uncertainty_dict=uncertainty_dict1,
            coverage=0.95,
            time_col=TIME_COL
        )
        model.train_df = df
        model._check_input()

    # ``residual_col`` is not a string.
    with pytest.raises(
            UncertaintyError,
            match="`residual_col` has to be a string or None, but found "):
        uncertainty_dict1 = deepcopy(uncertainty_dict)
        uncertainty_dict1["params"]["residual_col"] = 1
        model = SimpleConditionalResidualsModel(
            uncertainty_dict=uncertainty_dict1,
            coverage=0.95,
            time_col=TIME_COL
        )
        model.train_df = df
        model._check_input()

    # ``residual_col`` inferred from data.
    with LogCapture(LOGGER_NAME) as log_capture:
        uncertainty_dict1 = deepcopy(uncertainty_dict)
        uncertainty_dict1["params"]["residual_col"] = "residual_col"
        model = SimpleConditionalResidualsModel(
            uncertainty_dict=uncertainty_dict1,
            coverage=0.95,
            time_col=TIME_COL
        )
        model.train_df = df
        model._check_input()
        assert (
                   LOGGER_NAME,
                   "INFO",
                   f"`residual_col` {model.residual_col} is given but not found in `train_df.columns`, "
                   f"however, the prediction column {PREDICTED_COL} is found. "
                   f"Calculating residuals based on the prediction column."
               ) in log_capture.actual()

    # ``conditional_cols`` not a list of strings.
    with pytest.raises(
            UncertaintyError,
            match="`conditional_cols` \\['a', 1\\] must be a list of strings."):
        uncertainty_dict1 = deepcopy(uncertainty_dict)
        uncertainty_dict1["params"]["conditional_cols"] = ["a", 1]
        model = SimpleConditionalResidualsModel(
            uncertainty_dict=uncertainty_dict1,
            coverage=0.95,
            time_col=TIME_COL
        )
        model.train_df = df
        model._check_input()

    # ``conditional_cols`` not found in ``train_df``.
    with pytest.raises(
            UncertaintyError,
            match="The following conditional columns are not found in `train_df`: \\['x'\\]"):
        uncertainty_dict1 = deepcopy(uncertainty_dict)
        uncertainty_dict1["params"]["conditional_cols"] = ["dow", "dow_hr", "x"]
        model = SimpleConditionalResidualsModel(
            uncertainty_dict=uncertainty_dict1,
            coverage=0.95,
            time_col=TIME_COL
        )
        model.train_df = df
        model._check_input()

    # ``coverage`` is not in bound.
    with pytest.raises(
            UncertaintyError,
            match="Coverage must be between 0 and 1, found 1.95"):
        model = SimpleConditionalResidualsModel(
            uncertainty_dict=uncertainty_dict,
            coverage=1.95,
            time_col=TIME_COL
        )
        model.train_df = df
        model._check_input()

    # Smooth run.
    model = SimpleConditionalResidualsModel(
        uncertainty_dict=uncertainty_dict,
        coverage=0.99,
        time_col=TIME_COL
    )
    model.train_df = df
    model._check_input()
    assert model.value_col == VALUE_COL  # auto populated
    assert "dow" in model.train_df  # auto populated
    assert list(np.round(model.params["quantiles"], 3)) == [0.005, 0.995]  # auto populated


def test_fit_and_predict(df, uncertainty_dict):
    uncertainty_dict["params"]["conditional_cols"] = ["dow"]
    uncertainty_dict["params"]["residual_col"] = "residual_col"
    model = SimpleConditionalResidualsModel(
        uncertainty_dict=uncertainty_dict,
        coverage=0.99,
        time_col=TIME_COL
    )

    # fit
    model.fit(train_df=df)
    assert model.uncertainty_model is not None

    # predict
    pred = model.predict(fut_df=df)
    assert PREDICTED_LOWER_COL in pred.columns
    assert PREDICTED_UPPER_COL in pred.columns
    assert (pred[PREDICTED_LOWER_COL] + pred[PREDICTED_UPPER_COL]).round(2).equals(
        (pred[PREDICTED_COL] * 2).round(2))
    # Conditioning on ``dow``.
    assert (pred[PREDICTED_UPPER_COL].iloc[-1] - pred[PREDICTED_LOWER_COL].iloc[-1]
            != pred[PREDICTED_UPPER_COL].iloc[-2] - pred[PREDICTED_LOWER_COL].iloc[-2])

    # ``value_col`` not in ``fut_df``.
    with pytest.raises(
            UncertaintyError,
            match=f"The value column {VALUE_COL} is not found in `fut_df`."):
        model.predict(fut_df=df[[TIME_COL]])
