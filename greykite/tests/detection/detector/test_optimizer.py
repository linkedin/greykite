import numpy as np
import pandas as pd

from greykite.common.viz.timeseries_annotate import plot_lines_markers
from greykite.detection.detector.optimizer import CalcResult
from greykite.detection.detector.optimizer import Optimizer
from greykite.detection.detector.reward import Reward


def test_optimizer():
    """Tests ``Optimizer`` class."""
    optimizer = Optimizer()
    assert optimizer.reward is None
    assert optimizer.fit_info == {"param_full": None}


def test_optimizer1():
    """Tests ``Optimizer`` class.
    This is a simple test where the optimization is used to find the roots
    of the polynomial: ``x*2 + 2*x + 1``.
    In this simple, example the optimization does not depend on ``data``.
    """

    def distance_to_zero(x):
        """This is the reward function which checks how close to zero
        the result is."""
        return -abs(x)

    reward = Reward(reward_func=distance_to_zero)

    optimizer = Optimizer(
        reward=reward,
        param_iterable=[{"x": x} for x in np.arange(-5, 5, 0.1)])

    def calc_with_param(param, data=None):
        """This is the calculation step with ``param`` and ``data``.
            In this simple example, this does not depend on ``data.``
        """
        x = param["x"]
        return CalcResult(data=x**2 + 2*x + 1, model=None)

    optimizer.calc_with_param = calc_with_param

    # ``data`` is not needed to be passed below
    # because it does not appear in ``calc_with_param`` definition
    optim_res = optimizer.optimize_param()

    best_param = optim_res["best_param"]
    best_param_full = optim_res["best_param_full"]
    assert round(best_param["x"], 2) == -1.00
    assert round(best_param_full["x"], 2) == -1.00

    param_obj_list = optim_res["param_obj_list"]
    param_obj_df = pd.DataFrame.from_records(param_obj_list)

    fig = plot_lines_markers(
        df=param_obj_df,
        x_col="x",
        line_cols=["obj_value"])
    fig.layout.update(title="`Optimizer` parameter search for roots of `x**2 + 2*x + 1`")
    assert fig is not None
    # fig.show()


def test_optimizer2():
    """Tests ``Optimizer`` class.
    This is a slightly more complex test than above
    where the optimization is used to find the roots
    of the polynomial: ``x*p + p*x + 1``
    where ``p`` is is determined by ``data``.
    """

    def distance_to_zero(x):
        """This is the reward function which checks how close to zero
        the result is."""
        return -abs(x)

    reward = Reward(reward_func=distance_to_zero)

    optimizer = Optimizer(
        reward=reward,
        param_iterable=[{"x": x} for x in np.arange(-5, 5, 0.1)])

    def calc_with_param(param, data):
        """This is the calculation step with ``param`` and ``data``.
            In this example, ``p`` which is the polynomial power is
            determined / specified in ``data``.
        """
        x = param["x"]
        p = data["p"]
        return CalcResult(data=x**p + p*x + p, model=None)

    optimizer.calc_with_param = calc_with_param

    optim_res = optimizer.optimize_param(data={"p": 3})

    best_param = optim_res["best_param"]
    best_param_full = optim_res["best_param_full"]
    assert round(best_param["x"], 2) == -0.80
    assert round(best_param_full["x"], 2) == -0.80

    param_obj_list = optim_res["param_obj_list"]
    param_obj_df = pd.DataFrame.from_records(param_obj_list)

    fig = plot_lines_markers(
        df=param_obj_df,
        x_col="x",
        line_cols=["obj_value"])
    fig.layout.update(
        title="`Optimizer` parameter search for roots of `x**p + p*x + p`; p=3")
    assert fig is not None
    # fig.show()
