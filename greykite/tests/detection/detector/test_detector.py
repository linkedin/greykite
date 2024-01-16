import numpy as np
import pandas as pd
import pytest
from scipy import stats

from greykite.common.constants import PREDICTED_ANOMALY_COL
from greykite.common.viz.timeseries_annotate import plot_lines_markers
from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.detector.data import DetectorData as Data
from greykite.detection.detector.detector import Detector
from greykite.detection.detector.detector import build_anomaly_percent_reward
from greykite.detection.detector.optimizer import CalcResult
from greykite.detection.detector.reward import Reward


def test_build_anomaly_percent_reward():
    anomaly_percent_dict = {"range": (4, 6), "penalty": -1}
    reward = build_anomaly_percent_reward(anomaly_percent_dict)

    assert reward.min_unpenalized == -0.01
    assert reward.max_unpenalized == float("inf")
    assert reward.penalty == -1

    x = reward.apply(Data(y_pred=[True]*5 + [False]*95))
    assert x == 0

    x = reward.apply(Data(y_pred=[True]*6 + [False]*94))
    assert x == -0.01

    x = reward.apply(Data(y_pred=[True]*4 + [False]*96))
    assert x == -0.01


# This class is implemented to test the `Detector` class
class TukeyDetector(Detector):
    """A detector based on Tukey's outliar definition which is used in Boxplots
        (also invented by Tukey) as well to draw the whiskers.
        Reference: Exploratory Data Analysis, 1977, John Tukey
        Tukey defines outliars to be any points outside the interval range:
        ``(q1 - iqr_coef * iqr, q3 + iqr_coef * iqr)``
        where `q1` is the first quartile, `q3` is the third quartile and
        ``iqr = q3 - q1``.
        ``iqr_coef`` is the coefficient used and it is typically equal to 1.5
        In this detector, we choose the parameter using data.
    """
    def __init__(
            self,
            value_col,
            is_anomaly_col=None,
            reward=None,
            anomaly_percent_dict=None,
            param_iterable=None):

        super().__init__(
            reward=reward,
            anomaly_percent_dict=anomaly_percent_dict,
            param_iterable=param_iterable)

        self.is_anomaly_col = is_anomaly_col
        self.value_col = value_col
        if param_iterable is None:
            self.param_iterable = [{"iqr_coef": x} for x in np.arange(0, 5, 0.1)]

    def fit(
            self,
            data):
        df = data.df
        assert self.value_col in df.columns
        q1 = np.quantile(a=df[self.value_col], q=0.25)
        q3 = np.quantile(a=df[self.value_col], q=0.75)
        iqr = (q3 - q1)

        default_param = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "iqr_coef": None,
            "lower": None,
            "upper": None}

        y_true = df[self.is_anomaly_col]
        data = Data(
            df=df,
            y_true=y_true)

        param_iterable = self.param_iterable

        optim_res = self.optimize_param(
            data=data,
            param_iterable=param_iterable,
            default_param=default_param)

        self.fit_info = {
            "param": optim_res["best_param"],
            "param_full": optim_res["best_param_full"],
            "obj_value": optim_res["best_obj_value"],
            "param_obj_list": optim_res["param_obj_list"]}

        self.fitted_df = self.predict(
            Data(df=df.copy()))

    def calc_with_param(
            self,
            param,
            data):
        df = data.df
        assert self.value_col in df.columns
        param["upper"] = param["q3"] + (param["iqr"] * param["iqr_coef"])
        param["lower"] = param["q1"] - (param["iqr"] * param["iqr_coef"])

        df[PREDICTED_ANOMALY_COL] = (
            (df[self.value_col] < param["lower"]) |
            (df[self.value_col] > param["upper"]))

        data.y_pred = df[PREDICTED_ANOMALY_COL]

        return CalcResult(data=data)


# This class is implemented to test the `Detector` class
class NormalDetector(Detector):
    """A detector based on normal distribution.
        A normal distribution if fitted to data and then any points outside
        the range
            ``(mu - sig * z, mu + sig * z)``
        is considered an outliar, where

        - mu : mean of the data
        - sig : standard deviation of the data
        - z : the coefficient used for defining the confidence interval width.
            We assume ``z = stats.norm.ppf(p)`` for some ``p`` in ``(0.5, 1)`` range.

        This detector uses data to find the optimal ``p`` during fit.
        The optimizer implementation is inherited from
            `~greykite.detection.detector.detector.Detector`

     """
    def __init__(
            self,
            value_col,
            is_anomaly_col=None,
            reward=None,
            anomaly_percent_dict=None,
            param_iterable=None):

        super().__init__(
            reward=reward,
            anomaly_percent_dict=anomaly_percent_dict,
            param_iterable=param_iterable)

        self.is_anomaly_col = is_anomaly_col
        self.value_col = value_col
        if param_iterable is None:
            step = 0.005
            self.param_iterable = [
                {"prob_thresh": x} for x in np.arange(0.5 + step, 1 - step, step)]

    def fit(
            self,
            data):
        df = data.df
        assert self.value_col in df.columns
        mu = np.mean(df[self.value_col])
        sig = np.std(df[self.value_col])

        default_param = {
            "mu": mu,
            "sig": sig}

        y_true = None
        data = Data(
            df=df,
            y_true=y_true)

        param_iterable = self.param_iterable

        optim_res = self.optimize_param(
            data=data,
            param_iterable=param_iterable,
            default_param=default_param)

        self.fit_info = {
            "param": optim_res["best_param"],
            "param_full": optim_res["best_param_full"],
            "obj_value": optim_res["best_obj_value"],
            "param_obj_list": optim_res["param_obj_list"]}

        self.fitted_df = self.predict(
            Data(df=df.copy()))

    def calc_with_param(
            self,
            param,
            data):
        df = data.df
        assert self.value_col in df.columns
        err = stats.norm.ppf(param["prob_thresh"]) * param["sig"]
        param["upper"] = param["mu"] + err
        param["lower"] = param["mu"] - err
        param["err"] = err

        df[PREDICTED_ANOMALY_COL] = (
            (df[self.value_col] < param["lower"]) |
            (df[self.value_col] > param["upper"]))

        data.y_pred = df[PREDICTED_ANOMALY_COL]
        return CalcResult(data=data)


def test_detector():
    """Tests `Detector` class."""
    detector = Detector()
    assert detector.reward is not None
    assert detector.fit_info == {"param_full": None}
    detector.fit = lambda x: 30
    assert detector.fit(1) == 30
    assert detector.fit_info == {"param_full": None}
    assert detector.fitted_df is None
    assert detector.predict(data=None) is None


def test_normal_detector():
    """Tests `NormalDetector` class.
    This test is to demonstrate the usage of the `Detector` class."""
    size = 500
    np.random.seed(1317)
    y = np.random.normal(loc=0.0, scale=1.0, size=size)
    df = pd.DataFrame({"y": y})
    anomaly_percent_dict = {"range": (4, 6), "penalty": -1.0}

    detector = NormalDetector(
        value_col="y",
        anomaly_percent_dict=anomaly_percent_dict)

    reward = detector.reward
    x = reward.apply(Data(y_pred=[True, True, False, False]))
    assert x == -1.45

    x = reward.apply(Data(y_pred=[True]*5 + [False]*95))
    assert x == 0

    x = reward.apply(Data(y_pred=[True]*6 + [False]*94))
    assert x == -0.01

    x = reward.apply(Data(y_pred=[True]*4 + [False]*96))
    assert x == -0.01

    detector.fit(Data(df=df))
    assert detector.value_col == "y"

    param_full = detector.fit_info["param_full"]
    assert round(param_full["err"], 2) == 1.89
    assert round(param_full["prob_thresh"], 3) == 0.965

    param_obj_list = detector.fit_info["param_obj_list"]
    param_obj_df = pd.DataFrame.from_records(param_obj_list)
    fig = plot_lines_markers(
        df=param_obj_df,
        x_col="prob_thresh",
        line_cols=["obj_value"])
    fig.layout.update(title="'NormalDetector' parameter search for prob_thresh")
    assert fig is not None
    # fig.show()


def test_iqr_detector():
    """Tests `TukeyDetector` class.
    This test is to demonstrate the usage of the `Detector` class."""

    def reward_func(data):
        f1 = f1_score(
            y_true=data.y_true,
            y_pred=data.y_pred)
        return f1[True]

    reward = Reward(reward_func=reward_func)

    np.random.seed(seed=1317)
    normal_size = 300
    anomaly_size = 10
    anomalies = np.random.normal(loc=0, scale=5, size=anomaly_size)
    df = pd.DataFrame({
        "y": list(anomalies) + list(np.random.normal(size=normal_size)),
        "is_anomaly": [True]*anomaly_size + [False]*normal_size})

    new_df = pd.DataFrame({"y": [500, -10, -100, 0, 0.1, -0.2, 0.3, 200, 8]})

    detector = TukeyDetector(
        is_anomaly_col="is_anomaly",
        value_col="y",
        reward=reward)

    detector.fit(data=Data(df=df))

    param_full = detector.fit_info["param_full"]
    assert round(param_full["iqr"], 2) == 1.37
    assert round(param_full["iqr_coef"], 2) == 1.8
    assert round(param_full["q1"], 3) == -0.627
    assert round(param_full["q3"], 3) == 0.739
    assert round(param_full["lower"], 2) == -3.09
    assert round(param_full["upper"], 2) == 3.20

    param_obj_list = detector.fit_info["param_obj_list"]
    param_obj_df = pd.DataFrame.from_records(param_obj_list)
    fig = plot_lines_markers(
        df=param_obj_df,
        x_col="iqr_coef",
        line_cols=["obj_value"])
    fig.layout.update(title="'TukeyDetector' parameter search for iqr_coef")
    assert fig is not None
    # fig.show()

    assert detector.fit_info is not None
    assert detector.fitted_df is not None
    pred_data = detector.predict(data=Data(df=new_df))
    y_pred = pred_data.y_pred

    assert np.allclose(
        y_pred,
        np.array([True]*3 + [False]*4 + [True]*2))


def test_summary():
    """Tests `Detector` class summary method."""
    # Tests summary with NormalDetector
    size = 500
    np.random.seed(1317)
    y = np.random.normal(loc=0.0, scale=1.0, size=size)
    df = pd.DataFrame({"y": y})
    anomaly_percent_dict = {"range": (4, 6), "penalty": -1.0}

    detector = NormalDetector(
        value_col="y",
        anomaly_percent_dict=anomaly_percent_dict)
    detector.fit(Data(df=df))

    summary = detector.summary()
    assert "NormalDetector" in summary
    assert "Anomaly Duration" not in summary
    assert "Optimal Parameters" in summary

    # Tests error when `Detector` is not fitted.
    detector = Detector()
    with pytest.raises(ValueError, match="No data to summarize."):
        detector.summary()
