import numpy as np

from greykite.common.constants import PREDICTED_ANOMALY_COL
from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.common.testing_utils import sim_anomalous_data_and_forecasts
from greykite.detection.detector.best_forecast import BestForecastDetector
from greykite.detection.detector.data import ForecastDetectorData as Data
from greykite.detection.detector.optimizer import CalcResult
from greykite.detection.detector.reward import Reward


def test_best_forecast_detector():
    data = sim_anomalous_data_and_forecasts(
        sample_size=200,
        anomaly_num=20,
        seed=1317)

    # train data
    df_train = data["df_train"]
    forecast_dfs_train = data["forecast_dfs_train"]

    # test data
    df_test = data["df_test"]
    forecast_dfs_test = data["forecast_dfs_test"]

    def reward_func(data):
        f1 = f1_score(
            y_true=data.y_true,
            y_pred=data.y_pred)
        return f1[True]

    APE_PARAM_ITERABLE = [{"ape_thresh": x} for x in np.arange(0, 4, 0.05)]

    detector = BestForecastDetector(
        value_cols=["y"],
        pred_cols=["y_pred"],
        is_anomaly_col="is_anomaly",
        join_cols=["ts"],
        reward=Reward(reward_func),
        param_iterable=APE_PARAM_ITERABLE)

    assert detector.param_iterable == APE_PARAM_ITERABLE

    # Checks if the attributes are inherited from the `Detector` class
    assert detector.data is None
    assert detector.fitted_df is None
    assert detector.fit_info == {"param_full": None}

    joined_dfs = detector.join_with_forecasts(
        df=df_train,
        forecast_dfs=forecast_dfs_train)

    assert len(joined_dfs) == 2
    assert list(joined_dfs.keys()) == [0, 1]
    for joined_df in joined_dfs.values():
        assert "y_pred" in joined_df.columns
    assert detector.data is None

    # In order to apply `fit`, we need to implement
    # `add_features_one_df`
    # `calc_with_param`
    def add_features_one_df(joined_df):
        joined_df["ape"] = (
            abs(joined_df["y"] - joined_df["y_pred"]) /
            abs(joined_df["y"]))
        return joined_df

    def calc_with_param(param, data):
        pred_df = data.joined_dfs[param["forecast_id"]]
        y_pred = (pred_df["ape"] > param["ape_thresh"])
        pred_df[PREDICTED_ANOMALY_COL] = y_pred
        data.pred_df = pred_df
        data.y_pred = y_pred
        return CalcResult(data=data)

    detector.add_features_one_df = add_features_one_df
    detector.calc_with_param = calc_with_param

    detector.fit(Data(
        df=df_train,
        forecast_dfs=forecast_dfs_train))

    # Checking the fitted parameters
    param_full = detector.fit_info["param_full"]
    assert round(param_full["ape_thresh"], 2) == 0.15
    assert param_full["forecast_id"] == 0

    assert detector.data.joined_dfs is not None

    # Checks to see if the attached `joined_dfs` to `data`
    # is the same as previously calculated `joined_dfs` in the above
    assert len(detector.data.joined_dfs) == 2
    assert list(detector.data.joined_dfs.keys()) == [0, 1]

    common_cols = ["ts", "y", "is_anomaly", "y_pred"]
    for i in [0, 1]:
        joined_df_direct = joined_dfs[i]
        joined_df_from_detector = detector.data.joined_dfs[i]
        assert joined_df_direct[common_cols].equals(
            joined_df_from_detector[common_cols])

    # Since forecast 0 is best forecast we expect the corresponding
    # joined data to have the predictions within
    after_fit_cols = common_cols + ["ape", "is_anomaly_predicted"]
    joined_df_from_detector = detector.data.joined_dfs[0]
    assert list(joined_df_from_detector.columns) == after_fit_cols

    # The other joined_df will not have the predictions
    after_fit_cols = common_cols + ["ape"]
    joined_df_from_detector = detector.data.joined_dfs[1]
    assert list(joined_df_from_detector.columns) == after_fit_cols

    # Prediction step on test set
    data = detector.predict(Data(
        df=df_test,
        forecast_dfs=forecast_dfs_test,
        y_true=df_test[detector.is_anomaly_col]))

    test_obj_value = detector.reward.apply(data)

    assert round(test_obj_value, 3) == 0.421
