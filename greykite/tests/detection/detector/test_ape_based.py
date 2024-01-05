import pandas as pd
import plotly.express as px

from greykite.common.viz.timeseries_annotate import plot_lines_markers
from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.common.testing_utils import sim_anomalous_data_and_forecasts
from greykite.detection.detector.ad_utils import vertical_concat_dfs
from greykite.detection.detector.ape_based import APE_PARAM_ITERABLE
from greykite.detection.detector.ape_based import APEDetector
from greykite.detection.detector.data import ForecastDetectorData as Data
from greykite.detection.detector.reward import Reward


def test_ape_detector():
    """Tests `APEDetector`."""
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

    detector = APEDetector(
        reward=Reward(reward_func),
        value_cols=["y"],
        pred_cols=["y_pred"],
        is_anomaly_col="is_anomaly",
        join_cols=["ts"])

    assert detector.param_iterable == APE_PARAM_ITERABLE
    # Checks if the attributes are inherited from the `Detector` class
    assert detector.data is None
    assert detector.fitted_df is None
    assert detector.fit_info == {"param_full": None}

    joined_dfs = detector.join_with_forecasts(
        df=df_train,
        forecast_dfs=forecast_dfs_train)
    joined_dfs = detector.add_features(joined_dfs)

    assert len(joined_dfs) == 2
    assert list(joined_dfs.keys()) == [0, 1]
    for joined_df in joined_dfs.values():
        assert "y_pred" in joined_df.columns
    assert detector.data is None

    # Concats the joined data to plot
    joined_df_all = vertical_concat_dfs(
        df_list=list(joined_dfs.values()),
        join_cols=detector.join_cols,
        common_value_cols=["y", "is_anomaly"],
        different_value_cols=["ape", "y_pred"])

    fig = plot_lines_markers(
        df=joined_df_all,
        x_col="ts",
        line_cols=["ape0", "ape1"])
    fig.layout.update(title="Comparing two forecasts using APE")
    assert fig is not None
    fig.update_yaxes()
    # fig.show()

    # Calculates one reward value
    calc_result = detector.calc_with_param(
        data=Data(
            joined_dfs=joined_dfs,
            y_true=df_train["is_anomaly"]),
        param={"forecast_id": 0, "ape_thresh": 0.5})

    obj_value = detector.reward.apply(calc_result.data)

    assert round(obj_value, 3) == 0.25

    # Fits
    detector.fit(Data(
        df=df_train,
        forecast_dfs=forecast_dfs_train))

    param_obj_list = detector.fit_info["param_obj_list"]
    param_eval_df = pd.DataFrame.from_records(param_obj_list)
    assert list(param_eval_df.columns) == ["ape_thresh", "forecast_id", "obj_value"]

    param_eval_df["forecast_id"] = param_eval_df["forecast_id"].map(str)
    fig = px.line(
        param_eval_df,
        x="ape_thresh",
        y="obj_value",
        color="forecast_id",
        title="'APEDetector' result of parameter search for APE threshold")
    assert fig is not None
    # fig.show()

    # Prediction step on test set
    data = detector.predict(Data(
        df=df_test,
        forecast_dfs=forecast_dfs_test,
        y_true=df_test[detector.is_anomaly_col]))

    test_obj_value = detector.reward.apply(data)

    assert round(test_obj_value, 3) == 0.421
