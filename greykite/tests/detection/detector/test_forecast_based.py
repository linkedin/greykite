from greykite.detection.common.ad_evaluation import f1_score
from greykite.detection.common.testing_utils import sim_anomalous_data_and_forecasts
from greykite.detection.detector.data import ForecastDetectorData as Data
from greykite.detection.detector.forecast_based import ForecastBasedDetector
from greykite.detection.detector.reward import Reward


def test_forecast_based_detector():
    data = sim_anomalous_data_and_forecasts(
        sample_size=200,
        anomaly_num=20,
        seed=1317)

    df = data["df"]
    forecast_dfs = data["forecast_dfs"]

    def reward_func(y_true, y_pred):
        f1 = f1_score(
            y_true=y_true,
            y_pred=y_pred)
        return f1[True]

    detector = ForecastBasedDetector(
        reward=Reward(reward_func),
        value_cols=["y"],
        pred_cols=["y_pred"],
        is_anomaly_col="is_anomaly",
        join_cols=["ts"])

    # Checks if the attributes are inherited from the `Detector` class.
    assert detector.data is None
    assert detector.fitted_df is None
    assert detector.fit_info == {"param_full": None}

    joined_dfs = detector.join_with_forecasts(
        df=df,
        forecast_dfs=forecast_dfs)

    assert len(joined_dfs) == 2
    assert list(joined_dfs.keys()) == [0, 1]
    for joined_df in joined_dfs.values():
        assert "y_pred" in joined_df.columns

    assert detector.data is None

    detector.fit()
    # Since `fit` is not implemented and inherited from base class: `Detector`
    # it does not do anything
    assert detector.data is None

    data = Data(df=df, forecast_dfs=forecast_dfs)
    assert data.joined_dfs is None
    detector.prep_df_for_predict(data)

    assert data.joined_dfs is not None

    # Checks to see if the attached `joined_dfs` to `data`
    # is the same as previously calculated `joined_dfs` in the above
    assert len(data.joined_dfs) == 2
    assert list(data.joined_dfs.keys()) == [0, 1]

    for i in [0, 1]:
        joined_df_direct = joined_dfs[i]
        joined_df_from_detector = data.joined_dfs[i]
        assert joined_df_direct.equals(joined_df_from_detector)
