from greykite.detection.detector.data import DetectorData
from greykite.detection.detector.data import ForecastDetectorData


def test_detector_data():
    """Tests ``DetectorData`` data class."""
    data = DetectorData(df=None)

    assert data.df is None
    assert data.anomaly_df is None


def test_forecast_detector_data():
    """Tests ``ForecastDetectorData`` data class."""
    data = ForecastDetectorData(df=None)

    assert data.df is None
    assert data.forecast_dfs is None
    assert data.anomaly_df is None
