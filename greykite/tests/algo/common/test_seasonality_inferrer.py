from copy import deepcopy

import pandas as pd
import pytest

from greykite.algo.common.seasonality_inferrer import SeasonalityInferConfig
from greykite.algo.common.seasonality_inferrer import SeasonalityInferrer
from greykite.algo.common.seasonality_inferrer import TrendAdjustMethodEnum
from greykite.common import constants as cst
from greykite.common.testing_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests


@pytest.fixture
def df():
    df = generate_df_for_tests(
        freq="D",
        periods=1000
    )["df"]
    return df


@pytest.fixture
def configs():
    configs = [
        SeasonalityInferConfig(
            seas_name="yearly",
            col_name="ct1",
            period=1,
            max_order=10,
            aggregation_period="W",
            offset=1
        ),
        SeasonalityInferConfig(
            seas_name="weekly",
            col_name="year_woy_iso",
            period=7,
            max_order=10,
            aggregation_period=None,
            offset=None
        )
    ]
    return configs


def test_trend_adjust_method_enum():
    """Tests the trend adjust method Enum."""
    assert TrendAdjustMethodEnum.seasonal_average.value == "seasonal_average"
    assert TrendAdjustMethodEnum.overall_average.value == "overall_average"
    assert TrendAdjustMethodEnum.spline_fit.value == "spline_fit"


def test_init():
    """Tests instantiation."""
    model = SeasonalityInferrer()
    assert model.df is None
    assert model.time_col is None
    assert model.value_col is None
    assert model.fourier_series_orders is None
    assert model.FITTED_TREND_COL == "FITTED_TREND"


def test_process_params(configs):
    """Tests process input parameters."""
    model = SeasonalityInferrer()
    configs_new = model._process_params(
        configs=configs,
        adjust_trend_method=TrendAdjustMethodEnum.spline_fit.name,
        adjust_trend_param=None,
        fit_algorithm="linear",
        tolerance=0.0,
        plotting=False,
        aggregation_period=None,
        offset=None,
        criterion="bic"
    )
    assert configs_new == [
        SeasonalityInferConfig(
            seas_name="yearly",
            col_name="ct1",
            period=1,
            max_order=10,
            adjust_trend_method=TrendAdjustMethodEnum.spline_fit.name,
            adjust_trend_param=None,
            fit_algorithm="linear",
            tolerance=0.0,
            plotting=False,
            aggregation_period="W",
            offset=1,
            criterion="bic"
        ),
        SeasonalityInferConfig(
            seas_name="weekly",
            col_name="year_woy_iso",
            period=7,
            max_order=10,
            adjust_trend_method=TrendAdjustMethodEnum.spline_fit.name,
            adjust_trend_param=None,
            fit_algorithm="linear",
            tolerance=0.0,
            plotting=False,
            aggregation_period=None,
            offset=0,
            criterion="bic"
        )
    ]


def test_apply_default_value(configs):
    """Tests apply default value util function."""
    model = SeasonalityInferrer()
    # Overrides value.
    configs_new = model._apply_default_value(
        configs=deepcopy(configs),
        param_name="offset",
        override_value=2,
        default_value=10)
    assert configs_new[0].offset == 2
    assert configs_new[1].offset == 2
    # Does not override value and fills with default value.
    configs_new = model._apply_default_value(
        configs=deepcopy(configs),
        param_name="offset",
        override_value=None,
        default_value=10)
    assert configs_new[0].offset == 1
    assert configs_new[1].offset == 10
    # Override value not valid.
    with pytest.raises(
            ValueError,
            match=f"The parameter 'offset' has value 2, "
                  f"which is not valid. Valid values are '\\[1, 3\\]'."):
        model._apply_default_value(
            configs=deepcopy(configs),
            param_name="offset",
            override_value=2,
            default_value=1,
            allowed_values=[1, 3])
    # Given value not valid.
    with pytest.raises(
            ValueError,
            match=f"The parameter 'offset' in SeasonalityInferConfig has value 1, "
                  f"which is not valid. Valid values are '\\[2, 3\\]'."):
        model._apply_default_value(
            configs=deepcopy(configs),
            param_name="offset",
            override_value=None,
            default_value=2,
            allowed_values=[2, 3])


def test_adjust_trend(df):
    """Tests adjust trend methods."""
    # Seasonal average
    model = SeasonalityInferrer()
    df_adj = model._adjust_trend(
        df=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        method=TrendAdjustMethodEnum.seasonal_average.value,
        trend_average_col="year_woy_iso")
    assert len(df_adj.columns) == 3
    assert_equal(
        df[cst.VALUE_COL].values,
        (df_adj[cst.VALUE_COL] + df_adj[model.FITTED_TREND_COL]).values,
        rel=1e-10
    )
    assert_equal(
        df_adj.loc[:4, "y"],
        pd.Series([0, 2.7583, 3.6115, 0.5488, 0.7180], name="y"),
        rel=1e-3
    )
    assert_equal(
        df_adj.loc[:4, model.FITTED_TREND_COL],
        pd.Series([-4.4135, -0.7247, -0.7247, -0.7247, -0.7247], name=model.FITTED_TREND_COL),
        rel=1e-3
    )

    # Overall average
    df_adj = model._adjust_trend(
        df=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        method=TrendAdjustMethodEnum.overall_average.value)
    assert len(df_adj.columns) == 3
    assert_equal(
        df[cst.VALUE_COL].values,
        (df_adj[cst.VALUE_COL] + df_adj[model.FITTED_TREND_COL]).values,
        rel=1e-10
    )
    assert_equal(
        df_adj.loc[:4, "y"],
        pd.Series([-8.4033, -1.9561, -1.1030, -4.1656, -3.9965], name="y"),
        rel=1e-3
    )
    assert_equal(
        df_adj.loc[:4, model.FITTED_TREND_COL],
        pd.Series([3.9898, 3.9898, 3.9898, 3.9898, 3.9898], name=model.FITTED_TREND_COL),
        rel=1e-3
    )

    # Spline fit
    df_adj = model._adjust_trend(
        df=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        method=TrendAdjustMethodEnum.spline_fit.value,
        spline_fit_degree=3)
    assert len(df_adj.columns) == 3
    assert_equal(
        df[cst.VALUE_COL].values,
        (df_adj[cst.VALUE_COL] + df_adj[model.FITTED_TREND_COL]).values,
        rel=1e-10
    )
    assert_equal(
        df_adj.loc[:4, "y"],
        pd.Series([-3.6315, 2.8088, 3.6551, 0.5856, 0.7478], name="y"),
        rel=1e-3
    )
    assert_equal(
        df_adj.loc[:4, model.FITTED_TREND_COL],
        pd.Series([-0.7820, -0.7752, -0.7683, -0.7614, -0.7545], name=model.FITTED_TREND_COL),
        rel=1e-3
    )

    # Method not recognized
    method = "some_method"
    with pytest.raises(
            ValueError,
            match=f"The trend adjust method '{method}' is not a valid name. "
                  f"Available methods are \\['seasonal_average', 'overall_average', 'spline_fit', 'none'\\]."):
        model._adjust_trend(
            df=df,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            method=method)

    # ``trend_average_col`` not found.
    with pytest.raises(
            ValueError,
            match=f"The trend_average_col 'some_col' is neither found in df "):
        model._adjust_trend(
            df=df,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            method=TrendAdjustMethodEnum.seasonal_average.value,
            trend_average_col="some_col")

    # Negative spline degree.
    with pytest.raises(
            ValueError,
            match=f"Spline degree has be to a positive integer, "
                  f"but found -1."):
        model._adjust_trend(
            df=df,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            method=TrendAdjustMethodEnum.spline_fit.value,
            spline_fit_degree=-1)


def test_process_df(df):
    """Tests adjust trend and aggregation."""
    model = SeasonalityInferrer()
    df_adj = model._process_df(
        df=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        adjust_trend_method=TrendAdjustMethodEnum.seasonal_average.value,
        adjust_trend_params=None,
        aggregation_period="W-SUN"
    )
    assert pd.infer_freq(df_adj[cst.TIME_COL]) == "W-SUN"


def test_tolerance(df):
    """Tests ``tolerance``."""
    model = SeasonalityInferrer()
    result1 = model.infer_fourier_series_order(
        df=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        configs=[
            SeasonalityInferConfig(
                seas_name="yearly",
                col_name="toy",
                period=1.0,
                max_order=50,
                adjust_trend_param=dict(trend_average_col="year"),
                tolerance=0.0,
                aggregation_period="W",
                offset=0
            )
        ],
        adjust_trend_method="seasonal_average",
        fit_algorithm="linear",
        plotting=True,
        criterion="bic",
    )
    result2 = model.infer_fourier_series_order(
        df=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        configs=[
            SeasonalityInferConfig(
                seas_name="yearly",
                col_name="toy",
                period=1.0,
                max_order=50,
                adjust_trend_param=dict(trend_average_col="year"),
                tolerance=1.0,
                aggregation_period="W",
                offset=0
            )
        ],
        adjust_trend_method="seasonal_average",
        fit_algorithm="linear",
        plotting=True,
        criterion="bic",
    )
    # With `tolerance`, the new order is smaller.
    assert result2["best_orders"]["yearly"] < result1["best_orders"]["yearly"]


def test_infer_order(df):
    """Tests the full functionality of inferring orders."""
    model = SeasonalityInferrer()
    result = model.infer_fourier_series_order(
        df=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        configs=[
            SeasonalityInferConfig(
                seas_name="yearly",
                col_name="toy",
                period=1.0,
                max_order=50,
                adjust_trend_param=dict(trend_average_col="year"),
                tolerance=0.0,
                aggregation_period="W",
                offset=0
            ),
            SeasonalityInferConfig(
                seas_name="quarterly",
                col_name="toq",
                period=1.0,
                max_order=20,
                adjust_trend_param=dict(trend_average_col="year_quarter"),
                tolerance=0.0,
                aggregation_period="2D",
                offset=0
            ),
            SeasonalityInferConfig(
                seas_name="monthly",
                col_name="tom",
                period=1.0,
                max_order=20,
                adjust_trend_param=dict(trend_average_col="year_month"),
                tolerance=0.0,
                aggregation_period=None,
                offset=2
            ),
            SeasonalityInferConfig(
                seas_name="weekly",
                col_name="tow",
                period=7.0,
                max_order=10,
                adjust_trend_param=dict(trend_average_col="year_woy_iso"),
                tolerance=0.005,
                aggregation_period=None,
                offset=-100
            )
        ],
        adjust_trend_method="seasonal_average",
        fit_algorithm="linear",
        plotting=True,
        criterion="bic",
    )
    assert len(result["result"]) == 4
    assert result["result"][0]["seas_name"] == "yearly"
    assert result["result"][1]["seas_name"] == "quarterly"
    assert result["result"][2]["seas_name"] == "monthly"
    assert result["result"][3]["seas_name"] == "weekly"
    assert len(result["result"][0]["orders"]) == 50
    assert len(result["result"][1]["aics"]) == 20
    assert len(result["result"][2]["bics"]) == 20
    assert result["result"][2]["best_bic_order"] > 0
    assert result["best_orders"]["weekly"] == 0  # offset is truncated at 0
    assert len(result["result"][1]["fig"].data) == 8
