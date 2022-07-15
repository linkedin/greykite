import datetime

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

import greykite.common.constants as cst
from greykite.common.testing_utils import generate_df_for_tests
from greykite.sklearn.estimator.lag_based_estimator import LagBasedEstimator
from greykite.sklearn.estimator.multistage_forecast_estimator import MultistageForecastEstimator
from greykite.sklearn.estimator.multistage_forecast_estimator import MultistageForecastModelConfig
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator
from greykite.sklearn.uncertainty.uncertainty_methods import UncertaintyMethodEnum


@pytest.fixture
def params():
    params = dict(
        forecast_horizon=12,
        freq="H",
        model_configs=[
            MultistageForecastModelConfig(
                train_length="30D",
                fit_length="30D",
                agg_func="mean",
                agg_freq="D",
                estimator=SimpleSilverkiteEstimator,
                estimator_params=dict(
                    coverage=None,
                    forecast_horizon=1,
                    freq="D",
                    daily_seasonality=0,
                    weekly_seasonality=3,
                    quarterly_seasonality=5,
                    monthly_seasonality=0,
                    yearly_seasonality=0,
                    changepoints_dict=None,
                    autoreg_dict="auto",
                    holidays_to_model_separately="auto",
                    holiday_lookup_countries="auto",
                    holiday_pre_num_days=1,
                    holiday_post_num_days=1,
                    holiday_pre_post_num_dict=None,
                    daily_event_df_dict=None,
                    fit_algorithm_dict={
                        "fit_algorithm": "ridge",
                        "fit_algorithm_params": None,
                    },
                    feature_sets_enabled="auto",
                    max_daily_seas_interaction_order=5,
                    max_weekly_seas_interaction_order=2,
                    extra_pred_cols=[],
                    min_admissible_value=None,
                    max_admissible_value=None,
                    normalize_method="zero_to_one"
                )
            ),
            MultistageForecastModelConfig(
                train_length="7D",
                fit_length="7D",
                agg_func="mean",
                agg_freq=None,
                estimator=SimpleSilverkiteEstimator,
                estimator_params=dict(
                    coverage=None,
                    forecast_horizon=12,
                    freq="H",
                    growth_term=None,
                    daily_seasonality=12,
                    weekly_seasonality=0,
                    quarterly_seasonality=0,
                    monthly_seasonality=0,
                    yearly_seasonality=0,
                    autoreg_dict="auto",
                    holidays_to_model_separately=[],
                    holiday_lookup_countries=[],
                    holiday_pre_num_days=0,
                    holiday_post_num_days=0,
                    holiday_pre_post_num_dict=None,
                    daily_event_df_dict=None,
                    fit_algorithm_dict={
                        "fit_algorithm": "ridge",
                        "fit_algorithm_params": None,
                    },
                    feature_sets_enabled="auto",
                    max_daily_seas_interaction_order=5,
                    max_weekly_seas_interaction_order=2,
                    regressor_cols=None,
                    extra_pred_cols=None,
                    min_admissible_value=None,
                    max_admissible_value=None,
                    normalize_method="zero_to_one"
                )
            )
        ]
    )
    return params


@pytest.fixture
def hourly_data():
    df = generate_df_for_tests(
        freq="H",
        periods=24 * 7 * 8,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)["df"]
    return df


@pytest.fixture
def hourly_data_with_reg():
    df = generate_df_for_tests(
        freq="H",
        periods=24 * 7 * 8,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)["df"]
    df["regressor"] = np.arange(len(df))
    return df


@pytest.fixture
def daily_data():
    data = generate_df_for_tests(
        freq="D",
        periods=500,
        train_frac=0.99,
        train_start_date=datetime.datetime(2018, 1, 1),
        conti_year_origin=2018)
    return data


@pytest.fixture
def config_silverkite_daily():
    config = MultistageForecastModelConfig(
        train_length="500D",
        fit_length="500D",
        agg_func="mean",
        agg_freq="D",
        estimator=SimpleSilverkiteEstimator,
        estimator_params=dict(
            coverage=None,
            forecast_horizon=1,
            freq="D",
            daily_seasonality=0,
            weekly_seasonality=3,
            quarterly_seasonality=5,
            monthly_seasonality=0,
            yearly_seasonality=0,
            changepoints_dict=None,
            autoreg_dict=None,
            holidays_to_model_separately=None,
            holiday_lookup_countries=None,
            holiday_pre_num_days=0,
            holiday_post_num_days=0,
            holiday_pre_post_num_dict=None,
            daily_event_df_dict=None,
            fit_algorithm_dict={
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None,
            },
            feature_sets_enabled=None,
            max_daily_seas_interaction_order=5,
            max_weekly_seas_interaction_order=2,
            extra_pred_cols=[],
            min_admissible_value=None,
            normalize_method="zero_to_one"
        )
    )
    return config


@pytest.fixture
def config_silverkite_daily_2():
    config = MultistageForecastModelConfig(
        train_length="500D",
        fit_length="500D",
        agg_func="mean",
        agg_freq="D",
        estimator=SimpleSilverkiteEstimator,
        estimator_params=dict(
            coverage=None,
            forecast_horizon=1,
            freq="D",
            growth_term=None,
            daily_seasonality=0,
            weekly_seasonality=0,
            quarterly_seasonality=0,
            monthly_seasonality=0,
            yearly_seasonality=0,
            autoreg_dict="auto",
            holidays_to_model_separately=[],
            holiday_lookup_countries=[],
            holiday_pre_num_days=0,
            holiday_post_num_days=0,
            holiday_pre_post_num_dict=None,
            daily_event_df_dict=None,
            fit_algorithm_dict={
                "fit_algorithm": "linear",
                "fit_algorithm_params": None,
            },
            feature_sets_enabled=None,
            max_daily_seas_interaction_order=0,
            max_weekly_seas_interaction_order=0,
            regressor_cols=None,
            extra_pred_cols=None,
            min_admissible_value=None,
            max_admissible_value=None,
            normalize_method="zero_to_one"
        )
    )
    return config


def test_multistage_model_config():
    """Tests the default parameters in ``MultistageForecastModelConfig``."""
    config = MultistageForecastModelConfig()
    assert config.train_length == f"{7 * 56}D"
    assert config.fit_length is None
    assert config.agg_func == "nanmean"
    assert config.agg_freq is None
    assert config.estimator == SimpleSilverkiteEstimator
    assert config.estimator_params is None


def test_set_up(params):
    """Tests the set up of ``MultistageForecastEstimator``."""
    # Instatiation.
    model = MultistageForecastEstimator(**params)
    assert model.model_configs == params["model_configs"]
    assert model.forecast_horizon == params["forecast_horizon"]
    assert model.freq == params["freq"]
    assert model.train_lengths is None
    assert model.fit_lengths is None
    assert model.agg_funcs is None
    assert model.agg_freqs is None
    assert model.estimators is None
    assert model.estimator_params is None
    assert model.train_lengths_in_seconds is None
    assert model.fit_lengths_in_seconds is None
    assert model.fit_lengths_in_seconds is None
    assert model.max_ar_orders is None
    assert model.data_freq_in_seconds is None
    assert model.num_points_per_agg_freqs is None
    assert model.models is None
    assert model.fit_df is None
    assert model.train_end is None

    # Initialization for some derived parameters.
    model._initialize()
    assert model.train_lengths == ["30D", "7D"]
    assert model.fit_lengths == ["30D", "7D"]
    assert len(model.agg_funcs) == 2
    assert model.agg_freqs == ["D", "H"]
    assert model.estimators == [SimpleSilverkiteEstimator, SimpleSilverkiteEstimator]
    assert model.estimator_params == [config.estimator_params for config in params["model_configs"]]
    assert model.train_lengths_in_seconds == [60 * 60 * 24 * 30, 60 * 60 * 24 * 7]
    assert model.fit_lengths_in_seconds == [60 * 60 * 24 * 30, 60 * 60 * 24 * 7]
    assert len(model.models) == 2
    assert model.data_freq_in_seconds == 60 * 60


def test_get_agg_func(params):
    model = MultistageForecastEstimator(**params)
    with pytest.raises(
            ValueError,
            match="The aggregation function "):
        model._get_agg_func("some_function")


def test_get_freq_col(params):
    model = MultistageForecastEstimator(**params)
    model.time_col_ = "ttt"
    freq_col = model._get_freq_col(freq="D", index=0)
    assert freq_col == "ttt__0__D"


def test_get_non_time_cols(params):
    model = MultistageForecastEstimator(**params)
    model.time_col_ = "ttt"
    columns = ["ttt", "ttt__0__D", "ttt__2__5T", "ttt_H", "value", "reg"]
    non_time_cols = model._get_non_time_cols(columns=columns)
    assert non_time_cols == ["ttt_H", "value", "reg"]


def test_get_num_points_per_agg_freq(params):
    model = MultistageForecastEstimator(**params)
    assert model._get_num_points_per_agg_freq(
        data_freq="H",
        agg_freqs=["D", "2H"]) == [24, 2]


def test_add_agg_freq_cols(params, hourly_data_with_reg):
    model = MultistageForecastEstimator(**params)
    model.time_col_ = cst.TIME_COL
    model.value_col_ = cst.VALUE_COL
    model._initialize()
    df = model._add_agg_freq_cols(df=hourly_data_with_reg)
    assert df.shape[1] == 5  # includes the original 3 and 2 extra columns from the two aggregations.
    assert list(df.columns) == [cst.TIME_COL, cst.VALUE_COL, "regressor",
                                f"{cst.TIME_COL}__0__D", f"{cst.TIME_COL}__1__H"]
    assert df[f"{cst.TIME_COL}__0__D"].unique().shape[0] == 7 * 8  # data has 7 weeks.
    assert df[f"{cst.TIME_COL}__1__H"].unique().shape[0] == df.shape[0]  # hourly is the original freq.

    # Tests error.
    with pytest.raises(
            ValueError,
            match="The df size is zero. Does your"):
        model._add_agg_freq_cols(df.iloc[:0])


def test_drop_incomplete_agg(params, hourly_data_with_reg):
    model = MultistageForecastEstimator(**params)
    model.time_col_ = cst.TIME_COL
    model.value_col_ = cst.VALUE_COL
    model._initialize()
    df = model._add_agg_freq_cols(df=hourly_data_with_reg)
    df = df.iloc[:-1]  # removes the last row so the last period becomes incomplete.
    df_new = model._drop_incomplete_agg(
        df=df,
        agg_freq="D",
        location=-1,
        num_points_per_agg_freq=24,
        index=0
    )
    assert len(df_new) == len(hourly_data_with_reg) - 24  # minus 1 day.
    assert df_new.reset_index(drop=True).equals(df.iloc[:-23])


def test_aggregate_values(params, hourly_data_with_reg):
    model = MultistageForecastEstimator(**params)
    model.time_col_ = cst.TIME_COL
    model.value_col_ = cst.VALUE_COL
    model._initialize()
    df = model._add_agg_freq_cols(df=hourly_data_with_reg)
    df_agg = model._aggregate_values(
        df=df[[f"{cst.TIME_COL}__0__D", cst.VALUE_COL, "regressor"]],
        agg_freq="D",
        agg_func=np.nanmean,
        index=0
    )
    assert len(df_agg) == 7 * 8  # data is 8 weeks.
    assert round(df_agg[cst.VALUE_COL].iloc[0], 3) == round(df[cst.VALUE_COL].iloc[:24].mean(), 3)
    assert round(df_agg["regressor"].iloc[0], 3) == round(df["regressor"].iloc[:24].mean(), 3)


def test_get_agg_dfs(params, hourly_data_with_reg):
    model = MultistageForecastEstimator(**params)
    model.time_col_ = cst.TIME_COL
    model.value_col_ = cst.VALUE_COL
    model._initialize()
    df = model._add_agg_freq_cols(df=hourly_data_with_reg)
    df = df.iloc[1:-1]  # both the beginning period and the end period are incomplete.
    result = model._get_agg_dfs(
        df=df,
        agg_freq="D",
        agg_func=np.mean,
        train_length_in_seconds=60 * 60 * 24 * 30,
        fit_length_in_seconds=60 * 60 * 24 * 30,
        num_points_per_agg_freq=24,
        max_ar_order=5,
        index=0
    )
    assert len(result["train_df"]) == 30
    assert len(result["fit_df"]) == 32  # fit includes incomplete periods on purpose
    # ``past_df`` includes 1 more period to avoid errors
    # This is to ensure there is no gap between ``past_df`` and ``train_df``,
    # as well as to ensure we have at least the length of ``past_df`` needed for AR.
    # Extra terms of ``past_df`` will be handled in ``SilverkiteForecast``.
    assert len(result["past_df"]) == 6
    assert result["fit_df_has_incomplete_period"] is True


def test_get_silverkite_ar_max_order(params):
    model = MultistageForecastEstimator(**params)
    model.time_col_ = cst.TIME_COL
    model.value_col_ = cst.VALUE_COL
    model._initialize()
    assert model._get_silverkite_ar_max_order() == [21, 24 * 21]


def test_train_and_predict(params, hourly_data_with_reg):
    """Tests train and prediction functionality."""
    params["model_configs"][0].estimator_params["regressor_cols"] = ["regressor"]
    model = MultistageForecastEstimator(**params)
    # fit
    model.fit(hourly_data_with_reg)
    # predict training period
    pred = model.predict(hourly_data_with_reg)
    assert pred.shape[0] == hourly_data_with_reg.shape[0]
    # predict future period
    pred = model.predict(pd.DataFrame({
        cst.TIME_COL: pd.date_range(start=hourly_data_with_reg[cst.TIME_COL].max(), freq="H", periods=13)[1:],
        cst.VALUE_COL: np.nan,
        "regressor": 1
    }))
    assert pred.shape[0] == 12
    assert pred[cst.PREDICTED_COL].dropna().shape[0] == 12
    # checks values-
    assert model.fit_df[f"{cst.VALUE_COL}__0__D"].dropna().shape[0] >= 30  # daily training size
    assert model.fit_df[f"{cst.VALUE_COL}__1__H"].dropna().shape[0] >= 24 * 7  # hourly training size
    assert model.fit_df[f"{cst.PREDICTED_COL}__0__D"].dropna().shape[0] >= 30  # daily fit size
    assert model.fit_df[f"{cst.PREDICTED_COL}__1__H"].dropna().shape[0] >= 30  # daily fit size

    # makes sure the AR orders are correct
    assert "y_lag1" in model.models[0].model_dict["x_mat"].columns
    assert "y_lag12" in model.models[1].model_dict["x_mat"].columns
    # components plot
    plots = model.plot_components()
    assert len(plots) == 2
    # summary
    summaries = model.summary()
    assert len(summaries) == 2


def test_error(params, hourly_data_with_reg):
    model = MultistageForecastEstimator(**params)

    # Calling plot components or summary before fitting.
    with pytest.raises(
            ValueError,
            match="Please call `fit` before"):
        model.plot_components()

    with pytest.raises(
            ValueError,
            match="Please call `fit` before"):
        model.summary()

    # Minimum aggregation frequency is less than data frequency.
    params["model_configs"][0].agg_freq = "5T"
    with pytest.raises(
            ValueError,
            match="The minimum aggregation frequency"):
        model.fit(hourly_data_with_reg)


def test_incomplete_fit_df_warning(params, hourly_data_with_reg):
    model = MultistageForecastEstimator(**params)
    model.model_configs[0].estimator_params["regressor_cols"] = ["regressor"]
    with LogCapture(cst.LOGGER_NAME) as log_capture:
        model.fit(X=hourly_data_with_reg.iloc[:-1])  # The last period is incomplete.
        log_capture.check(
            (cst.LOGGER_NAME,
             "WARNING",
             "There are incomplete periods in `fit_df`, thus the regressor values are "
             "biased after aggregation.")
        )


def test_missing_timestamps_during_aggregation(params, hourly_data_with_reg):
    model = MultistageForecastEstimator(**params)
    model.time_col_ = cst.TIME_COL
    model.value_col_ = cst.VALUE_COL
    model._initialize()
    df = model._add_agg_freq_cols(df=hourly_data_with_reg)
    df = df.iloc[1:-1]  # both the beginning period and the end period are incomplete.
    # Removes one timestamp in the middle.
    df = pd.concat([df.iloc[:50], df.iloc[51:]], axis=0).reset_index(drop=True)
    with LogCapture(cst.LOGGER_NAME) as log_capture:
        model._drop_incomplete_agg_and_aggregate_values(
            df=df,
            agg_freq="D",
            agg_func=np.mean,
            num_points_per_agg_freq=24,
            drop_incomplete=True,
            index=0
        )
        log_capture.check(
            (cst.LOGGER_NAME,
             "WARNING",
             "There are missing timestamps in `df` when performing aggregation with "
             "frequency D. These points are             ts   y\nts                "
             "\n2018-01-03  23  23. "
             "This may cause the aggregated values to be biased.")
        )


def test_infer_forecast_horizons(hourly_data_with_reg, params):
    """Tests that the estimator is able to infer the correct forecast horizon for
    each stage of model, under different situations.
    """
    model = MultistageForecastEstimator(**params)
    # The default forecast horizon is 12.
    # We truncate df to have the future period overlapping 2 days.
    df = hourly_data_with_reg.iloc[:-3]
    model.fit(df)
    assert model.forecast_horizons == (2, 12)
    # Now we do not truncate df to have the future period overlapping 1 day.
    model.fit(hourly_data_with_reg)
    assert model.forecast_horizons == (1, 12)


def test_short_fit_length(params):
    params["model_configs"][0].fit_length = "29D"
    model = MultistageForecastEstimator(**params)
    with LogCapture(cst.LOGGER_NAME) as log_capture:
        model._initialize()
        log_capture.check(
            (cst.LOGGER_NAME,
             "INFO",
             "Some `fit_length` is None or is shorter than `train_length`. "
             "These `fit_length` have been replaced with `train_length`.")
        )


def test_uncertainty(hourly_data_with_reg, params):
    # ``uncertainty_dict`` is given.
    params["uncertainty_dict"] = dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params={}
    )
    model = MultistageForecastEstimator(**params)
    # fit
    model.fit(hourly_data_with_reg)
    # predict
    pred = model.predict(pd.DataFrame({
        cst.TIME_COL: pd.date_range(start=hourly_data_with_reg[cst.TIME_COL].max(), freq="H", periods=13)[1:],
        cst.VALUE_COL: np.nan,
    }))
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred
    assert (pred[cst.PREDICTED_LOWER_COL] + pred[cst.PREDICTED_UPPER_COL]).round(2).equals(
        (pred[cst.PREDICTED_COL] * 2).round(2))

    # ``uncertainty_dict`` is not given but coverage is given.
    del params["uncertainty_dict"]
    params["coverage"] = 0.95
    model = MultistageForecastEstimator(**params)
    # fit
    model.fit(hourly_data_with_reg)
    # predict
    pred = model.predict(pd.DataFrame({
        cst.TIME_COL: pd.date_range(start=hourly_data_with_reg[cst.TIME_COL].max(), freq="H", periods=13)[1:],
        cst.VALUE_COL: np.nan,
    }))
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred
    assert (pred[cst.PREDICTED_LOWER_COL] + pred[cst.PREDICTED_UPPER_COL]).round(2).equals(
        (pred[cst.PREDICTED_COL] * 2).round(2))

    # ``uncertainty_dict`` and ``coverage`` are not given.
    del params["coverage"]
    model = MultistageForecastEstimator(**params)
    # fit
    model.fit(hourly_data_with_reg)
    # predict
    pred = model.predict(pd.DataFrame({
        cst.TIME_COL: pd.date_range(start=hourly_data_with_reg[cst.TIME_COL].max(), freq="H", periods=13)[1:],
        cst.VALUE_COL: np.nan,
    }))
    assert cst.PREDICTED_LOWER_COL not in pred
    assert cst.PREDICTED_UPPER_COL not in pred


def test_uncertainty_nonstandard_cols(hourly_data_with_reg, params):
    params["uncertainty_dict"] = dict(
        uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
        params={}
    )
    model = MultistageForecastEstimator(**params)
    # fit
    model.fit(
        hourly_data_with_reg.rename(columns={
            cst.TIME_COL: "t",
            cst.VALUE_COL: "z"
        }),
        time_col="t",
        value_col="z"
    )
    # predict
    pred = model.predict(pd.DataFrame({
        "t": pd.date_range(start=hourly_data_with_reg[cst.TIME_COL].max(), freq="H", periods=13)[1:],
        "z": np.nan,
    }))
    assert cst.PREDICTED_LOWER_COL in pred
    assert cst.PREDICTED_UPPER_COL in pred
    assert (pred[cst.PREDICTED_LOWER_COL] + pred[cst.PREDICTED_UPPER_COL]).round(2).equals(
        (pred[cst.PREDICTED_COL] * 2).round(2))


def test_uncertainty_with_error(hourly_data_with_reg, params):
    """Tests model still produces results when uncertainty model fails."""
    with LogCapture(cst.LOGGER_NAME) as log_capture:
        params["coverage"] = 0.95
        params["uncertainty_dict"] = dict(
            uncertainty_method=UncertaintyMethodEnum.simple_conditional_residuals.name,
            params={
                "value_col": "non_exist"
            }
        )
        model = MultistageForecastEstimator(**params)
        # fit
        model.fit(hourly_data_with_reg)
        # predict
        pred = model.predict(pd.DataFrame({
            cst.TIME_COL: pd.date_range(start=hourly_data_with_reg[cst.TIME_COL].max(), freq="H", periods=13)[1:],
            cst.VALUE_COL: np.nan,
        }))
        assert pred is not None
        assert cst.PREDICTED_LOWER_COL not in pred
        assert cst.PREDICTED_UPPER_COL not in pred
        assert (
            cst.LOGGER_NAME,
            "WARNING",
            "The following errors occurred during fitting the uncertainty model, "
            "the uncertainty model is skipped. `value_col` non_exist not found in `train_df`."
        ) in log_capture.actual()


def test_same_agg_freq(daily_data, config_silverkite_daily, config_silverkite_daily_2):
    """Tests two stages with the same aggregation frequency.
    Data is daily, and both stages uses daily aggregation.
    """
    df = daily_data["train_df"]
    df_test = daily_data["test_df"]
    forecast_horizon = len(df_test)
    config_silverkite_daily.estimator_params["forecast_horizon"] = forecast_horizon
    config_silverkite_daily_2.estimator_params["forecast_horizon"] = forecast_horizon
    params = dict(
        forecast_horizon=forecast_horizon,
        freq="D",
        model_configs=[
            config_silverkite_daily,
            config_silverkite_daily_2
        ]
    )
    model = MultistageForecastEstimator(**params)
    # Tests fit
    model.fit(df)
    # Checks models
    # Intercept, ct1, 6 weekly seas terms, 10 quarterly seas terms
    assert model.models[0].model_dict["x_mat"].shape[1] == 18
    # Intercept, 6 ar terms
    assert model.models[1].model_dict["x_mat"].shape[1] == 7

    # Tests prediction
    df_fit = model.predict(df)
    assert df_fit[cst.PREDICTED_COL].isna().sum() == 0
    df_predict = model.predict(df_test)
    assert df_predict[cst.PREDICTED_COL].isna().sum() == 0


def test_silverkite_wow_daily(daily_data, config_silverkite_daily):
    """Tests Silverkite + WOW estimator."""

    df = daily_data["train_df"]
    df_test = daily_data["test_df"]
    forecast_horizon = len(df_test)
    config_silverkite_daily.estimator_params["forecast_horizon"] = forecast_horizon
    params = dict(
        forecast_horizon=forecast_horizon,
        freq="D",
        model_configs=[
            config_silverkite_daily,
            MultistageForecastModelConfig(
                train_length="500D",
                fit_length="500D",
                agg_func="mean",
                agg_freq="D",
                estimator=LagBasedEstimator,
                estimator_params=dict(
                    lags=[1],
                    lag_unit="week",
                    agg_func="mean",
                    series_na_fill_func=lambda x: x.bfill().ffill()
                )
            )
        ]
    )
    model = MultistageForecastEstimator(**params)
    # Tests fit
    model.fit(df)
    # Checks models
    # Intercept, ct1, 6 weekly seas terms, 10 quarterly seas terms
    assert model.models[0].model_dict["x_mat"].shape[1] == 18

    # Tests prediction
    df_fit = model.predict(df)
    assert df_fit[cst.PREDICTED_COL].isna().sum() == 0
    df_predict = model.predict(df_test)
    assert df_predict[cst.PREDICTED_COL].isna().sum() == 0


def test_silverkite_wow_hourly(hourly_data, config_silverkite_daily):
    """Tests Silverkite + WOW estimator."""
    df = hourly_data.iloc[:-24]
    df_test = hourly_data.iloc[-24:].reset_index(drop=True)
    forecast_horizon = len(df_test)
    config_silverkite_daily.estimator_params["forecast_horizon"] = forecast_horizon
    params = dict(
        forecast_horizon=forecast_horizon,
        freq="H",
        model_configs=[
            config_silverkite_daily,
            MultistageForecastModelConfig(
                train_length="500D",
                fit_length="500D",
                agg_func="mean",
                agg_freq="H",
                estimator=LagBasedEstimator,
                estimator_params=dict(
                    lags=[1],
                    lag_unit="week",
                    agg_func="mean",
                    series_na_fill_func=lambda x: x.bfill().ffill()
                )
            )
        ]
    )
    model = MultistageForecastEstimator(**params)
    # Tests fit
    model.fit(df)
    # Checks models
    # Intercept, ct1, 6 weekly seas terms, 10 quarterly seas terms
    assert model.models[0].model_dict["x_mat"].shape[1] == 18

    # Tests prediction
    df_fit = model.predict(df)
    assert df_fit[cst.PREDICTED_COL].isna().sum() == 0
    df_predict = model.predict(df_test)
    assert df_predict[cst.PREDICTED_COL].isna().sum() == 0
