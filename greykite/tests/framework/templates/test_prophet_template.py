import datetime
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal

import greykite.common.constants as cst
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.framework.constants import COMPUTATION_N_JOBS
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.prophet_template import ProphetTemplate
from greykite.framework.utils.framework_testing_utils import assert_basic_pipeline_equal
from greykite.sklearn.estimator.prophet_estimator import ProphetEstimator


try:
    import prophet
    from prophet.make_holidays import make_holidays_df
    prophet
except ModuleNotFoundError:
    pass


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
@pytest.fixture
def default_holidays():
    """Default holidays by country params"""
    start_year = 2015
    end_year = 2030
    holiday_pre_num_days = [2]
    holiday_post_num_days = [2]
    expected_holidays = ProphetTemplate().get_prophet_holidays(
        year_list=list(range(start_year - 1, end_year + 2)),  # covers one more year on both sides.
        countries="auto",
        lower_window=-holiday_pre_num_days[0],
        upper_window=holiday_post_num_days[0])
    return expected_holidays


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_property():
    """Tests properties"""
    assert ProphetTemplate().allow_model_template_list is False
    assert ProphetTemplate().allow_model_components_param_list is False

    template = ProphetTemplate()
    assert template.DEFAULT_MODEL_TEMPLATE == "PROPHET"
    assert isinstance(template.estimator, ProphetEstimator)
    assert template.estimator.coverage == 0.80
    assert template.apply_forecast_config_defaults().model_template == "PROPHET"

    estimator = ProphetEstimator(coverage=0.99)
    template = ProphetTemplate(estimator=estimator)
    assert template.estimator is estimator


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_get_prophet_holidays():
    """Tests get_prophet_holidays"""
    year_list = list(range(2014, 2030+2))
    holiday_lookup_countries = ["UnitedStates", "UnitedKingdom", "India", "France", "China"]

    # Default holidays from get_prophet_holidays
    actual_holidays = ProphetTemplate().get_prophet_holidays(
        year_list=year_list)

    # Ensures all given countries' holidays are captured in the given `year_list`.
    # Loops through all country level holidays and confirm they are available in actual_holidays.
    # Suppresses the warnings such as "We only support Diwali and Holi holidays from 2010 to 2025"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ctry in holiday_lookup_countries:
            expected_ctry_holidays = make_holidays_df(
                year_list=year_list,
                country=ctry)
            # sort df and reset index to ensure assert_frame_equal works well. Without it, assert throws an error.
            expected_ctry_holidays = expected_ctry_holidays.sort_values(by=["ds", "holiday"]).reset_index(drop=True)

            actual = actual_holidays[["ds", "holiday"]]  # All actual holidays
            actual_ctry_holidays = actual.merge(  # Ensure country-level holidays are a subset of actual holidays
                expected_ctry_holidays,
                on=["ds", "holiday"],
                how="inner",
                validate="1:1")  # Ensures 1:1 mapping
            actual_ctry_holidays = actual_ctry_holidays.sort_values(by=["ds", "holiday"]).reset_index(drop=True)
            assert_frame_equal(expected_ctry_holidays, actual_ctry_holidays)

    # there are no duplicates at date and holiday level in the final holidays df
    actual_holidays_rows = actual_holidays.shape[0]
    unique_date_holiday = actual_holidays["ds"].astype(str)+" "+actual_holidays["holiday"]
    unique_date_holiday_combinations = pd.unique(unique_date_holiday).shape[0]
    assert unique_date_holiday_combinations == actual_holidays_rows

    # Tests custom params
    lower_window = -1
    upper_window = 4
    countries = ["UnitedKingdom", "Australia"]
    actual_holidays = ProphetTemplate().get_prophet_holidays(
        countries=countries,
        year_list=year_list,
        lower_window=lower_window,
        upper_window=upper_window)
    assert "Australia Day (Observed)" in actual_holidays["holiday"].values
    assert "Chinese New Year" not in actual_holidays["holiday"].values

    # all of the expected columns are available in the output
    actual_columns = list(actual_holidays.columns)
    expected_columns = ["ds", "holiday", "lower_window", "upper_window"]
    assert actual_columns == expected_columns

    # lower_window and upper_window are accurately assigned
    assert actual_holidays["lower_window"].unique() == lower_window
    assert actual_holidays["upper_window"].unique() == upper_window

    # no countries
    actual_holidays = ProphetTemplate().get_prophet_holidays(
        countries=[],
        year_list=year_list,
        lower_window=lower_window,
        upper_window=upper_window)
    assert actual_holidays is None
    actual_holidays = ProphetTemplate().get_prophet_holidays(
        countries=None,
        year_list=year_list,
        lower_window=lower_window,
        upper_window=upper_window)
    assert actual_holidays is None

    # single country
    with pytest.raises(ValueError, match="`countries` should be a list, found Australia"):
        ProphetTemplate().get_prophet_holidays(
            countries="Australia",
            year_list=year_list,
            lower_window=lower_window,
            upper_window=upper_window)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_get_regressor_cols():
    """Tests get_regressor_names"""
    # `add_regressor_dict` is a list of dict
    template = ProphetTemplate()
    model_components = ModelComponentsParam(
        regressors={
            "add_regressor_dict": [
                {
                    "regressor1": {
                        "prior_scale": 10,
                        "standardize": True,
                        "mode": "additive"
                    },
                    "regressor2": {
                        "prior_scale": 15,
                        "standardize": False,
                        "mode": "additive"
                    },
                    "regressor3": {}
                },
                None,
                {
                    "regressor1": {
                        "prior_scale": 10,
                        "standardize": True,
                        "mode": "additive"
                    },
                    "regressor4": {
                        "prior_scale": 15,
                        "standardize": False,
                        "mode": "additive"
                    },
                    "regressor5": {}
                }
            ]
        }
    )
    template.config = ForecastConfig(model_components_param=model_components)
    assert set(template.get_regressor_cols()) == {
        "regressor1",
        "regressor2",
        "regressor3",
        "regressor4",
        "regressor5"}

    # `add_regressor_dict` is a single dict
    model_components = ModelComponentsParam(
        regressors={
            "add_regressor_dict": {
                "regressor1": {
                    "prior_scale": 10,
                    "standardize": True,
                    "mode": "additive"
                },
                "regressor2": {
                    "prior_scale": 15,
                    "standardize": False,
                    "mode": "additive"
                },
                "regressor3": {}
            }
        }
    )
    template.config = ForecastConfig(model_components_param=model_components)
    assert set(template.get_regressor_cols()) == {
        "regressor1",
        "regressor2",
        "regressor3"}

    # no regressors
    model_components = ModelComponentsParam()
    template.config = ForecastConfig(model_components_param=model_components)
    assert template.get_regressor_cols() is None

    model_components = ModelComponentsParam(regressors={})
    template.config = ForecastConfig(model_components_param=model_components)
    assert template.get_regressor_cols() is None

    model_components = ModelComponentsParam(regressors={
        "add_regressor_dict": []
    })
    template.config = ForecastConfig(model_components_param=model_components)
    assert template.get_regressor_cols() is None

    model_components = ModelComponentsParam(regressors={
        "add_regressor_dict": [{}, None]
    })
    template.config = ForecastConfig(model_components_param=model_components)
    assert template.get_regressor_cols() is None


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_hyperparameter_grid_default():
    """Tests get_hyperparameter_grid and apply_prophet_model_components_defaults"""
    template = ProphetTemplate()
    template.config = template.apply_forecast_config_defaults()
    # both model_components, time_properties are None
    hyperparameter_grid = template.get_hyperparameter_grid()
    expected_holidays = template.get_prophet_holidays(
        year_list=list(range(2015-1, 2030+2)),
        countries="auto",
        lower_window=-2,
        upper_window=2)
    expected_grid = {
        "estimator__growth": ["linear"],
        "estimator__seasonality_mode": ["additive"],
        "estimator__seasonality_prior_scale": [10.0],
        "estimator__yearly_seasonality": ["auto"],
        "estimator__weekly_seasonality": ["auto"],
        "estimator__daily_seasonality": ["auto"],
        "estimator__add_seasonality_dict": [None],
        "estimator__holidays": [expected_holidays],
        "estimator__holidays_prior_scale": [10.0],
        "estimator__changepoint_prior_scale": [0.05],
        "estimator__changepoints": [None],
        "estimator__n_changepoints": [25],
        "estimator__changepoint_range": [0.8],
        "estimator__mcmc_samples": [0],
        "estimator__uncertainty_samples": [1000],
        "estimator__add_regressor_dict": [None]
    }
    assert_equal(actual=hyperparameter_grid, expected=expected_grid)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_hyperparameter_grid_seasonality_growth(default_holidays):
    """Tests get_hyperparameter_grid for basic seasonality, growth and other default params"""
    seasonality = {"yearly_seasonality": [True], "weekly_seasonality": [False]}
    growth = {"growth_term": ["linear"]}
    model_components = ModelComponentsParam(
        seasonality=seasonality,
        growth=growth)
    template = ProphetTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_components_param = model_components
    hyperparameter_grid = template.get_hyperparameter_grid()
    # Expected Values
    expected_holidays = default_holidays
    expected_grid = {
        "estimator__growth": ["linear"],
        "estimator__seasonality_mode": ["additive"],
        "estimator__seasonality_prior_scale": [10.0],
        "estimator__yearly_seasonality": [True],
        "estimator__weekly_seasonality": [False],
        "estimator__daily_seasonality": ["auto"],
        "estimator__add_seasonality_dict": [None],
        "estimator__holidays": [expected_holidays],
        "estimator__holidays_prior_scale": [10.0],
        "estimator__changepoint_prior_scale": [0.05],
        "estimator__changepoints": [None],
        "estimator__n_changepoints": [25],
        "estimator__changepoint_range": [0.8],
        "estimator__mcmc_samples": [0],
        "estimator__uncertainty_samples": [1000],
        "estimator__add_regressor_dict": [None]
    }
    # Assertions
    assert_equal(actual=hyperparameter_grid, expected=expected_grid)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_hyperparameter_grid_events():
    """Tests get_prophet_hyperparameter_grid for selected Countries" holidays"""
    # holiday params
    start_year = 2018
    end_year = 2022
    holiday_pre_num_days = [1]
    holiday_post_num_days = [1]
    holiday_lookup_countries = ["UnitedStates", "China", "India"]
    holidays_prior_scale = [5.0, 10.0, 15.0]
    events = {
        "holiday_lookup_countries": holiday_lookup_countries,
        "holiday_pre_num_days": holiday_pre_num_days,
        "holiday_post_num_days": holiday_post_num_days,
        "start_year": start_year,
        "end_year": end_year,
        "holidays_prior_scale": holidays_prior_scale
    }
    model_components = ModelComponentsParam(events=events)
    template = ProphetTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_components_param = model_components
    hyperparameter_grid = template.get_hyperparameter_grid()

    # Expected Values
    # Holidays df, based on given holidays params
    expected_holidays = template.get_prophet_holidays(
        year_list=list(range(start_year - 1, end_year + 2)),
        countries=holiday_lookup_countries,
        lower_window=-holiday_pre_num_days[0],
        upper_window=holiday_post_num_days[0])

    expected_grid = {
        "estimator__growth": ["linear"],
        "estimator__seasonality_mode": ["additive"],
        "estimator__seasonality_prior_scale": [10.0],
        "estimator__yearly_seasonality": ["auto"],
        "estimator__weekly_seasonality": ["auto"],
        "estimator__daily_seasonality": ["auto"],
        "estimator__add_seasonality_dict": [None],
        "estimator__holidays": [expected_holidays],
        "estimator__holidays_prior_scale": [5.0, 10.0, 15.0],
        "estimator__changepoint_prior_scale": [0.05],
        "estimator__changepoints": [None],
        "estimator__n_changepoints": [25],
        "estimator__changepoint_range": [0.8],
        "estimator__mcmc_samples": [0],
        "estimator__uncertainty_samples": [1000],
        "estimator__add_regressor_dict": [None]
    }
    # Assertions
    assert_equal(actual=hyperparameter_grid, expected=expected_grid)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_hyperparameter_grid_exception():
    """Tests prophet_hyperparameter_grid exceptions"""
    # unknown argument
    with pytest.raises(
            ValueError,
            match=r"Unexpected key\(s\) found"):
        model_components = ModelComponentsParam(
            seasonality={
                "unknown_seasonality": ["additive"]
            }
        )
        template = ProphetTemplate()
        template.config = template.apply_forecast_config_defaults()
        template.config.model_components_param = model_components
        template.get_hyperparameter_grid()

    # regressor must be specified under `add_regressor_dict`, not directly
    with pytest.raises(
            ValueError,
            match=r"Unexpected key\(s\) found"):
        model_components = ModelComponentsParam(
            regressors={
                "regressor1": {
                    "prior_scale": 10,
                    "standardize": True,
                    "mode": "additive"
                },
                "regressor2": {
                    "prior_scale": 15,
                    "standardize": False,
                    "mode": "additive"
                },
                "regressor3": {}
            }
        )
        template = ProphetTemplate()
        template.config = template.apply_forecast_config_defaults()
        template.config.model_components_param = model_components
        template.get_hyperparameter_grid()


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_hyperparameter_grid_warn():
    """Tests get_prophet_hyperparameter_grid warnings"""
    # holiday params
    start_year = 2018
    end_year = 2022
    holiday_pre_num_days = [1, 2, 3, 4]
    holiday_post_num_days = [1, 2, 3, 4]
    holiday_lookup_countries = [["UnitedStates"], ["UnitedStates", "China", "India"]]
    holidays_prior_scale = [5.0, 10.0, 15.0]
    events = {
        "holiday_lookup_countries": holiday_lookup_countries,
        "holiday_pre_num_days": holiday_pre_num_days,
        "holiday_post_num_days": holiday_post_num_days,
        "start_year": start_year,
        "end_year": end_year,
        "holidays_prior_scale": holidays_prior_scale
    }
    model_components = ModelComponentsParam(events=events)
    with pytest.warns(Warning) as record:
        template = ProphetTemplate()
        template.config = template.apply_forecast_config_defaults()
        template.config.model_components_param = model_components
        template.get_hyperparameter_grid()
        assert f"`events['holiday_pre_num_days']` list has more than 1 element. "\
               f"We currently support only 1 element. "\
               f"Using 1." in record[0].message.args[0]
        assert f"`events['holiday_post_num_days']` list has more than 1 element. " \
               f"We currently support only 1 element. " \
               f"Using 1." in record[1].message.args[0]
        # Extra spaces for holidays to align with actual warning in the function, because of an extra tab.
        assert f"`events['holiday_lookup_countries']` contains multiple options. "\
               f"We currently support only 1 option. Using ['UnitedStates']." in record[2].message.args[0]

    # other invalid holiday_lookup_countries
    with pytest.warns(Warning) as record:
        events["holiday_pre_num_days"] = [1]
        events["holiday_post_num_days"] = [0]
        events["holiday_lookup_countries"] = ["auto", None]
        model_components = ModelComponentsParam(events=events)
        template = ProphetTemplate()
        template.config = template.apply_forecast_config_defaults()
        template.config.model_components_param = model_components
        template.get_hyperparameter_grid()
        assert f"`events['holiday_lookup_countries']` contains multiple options. " \
               f"We currently support only 1 option. Using auto." in record[0].message.args[0]

    # no warning if only one list of holiday_lookup_countries is provided
    with pytest.warns(None):
        events["holiday_pre_num_days"] = [1]
        events["holiday_post_num_days"] = [0]
        events["holiday_lookup_countries"] = [["UnitedStates", "China", "UnitedKingdom", "India"]]
        template = ProphetTemplate()
        template.config = template.apply_forecast_config_defaults()
        template.config.model_components_param = ModelComponentsParam(events=events)
        hyp1 = template.get_hyperparameter_grid()

        events["holiday_lookup_countries"] = ["UnitedStates", "China", "UnitedKingdom", "India"]
        template = ProphetTemplate()
        template.config = template.apply_forecast_config_defaults()
        template.config.model_components_param = ModelComponentsParam(events=events)
        hyp2 = template.get_hyperparameter_grid()
        assert_equal(hyp1, hyp2)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_hyperparameter_grid_custom_seasonality(default_holidays):
    """Tests get_prophet_hyperparameter_grid for custom seasonality params, other params being defaults"""
    seasonality = {
        "seasonality_mode": ["additive", "multiplicative"],
        "seasonality_prior_scale": [5.0, 10.0, 15.0],
        "yearly_seasonality": [True, False],
        "weekly_seasonality": [True, False],
        "daily_seasonality": [True, False],
        "add_seasonality_dict": [
            {
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 20,
                    "prior_scale": 20.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 15
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 35,
                    "prior_scale": 30.0
                }
            },
            {
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 10,
                    "prior_scale": 20.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 3
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 5,
                    "prior_scale": 20.0
                }
            },
            {
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 10,
                    "prior_scale": 30.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 5
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 15,
                    "prior_scale": 20.0
                }
            },
            {
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 15,
                    "prior_scale": 20.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 10
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 25,
                    "prior_scale": 20.0
                }
            }]
    }
    model_components = ModelComponentsParam(seasonality=seasonality)
    template = ProphetTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_components_param = model_components
    hyperparameter_grid = template.get_hyperparameter_grid()

    # Expected Values
    expected_holidays = default_holidays
    expected_grid = {
        "estimator__growth": ["linear"],
        "estimator__seasonality_mode": ["additive", "multiplicative"],
        "estimator__seasonality_prior_scale": [5.0, 10.0, 15.0],
        "estimator__yearly_seasonality": [True, False],
        "estimator__weekly_seasonality": [True, False],
        "estimator__daily_seasonality": [True, False],
        "estimator__add_seasonality_dict": [
            {
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 20,
                    "prior_scale": 20.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 15
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 35,
                    "prior_scale": 30.0
                }
            },
            {
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 10,
                    "prior_scale": 20.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 3
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 5,
                    "prior_scale": 20.0
                }
            },
            {
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 10,
                    "prior_scale": 30.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 5
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 15,
                    "prior_scale": 20.0
                }
            },
            {
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 15,
                    "prior_scale": 20.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 10
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 25,
                    "prior_scale": 20.0
                }
            }],
        "estimator__holidays": [expected_holidays],
        "estimator__holidays_prior_scale": [10.0],
        "estimator__changepoint_prior_scale": [0.05],
        "estimator__changepoints": [None],
        "estimator__n_changepoints": [25],
        "estimator__changepoint_range": [0.8],
        "estimator__mcmc_samples": [0],
        "estimator__uncertainty_samples": [1000],
        "estimator__add_regressor_dict": [None]
    }
    # Assertions
    assert_equal(actual=hyperparameter_grid, expected=expected_grid)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_hyperparameter_grid_changepoints_uncertainty_custom(default_holidays):
    """Tests get_prophet_hyperparameter_grid for selected
    changepoints, regressor, and uncertainty"""
    changepoints = {
        "changepoint_prior_scale": [0.05, 1.0, 5.0, 10.0, 15.0],
        "changepoints": [None, ["2018-10-11", "2018-11-11", "2019-01-17"]],
        "n_changepoints": [25, 50, 100],
        "changepoint_range": [0.8, 0.9]
    }
    uncertainty = {
        "mcmc_samples": [0, 1000],
        "uncertainty_samples": [1000, 2000]
    }
    regressors = {
        "add_regressor_dict": [{
            "reg_col1": {
                "prior_scale": 10.0,
                "standardize": False,
                "mode": "additive"
            },
            "reg_col2": {
                "prior_scale": 20.0,
                "standardize": True,
                "mode": "multiplicative"
            }
        },
            {
                "reg_col1": {
                    "prior_scale": 20.0,
                    "standardize": True,
                    "mode": "additive"
                },
                "reg_col2": {
                    "prior_scale": 40.0,
                    "standardize": False,
                    "mode": "multiplicative"
                    }
                }]
    }
    model_components = ModelComponentsParam(
        changepoints=changepoints,
        regressors=regressors,
        uncertainty=uncertainty)
    template = ProphetTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_components_param = model_components
    hyperparameter_grid = template.get_hyperparameter_grid()

    # Expected Values
    expected_holidays = default_holidays
    expected_grid = {
        "estimator__growth": ["linear"],
        "estimator__seasonality_mode": ["additive"],
        "estimator__seasonality_prior_scale": [10.0],
        "estimator__yearly_seasonality": ["auto"],
        "estimator__weekly_seasonality": ["auto"],
        "estimator__daily_seasonality": ["auto"],
        "estimator__add_seasonality_dict": [None],
        "estimator__holidays": [expected_holidays],
        "estimator__holidays_prior_scale": [10.0],
        "estimator__changepoint_prior_scale": [0.05, 1.0, 5.0, 10.0, 15.0],
        "estimator__changepoints": [None, ["2018-10-11", "2018-11-11", "2019-01-17"]],
        "estimator__n_changepoints": [25, 50, 100],
        "estimator__changepoint_range": [0.8, 0.9],
        "estimator__mcmc_samples": [0, 1000],
        "estimator__uncertainty_samples": [1000, 2000],
        "estimator__add_regressor_dict": [
            {
                "reg_col1": {
                    "prior_scale": 10.0,
                    "standardize": False,
                    "mode": "additive"
                },
                "reg_col2": {
                    "prior_scale": 20.0,
                    "standardize": True,
                    "mode": "multiplicative"
                }
            },
            {
                "reg_col1": {
                    "prior_scale": 20.0,
                    "standardize": True,
                    "mode": "additive"
                },
                "reg_col2": {
                    "prior_scale": 40.0,
                    "standardize": False,
                    "mode": "multiplicative"
                }
            }]
        }
    assert_equal(actual=hyperparameter_grid, expected=expected_grid)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_hyperparameter_grid_auto_list(default_holidays):
    """Tests `get_prophet_hyperparameter_grid` automatic list conversion
    via `dictionaries_values_to_lists`. Holidays are tested separately
    because they are not directly passed to ProphetEstimator."""
    growth = {
        "growth_term": "linear"
    }
    seasonality = {
        "seasonality_mode": "multiplicative",
        "seasonality_prior_scale": 10.0,
        "yearly_seasonality": False,
        "weekly_seasonality": False,
        "daily_seasonality": True,
        "add_seasonality_dict":
            {
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 20,
                    "prior_scale": 20.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 15
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 35,
                    "prior_scale": 30.0
                }
            }
    }

    changepoints = {
        "changepoint_prior_scale": 0.05,
        "changepoints": ["2018-10-11", "2018-11-11", "2019-01-17"],
        "n_changepoints": 25,
        "changepoint_range": 0.8
    }
    uncertainty = {
        "mcmc_samples": 0,
        "uncertainty_samples": 1000
    }
    regressors = {
        "add_regressor_dict": {
            "reg_col1": {
                "prior_scale": 10.0,
                "standardize": False,
                "mode": "additive"
            },
            "reg_col2": {
                "prior_scale": 20.0,
                "standardize": True,
                "mode": "multiplicative"
            }
        }
    }
    model_components = ModelComponentsParam(
        growth=growth,
        seasonality=seasonality,
        changepoints=changepoints,
        regressors=regressors,
        uncertainty=uncertainty)
    template = ProphetTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_components_param = model_components
    hyperparameter_grid = template.get_hyperparameter_grid()

    # Expected Values
    expected_grid = {
        "estimator__growth": ["linear"],
        "estimator__seasonality_mode": ["multiplicative"],
        "estimator__seasonality_prior_scale": [10.0],
        "estimator__yearly_seasonality": [False],
        "estimator__weekly_seasonality": [False],
        "estimator__daily_seasonality": [True],
        "estimator__add_seasonality_dict": [seasonality["add_seasonality_dict"]],
        "estimator__holidays": [default_holidays],
        "estimator__holidays_prior_scale": [10.0],
        "estimator__changepoint_prior_scale": [0.05],
        "estimator__changepoints": [["2018-10-11", "2018-11-11", "2019-01-17"]],
        "estimator__n_changepoints": [25],
        "estimator__changepoint_range": [0.8],
        "estimator__mcmc_samples": [0],
        "estimator__uncertainty_samples": [1000],
        "estimator__add_regressor_dict": [
            {
                "reg_col1": {
                    "prior_scale": 10.0,
                    "standardize": False,
                    "mode": "additive"
                },
                "reg_col2": {
                    "prior_scale": 20.0,
                    "standardize": True,
                    "mode": "multiplicative"
                }
            }]
    }
    assert_equal(actual=hyperparameter_grid, expected=expected_grid)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_hyperparameter_override(default_holidays):
    """Tests the hyperparameter_override functionality.
    Use hyperparameter_override to override parameters and
    create multiple sets of grids.
    """
    model_components = ModelComponentsParam(
        seasonality={
            "yearly_seasonality": [True, False],
            "weekly_seasonality": False,
        },
        hyperparameter_override=[
            {
                "input__response__null__max_frac": 0.1,
                "estimator__yearly_seasonality": True,
                "estimator__growth": ["logistic"],
            },
            {}]
    )
    template = ProphetTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_components_param = model_components
    hyperparameter_grid = template.get_hyperparameter_grid()
    expected_grid = {
        "estimator__growth": ["linear"],
        "estimator__seasonality_mode": ["additive"],
        "estimator__seasonality_prior_scale": [10.0],
        "estimator__yearly_seasonality": [True, False],
        "estimator__weekly_seasonality": [False],
        "estimator__daily_seasonality": ["auto"],
        "estimator__add_seasonality_dict": [None],
        "estimator__holidays": [default_holidays],
        "estimator__holidays_prior_scale": [10.0],
        "estimator__changepoint_prior_scale": [0.05],
        "estimator__changepoints": [None],
        "estimator__n_changepoints": [25],
        "estimator__changepoint_range": [0.8],
        "estimator__mcmc_samples": [0],
        "estimator__uncertainty_samples": [1000],
        "estimator__add_regressor_dict": [None]
    }
    updated_grid = expected_grid.copy()
    updated_grid["input__response__null__max_frac"] = [0.1]
    updated_grid["estimator__yearly_seasonality"] = [True]
    updated_grid["estimator__growth"] = ["logistic"]
    assert_equal(hyperparameter_grid, [updated_grid, expected_grid])


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_apply_template_decorator():
    data = generate_df_for_tests(freq="D", periods=10)
    df = data["df"]
    template = ProphetTemplate()
    with pytest.raises(
            ValueError,
            match="ProphetTemplate only supports config.model_template='PROPHET', found 'UNKNOWN'"):
        template.apply_template_for_pipeline_params(
            df=df,
            config=ForecastConfig(model_template="UNKNOWN")
        )


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_template_default():
    """Tests prophet_template with default values, for limited data"""
    # prepares input data
    num_days = 10
    data = generate_df_for_tests(freq="D", periods=num_days, train_start_date="2018-01-01")
    df = data["df"]
    template = ProphetTemplate()
    config = ForecastConfig(model_template="PROPHET")
    params = template.apply_template_for_pipeline_params(
        df=df,
        config=config
    )
    # not modified
    assert config == ForecastConfig(model_template="PROPHET")
    # checks result
    metric = EvaluationMetricEnum.MeanAbsolutePercentError
    pipeline = params.pop("pipeline", None)
    expected_params = dict(
        df=df,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        date_format=None,
        freq=None,
        train_end_date=None,
        anomaly_info=None,
        # model
        regressor_cols=None,
        lagged_regressor_cols=None,
        estimator=None,
        hyperparameter_grid=template.hyperparameter_grid,
        hyperparameter_budget=None,
        n_jobs=COMPUTATION_N_JOBS,
        verbose=1,
        # forecast
        forecast_horizon=None,
        coverage=None,
        test_horizon=None,
        periods_between_train_test=None,
        agg_periods=None,
        agg_func=None,
        # evaluation
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better(),
        cv_report_metrics=CV_REPORT_METRICS_ALL,
        null_model_params=None,
        relative_error_tolerance=None,
        # CV
        cv_horizon=None,
        cv_min_train_periods=None,
        cv_expanding_window=True,
        cv_use_most_recent_splits=None,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=None,
        cv_max_splits=3
    )
    assert_basic_pipeline_equal(pipeline, template.pipeline)
    assert_equal(params, expected_params)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_prophet_template_custom():
    """Tests prophet_template with custom values, with long range input"""
    # prepares input data
    data = generate_df_with_reg_for_tests(
        freq="H",
        periods=300*24,
        remove_extra_cols=True,
        mask_test_actuals=True)
    df = data["df"]
    time_col = "some_time_col"
    value_col = "some_value_col"
    df.rename({
        cst.TIME_COL: time_col,
        cst.VALUE_COL: value_col
    }, axis=1, inplace=True)
    # prepares params and calls template
    metric = EvaluationMetricEnum.MeanAbsoluteError
    # anomaly adjustment adds 10.0 to every record
    adjustment_size = 10.0
    anomaly_df = pd.DataFrame({
        cst.START_TIME_COL: [df[time_col].min()],
        cst.END_TIME_COL: [df[time_col].max()],
        cst.ADJUSTMENT_DELTA_COL: [adjustment_size],
        cst.METRIC_COL: [value_col]
    })
    anomaly_info = {
        "value_col": cst.VALUE_COL,
        "anomaly_df": anomaly_df,
        "start_time_col": cst.START_TIME_COL,
        "end_time_col": cst.END_TIME_COL,
        "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL,
        "filter_by_dict": {cst.METRIC_COL: cst.VALUE_COL},
        "adjustment_method": "add"
    }
    metadata = MetadataParam(
        time_col=time_col,
        value_col=value_col,
        freq="H",
        date_format="%Y-%m-%d-%H",
        train_end_date=datetime.datetime(2019, 7, 1),
        anomaly_info=anomaly_info,
    )
    evaluation_metric = EvaluationMetricParam(
        cv_selection_metric=metric.name,
        cv_report_metrics=[EvaluationMetricEnum.MedianAbsolutePercentError.name],
        agg_periods=24,
        agg_func=np.max,
        null_model_params={
            "strategy": "quantile",
            "constant": None,
            "quantile": 0.8
        },
        relative_error_tolerance=0.01
    )
    evaluation_period = EvaluationPeriodParam(
        test_horizon=1,
        periods_between_train_test=2,
        cv_horizon=3,
        cv_min_train_periods=4,
        cv_expanding_window=True,
        cv_use_most_recent_splits=True,
        cv_periods_between_splits=5,
        cv_periods_between_train_test=6,
        cv_max_splits=7
    )
    model_components = ModelComponentsParam(
        seasonality={
            "yearly_seasonality": [True],
            "weekly_seasonality": [False],
            "daily_seasonality": [4],
            "add_seasonality_dict": [{
                    "yearly": {
                        "period": 365.25,
                        "fourier_order": 20,
                        "prior_scale": 20.0
                    },
                    "quarterly": {
                        "period": 365.25/4,
                        "fourier_order": 15
                    },
                    "weekly": {
                        "period": 7,
                        "fourier_order": 35,
                        "prior_scale": 30.0
                    }
                }]
        },
        growth={
            "growth_term": "linear"
        },
        events={
            "holiday_lookup_countries": ["UnitedStates", "UnitedKingdom", "India"],
            "holiday_pre_num_days": [2],
            "holiday_post_num_days": [3],
            "holidays_prior_scale": [5.0]
        },
        regressors={
            "add_regressor_dict": [{
                "regressor1": {
                    "prior_scale": 10.0,
                    "mode": 'additive'
                },
                "regressor2": {
                    "prior_scale": 20.0,
                    "mode": 'multiplicative'
                },
            }]
        },
        changepoints={
            "changepoint_prior_scale": [0.05],
            "changepoints": [None],
            "n_changepoints": [50],
            "changepoint_range": [0.9]
        },
        uncertainty={
            "mcmc_samples": [500],
            "uncertainty_samples": [2000]
        },
        hyperparameter_override={
            "input__response__null__impute_algorithm": "ts_interpolate",
            "input__response__null__impute_params": {"orders": [7, 14]},
            "input__regressors_numeric__normalize__normalize_algorithm": "RobustScaler",
        }
    )
    computation = ComputationParam(
        hyperparameter_budget=10,
        n_jobs=None,
        verbose=1
    )
    forecast_horizon = 20
    coverage = 0.7
    config = ForecastConfig(
        model_template=ModelTemplateEnum.PROPHET.name,
        metadata_param=metadata,
        forecast_horizon=forecast_horizon,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        model_components_param=model_components,
        computation_param=computation
    )
    template = ProphetTemplate()
    params = template.apply_template_for_pipeline_params(
        df=df,
        config=config
    )
    pipeline = params.pop("pipeline", None)
    # Adding start_year and end_year based on the input df
    model_components.events["start_year"] = df[time_col].min().year
    model_components.events["end_year"] = df[time_col].max().year
    expected_params = dict(
        df=df,
        time_col=time_col,
        value_col=value_col,
        date_format=metadata.date_format,
        freq=metadata.freq,
        train_end_date=metadata.train_end_date,
        anomaly_info=metadata.anomaly_info,
        # model
        regressor_cols=template.regressor_cols,
        lagged_regressor_cols=template.lagged_regressor_cols,
        estimator=None,
        hyperparameter_grid=template.hyperparameter_grid,
        hyperparameter_budget=computation.hyperparameter_budget,
        n_jobs=computation.n_jobs,
        verbose=computation.verbose,
        # forecast
        forecast_horizon=forecast_horizon,
        coverage=coverage,
        test_horizon=evaluation_period.test_horizon,
        periods_between_train_test=evaluation_period.periods_between_train_test,
        agg_periods=evaluation_metric.agg_periods,
        agg_func=evaluation_metric.agg_func,
        # evaluation
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better(),
        cv_report_metrics=evaluation_metric.cv_report_metrics,
        null_model_params=evaluation_metric.null_model_params,
        relative_error_tolerance=evaluation_metric.relative_error_tolerance,
        # CV
        cv_horizon=evaluation_period.cv_horizon,
        cv_min_train_periods=evaluation_period.cv_min_train_periods,
        cv_expanding_window=evaluation_period.cv_expanding_window,
        cv_use_most_recent_splits=evaluation_period.cv_use_most_recent_splits,
        cv_periods_between_splits=evaluation_period.cv_periods_between_splits,
        cv_periods_between_train_test=evaluation_period.cv_periods_between_train_test,
        cv_max_splits=evaluation_period.cv_max_splits
    )
    assert_basic_pipeline_equal(pipeline, template.pipeline)
    assert_equal(params, expected_params)


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_run_prophet_template_custom():
    """Tests running prophet template through the pipeline"""
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=50,
        train_frac=0.8,
        conti_year_origin=2018,
        remove_extra_cols=True,
        mask_test_actuals=True)
    # select relevant columns for testing
    relevant_cols = [cst.TIME_COL, cst.VALUE_COL, "regressor1", "regressor2", "regressor3"]
    df = data["df"][relevant_cols]
    forecast_horizon = data["fut_time_num"]

    # Model components - custom holidays; other params as defaults
    model_components = ModelComponentsParam(
        seasonality={
            "seasonality_mode": ["additive"],
            "yearly_seasonality": ["auto"],
            "weekly_seasonality": [True],
            "daily_seasonality": ["auto"],
        },
        growth={
            "growth_term": ["linear"]
        },
        events={
            "holiday_pre_num_days": [1],
            "holiday_post_num_days": [1],
            "holidays_prior_scale": [1.0]
        },
        changepoints={
            "changepoint_prior_scale": [0.05],
            "n_changepoints": [1],
            "changepoint_range": [0.5],
        },
        regressors={
            "add_regressor_dict": [{
                "regressor1": {
                    "prior_scale": 10,
                    "standardize": True,
                    "mode": "additive"
                },
                "regressor2": {
                    "prior_scale": 15,
                    "standardize": False,
                    "mode": "additive"
                },
                "regressor3": {}
            }]
        },
        uncertainty={
            "uncertainty_samples": [10]
        }
    )

    metadata = MetadataParam(
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        freq="D",
    )
    evaluation_period = EvaluationPeriodParam(
        test_horizon=5,  # speeds up test case
        periods_between_train_test=5,
        cv_horizon=0,  # speeds up test case
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.PROPHET.name,
        metadata_param=metadata,
        forecast_horizon=forecast_horizon,
        coverage=0.95,
        model_components_param=model_components,
        evaluation_period_param=evaluation_period,
    )
    result = Forecaster().run_forecast_config(
        df=df,
        config=config,
    )

    forecast_df = result.forecast.df_test.reset_index(drop=True)
    expected_cols = ["ts", "actual", "forecast", "forecast_lower", "forecast_upper"]
    assert list(forecast_df.columns) == expected_cols
    assert result.backtest.coverage == 0.95, "coverage is not correct"
    # NB: coverage is poor because of very small dataset size and low uncertainty_samples
    assert result.backtest.train_evaluation[cst.PREDICTION_BAND_COVERAGE] == pytest.approx(0.742, rel=1e-3), \
        "training coverage is None or less than expected"
    assert result.backtest.test_evaluation[cst.PREDICTION_BAND_COVERAGE] == pytest.approx(0.800, rel=1e-3), \
        "testing coverage is None or less than expected"
    assert result.backtest.train_evaluation["MSE"] == pytest.approx(3.3942, rel=1e-3), \
        "training MSE is None or more than expected"
    assert result.backtest.test_evaluation["MSE"] == pytest.approx(1.9477, rel=1e-3), \
        "testing MSE is None or more than expected"
    assert result.forecast.train_evaluation[cst.PREDICTION_BAND_COVERAGE] == pytest.approx(0.7805, rel=1e-3), \
        "forecast coverage is None or less than expected"
    assert result.forecast.train_evaluation["MSE"] == pytest.approx(3.6025, rel=1e-3), \
        "forecast MSE is None or more than expected"

    # ensure regressors were used in the model
    prophet_estimator = result.model.steps[-1][-1]
    regressors = prophet_estimator.model.extra_regressors
    assert regressors.keys() == {"regressor1", "regressor2", "regressor3"}
    assert regressors["regressor1"]["prior_scale"] == 10.0
    assert regressors["regressor1"]["standardize"] is True
    assert regressors["regressor1"]["mode"] == "additive"
    assert regressors["regressor2"]["prior_scale"] == 15.0
    assert regressors["regressor3"]["standardize"] == "auto"
