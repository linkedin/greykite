import inspect
import os
import sys
from collections import OrderedDict

import pytest

from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.pickle_utils import dump_obj
from greykite.framework.templates.pickle_utils import load_obj
from greykite.framework.templates.pickle_utils import recursive_rm_dir
from greykite.framework.utils.result_summary import summarize_grid_search_results


try:
    import prophet  # noqa
except ModuleNotFoundError:
    pass


@pytest.fixture
def df():
    data = generate_df_for_tests(
        freq="D",
        periods=365)
    return data["df"]


@pytest.fixture
def result(df):
    forecaster = Forecaster()
    # Run the forecast
    result = forecaster.run_forecast_config(
        df=df,  # includes the regressor
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            forecast_horizon=7,
            coverage=0.8,
            metadata_param=MetadataParam(
                time_col="ts",
                value_col="y",
                freq="D"
            ),
            evaluation_period_param=EvaluationPeriodParam(
                cv_max_splits=1,
                cv_horizon=7,
                test_horizon=7,
                cv_min_train_periods=80
            ),
            model_components_param=ModelComponentsParam(
                custom={"fit_algorithm_dict": {"fit_algorithm": "linear"}},
                autoregression={"autoreg_dict": "auto"}
            ),
            computation_param=ComputationParam(n_jobs=-1),
        )
    )
    return result


class X:
    def __init__(self, a):
        self.a = a


def test_recursive_rm_dir():
    dir_name = "dir_to_be_removed"
    # Empty dir.
    os.mkdir(dir_name)
    files = os.listdir(".")
    assert dir_name in files
    recursive_rm_dir(dir_name)
    files = os.listdir(".")
    assert dir_name not in files
    # Single file.
    f = open(dir_name, "a")
    f.write("This file is to be removed.")
    f.close()
    files = os.listdir(".")
    assert dir_name in files
    recursive_rm_dir(dir_name)
    files = os.listdir(".")
    assert dir_name not in files
    # Nested dir and files.
    os.mkdir(dir_name)
    f = open(os.path.join(dir_name, f"{dir_name}_file"), "a")
    f.write("This file is to be removed.")
    f.close()
    os.mkdir(os.path.join(dir_name, f"{dir_name}_dir"))
    f = open(os.path.join(dir_name, f"{dir_name}_dir", f"{dir_name}_file"), "a")
    f.write("This file is to be removed.")
    f.close()
    files = os.listdir(".")
    assert dir_name in files
    recursive_rm_dir(dir_name)
    files = os.listdir(".")
    assert dir_name not in files


def test_list():
    x = [1, "a", [3]]
    dump_obj(x, "list", overwrite_exist_dir=True)
    y = load_obj("list")
    assert x == y
    recursive_rm_dir("list")


def test_tuple():
    x = (1, "a", [3])
    dump_obj(x, "tuple", overwrite_exist_dir=True)
    y = load_obj("tuple")
    assert x == y
    recursive_rm_dir("tuple")


def test_dict():
    x = {"a": 1, 1: 3, 5: ["b"]}
    dump_obj(x, "dict", overwrite_exist_dir=True)
    y = load_obj("dict")
    assert x == y
    recursive_rm_dir("dict")


def test_ordered_dict():
    x = OrderedDict({"a": 1, 1: 3, 5: ["b"]})
    dump_obj(x, "odict", overwrite_exist_dir=True)
    y = load_obj("odict")
    assert x == y
    recursive_rm_dir("odict")


def test_class():
    x = X(1)
    dump_obj(x, "class", overwrite_exist_dir=True)
    y = load_obj("class")
    assert x.__class__ == y.__class__
    assert x.a == y.a
    recursive_rm_dir("class")


def test_forecast_result_silverkite(df, result):
    dump_obj(
        result,
        dir_name="silverkite",
        dump_design_info=True,
        overwrite_exist_dir=True
    )
    result_rec = load_obj(
        dir_name="silverkite",
        load_design_info=True
    )
    recursive_rm_dir("silverkite")

    # Tests loaded results
    # Grid search cv results
    assert_equal(
        summarize_grid_search_results(result.grid_search),
        summarize_grid_search_results(result_rec.grid_search)
    )
    # Grid search attributes
    for key in result.grid_search.__dict__.keys():
        if key not in ["scoring", "estimator", "refit", "cv", "error_score", "cv_results_",
                       "scorer_", "best_estimator_"]:
            assert_equal(
                result.grid_search.__dict__[key],
                result_rec.grid_search.__dict__[key])

    # Model
    assert_equal(
        result.model[-1].predict(df),
        result_rec.model[-1].predict(df)
    )
    assert result.model[-1].model_dict["x_design_info"] is not None
    # Model: estimator
    for key in result.model[-1].__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model_dict"]:
            assert_equal(
                result.model[-1].__dict__[key],
                result_rec.model[-1].__dict__[key])
    assert_equal(
        inspect.getsource(result.model[-1].__dict__["score_func"]),
        inspect.getsource(result_rec.model[-1].__dict__["score_func"])
    )
    # Model: estimator/model_dict
    for key in result.model[-1].model_dict.keys():
        # Functions and classes are not testable.
        if key not in ["x_design_info", "fs_func", "ml_model", "plt_pred",
                       "autoreg_dict", "changepoint_detector", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                result.model[-1].model_dict[key],
                result_rec.model[-1].model_dict[key])
        # Tests function source code.
        elif key in ["fs_func", "plt_pred", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                inspect.getsource(result.model[-1].model_dict[key]),
                inspect.getsource(result_rec.model[-1].model_dict[key]))
    # Model: estimator/model_dict/autoreg_dict
    for key in result.model[-1].model_dict["autoreg_dict"].keys():
        if key not in ["series_na_fill_func"]:
            assert_equal(
                result.model[-1].model_dict["autoreg_dict"][key],
                result_rec.model[-1].model_dict["autoreg_dict"][key])
    assert_equal(
        inspect.getsource(result.model[-1].model_dict["autoreg_dict"]["series_na_fill_func"]),
        inspect.getsource(result_rec.model[-1].model_dict["autoreg_dict"]["series_na_fill_func"]))

    # Forecast
    assert_equal(
        result.forecast.estimator.predict(df),
        result_rec.forecast.estimator.predict(df)
    )
    assert result.forecast.estimator.model_dict["x_design_info"] is not None
    # Forecast: attributes
    for key in result.forecast.__dict__.keys():
        if key not in ["r2_loss_function", "estimator"]:
            assert_equal(
                result.forecast.__dict__[key],
                result_rec.forecast.__dict__[key])
    assert_equal(
        inspect.getsource(result.forecast.__dict__["r2_loss_function"]),
        inspect.getsource(result_rec.forecast.__dict__["r2_loss_function"]))
    # Forecast: estimator
    for key in result.forecast.estimator.__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model_dict"]:
            assert_equal(
                result.forecast.estimator.__dict__[key],
                result_rec.forecast.estimator.__dict__[key])
    assert_equal(
        inspect.getsource(result.forecast.estimator.__dict__["score_func"]),
        inspect.getsource(result_rec.forecast.estimator.__dict__["score_func"])
    )
    # Model: estimator/model_dict
    for key in result.forecast.estimator.model_dict.keys():
        # Functions and classes are not testable.
        if key not in ["x_design_info", "fs_func", "ml_model", "plt_pred",
                       "autoreg_dict", "changepoint_detector", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                result.forecast.estimator.model_dict[key],
                result_rec.forecast.estimator.model_dict[key])
        # Tests function source code.
        elif key in ["fs_func", "plt_pred", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                inspect.getsource(result.forecast.estimator.model_dict[key]),
                inspect.getsource(result_rec.forecast.estimator.model_dict[key]))
    # Model: estimator/model_dict/autoreg_dict
    for key in result.forecast.estimator.model_dict["autoreg_dict"].keys():
        if key not in ["series_na_fill_func"]:
            assert_equal(
                result.forecast.estimator.model_dict["autoreg_dict"][key],
                result_rec.forecast.estimator.model_dict["autoreg_dict"][key])
    assert_equal(
        inspect.getsource(result.forecast.estimator.model_dict["autoreg_dict"]["series_na_fill_func"]),
        inspect.getsource(result_rec.forecast.estimator.model_dict["autoreg_dict"]["series_na_fill_func"]))

    # Backtest
    assert_equal(
        result.backtest.estimator.predict(df),
        result_rec.backtest.estimator.predict(df)
    )
    assert result.backtest.estimator.model_dict["x_design_info"] is not None
    # Backtest: attributes
    for key in result.backtest.__dict__.keys():
        if key not in ["r2_loss_function", "estimator"]:
            assert_equal(
                result.backtest.__dict__[key],
                result_rec.backtest.__dict__[key])
    assert_equal(
        inspect.getsource(result.backtest.__dict__["r2_loss_function"]),
        inspect.getsource(result_rec.backtest.__dict__["r2_loss_function"]))
    # Backtest: estimator
    for key in result.backtest.estimator.__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model_dict"]:
            assert_equal(
                result.backtest.estimator.__dict__[key],
                result_rec.backtest.estimator.__dict__[key])
    assert_equal(
        inspect.getsource(result.backtest.estimator.__dict__["score_func"]),
        inspect.getsource(result_rec.backtest.estimator.__dict__["score_func"])
    )
    # Model: estimator/model_dict
    for key in result.backtest.estimator.model_dict.keys():
        # Functions and classes are not testable.
        if key not in ["x_design_info", "fs_func", "ml_model", "plt_pred",
                       "autoreg_dict", "changepoint_detector", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                result.backtest.estimator.model_dict[key],
                result_rec.backtest.estimator.model_dict[key])
        # Tests function source code.
        elif key in ["fs_func", "plt_pred", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                inspect.getsource(result.backtest.estimator.model_dict[key]),
                inspect.getsource(result_rec.backtest.estimator.model_dict[key]))
    # Model: estimator/model_dict/autoreg_dict
    for key in result.backtest.estimator.model_dict["autoreg_dict"].keys():
        if key not in ["series_na_fill_func"]:
            assert_equal(
                result.backtest.estimator.model_dict["autoreg_dict"][key],
                result_rec.backtest.estimator.model_dict["autoreg_dict"][key])
    assert_equal(
        inspect.getsource(result.backtest.estimator.model_dict["autoreg_dict"]["series_na_fill_func"]),
        inspect.getsource(result_rec.backtest.estimator.model_dict["autoreg_dict"]["series_na_fill_func"]))

    # Timeseries
    for key in result.timeseries.__dict__.keys():
        assert_equal(
            result.timeseries.__dict__[key],
            result_rec.timeseries.__dict__[key])


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_forecast_result_prophet(df):
    forecaster = Forecaster()
    # Run the forecast
    result = forecaster.run_forecast_config(
        df=df,  # includes the regressor
        config=ForecastConfig(
            model_template=ModelTemplateEnum.PROPHET.name,
            forecast_horizon=7,
            coverage=0.8,
            metadata_param=MetadataParam(
                time_col="ts",
                value_col="y",
                freq="D"
            ),
            evaluation_period_param=EvaluationPeriodParam(
                cv_max_splits=1,
                cv_horizon=7,
                test_horizon=7,
                cv_min_train_periods=80
            ),
            computation_param=ComputationParam(n_jobs=-1),
        )
    )
    dump_obj(
        result,
        dir_name="prophet",
        dump_design_info=True,
        overwrite_exist_dir=True
    )
    result_rec = load_obj(
        dir_name="prophet",
        load_design_info=True
    )
    recursive_rm_dir("prophet")

    # Tests loaded results

    # Grid search cv results
    # There is one element that can not be tested directly.
    # This element is a pandas.BlockManager object.
    # We compare it separately.

    # Only one entry is different.
    assert ((summarize_grid_search_results(result.grid_search).values
             != summarize_grid_search_results(result_rec.grid_search).values).sum().sum() == 1)
    # The entry is location 77.
    assert (summarize_grid_search_results(result.grid_search).iloc[0, 77]
            != summarize_grid_search_results(result_rec.grid_search).iloc[0, 77])
    # Use .equal to test equality, they are actually equal.
    assert (summarize_grid_search_results(result.grid_search).iloc[0, 77].equals(
        summarize_grid_search_results(result_rec.grid_search).iloc[0, 77]))

    # Grid search attributes
    for key in result.grid_search.__dict__.keys():
        if key not in ["scoring", "estimator", "refit", "cv", "error_score", "cv_results_",
                       "scorer_", "best_estimator_"]:
            assert_equal(
                result.grid_search.__dict__[key],
                result_rec.grid_search.__dict__[key])

    # Model
    for key in result.model[-1].__dict__.keys():
        if key not in ["score_func", "model"]:
            assert_equal(
                result.model[-1].__dict__[key],
                result_rec.model[-1].__dict__[key])
    assert_equal(
        inspect.getsource(result.model[-1].__dict__["score_func"]),
        inspect.getsource(result_rec.model[-1].__dict__["score_func"])
    )

    # Forecast
    for key in result.forecast.__dict__.keys():
        if key not in ["r2_loss_function", "estimator"]:
            assert_equal(
                result.forecast.__dict__[key],
                result_rec.forecast.__dict__[key])
    assert_equal(
        inspect.getsource(result.forecast.__dict__["r2_loss_function"]),
        inspect.getsource(result_rec.forecast.__dict__["r2_loss_function"]))
    # Forecast: estimator
    for key in result.forecast.estimator.__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model"]:
            assert_equal(
                result.forecast.estimator.__dict__[key],
                result_rec.forecast.estimator.__dict__[key])
    assert_equal(
        inspect.getsource(result.forecast.estimator.__dict__["score_func"]),
        inspect.getsource(result_rec.forecast.estimator.__dict__["score_func"])
    )

    # Backtest
    for key in result.backtest.__dict__.keys():
        if key not in ["r2_loss_function", "estimator"]:
            assert_equal(
                result.backtest.__dict__[key],
                result_rec.backtest.__dict__[key])
    assert_equal(
        inspect.getsource(result.backtest.__dict__["r2_loss_function"]),
        inspect.getsource(result_rec.backtest.__dict__["r2_loss_function"]))
    # Backtest: estimator
    for key in result.backtest.estimator.__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model"]:
            assert_equal(
                result.backtest.estimator.__dict__[key],
                result_rec.backtest.estimator.__dict__[key])
    assert_equal(
        inspect.getsource(result.backtest.estimator.__dict__["score_func"]),
        inspect.getsource(result_rec.backtest.estimator.__dict__["score_func"])
    )

    # Test the predict methods.
    # We only test the forecasted values, because the prediction intervals are
    # randomly sampled and could be different.
    # We test the prediction after everything because it may alter the class attributes.
    assert_equal(
        result.model[-1].predict(df)["forecast"],
        result_rec.model[-1].predict(df)["forecast"]
    )
    assert_equal(
        result.forecast.estimator.predict(df)["forecast"],
        result_rec.forecast.estimator.predict(df)["forecast"]
    )
    assert_equal(
        result.backtest.estimator.predict(df)["forecast"],
        result_rec.backtest.estimator.predict(df)["forecast"]
    )


def test_forecast_result_one_by_one(df):
    forecaster = Forecaster()
    # Run the forecast
    result = forecaster.run_forecast_config(
        df=df,  # includes the regressor
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            forecast_horizon=7,
            coverage=0.8,
            metadata_param=MetadataParam(
                time_col="ts",
                value_col="y",
                freq="D"
            ),
            evaluation_period_param=EvaluationPeriodParam(
                cv_max_splits=1,
                cv_horizon=7,
                test_horizon=7,
                cv_min_train_periods=80
            ),
            computation_param=ComputationParam(n_jobs=-1),
            model_components_param=ModelComponentsParam(
                custom={"fit_algorithm_dict": {"fit_algorithm": "linear"}},
                autoregression={"autoreg_dict": "auto"}
            ),
            forecast_one_by_one=[2, 5]
        )
    )
    dump_obj(
        result,
        dir_name="onebyone",
        dump_design_info=True,
        overwrite_exist_dir=True
    )
    result_rec = load_obj(
        dir_name="onebyone",
        load_design_info=True
    )
    recursive_rm_dir("onebyone")

    # Tests loaded results
    # Grid search cv results
    assert_equal(
        summarize_grid_search_results(result.grid_search),
        summarize_grid_search_results(result_rec.grid_search)
    )
    # Grid search attributes
    for key in result.grid_search.__dict__.keys():
        if key not in ["scoring", "estimator", "refit", "cv", "error_score", "cv_results_",
                       "scorer_", "best_estimator_"]:
            assert_equal(
                result.grid_search.__dict__[key],
                result_rec.grid_search.__dict__[key])

    # Model
    assert_equal(
        result.model[-1].predict(df),
        result_rec.model[-1].predict(df)
    )
    assert result.model[-1].estimators[0].model_dict["x_design_info"] is not None
    # Model: estimator
    for key in result.model[-1].estimators[0].__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model_dict"]:
            assert_equal(
                result.model[-1].estimators[0].__dict__[key],
                result_rec.model[-1].estimators[0].__dict__[key])
    assert_equal(
        inspect.getsource(result.model[-1].estimators[0].__dict__["score_func"]),
        inspect.getsource(result_rec.model[-1].estimators[0].__dict__["score_func"])
    )
    # Model: estimator/model_dict
    for key in result.model[-1].estimators[0].model_dict.keys():
        # Functions and classes are not testable.
        if key not in ["x_design_info", "fs_func", "ml_model", "plt_pred",
                       "autoreg_dict", "changepoint_detector", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                result.model[-1].estimators[0].model_dict[key],
                result_rec.model[-1].estimators[0].model_dict[key])
        # Tests function source code.
        elif key in ["fs_func", "plt_pred", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                inspect.getsource(result.model[-1].estimators[0].model_dict[key]),
                inspect.getsource(result_rec.model[-1].estimators[0].model_dict[key]))
    # Model: estimator/model_dict/autoreg_dict
    for key in result.model[-1].estimators[0].model_dict["autoreg_dict"].keys():
        if key not in ["series_na_fill_func"]:
            assert_equal(
                result.model[-1].estimators[0].model_dict["autoreg_dict"][key],
                result_rec.model[-1].estimators[0].model_dict["autoreg_dict"][key])
    assert_equal(
        inspect.getsource(result.model[-1].estimators[0].model_dict["autoreg_dict"]["series_na_fill_func"]),
        inspect.getsource(result_rec.model[-1].estimators[0].model_dict["autoreg_dict"]["series_na_fill_func"]))

    # Forecast
    assert_equal(
        result.forecast.estimator.predict(df),
        result_rec.forecast.estimator.predict(df)
    )
    assert result.forecast.estimator.estimators[0].model_dict["x_design_info"] is not None
    # Forecast: attributes
    for key in result.forecast.__dict__.keys():
        if key not in ["r2_loss_function", "estimator"]:
            assert_equal(
                result.forecast.__dict__[key],
                result_rec.forecast.__dict__[key])
    assert_equal(
        inspect.getsource(result.forecast.__dict__["r2_loss_function"]),
        inspect.getsource(result_rec.forecast.__dict__["r2_loss_function"]))
    # Forecast: estimator
    for key in result.forecast.estimator.estimators[0].__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model_dict"]:
            assert_equal(
                result.forecast.estimator.estimators[0].__dict__[key],
                result_rec.forecast.estimator.estimators[0].__dict__[key])
    assert_equal(
        inspect.getsource(result.forecast.estimator.estimators[0].__dict__["score_func"]),
        inspect.getsource(result_rec.forecast.estimator.estimators[0].__dict__["score_func"])
    )
    # Model: estimator/model_dict
    for key in result.forecast.estimator.estimators[0].model_dict.keys():
        # Functions and classes are not testable.
        if key not in ["x_design_info", "fs_func", "ml_model", "plt_pred",
                       "autoreg_dict", "changepoint_detector", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                result.forecast.estimator.estimators[0].model_dict[key],
                result_rec.forecast.estimator.estimators[0].model_dict[key])
        # Tests function source code.
        elif key in ["fs_func", "plt_pred", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                inspect.getsource(result.forecast.estimator.estimators[0].model_dict[key]),
                inspect.getsource(result_rec.forecast.estimator.estimators[0].model_dict[key]))
    # Model: estimator/model_dict/autoreg_dict
    for key in result.forecast.estimator.estimators[0].model_dict["autoreg_dict"].keys():
        if key not in ["series_na_fill_func"]:
            assert_equal(
                result.forecast.estimator.estimators[0].model_dict["autoreg_dict"][key],
                result_rec.forecast.estimator.estimators[0].model_dict["autoreg_dict"][key])
    assert_equal(
        inspect.getsource(result.forecast.estimator.estimators[0].model_dict["autoreg_dict"]["series_na_fill_func"]),
        inspect.getsource(result_rec.forecast.estimator.estimators[0].model_dict["autoreg_dict"]["series_na_fill_func"]))

    # Backtest
    assert_equal(
        result.backtest.estimator.predict(df),
        result_rec.backtest.estimator.predict(df)
    )
    assert result.backtest.estimator.estimators[0].model_dict["x_design_info"] is not None
    # Forecast: attributes
    for key in result.backtest.__dict__.keys():
        if key not in ["r2_loss_function", "estimator"]:
            assert_equal(
                result.backtest.__dict__[key],
                result_rec.backtest.__dict__[key])
    assert_equal(
        inspect.getsource(result.backtest.__dict__["r2_loss_function"]),
        inspect.getsource(result_rec.backtest.__dict__["r2_loss_function"]))
    # Forecast: estimator
    for key in result.backtest.estimator.estimators[0].__dict__.keys():
        if key not in ["score_func", "silverkite", "silverkite_diagnostics", "model_dict"]:
            assert_equal(
                result.backtest.estimator.estimators[0].__dict__[key],
                result_rec.backtest.estimator.estimators[0].__dict__[key])
    assert_equal(
        inspect.getsource(result.backtest.estimator.estimators[0].__dict__["score_func"]),
        inspect.getsource(result_rec.backtest.estimator.estimators[0].__dict__["score_func"])
    )
    # Model: estimator/model_dict
    for key in result.backtest.estimator.estimators[0].model_dict.keys():
        # Functions and classes are not testable.
        if key not in ["x_design_info", "fs_func", "ml_model", "plt_pred",
                       "autoreg_dict", "changepoint_detector", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                result.backtest.estimator.estimators[0].model_dict[key],
                result_rec.backtest.estimator.estimators[0].model_dict[key])
        # Tests function source code.
        elif key in ["fs_func", "plt_pred", "autoreg_func", "normalize_df_func"]:
            assert_equal(
                inspect.getsource(result.backtest.estimator.estimators[0].model_dict[key]),
                inspect.getsource(result_rec.backtest.estimator.estimators[0].model_dict[key]))
    # Model: estimator/model_dict/autoreg_dict
    for key in result.backtest.estimator.estimators[0].model_dict["autoreg_dict"].keys():
        if key not in ["series_na_fill_func"]:
            assert_equal(
                result.backtest.estimator.estimators[0].model_dict["autoreg_dict"][key],
                result_rec.backtest.estimator.estimators[0].model_dict["autoreg_dict"][key])
    assert_equal(
        inspect.getsource(result.backtest.estimator.estimators[0].model_dict["autoreg_dict"]["series_na_fill_func"]),
        inspect.getsource(result_rec.backtest.estimator.estimators[0].model_dict["autoreg_dict"]["series_na_fill_func"]))

    # Timeseries
    for key in result.timeseries.__dict__.keys():
        assert_equal(
            result.timeseries.__dict__[key],
            result_rec.timeseries.__dict__[key])


def test_no_design_info(result):
    # Does not dump design info, load design info
    dump_obj(
        result,
        dir_name="silverkite_dump_design_false_load_design_true",
        dump_design_info=False,
        overwrite_exist_dir=True
    )
    result_rec = load_obj(
        dir_name="silverkite_dump_design_false_load_design_true",
        load_design_info=True
    )
    recursive_rm_dir("silverkite_dump_design_false_load_design_true")
    assert "x_design_info" in result.model[-1].model_dict.keys()
    assert "x_design_info" not in result_rec.model[-1].model_dict.keys()
    assert "x_design_info" in result.forecast.estimator.model_dict.keys()
    assert "x_design_info" not in result_rec.forecast.estimator.model_dict.keys()
    assert "x_design_info" in result.backtest.estimator.model_dict.keys()
    assert "x_design_info" not in result_rec.backtest.estimator.model_dict.keys()

    # Does not dump design info, does not load design info
    dump_obj(
        result,
        dir_name="silverkite_dump_design_false_load_design_false",
        dump_design_info=False,
        overwrite_exist_dir=True
    )
    result_rec = load_obj(
        dir_name="silverkite_dump_design_false_load_design_false",
        load_design_info=False
    )
    recursive_rm_dir("silverkite_dump_design_false_load_design_false")
    assert "x_design_info" in result.model[-1].model_dict.keys()
    assert "x_design_info" not in result_rec.model[-1].model_dict.keys()
    assert "x_design_info" in result.forecast.estimator.model_dict.keys()
    assert "x_design_info" not in result_rec.forecast.estimator.model_dict.keys()
    assert "x_design_info" in result.backtest.estimator.model_dict.keys()
    assert "x_design_info" not in result_rec.backtest.estimator.model_dict.keys()

    # Dumps design info, does not load design info
    dump_obj(
        result,
        dir_name="silverkite_dump_design_true_load_design_false",
        dump_design_info=True,
        overwrite_exist_dir=True
    )
    result_rec = load_obj(
        dir_name="silverkite_dump_design_true_load_design_false",
        load_design_info=False
    )
    recursive_rm_dir("silverkite_dump_design_true_load_design_false")
    assert "x_design_info" in result.model[-1].model_dict.keys()
    assert result_rec.model[-1].model_dict["x_design_info"] is None
    assert "x_design_info" in result.forecast.estimator.model_dict.keys()
    assert result_rec.forecast.estimator.model_dict["x_design_info"] is None
    assert "x_design_info" in result.backtest.estimator.model_dict.keys()
    assert result_rec.backtest.estimator.model_dict["x_design_info"] is None


def test_overwrite():
    dump_obj(
        [1, "a", {"b": 3}],
        dir_name="overwrite",
        dump_design_info=True,
        overwrite_exist_dir=True
    )
    loaded_obj = load_obj(
        dir_name="overwrite"
    )
    assert loaded_obj == [1, "a", {"b": 3}]
    # Overwrites the same dir with a different object.
    dump_obj(
        [2, "b", {"c": 4}],
        dir_name="overwrite",
        dump_design_info=True,
        overwrite_exist_dir=True
    )
    loaded_obj = load_obj(
        dir_name="overwrite"
    )
    assert loaded_obj == [2, "b", {"c": 4}]
    recursive_rm_dir("overwrite")


def test_errors(result):
    # Directory already exists.
    dump_obj(
        result,
        dir_name="name",
        dump_design_info=False,
        overwrite_exist_dir=True
    )
    with pytest.raises(
            FileExistsError,
            match="The directory already exists. "
                  "Please either specify a new directory or "
                  "set overwrite_exist_dir to True to overwrite it."):
        dump_obj(
            result,
            dir_name="name",
            dump_design_info=False,
            overwrite_exist_dir=False
        )
    recursive_rm_dir("name")

    # Empty dir load.
    os.mkdir("this_dir_is_empty")
    with pytest.raises(
            ValueError,
            match="dir is empty!"):
        load_obj(
            dir_name="this_dir_is_empty"
        )
    recursive_rm_dir("this_dir_is_empty")

    # Multiple elements in top level dir.
    os.mkdir("multiple_dir")
    dump_obj(1, "multiple_dir", "one", top_level=False)
    dump_obj(2, "multiple_dir", "two", top_level=False)
    with pytest.raises(
            ValueError,
            match="Multiple elements found in top level."):
        load_obj(
            dir_name="multiple_dir"
        )
    recursive_rm_dir("multiple_dir")
