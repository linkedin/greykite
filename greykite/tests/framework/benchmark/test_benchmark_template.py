import os

from greykite.common.data_loader import DataLoader
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.testing_utils import generate_df_for_tests
from greykite.framework.benchmark.benchmark_template import benchmark_silverkite_template
from greykite.framework.benchmark.benchmark_template import get_default_benchmark_parameters
from greykite.framework.benchmark.benchmark_template import get_default_benchmark_real_datasets
from greykite.framework.benchmark.benchmark_template import get_default_benchmark_silverkite_parameters
from greykite.framework.benchmark.benchmark_template import get_default_benchmark_simulated_datasets
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import MetadataParam


def test_benchmark_silverkite_template_with_simulated_data():
    # setting every list to 1 item to speed up test case
    forecast_horizons = [30]
    max_cvs = [3]
    fit_algorithms = ["linear"]
    metric = EvaluationMetricEnum.MeanSquaredError
    evaluation_metric = EvaluationMetricParam(cv_selection_metric=metric.name)

    # Simulated data
    data_name = "daily_simulated"
    train_period = 365
    data = generate_df_for_tests(freq="D", periods=train_period)
    df = data["df"]
    time_col, value_col = df.columns
    metadata = MetadataParam(time_col=time_col, value_col=value_col, freq="D")
    result_silverkite_simulated = benchmark_silverkite_template(
        data_name=data_name,
        df=df,
        metadata=metadata,
        evaluation_metric=evaluation_metric,
        forecast_horizons=forecast_horizons,
        fit_algorithms=fit_algorithms,
        max_cvs=max_cvs)

    result_silverkite_simulated = result_silverkite_simulated[0]
    assert result_silverkite_simulated["data_name"] == data_name
    assert result_silverkite_simulated["forecast_model_name"] == "silverkite_linear"
    assert result_silverkite_simulated["train_period"] == train_period
    assert result_silverkite_simulated["forecast_horizon"] == 30
    assert result_silverkite_simulated["cv_folds"] == 3
    # Not checking the other parameters as it will add ~10 secs for every mint build


def test_benchmark_silverkite_template_with_real_data():
    # setting every list to 1 item to speed up test case
    forecast_horizons = [30]
    max_cvs = [3]
    fit_algorithms = ["linear"]
    metric = EvaluationMetricEnum.MeanSquaredError
    evaluation_metric = EvaluationMetricParam(cv_selection_metric=metric.name)

    # real data
    dl = DataLoader()
    data_path = dl.get_data_home(data_sub_dir="daily")
    data_name = "daily_female_births"
    df = dl.get_df(data_path=data_path, data_name="daily_female_births")
    time_col = "Date"
    value_col = "Births"
    metadata = MetadataParam(time_col=time_col, value_col=value_col, freq="D")
    result_silverkite_real = benchmark_silverkite_template(
        data_name=data_name,
        df=df,
        metadata=metadata,
        evaluation_metric=evaluation_metric,
        forecast_horizons=forecast_horizons,
        fit_algorithms=fit_algorithms,
        max_cvs=max_cvs)

    result_silverkite_real = result_silverkite_real[0]
    assert result_silverkite_real["data_name"] == data_name
    assert result_silverkite_real["forecast_model_name"] == "silverkite_linear"
    assert result_silverkite_real["train_period"] == df.shape[0]
    assert result_silverkite_real["forecast_horizon"] == 30
    assert result_silverkite_real["cv_folds"] == 3
    # Not checking the other parameters as it will add ~10 secs for every mint build


def test_get_default_benchmark_simulated_datasets():
    simulated_datasets = get_default_benchmark_simulated_datasets()

    assert simulated_datasets[0] == ("daily_simulated", "D", 3*30, [30])
    assert simulated_datasets[1] == ("daily_simulated", "D", 2*365, [365])
    assert simulated_datasets[2] == ("hourly_simulated", "H", 7*24, [24])


def test_get_default_benchmark_real_datasets():
    real_datasets = get_default_benchmark_real_datasets()

    assert real_datasets[0] == ("daily_peyton_manning", "D", "ts", "y", [30, 365])
    assert real_datasets[1] == ("daily_female_births", "D", "Date", "Births", [30, 3*30])


def test_get_default_benchmark_silverkite_parameters():
    silverkite_parameters = get_default_benchmark_silverkite_parameters()

    assert silverkite_parameters["fit_algorithms"] == ("linear", "lasso", "ridge", "rf", "sgd")
    assert silverkite_parameters["max_cvs"] == [3]


def test_get_default_benchmark_parameters():
    benchmark_parameters = get_default_benchmark_parameters()

    assert benchmark_parameters["metric"] == EvaluationMetricEnum.MeanSquaredError
    assert os.path.basename(os.path.normpath(benchmark_parameters["data_directory"])) == "benchmark_output"
