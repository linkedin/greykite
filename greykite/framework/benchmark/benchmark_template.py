# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Sayan Patra
"""Functions to run benchmarking."""

import itertools
import os
import timeit
from pathlib import Path

from greykite.common.evaluation import EvaluationMetricEnum
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum


def benchmark_silverkite_template(
        data_name,
        df,
        forecast_horizons,
        fit_algorithms,
        max_cvs,
        metadata=None,
        evaluation_metric=None):
    """Benchmarks silverkite template and returns the output as a list

    :param data_name: str
        Name of the dataset we are performing benchmarking on
        For real datasets, the data_name matches the corresponding filename in the data/ folder
        For simulated datasets, we follow the convention "<freq>_simulated" e.g. "daily_simulated"
    :param df: pd.DataFrame
        Dataframe containing the time and value columns
    :param forecast_horizons: List[int]
        One forecast is created for every given forecast_horizon
    :param fit_algorithms: List[str]
        Names of predictive models to fit.
        Options are "linear", "lasso", "ridge", "rf" etc.
    :param max_cvs: List[int] or None
        Number of maximum CV folds to use.
    :param metadata: :class:`~greykite.framework.templates.autogen.forecast_config.MetadataParam` or None, default None
        Information about the input data. See
        :class:`~greykite.framework.templates.autogen.forecast_config.MetadataParam`.
    :param evaluation_metric: :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam` or None, default None
        What metrics to evaluate. See
        :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam`.
    :return: .csv file
        Each row of the .csv file records the following outputs from one run of the silverkite template:

            - "data_name": Fixed string "<freq>_simulated", or name of the dataset in data/ folder
            - "forecast_model_name": "silverkite_<fit_algorithm>" e.g. "silverkite_linear" or "prophet"
            - "train_period": train_period
            - "forecast_horizon": forecast_horizon
            - "fit_algorithm": fit algorithm name
            - "cv_folds": max_cv
            - "runtime_sec": runtime in seconds
            - "train_mae": Mean Absolute Error of training data in backtest
            - "train_mape": Mean Absolute Percent Error of training data in backtest
            - "test_mae": Mean Absolute Error of testing data in backtest
            - "test_mape": Mean Absolute Percent Error of testing data in backtest
    """
    benchmark_results = []

    for forecast_horizon, fit_algorithm, max_cv in itertools.product(forecast_horizons, fit_algorithms, max_cvs):
        model_components = ModelComponentsParam(
            custom={
                "fit_algorithm_dict": {
                    "fit_algorithm": fit_algorithm,
                },
                "feature_sets_enabled": True
            }
        )
        evaluation_period = EvaluationPeriodParam(
            cv_max_splits=max_cv
        )

        start_time = timeit.default_timer()
        forecaster = Forecaster()
        result = forecaster.run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                forecast_horizon=forecast_horizon,
                metadata_param=metadata,
                evaluation_metric_param=evaluation_metric,
                model_components_param=model_components,
                evaluation_period_param=evaluation_period,
            )
        )
        runtime = timeit.default_timer() - start_time

        output_dict = dict(
            data_name=data_name,
            forecast_model_name=f"silverkite_{fit_algorithm}",
            train_period=df.shape[0],
            forecast_horizon=forecast_horizon,
            cv_folds=result.grid_search.n_splits_,
            runtime_sec=round(runtime, 3),
            train_mae=result.backtest.train_evaluation["MAE"].round(3),
            train_mape=result.backtest.train_evaluation["MAPE"].round(3),
            test_mae=result.backtest.test_evaluation["MAE"].round(3),
            test_mape=result.backtest.test_evaluation["MAPE"].round(3)
        )
        benchmark_results.append(output_dict)

    return benchmark_results


def get_default_benchmark_real_datasets():
    """Default parameter sets to framework.benchmark real datasets. The datasets are located in data folder.
    Every tuple has the following structure:
    (data_name, frequency, time_col, value_col, forecast_horizon)"""
    real_datasets = [
        # daily_peyton_manning, 8 years of data
        ("daily_peyton_manning", "D", "ts", "y", [30, 365]),
        # daily_female_births, 1 year of data
        ("daily_female_births", "D", "Date", "Births", [30, 3*30])
    ]

    return real_datasets


def get_default_benchmark_silverkite_parameters():
    """Default parameter sets for benchmarking silverkite template"""
    return dict(
        fit_algorithms=("linear", "lasso", "ridge", "rf", "sgd"),
        max_cvs=[3]
    )


def get_default_benchmark_parameters():
    """Default parameter sets for benchmarking"""
    directory = Path(__file__).parents[4]  # src/ root
    directory = os.path.abspath(directory)
    data_directory = os.path.join(directory, "benchmark_output")
    return dict(
        metric=EvaluationMetricEnum.MeanSquaredError,
        data_directory=data_directory
    )


def get_default_benchmark_simulated_datasets():
    """Default parameter sets to generate simulated data for benchmarking.
    The training periods and forecast horizon are chosen to complement default real datasets.
    Every tuple has the following structure:
    (data_name, frequency, training_periods, forecast_horizon)"""
    simulation_parameters = [
        # daily data
        ("daily_simulated", "D", 3*30, [30]),
        ("daily_simulated", "D", 2*365, [365]),
        # hourly data
        ("hourly_simulated", "H", 7*24, [24]),
        ("hourly_simulated", "H", 30*24, [7*24]),
        ("hourly_simulated", "H", 365*24, [6*30*24]),
        ("hourly_simulated", "H", 4*365*24, [365*24])
    ]

    return simulation_parameters
