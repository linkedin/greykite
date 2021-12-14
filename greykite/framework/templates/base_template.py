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
# original author: Albert Chen
"""Base class for templates.
Contains common code used by multiple templates.
"""
import functools
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional

import pandas as pd

from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.time_properties_forecast import get_forecast_time_properties
from greykite.framework.pipeline.utils import get_basic_pipeline
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.forecast_config_defaults import ForecastConfigDefaults
from greykite.framework.templates.template_interface import TemplateInterface
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator


class BaseTemplate(TemplateInterface, ForecastConfigDefaults, ABC):
    """Base template with common code used by multiple templates.

    Provides a particular modular approach to implement
    `~greykite.framework.templates.template_interface.TemplateInterface.apply_template_for_pipeline_params`.

    Includes the config defaults from
    `~greykite.framework.templates.forecast_config_defaults.ForecastConfigDefaults`.

    Subclasses must provide these properties / functions used by ``apply_template_for_pipeline_params``:

        - estimator (__init__ default value)
        - get_regressor_cols
        - get_lagged_regressor_info
        - get_hyperparameter_grid

    Subclasses may optionally want to override:

        - get_pipeline
        - get_forecast_time_properties
        - apply_metadata_defaults
        - apply_evaluation_metric_defaults
        - apply_evaluation_period_defaults
        - apply_computation_defaults
        - apply_model_components_defaults
        - apply_forecast_config_defaults
    """
    def __init__(self, estimator: BaseForecastEstimator):
        # See attributes of `TemplateInterface` and `ForecastConfigDefaults`.
        # Note that `self.config` includes modifications after applying default values.
        super().__init__()

        self._estimator: BaseForecastEstimator = estimator
        """The estimator instance to use as the final step in the pipeline.
        An instance of `greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`.
        """

        # Attributes used by `~greykite.framework.templates.base_template.apply_template_for_pipeline_params`.
        self.score_func = None
        """Score function used to select optimal model in CV."""
        self.score_func_greater_is_better = None
        """True if ``score_func`` is a score function, meaning higher is better,
        and False if it is a loss function, meaning lower is better.
        """
        self.regressor_cols = None
        """A list of regressor columns used in the training and prediction DataFrames.
        If None, no regressor columns are used.
        """
        self.lagged_regressor_cols = None
        """A list of lagged regressor columns used in the training and prediction DataFrames.
        If None, no lagged regressor columns are used.
        """
        self.pipeline = None
        """Pipeline to fit. The final named step must be called "estimator"."""
        self.time_properties = None
        """Time properties dictionary (likely produced by
        `~greykite.common.time_properties_forecast.get_forecast_time_properties`)
        """
        self.hyperparameter_grid = None
        """Sets properties of the steps in the pipeline,
        and specifies combinations to search over.
        Should be valid input to `sklearn.model_selection.GridSearchCV` (param_grid)
        or `sklearn.model_selection.RandomizedSearchCV` (param_distributions).
        """

    @property
    def estimator(self):
        """The estimator instance to use as the final step in the pipeline.
        An instance of `greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`.
        """
        return self._estimator

    @abstractmethod
    def get_regressor_cols(self):
        """Returns regressor column names.

        To be implemented by subclass.

        Available parameters:

            - self.df
            - self.config
            - self.score_func
            - self.score_func_greater_is_better

        Returns
        -------
        regressor_cols : `list` [`str`] or None
            See `~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        """
        pass

    def get_lagged_regressor_info(self):
        """Returns lagged regressor column names and minimal/maximal lag order. The lag order
        can be used to check potential imputation in the computation of lags.

        Can be overridden by subclass.

        Returns
        -------
        lagged_regressor_info : `dict`
            A dictionary that includes the lagged regressor column names and maximal/minimal lag order
            The keys are:

                lagged_regressor_cols : `list` [`str`] or None
                    See `~greykite.framework.pipeline.pipeline.forecast_pipeline`.
                overall_min_lag_order : `int` or None
                overall_max_lag_order : `int` or None
        """
        lagged_regressor_info = {
            "lagged_regressor_cols": None,
            "overall_min_lag_order": None,
            "overall_max_lag_order": None
        }
        return lagged_regressor_info

    def get_pipeline(self):
        """Returns pipeline.

        Implementation may be overridden by subclass
        if a different pipeline is desired.

        Uses ``self.estimator``, ``self.score_func``,
        ``self.score_func_greater_is_better``, ``self.config``,
        ``self.regressor_cols``.

        Available parameters:

            - self.df
            - self.config
            - self.score_func
            - self.score_func_greater_is_better
            - self.regressor_cols
            - self.estimator

        Returns
        -------
        pipeline : `sklearn.pipeline.Pipeline`
            See `~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        """
        return get_basic_pipeline(
            estimator=self.estimator,
            score_func=self.score_func,
            score_func_greater_is_better=self.score_func_greater_is_better,
            agg_periods=self.config.evaluation_metric_param.agg_periods,
            agg_func=self.config.evaluation_metric_param.agg_func,
            relative_error_tolerance=self.config.evaluation_metric_param.relative_error_tolerance,
            coverage=self.config.coverage,
            null_model_params=self.config.evaluation_metric_param.null_model_params,
            regressor_cols=self.regressor_cols,
            lagged_regressor_cols=self.lagged_regressor_cols)

    def get_forecast_time_properties(self):
        """Returns forecast time parameters.

        Uses ``self.df``, ``self.config``, ``self.regressor_cols``.

        Available parameters:

            - self.df
            - self.config
            - self.score_func
            - self.score_func_greater_is_better
            - self.regressor_cols
            - self.lagged_regressor_cols
            - self.estimator
            - self.pipeline

        Returns
        -------
        time_properties : `dict` [`str`, `any`] or None, default None
            Time properties dictionary (likely produced by
            `~greykite.common.time_properties_forecast.get_forecast_time_properties`)
            with keys:

            ``"period"`` : `int`
                Period of each observation (i.e. minimum time between observations, in seconds).
            ``"simple_freq"`` : `SimpleTimeFrequencyEnum`
                ``SimpleTimeFrequencyEnum`` member corresponding to data frequency.
            ``"num_training_points"`` : `int`
                Number of observations for training.
            ``"num_training_days"`` : `int`
                Number of days for training.
            ``"start_year"`` : `int`
                Start year of the training period.
            ``"end_year"`` : `int`
                End year of the forecast period.
            ``"origin_for_time_vars"`` : `float`
                Continuous time representation of the first date in ``df``.
        """
        return get_forecast_time_properties(
            df=self.df,
            time_col=self.config.metadata_param.time_col,
            value_col=self.config.metadata_param.value_col,
            freq=self.config.metadata_param.freq,
            date_format=self.config.metadata_param.date_format,
            train_end_date=self.config.metadata_param.train_end_date,
            regressor_cols=self.regressor_cols,
            lagged_regressor_cols=self.lagged_regressor_cols,
            forecast_horizon=self.config.forecast_horizon)

    @abstractmethod
    def get_hyperparameter_grid(self):
        """Returns hyperparameter grid.

        To be implemented by subclass.

        Available parameters:

            - self.df
            - self.config
            - self.score_func
            - self.score_func_greater_is_better
            - self.regressor_cols
            - self.estimator
            - self.pipeline
            - self.time_properties

        Returns
        -------
        hyperparameter_grid : `dict`, `list` [`dict`] or None
            See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
            The output dictionary values are lists, combined in grid search.
        """
        pass

    def apply_template_decorator(func):
        """Decorator for ``apply_template_for_pipeline_params`` function.

        By default, this applies ``apply_forecast_config_defaults`` to ``config``.

        Subclass may override this for pre/post processing of
        ``apply_template_for_pipeline_params``, such as input validation.
        In this case, ``apply_template_for_pipeline_params`` must also be implemented in the subclass.
        """
        @functools.wraps(func)
        def process_wrapper(self, df: pd.DataFrame, config: Optional[ForecastConfig] = None):
            # Sets defaults and makes a copy of ``config``
            # All subclasses should keep this line.
            config = self.apply_forecast_config_defaults(config)
            # <optional processing before calling `func`, if needed>
            pipeline_params = func(self, df, config)
            # <optional postprocessing after calling `func`, if needed>
            return pipeline_params
        return process_wrapper

    @apply_template_decorator
    def apply_template_for_pipeline_params(
            self,
            df: pd.DataFrame,
            config: Optional[ForecastConfig] = None) -> Dict:
        """Implements template interface method.
        Takes input data and optional configuration parameters
        to customize the model. Returns a set of parameters to call
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.

        See template interface for parameters and return value.

        Uses the methods in this class to set:

            - ``"regressor_cols"`` : get_regressor_cols()
            - ``lagged_regressor_cols`` : get_lagged_regressor_info()
            - ``"pipeline"`` : get_pipeline()
            - ``"time_properties"`` : get_forecast_time_properties()
            - ``"hyperparameter_grid"`` : get_hyperparameter_grid()

        All other parameters are taken directly from ``config``.
        """
        self.df = df
        self.config = config
        # Defines score_func, score_func_greater_is_better
        # Sets `score_func` to a string instead of a function, so CV results are
        # reported as "mean_test_{short_name}" instead of "mean_test_score".
        metric = EvaluationMetricEnum[config.evaluation_metric_param.cv_selection_metric]
        self.score_func = metric.name
        self.score_func_greater_is_better = metric.get_metric_greater_is_better()

        self.regressor_cols = self.get_regressor_cols()
        self.lagged_regressor_cols = self.get_lagged_regressor_info().get("lagged_regressor_cols", None)
        self.pipeline = self.get_pipeline()
        self.time_properties = self.get_forecast_time_properties()
        self.hyperparameter_grid = self.get_hyperparameter_grid()

        self.pipeline_params = dict(
            # input
            df=self.df,
            time_col=self.config.metadata_param.time_col,
            value_col=self.config.metadata_param.value_col,
            date_format=self.config.metadata_param.date_format,
            freq=self.config.metadata_param.freq,
            train_end_date=self.config.metadata_param.train_end_date,
            anomaly_info=self.config.metadata_param.anomaly_info,
            # model
            pipeline=self.pipeline,
            regressor_cols=self.regressor_cols,
            lagged_regressor_cols=self.lagged_regressor_cols,
            estimator=None,  # ignored when `pipeline` is provided
            hyperparameter_grid=self.hyperparameter_grid,
            hyperparameter_budget=self.config.computation_param.hyperparameter_budget,
            n_jobs=self.config.computation_param.n_jobs,
            verbose=self.config.computation_param.verbose,
            # forecast
            forecast_horizon=self.config.forecast_horizon,
            coverage=self.config.coverage,
            test_horizon=self.config.evaluation_period_param.test_horizon,
            periods_between_train_test=self.config.evaluation_period_param.periods_between_train_test,
            agg_periods=self.config.evaluation_metric_param.agg_periods,
            agg_func=self.config.evaluation_metric_param.agg_func,
            # evaluation
            score_func=self.score_func,
            score_func_greater_is_better=self.score_func_greater_is_better,
            cv_report_metrics=self.config.evaluation_metric_param.cv_report_metrics,
            null_model_params=self.config.evaluation_metric_param.null_model_params,
            relative_error_tolerance=self.config.evaluation_metric_param.relative_error_tolerance,
            # CV
            cv_horizon=self.config.evaluation_period_param.cv_horizon,
            cv_min_train_periods=self.config.evaluation_period_param.cv_min_train_periods,
            cv_expanding_window=self.config.evaluation_period_param.cv_expanding_window,
            cv_use_most_recent_splits=self.config.evaluation_period_param.cv_use_most_recent_splits,
            cv_periods_between_splits=self.config.evaluation_period_param.cv_periods_between_splits,
            cv_periods_between_train_test=self.config.evaluation_period_param.cv_periods_between_train_test,
            cv_max_splits=self.config.evaluation_period_param.cv_max_splits,
        )
        return self.pipeline_params

    # `apply_template_decorator` needs to be a static method and take `func` as the
    # only argument (not self). It also needs to be defined in this class to allow
    # override. If we use the @staticmethod decorator above, this error appears:
    #   `TypeError: 'staticmethod' object is not callable`
    # However, if we call staticmethod at the bottom of the class, after it is
    # applied to `apply_template_for_pipeline_params`, it works.
    apply_template_decorator = staticmethod(apply_template_decorator)
