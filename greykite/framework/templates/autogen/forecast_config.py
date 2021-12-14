# flake8: noqa
# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = forecast_config_from_dict(json.loads(json_string))

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union
from typing import cast


T = TypeVar("T")


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


def from_list_dict(f: Callable[[Any], T], x: Any) -> List[Dict[str, T]]:
    """Parses list of dictionaries, applying `f` to the dictionary values.
    All items must be dictionaries.
    """
    assert isinstance(x, list)
    assert all(isinstance(d, dict) for d in x)
    return [ { k: f(v) for (k, v) in d.items() } for d in x]


def from_list_dict_or_none(f: Callable[[Any], T], x: Any) -> List[Optional[Dict[str, T]]]:
    """Parses list of dictionaries or None elements, applying `f` to the dictionary values.
    If an element in the list is None, it is returned directly.
    """
    assert isinstance(x, list)
    assert all(d is None or isinstance(d, dict)for d in x)
    return [ { k: f(v) for (k, v) in d.items() } if isinstance(d, dict) else d for d in x]


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_list_str(x: Any) -> List[str]:
    assert isinstance(x, list)
    assert all(isinstance(item, str) for item in x)
    return x


def from_list_int(x: Any) -> List[int]:
    assert isinstance(x, list)
    assert all(isinstance(item, int) for item in x)
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class ComputationParam:
    """How to compute the result."""
    hyperparameter_budget: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    n_jobs: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    verbose: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""

    @staticmethod
    def from_dict(obj: Any) -> 'ComputationParam':
        assert isinstance(obj, dict)
        hyperparameter_budget = from_union([from_int, from_none], obj.get("hyperparameter_budget"))
        n_jobs = from_union([from_int, from_none], obj.get("n_jobs"))
        verbose = from_union([from_int, from_none], obj.get("verbose"))
        return ComputationParam(
            hyperparameter_budget=hyperparameter_budget,
            n_jobs=n_jobs,
            verbose=verbose)

    def to_dict(self) -> dict:
        result: dict = {}
        result["hyperparameter_budget"] = from_union([from_int, from_none], self.hyperparameter_budget)
        result["n_jobs"] = from_union([from_int, from_none], self.n_jobs)
        result["verbose"] = from_union([from_int, from_none], self.verbose)
        return result


@dataclass
class EvaluationMetricParam:
    """What metrics to evaluate"""
    agg_func: Optional[Callable] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    agg_periods: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    cv_report_metrics: Optional[Union[str, List[str]]] = None
    """See `score_func` in :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    cv_selection_metric: Optional[str] = None
    """See `score_func` in :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    null_model_params: Optional[Dict[str, Any]] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    relative_error_tolerance: Optional[float] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""

    @staticmethod
    def from_dict(obj: Any) -> 'EvaluationMetricParam':
        assert isinstance(obj, dict)
        agg_func = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("agg_func"))
        agg_periods = from_union([from_int, from_none], obj.get("agg_periods"))
        cv_report_metrics = from_union([from_str, from_list_str, from_none], obj.get("cv_report_metrics"))
        cv_selection_metric = from_union([from_str, from_none], obj.get("cv_selection_metric"))
        null_model_params = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("null_model_params"))
        relative_error_tolerance = from_union([from_float, from_none], obj.get("relative_error_tolerance"))
        return EvaluationMetricParam(
            agg_func=agg_func,
            agg_periods=agg_periods,
            cv_report_metrics=cv_report_metrics,
            cv_selection_metric=cv_selection_metric,
            null_model_params=null_model_params,
            relative_error_tolerance=relative_error_tolerance)

    def to_dict(self) -> dict:
        result: dict = {}
        result["agg_func"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.agg_func)
        result["agg_periods"] = from_union([from_int, from_none], self.agg_periods)
        result["cv_report_metrics"] = from_union([from_str, from_list_str, from_none], self.cv_report_metrics)
        result["cv_selection_metric"] = from_union([from_str, from_none], self.cv_selection_metric)
        result["null_model_params"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.null_model_params)
        result["relative_error_tolerance"] = from_union([to_float, from_none], self.relative_error_tolerance)
        return result


@dataclass
class EvaluationPeriodParam:
    """How to split data for evaluation."""
    cv_expanding_window: Optional[bool] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    cv_horizon: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    cv_max_splits: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    cv_min_train_periods: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    cv_periods_between_splits: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    cv_periods_between_train_test: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    cv_use_most_recent_splits: Optional[bool] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""
    periods_between_train_test: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`"""
    test_horizon: Optional[int] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`."""

    @staticmethod
    def from_dict(obj: Any) -> 'EvaluationPeriodParam':
        assert isinstance(obj, dict)
        cv_expanding_window = from_union([from_bool, from_none], obj.get("cv_expanding_window"))
        cv_horizon = from_union([from_int, from_none], obj.get("cv_horizon"))
        cv_max_splits = from_union([from_int, from_none], obj.get("cv_max_splits"))
        cv_min_train_periods = from_union([from_int, from_none], obj.get("cv_min_train_periods"))
        cv_periods_between_splits = from_union([from_int, from_none], obj.get("cv_periods_between_splits"))
        cv_periods_between_train_test = from_union([from_int, from_none], obj.get("cv_periods_between_train_test"))
        cv_use_most_recent_splits = from_union([from_bool, from_none], obj.get("cv_use_most_recent_splits"))
        periods_between_train_test = from_union([from_int, from_none], obj.get("periods_between_train_test"))
        test_horizon = from_union([from_int, from_none], obj.get("test_horizon"))
        return EvaluationPeriodParam(
            cv_expanding_window=cv_expanding_window,
            cv_horizon=cv_horizon,
            cv_max_splits=cv_max_splits,
            cv_min_train_periods=cv_min_train_periods,
            cv_periods_between_splits=cv_periods_between_splits,
            cv_periods_between_train_test=cv_periods_between_train_test,
            cv_use_most_recent_splits=cv_use_most_recent_splits,
            periods_between_train_test=periods_between_train_test,
            test_horizon=test_horizon)

    def to_dict(self) -> dict:
        result: dict = {}
        result["cv_expanding_window"] = from_union([from_bool, from_none], self.cv_expanding_window)
        result["cv_horizon"] = from_union([from_int, from_none], self.cv_horizon)
        result["cv_max_splits"] = from_union([from_int, from_none], self.cv_max_splits)
        result["cv_min_train_periods"] = from_union([from_int, from_none], self.cv_min_train_periods)
        result["cv_periods_between_splits"] = from_union([from_int, from_none], self.cv_periods_between_splits)
        result["cv_periods_between_train_test"] = from_union([from_int, from_none], self.cv_periods_between_train_test)
        result["cv_use_most_recent_splits"] = from_union([from_bool, from_none], self.cv_use_most_recent_splits)
        result["periods_between_train_test"] = from_union([from_int, from_none], self.periods_between_train_test)
        result["test_horizon"] = from_union([from_int, from_none], self.test_horizon)
        return result


@dataclass
class MetadataParam:
    """Properties of the input data"""
    anomaly_info: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    """Anomaly adjustment info. Anomalies in ``df`` are corrected before any forecasting is
    done. If None, no adjustments are made.
    See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
    """
    date_format: Optional[str] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`"""
    freq: Optional[str] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`"""
    time_col: Optional[str] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`"""
    train_end_date: Optional[str] = None
    """Last date to use for fitting the model. Forecasts are generated after this date.
    If None, it is set to the last date with a non-null value in value_col df.
    See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
    """
    value_col: Optional[str] = None
    """See :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`"""

    @staticmethod
    def from_dict(obj: Any) -> 'MetadataParam':
        assert isinstance(obj, dict)
        anomaly_info = from_union([
            lambda x: from_dict(lambda x: x, x),
            lambda x: from_list_dict(lambda x: x, x),
            from_none], obj.get("anomaly_info"))
        date_format = from_union([from_str, from_none], obj.get("date_format"))
        freq = from_union([from_str, from_none], obj.get("freq"))
        time_col = from_union([from_str, from_none], obj.get("time_col"))
        train_end_date = from_union([from_str, from_none], obj.get("train_end_date"))
        value_col = from_union([from_str, from_none], obj.get("value_col"))
        return MetadataParam(anomaly_info, date_format, freq, time_col, train_end_date, value_col)

    def to_dict(self) -> dict:
        result: dict = {}
        result["anomaly_info"] = from_union([
            lambda x: from_dict(lambda x: x, x),
            lambda x: from_list_dict(lambda x: x, x),
            from_none], self.anomaly_info)
        result["date_format"] = from_union([from_str, from_none], self.date_format)
        result["freq"] = from_union([from_str, from_none], self.freq)
        result["time_col"] = from_union([from_str, from_none], self.time_col)
        result["train_end_date"] = from_union([from_str, from_none], self.train_end_date)
        result["value_col"] = from_union([from_str, from_none], self.value_col)
        return result


@dataclass
class ModelComponentsParam:
    """Parameters to tune the model."""
    autoregression: Optional[Dict[str, Any]] = None
    """For modeling autoregression, see template for details"""
    changepoints: Optional[Dict[str, Any]] = None
    """For modeling changepoints, see template for details"""
    custom: Optional[Dict[str, Any]] = None
    """Additional parameters used by template, see template for details"""
    events: Optional[Dict[str, Any]] = None
    """For modeling events, see template for details"""
    growth: Optional[Dict[str, Any]] = None
    """For modeling growth (trend), see template for details"""
    hyperparameter_override: Optional[Union[Dict, List[Optional[Dict]]]] = None
    """After the above model components are used to create a hyperparameter grid,
    the result is updated by this dictionary, to create new keys or override existing ones.
    Allows for complete customization of the grid search.
    """
    regressors: Optional[Dict[str, Any]] = None
    """For modeling regressors, see template for details"""
    lagged_regressors: Optional[Dict[str, Any]] = None
    """For modeling lagged regressors, see template for details"""
    seasonality: Optional[Dict[str, Any]] = None
    """For modeling seasonality, see template for details"""
    uncertainty: Optional[Dict[str, Any]] = None
    """For modeling uncertainty, see template for details"""

    @staticmethod
    def from_dict(obj: Any) -> 'ModelComponentsParam':
        assert isinstance(obj, dict)
        autoregression = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("autoregression"))
        changepoints = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("changepoints"))
        custom = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("custom"))
        events = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("events"))
        growth = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("growth"))
        hyperparameter_override = from_union([
            lambda x: from_dict(lambda x: x, x),
            lambda x: from_list_dict_or_none(lambda x: x, x),
            from_none], obj.get("hyperparameter_override"))
        regressors = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("regressors"))
        lagged_regressors = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("lagged_regressors"))
        seasonality = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("seasonality"))
        uncertainty = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("uncertainty"))
        return ModelComponentsParam(
            autoregression=autoregression,
            changepoints=changepoints,
            custom=custom,
            events=events,
            growth=growth,
            hyperparameter_override=hyperparameter_override,
            regressors=regressors,
            lagged_regressors=lagged_regressors,
            seasonality=seasonality,
            uncertainty=uncertainty)

    def to_dict(self) -> dict:
        result: dict = {}
        result["autoregression"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.autoregression)
        result["changepoints"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.changepoints)
        result["custom"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.custom)
        result["events"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.events)
        result["growth"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.growth)
        result["hyperparameter_override"] = from_union([
            lambda x: from_dict(lambda x: x, x),
            lambda x: from_list_dict_or_none(lambda x: x, x),
            from_none], self.hyperparameter_override)
        result["regressors"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.regressors)
        result["lagged_regressors"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.lagged_regressors)
        result["seasonality"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.seasonality)
        result["uncertainty"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.uncertainty)
        return result


@dataclass
class ForecastConfig:
    """Config for providing parameters to the Forecast library"""
    computation_param: Optional[ComputationParam] = None
    """How to compute the result. See
    :class:`~greykite.framework.templates.autogen.forecast_config.ComputationParam`.
    """
    coverage: Optional[float] = None
    """Intended coverage of the prediction bands (0.0 to 1.0).
    If None, the upper/lower predictions are not returned.
    """
    evaluation_metric_param: Optional[EvaluationMetricParam] = None
    """What metrics to evaluate. See
    :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationMetricParam`.
    """
    evaluation_period_param: Optional[EvaluationPeriodParam] = None
    """How to split data for evaluation. See
    :class:`~greykite.framework.templates.autogen.forecast_config.EvaluationPeriodParam`.
    """
    forecast_horizon: Optional[int] = None
    """Number of periods to forecast into the future. Must be > 0.
    If None, default is determined from input data frequency.
    """
    forecast_one_by_one: Optional[Union[bool, int, List[int]]] = None
    """The options to activate the forecast one-by-one algorithm.
    See :class:`~greykite.sklearn.estimator.one_by_one_estimator.OneByOneEstimator`.
    Can be boolean, int, of list of int.
    If int, it has to be less than or equal to the forecast horizon.
    If list of int, the sum has to be the forecast horizon.
    """
    metadata_param: Optional[MetadataParam] = None
    """Information about the input data. See
    :class:`~greykite.framework.templates.autogen.forecast_config.MetadataParam`.
    """
    model_components_param: Optional[Union[ModelComponentsParam, List[Optional[ModelComponentsParam]]]] = None
    """Parameters to tune the model. Typically a single ModelComponentsParam, but the `SimpleSilverkiteTemplate`
    template also allows a list of ModelComponentsParam for grid search. A single ModelComponentsParam
    corresponds to one grid, and a list corresponds to a list of grids.
    See :class:`~greykite.framework.templates.autogen.forecast_config.ModelComponentsParam`.
    """
    model_template: Optional[Union[str, dataclass, List[Union[str, dataclass]]]] = None
    """Name of the model template. Typically a single string, but the `SimpleSilverkiteTemplate`
    template also allows a list of string for grid search.
    See :class:`~greykite.framework.templates.model_templates.ModelTemplateEnum`
    for valid names.
    """

    @staticmethod
    def from_dict(obj: Any) -> 'ForecastConfig':
        assert isinstance(obj, dict)
        computation_param = from_union([ComputationParam.from_dict, from_none], obj.get("computation_param"))
        coverage = from_union([from_float, from_none], obj.get("coverage"))
        evaluation_metric_param = from_union([EvaluationMetricParam.from_dict, from_none], obj.get("evaluation_metric_param"))
        evaluation_period_param = from_union([EvaluationPeriodParam.from_dict, from_none], obj.get("evaluation_period_param"))
        forecast_horizon = from_union([from_int, from_none], obj.get("forecast_horizon"))
        forecast_one_by_one = from_union([from_int, from_bool, from_none, from_list_int], obj.get("forecast_one_by_one"))
        metadata_param = from_union([MetadataParam.from_dict, from_none], obj.get("metadata_param"))
        if not isinstance(obj.get("model_components_param"), list):
            obj["model_components_param"] = [obj.get("model_components_param")]
        model_components_param = [from_union([ModelComponentsParam.from_dict, from_none], mcp) for mcp in obj.get("model_components_param")]
        if not isinstance(obj.get("model_template"), list):
            obj["model_template"] = [obj.get("model_template")]
        model_template = [from_union([from_str, from_none], mt) for mt in obj.get("model_template")]
        return ForecastConfig(
            computation_param=computation_param,
            coverage=coverage,
            evaluation_metric_param=evaluation_metric_param,
            evaluation_period_param=evaluation_period_param,
            forecast_horizon=forecast_horizon,
            forecast_one_by_one=forecast_one_by_one,
            metadata_param=metadata_param,
            model_components_param=model_components_param,
            model_template=model_template)

    def to_dict(self) -> dict:
        result: dict = {}
        result["computation_param"] = from_union([lambda x: to_class(ComputationParam, x), from_none], self.computation_param)
        result["coverage"] = from_union([to_float, from_none], self.coverage)
        result["evaluation_metric_param"] = from_union([lambda x: to_class(EvaluationMetricParam, x), from_none], self.evaluation_metric_param)
        result["evaluation_period_param"] = from_union([lambda x: to_class(EvaluationPeriodParam, x), from_none], self.evaluation_period_param)
        result["forecast_horizon"] = from_union([from_int, from_none], self.forecast_horizon)
        result["forecast_one_by_one"] = from_union([from_int, from_bool, from_none, from_list_int], self.forecast_one_by_one)
        result["metadata_param"] = from_union([lambda x: to_class(MetadataParam, x), from_none], self.metadata_param)
        if not isinstance(self.model_components_param, list):
            self.model_components_param = [self.model_components_param]
        result["model_components_param"] = [from_union([lambda x: to_class(ModelComponentsParam, x), from_none], mcp) for mcp in self.model_components_param]
        if not isinstance(self.model_template, list):
            self.model_template = [self.model_template]
        result["model_template"] = [from_union([from_str, from_none], mt) for mt in self.model_template]
        return result


def forecast_config_from_dict(s: Any) -> ForecastConfig:
    return ForecastConfig.from_dict(s)


def forecast_config_to_dict(x: ForecastConfig) -> Any:
    return to_class(ForecastConfig, x)
