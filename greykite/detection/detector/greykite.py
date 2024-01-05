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
# original author: Reza Hosseini, Sayan Patra

from copy import deepcopy

import numpy as np
import pandas as pd

from greykite.algo.uncertainty.conditional.conf_interval import conf_interval
from greykite.algo.uncertainty.conditional.conf_interval import predict_ci
from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import ERR_STD_COL
from greykite.common.constants import PREDICTED_ANOMALY_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import QUANTILE_SUMMARY_COL
from greykite.common.constants import RESIDUAL_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import ElementwiseEvaluationMetricEnum
from greykite.common.features.adjust_anomalous_data import adjust_anomalous_data
from greykite.common.features.timeseries_features import add_daily_events
from greykite.common.features.timeseries_features import add_time_features_df
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.time_properties import describe_timeseries
from greykite.common.time_properties import get_canonical_data
from greykite.common.time_properties import infer_freq
from greykite.detection.detector.ad_utils import add_new_params_to_records
from greykite.detection.detector.ad_utils import get_anomaly_df
from greykite.detection.detector.ad_utils import get_anomaly_df_from_outliers
from greykite.detection.detector.ad_utils import get_canonical_anomaly_df
from greykite.detection.detector.config import ADConfig
from greykite.detection.detector.config_to_reward import config_to_reward
from greykite.detection.detector.constants import DEFAULT_COVERAGE_GRID
from greykite.detection.detector.constants import PHASE_PREDICT
from greykite.detection.detector.constants import Z_SCORE_COL
from greykite.detection.detector.detector import Detector
from greykite.detection.detector.optimizer import CalcResult
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster


DETECTOR_PREDICT_COLS = [
    TIME_COL,
    ACTUAL_COL,
    PREDICTED_COL,
    PREDICTED_LOWER_COL,
    PREDICTED_UPPER_COL,
    PREDICTED_ANOMALY_COL,
    Z_SCORE_COL]
"""The standard columns returned by the greykite detector's `predict` method."""


class GreykiteDetector(Detector):
    """This class enables Greykite based anomaly detection algorithms.
        It takes a ``forecast_config`` and ``ad_config`` (see Parameters) and builds a detector which
        uses the forecast as baseline.

        The fit consists of following stages:

        - Fit a forecast model using the ``forecast_config`` passed
        - Fit a volatility model using
            `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`
            to optimize over

            -- ``volatility_features_list``
            -- ``coverage_grid``

            specified in ``ad_config`` passed.

    Parameters
    ----------
    ad_config: `~greykite.detection.detector.config.ADConfig` or None, default None
        Config object for anomaly detection to use.
    forecast_config : `~greykite.framework.templates.model_templates.ForecastConfig` or None, default None
        Config object for forecast to use.
    reward : See docstring for
        `~greykite.detection.detector.detector.Detector`

    Attributes
    ----------
    anomaly_percent_dict : `dict` or None,
        See attributes of `~greykite.detection.detector.detector.Detector`
    forecast_result : `~greykite.framework.pipeline.pipeline.ForecastResult` or None
        See class:`~greykite.framework.pipeline.pipeline.ForecastResult`
        for details.
    forecast_estimator : `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator` or None
        See class: `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`
        for more details.
    ci_model : `dict`
        This is the fitted volatility model which is the returned dictionary from
        `greykite.algo.uncertainty.conditional.conf_interval`.
    fit_info : `dict` or None
        See attributes of `~greykite.detection.detector.detector.Detector`.
    anomaly_df : `pandas.DataFrame` or None
        A dataframe which includes the start and end times of observed anomalies.
    fit_data : `~greykite.detection.detector.data.DetectorData` or None
        The data used in the `fit` method.
    """
    def __init__(
            self,
            forecast_config=None,
            ad_config=None,
            reward=None):
        """Initializes the GreykiteDetector class."""
        if forecast_config is None:
            forecast_config = ForecastConfig()
        else:
            forecast_config = deepcopy(forecast_config)
        if ad_config is None:
            ad_config = ADConfig()
        else:
            ad_config = deepcopy(ad_config)

        # Constructs an object of class `Reward` from `ad_config`.
        reward_from_config = config_to_reward(ad_config)
        if reward is None:
            reward = reward_from_config
        else:
            reward = reward + reward_from_config

        param_iterable = None
        if ad_config.coverage_grid is None:
            ad_config.coverage_grid = DEFAULT_COVERAGE_GRID  # coverage grid can not be empty
        param_iterable = add_new_params_to_records(
            new_params={"coverage": ad_config.coverage_grid},
            records=param_iterable)

        if ad_config.volatility_features_list is None:
            ad_config.volatility_features_list = [[]]  # volatility features can be empty
        param_iterable = add_new_params_to_records(
            new_params={"volatility_features": ad_config.volatility_features_list},
            records=param_iterable)

        if ad_config.ape_grid is not None:
            metric = ElementwiseEvaluationMetricEnum.AbsolutePercentError
            param_iterable = add_new_params_to_records(
                new_params={metric.get_metric_name(): ad_config.ape_grid},
                records=param_iterable)

        if ad_config.sape_grid is not None:
            metric = ElementwiseEvaluationMetricEnum.SymmetricAbsolutePercentError
            param_iterable = add_new_params_to_records(
                new_params={metric.get_metric_name(): ad_config.sape_grid},
                records=param_iterable)

        super().__init__(
            reward=reward,
            anomaly_percent_dict=None,
            # In the above, if anomaly percent appears in `ad_config`,
            # it is already baked into `reward` via `config_to_reward` call above.
            param_iterable=param_iterable)

        # Attributes
        self.forecast_config = forecast_config
        self.ad_config = ad_config

        # These will be set by the `fit` method.
        self.forecast_result = None
        self.forecast_estimator = None
        self.anomaly_df = None
        self.ci_model = None
        self.fit_info = None
        self.fit_data = None

    def fit(
            self,
            data):
        """The fit method for Greykite based method.
        The fit consists of following stages:

            - Fit a forecast model using the ``forecast_config`` passed
            - Fit a volatility model using
                `~greykite.algo.uncertainty.conditional.conf_interval.conf_interval`
                to optimize over

                -- ``volatility_features_list``
                -- ``coverage_grid``

                specified in ``ad_config`` passed.

        Parameters
        ----------
        data : `~greykite.detection.detector.data.DetectorData`
            The input data which needs to include the input timeseries in ``df``
            attribute. We assume df includes minimally these two columns:

            - ``TIME_COL``
            - ``VALUE_COL``

            and if labels are also available in this datasets we expect the labels
            to be available in ``ANOMALY_COL``
        """
        fit_data = deepcopy(data)
        df = fit_data.df.copy()
        if df is None:
            raise ValueError("observed dataframe (df) must be available in fit data")
        if VALUE_COL not in df.columns:
            raise ValueError(f"observed dataframe (df) must be include {VALUE_COL} column")
        if TIME_COL not in df.columns:
            raise ValueError(f"observed dataframe (df) must be include {TIME_COL} column")

        # Initializes the forecaster and extracts the forecast parameters.
        forecaster = Forecaster()
        forecast_params = forecaster.apply_forecast_config(
            df=df,
            config=self.forecast_config)
        freq = forecast_params["freq"] or infer_freq(df, time_col=TIME_COL)
        if freq is None:
            raise ValueError("Frequency could not be inferred as timestamps were too irregular.")

        # Sets train end date to be the last date in the data.
        # This way the Forecaster does not drop anomalous dates from the training data if
        # the anomaly is at the end of the data.
        train_end_date = forecast_params.get("train_end_date", None) or df[TIME_COL].max()
        date_format = forecast_params["date_format"]

        # Logs warnings if the data is not regularly spaced.
        # The missing timestamps are filled in the `get_canonical_data` function.
        time_stats = describe_timeseries(df=df, time_col=TIME_COL)
        if not time_stats["regular_increments"]:
            log_message(
                "Input time series data is not regularly spaced. "
                f"Minimum time increment: {time_stats['min_delta']}. "
                f"Maximum time increment: {time_stats['max_delta']}. "
                "We will attempt to fill in missing times.",
                LoggingLevelEnum.WARNING)

        # Logs warnings if the data has repeated timestamps.
        # The repeated timestamps are removed in the `get_canonical_data` function.
        ts_unique, ts_count = np.unique(
            df[TIME_COL],
            return_counts=True)
        if max(ts_count) > 1:
            log_message(
                "The data timestamps had repeated values. "
                f"One timestamp had {max(ts_count)} repetitions. "
                "We will attempt to remove repeated timestamps and only keep first instance.",
                LoggingLevelEnum.WARNING)

        # Gets the canonical data.
        canonical_data_dict = get_canonical_data(
            df=df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            freq=freq,
            date_format=date_format,
            train_end_date=train_end_date,
            anomaly_info=None)
        df = canonical_data_dict["df"].reset_index(drop=True)

        # The following code checks for the existence of anomaly labels in the input `data`
        # and `forecast_config`, extracts them if available and merges them into a single dataframe.
        # Updated data is saved in `self.fit_data`.
        fit_data.df = df
        self.fit_data = self.merge_anomaly_info(fit_data=fit_data, freq=freq)

        # Builds the anomaly info.
        if self.anomaly_df is None:
            anomaly_info = None
        else:
            anomaly_info = {
                "value_col": VALUE_COL,
                "anomaly_df": self.anomaly_df}

        # Updates `metadata_param` in the `forecast_config`.
        if self.forecast_config.metadata_param is None:
            self.forecast_config.metadata_param = MetadataParam()
        self.forecast_config.metadata_param.time_col = TIME_COL
        self.forecast_config.metadata_param.value_col = VALUE_COL
        self.forecast_config.metadata_param.train_end_date = train_end_date
        self.forecast_config.metadata_param.freq = freq
        self.forecast_config.metadata_param.anomaly_info = anomaly_info

        # Fits the forecast model.
        self.forecast_result = forecaster.run_forecast_config(
            df=df,
            config=self.forecast_config)
        forecast_estimator = self.forecast_result.model[-1]
        self.forecast_estimator = forecast_estimator

        default_param = {}
        optim_res = self.optimize_param(
            data=self.fit_data,
            param_iterable=self.param_iterable,
            default_param=default_param,
            phase="fit")

        self.fit_info = {
            "param": optim_res["best_param"],
            "param_full": optim_res["best_param_full"],
            "obj_value": optim_res["best_obj_value"],
            "param_obj_list": optim_res["param_obj_list"],
            "best_calc_result": optim_res["best_calc_result"]}
        self.fitted_df = self.fit_info["best_calc_result"].data.pred_df

    def calc_with_param(
            self,
            param,
            data=None,
            phase=PHASE_PREDICT):
        """Predicts anomalies assuming the parameters:

            - ``volatility_features``
            - ``coverage``

        are passed. This will enable optimization over these parameters
        in the `fit` phase.

        Parameters
        ----------
        param : `dict`
            The parameter to optimize over if desired.
        data : `~greykite.detection.detector.data.DetectorData`
            The input data which needs to include the input timeseries in ``df``
            attribute. We assume df includes minimally these two columns:

                - ``TIME_COL``
                - ``VALUE_COL``

            and if labels are also available in this datasets we expect the labels
            to be available in ``ANOMALY_COL``
        phase : `str`, default ``PHASE_PREDICT``
            If ``PHASE_PREDICT`` the baseline data will be generated and otherwise we
            assume we are in fitting phase and will extract the baseline from the
            fitted model.

        Returns
        -------
        cal_result: A calculation result from the optimizer.
        See `~greykite.detection.detector.optimizer.CalcResult`
        """
        df = data.df.copy()
        model = self.forecast_estimator
        trained_model = model.model_dict

        if phase == PHASE_PREDICT:
            # `forecast_pred_df` has the forecasted values.
            forecast_pred_df = model.predict(X=df)
            x_mat = model.forecast_x_mat
            # Extracts only the time column and forecast column.
            forecast_pred_df = forecast_pred_df[[TIME_COL, PREDICTED_COL]]
        else:
            x_mat = trained_model["x_mat"]
            forecast_pred_df = trained_model["fitted_df"].copy()
            # Extracts only the time column and forecast column.
            # Forecasts are in the original `value_col` in fitted data.
            forecast_pred_df = forecast_pred_df[[TIME_COL, VALUE_COL]]
            forecast_pred_df.columns = [TIME_COL, PREDICTED_COL]
            # Asserts that the returned dataframe has the correct size.
            assert len(forecast_pred_df) == len(df), (
                f"length of `forecast_pred_df`: {len(forecast_pred_df)},"
                f" must be same as length of `df`: {len(df)}")

        volatility_df = pd.merge(df, forecast_pred_df, on=TIME_COL)
        assert len(volatility_df) == len(df), "length of `volatility_df` must be same `df`"

        # Adds time features
        volatility_df = add_time_features_df(
            df=volatility_df,
            time_col=TIME_COL,
            conti_year_origin=trained_model["origin_for_time_vars"])

        # Adds daily events (e.g. holidays)
        # if daily event data are given, we add them to temporal features data
        # `date_col` below is used to join with `daily_events` data given
        # in `daily_event_df_dict`.
        # Note: events must be provided for both train and forecast time range.
        daily_event_df_dict = trained_model["daily_event_df_dict"]
        if daily_event_df_dict is not None:
            daily_event_neighbor_impact = trained_model["daily_event_neighbor_impact"]
            volatility_df = add_daily_events(
                df=volatility_df,
                event_df_dict=daily_event_df_dict,
                date_col="date",
                neighbor_impact=daily_event_neighbor_impact)

        coverage = param.get("coverage", None)
        volatility_features = param.get("volatility_features", [])  # volatility features can be empty
        ape = param.get(ElementwiseEvaluationMetricEnum.AbsolutePercentError.get_metric_name(), None)
        sape = param.get(ElementwiseEvaluationMetricEnum.SymmetricAbsolutePercentError.get_metric_name(), None)
        sigma_scaler = trained_model["sigma_scaler"]
        h_mat = trained_model["h_mat"]
        x_mean = trained_model["x_mean"]
        if self.ad_config.variance_scaling is False or self.ad_config.variance_scaling is None:
            # Enables default variance scaling.
            sigma_scaler = None
            h_mat = None

        if phase == "fit":
            alpha = (1 - coverage)
            q_lower = alpha / 2.0
            q_upper = 1 - q_lower
            quantiles = (q_lower, q_upper)

            # Residual column is calculated based on the adjusted values so that the anomalies and outliers
            # do not impact the training of the volatility model.
            volatility_df[RESIDUAL_COL] = volatility_df[f"adjusted_{VALUE_COL}"] - volatility_df[PREDICTED_COL]
            ci_model = conf_interval(
                df=volatility_df,
                distribution_col=RESIDUAL_COL,
                offset_col=PREDICTED_COL,
                conditional_cols=volatility_features,
                quantiles=quantiles,
                sigma_scaler=sigma_scaler,
                h_mat=h_mat,
                x_mean=x_mean,
                min_admissible_value=self.ad_config.min_admissible_value,
                max_admissible_value=self.ad_config.max_admissible_value)
        else:
            ci_model = self.fit_info["best_calc_result"].model

        # Calculates the prediction intervals using the fitted `ci_model`.
        # Note that if there is no variance scaling (since variance scaling only applies to ridge / linear regression),
        # `ci_model` has contained this information since both `sigma_scaler` and `h_mat` were set to `None`.
        # In this case, the `predict_ci` function behaves the same as `x_mat` is not passed.
        # Also note that the default behavior in greykite is to scale the variance.
        ci_df = predict_ci(new_df=volatility_df, ci_model=ci_model, x_mat=x_mat)
        # Adds the z score column.
        ci_df[Z_SCORE_COL] = (ci_df[VALUE_COL] - ci_df[PREDICTED_COL]) / ci_df[ERR_STD_COL]

        cols = [PREDICTED_COL, QUANTILE_SUMMARY_COL, ERR_STD_COL, Z_SCORE_COL]
        pred_df = pd.concat([df, ci_df[cols]], axis=1)
        pred_df[PREDICTED_LOWER_COL] = pred_df[QUANTILE_SUMMARY_COL].map(lambda x: x[0])
        pred_df[PREDICTED_UPPER_COL] = pred_df[QUANTILE_SUMMARY_COL].map(lambda x: x[1])

        # Since, we like to return the observed values in `ACTUAL_COL`,
        # the column `ACTUAL_COL` is added and it is set to be equal to
        # the `VALUE_COL` which includes the observed values.
        pred_df[ACTUAL_COL] = pred_df[VALUE_COL]

        # Anomaly is declared when the actual is outside the confidence interval.
        y_pred = (
            (pred_df[ACTUAL_COL] < pred_df[PREDICTED_LOWER_COL]) |
            (pred_df[ACTUAL_COL] > pred_df[PREDICTED_UPPER_COL]))

        # Adds Absolute Percent Error (APE) threshold check if ape is not None.
        if ape is not None:
            metric = ElementwiseEvaluationMetricEnum.AbsolutePercentError
            pred_df[metric.get_metric_name()] = pred_df.apply(
                lambda row: metric.get_metric_func()(row[ACTUAL_COL], row[PREDICTED_COL]), axis=1)
            y_pred = y_pred & (pred_df[metric.get_metric_name()] > ape)

        # Adds Symmetric Absolute Percent Error (SAPE) threshold check if sape is not None.
        if sape is not None:
            metric = ElementwiseEvaluationMetricEnum.SymmetricAbsolutePercentError
            pred_df[metric.get_metric_name()] = pred_df.apply(
                lambda row: metric.get_metric_func()(row[ACTUAL_COL], row[PREDICTED_COL]), axis=1)
            y_pred = y_pred & (pred_df[metric.get_metric_name()] > sape)

        pred_df[PREDICTED_ANOMALY_COL] = y_pred

        # Only subset the columns needed / prescribed.
        pred_df = pred_df[DETECTOR_PREDICT_COLS]
        if ANOMALY_COL in df.columns:
            pred_df[ANOMALY_COL] = df[ANOMALY_COL]
        else:
            pred_df[ANOMALY_COL] = None
        data.y_pred = y_pred
        data.pred_df = pred_df

        return CalcResult(data=data, model=ci_model)

    def merge_anomaly_info(self, fit_data, freq):
        """This function combines anomalies information that may exist in various places
        and combine them into a consistent `anomaly_df`: a point will be an anomaly if it appears in
        ANY of the given inputs.

        1. As `ANOMALY_COL` column in the input fit_data (`fit_data.df`)
        2. As a vector in the input fit_data (`fit_data.y_true`)
        3. As a separate dataframe in the input fit_data (`fit_data.anomaly_df`)
        4. As a separate dataframe in the `metadata_param` in the `forecast_config`.

        It then updates the anomaly information to be the final one across various inputs.

        Parameters
        ----------
        fit_data : `~greykite.detection.detector.fit_data.DetectorData` or None, default None
            The input fit_data to the `fit` method.
        freq : `str`
            The frequency of the input fit_data.

        Returns
        -------
        fit_data : `~greykite.detection.detector.data.DetectorData`
            The updated fit_data which includes the merged anomaly information.
        """
        df = fit_data.df.copy()
        merged_anomaly_df = pd.DataFrame()

        # Adds anomalies from `fit_data.df`.
        if ANOMALY_COL in df.columns:
            anomaly_df = get_anomaly_df(
                df=df,
                time_col=TIME_COL,
                anomaly_col=ANOMALY_COL)
            merged_anomaly_df = pd.concat([merged_anomaly_df, anomaly_df])

        # Adds anomalies from `fit_data.y_true`.
        if fit_data.y_true is not None:
            assert len(fit_data.y_true) == len(df), (
                f"length of `y_true`: {len(fit_data.y_true)}, must be same as length of `df`: {len(df)}")
            # Builds a temporary df to be used in `get_anomaly_df`.
            temp_df = pd.DataFrame({
                TIME_COL: df[TIME_COL],
                ANOMALY_COL: fit_data.y_true})
            anomaly_df = get_anomaly_df(
                df=temp_df,
                time_col=TIME_COL,
                anomaly_col=ANOMALY_COL)
            merged_anomaly_df = pd.concat([merged_anomaly_df, anomaly_df])

        # Adds anomalies from `fit_data.anomaly_df`.
        if fit_data.anomaly_df is not None:
            fit_data.anomaly_df = fit_data.anomaly_df[[START_TIME_COL, END_TIME_COL]]
            merged_anomaly_df = pd.concat([merged_anomaly_df, fit_data.anomaly_df])

        # Adds anomalies from `metadata_param` in `forecast_config`.
        metadata_param = self.forecast_config.metadata_param
        if (metadata_param is not None) and (metadata_param.anomaly_info is not None):
            anomaly_df = metadata_param.anomaly_info.get("anomaly_df", None)
            if anomaly_df is not None:
                anomaly_df = anomaly_df[[START_TIME_COL, END_TIME_COL]]
            merged_anomaly_df = pd.concat([merged_anomaly_df, anomaly_df])

        # Input fit_data might have huge outliers which have a large impact on model fit.
        # At this step, the function identifies outliers based on z-scores
        # and constructs `anomaly_df_outliers`to save the information,
        # which is added to `merged_anomaly_df`.
        # Does 1.0 percent trimming to ensure the standard deviation and mean,
        # used in z-score calculation are robust.
        anomaly_df_outliers = get_anomaly_df_from_outliers(
            df=df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            freq=freq,
            trim_percent=1.0)
        if not anomaly_df_outliers.empty:
            log_message(
                f"Found the following outliers: \n{anomaly_df_outliers[START_TIME_COL]}. "
                " Adding these to `anomaly_df`.",
                LoggingLevelEnum.WARNING)
            merged_anomaly_df = pd.concat([merged_anomaly_df, anomaly_df_outliers])

        # Gets canonical anomaly df.
        if merged_anomaly_df.empty:
            merged_anomaly_df = None
            log_message(
                "No anomalies are provided and no outliers have been found. "
                "Setting 'anomaly_df' to None.",
                LoggingLevelEnum.WARNING)
        else:
            merged_anomaly_df = get_canonical_anomaly_df(
                anomaly_df=merged_anomaly_df,
                freq=freq)

        # Updates `anomaly_df` in respective places.
        if merged_anomaly_df is not None:
            # Directly adjusts anomalies in input fit_data,
            # so that prediction intervals are not impacted by outliers as well
            # in `predict_with_params` during the "fit" phase.
            # Values of the anomaly locations will become none in "adjusted_{VALUE_COL}" column.
            adj_df_info = adjust_anomalous_data(
                df=df,
                time_col=TIME_COL,
                value_col=VALUE_COL,
                anomaly_df=merged_anomaly_df)
            df = adj_df_info["augmented_df"]
            df[ANOMALY_COL] = df[ANOMALY_COL].astype(bool)

            if self.forecast_config.metadata_param is None:
                self.forecast_config.metadata_param = MetadataParam()
            if self.forecast_config.metadata_param.anomaly_info is None:
                self.forecast_config.metadata_param.anomaly_info = {
                    "value_col": VALUE_COL,
                    "anomaly_df": merged_anomaly_df}
            else:
                self.forecast_config.metadata_param.anomaly_info.update({
                    "value_col": VALUE_COL,
                    "anomaly_df": merged_anomaly_df})
        else:
            df[f"adjusted_{VALUE_COL}"] = df[VALUE_COL]
            df[ANOMALY_COL] = None
        fit_data.y_true = df[ANOMALY_COL]
        fit_data.df = df
        fit_data.anomaly_df = merged_anomaly_df
        self.anomaly_df = merged_anomaly_df

        return fit_data

    def summary(self):
        """Returns a summary of the fitted model.
        Fetches the summary from the forecast estimator and adds it to the summary
        of the base class.
        """
        content = super().summary()
        content += self.forecast_estimator.summary().__str__()

        return content
