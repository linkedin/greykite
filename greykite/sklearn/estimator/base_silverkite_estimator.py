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
# original author: Albert Chen, Reza Hosseini, Sayan Patra
"""sklearn estimator with common functionality between
SilverkiteEstimator and SimpleSilverkiteEstimator.
"""

from typing import Dict
from typing import Optional

import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error

from greykite.algo.common.col_name_utils import create_pred_category
from greykite.algo.common.ml_models import breakdown_regression_based_prediction
from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.algo.forecast.silverkite.forecast_silverkite_helper import get_silverkite_uncertainty_dict
from greykite.common import constants as cst
from greykite.common.features.timeseries_lags import min_max_lag_order
from greykite.common.time_properties import min_gap_in_seconds
from greykite.common.time_properties_forecast import get_simple_time_frequency_from_period
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator
from greykite.sklearn.estimator.silverkite_diagnostics import SilverkiteDiagnostics
from greykite.sklearn.uncertainty.uncertainty_methods import UncertaintyMethodEnum


class BaseSilverkiteEstimator(BaseForecastEstimator):
    """A base class for forecast estimators that fit using
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.

    Notes
    -----
    Allows estimators that fit using
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`
    to share the same functions for input data validation,
    fit postprocessing, predict, summary, plot_components, etc.

    Subclasses should:

        - Implement their own ``__init__`` that uses a superset of the parameters here.
        - Implement their own ``fit``, with this sequence of steps:

            - calls ``super().fit``
            - calls ``SilverkiteForecast.forecast`` or ``SimpleSilverkiteForecast.forecast_simple`` and stores the result in ``self.model_dict``
            - calls ``super().finish_fit``

    Uses ``coverage`` to set prediction band width. Even though
    coverage is not needed by ``forecast_silverkite``, it is included
    in every ``BaseForecastEstimator`` to be used universally for
    forecast evaluation.

    Therefore, ``uncertainty_dict`` must be consistent with ``coverage``
    if provided as a dictionary. If ``uncertainty_dict`` is None or
    "auto", an appropriate default value is set, according to ``coverage``.

    Parameters
    ----------
    score_func : callable, optional, default mean_squared_error
        See `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`.
    coverage : `float` between [0.0, 1.0] or None, optional
        See `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`.
    null_model_params : `dict`, optional
        Dictionary with arguments to define DummyRegressor null model, default is `None`.
        See `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`.
    uncertainty_dict : `dict` or `str` or None, optional
        How to fit the uncertainty model.
        See `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
        Note that this is allowed to be "auto". If None or "auto", will be set to
        a default value by ``coverage`` before calling ``forecast_silverkite``.

    Attributes
    ----------
    silverkite : Class or a derived class of `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast`
        The silverkite algorithm instance used for forecasting
    silverkite_diagnostics : Class or a derived class of `~greykite.sklearn.estimator.silverkite_diagnostics.SilverkiteDiagnostics`
        The silverkite class used for plotting and generating model summary.
    model_dict : `dict` or None
        A dict with fitted model and its attributes.
        The output of `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
    pred_cols : `list` [`str`] or None
        Names of the features used in the model.
    feature_cols : `list` [`str`] or None
        Column names of the patsy design matrix built by
        `~greykite.algo.common.ml_models.design_mat_from_formula`.
    df : `pandas.DataFrame` or None
        The training data used to fit the model.
    coef_ : `pandas.DataFrame` or None
        Estimated coefficient matrix for the model.
        Not available for ``random forest`` and ``gradient boosting`` methods and
        set to the default value `None`.
    _pred_category : `dict` or None
        A dictionary with keys being the predictor category and
        values being the predictors belonging to the category.
        For details, see
        `~greykite.sklearn.estimator.base_silverkite_estimator.BaseSilverkiteEstimator.pred_category`.
    extra_pred_cols : `list` or None
        User provided extra predictor names, for details, see
        `~greykite.sklearn.estimator.simple_silverkite_estimator.SimpleSilverkiteEstimator`
        or
        `~greykite.sklearn.estimator.silverkite_estimator.SilverkiteEstimator`.
    past_df : `pandas.DataFrame` or None
        The extra past data before training data used to generate autoregression terms.
    forecast : `pandas.DataFrame` or None
        Output of ``predict_silverkite``, set by ``self.predict``.
    forecast_x_mat : `pandas.DataFrame` or None
        The design matrix of the model at the predict time.
    model_summary : `class` or `None`
        The `~greykite.algo.common.model_summary.ModelSummary` class.

    See Also
    --------
    `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`
        Function performing the fit and predict.

    Notes
    -----
    The subclasses will pass ``fs_components_df`` to ``forecast_silverkite``. The model terms
    it creates internally are used to generate the component plots.

        - `~greykite.common.features.timeseries_features.fourier_series_multi_fcn` uses
          ``fs_components_df["names"]`` (e.g. ``tod``, ``tow``) to build the fourier series
          and to create column names.

        - ``fs_components_df["seas_names"]`` (e.g. ``daily``, ``weekly``) is appended
          to the column names, if provided.

    `~greykite.sklearn.estimator.silverkite_diagnostics.SilverkiteDiagnostics.plot_silverkite_components` groups
    based on ``fs_components_df["seas_names"]`` passed to ``forecast_silverkite`` during fit.
    E.g. any column containing ``daily`` is added to daily seasonality effect. The reason
    is as follows:

        1. User can provide ``tow`` and ``str_dow`` for weekly seasonality.
        These should be aggregated, and we can do that only based on "seas_names".
        2. yearly and quarterly seasonality both use ``ct1`` as "names" column.
        Only way to distinguish those effects is via "seas_names".
        3. ``ct1`` is also used for growth. If it is interacted with seasonality, the columns become
        indistinguishable without "seas_names".

    Additionally, the function sets yaxis labels based on ``seas_names``:
    ``daily`` as ylabel is much more informative than ``tod`` as ylabel in component plots.
    """
    def __init__(
            self,
            silverkite: SilverkiteForecast = SilverkiteForecast(),
            silverkite_diagnostics: SilverkiteDiagnostics = SilverkiteDiagnostics(),
            score_func: callable = mean_squared_error,
            coverage: float = None,
            null_model_params: Optional[Dict] = None,
            uncertainty_dict: Optional[Dict] = None):
        # initializes null model
        super().__init__(
            score_func=score_func,
            coverage=coverage,
            null_model_params=null_model_params)

        # required in subclasses __init__
        self.uncertainty_dict = uncertainty_dict

        # set by `fit`
        # fitted model in dictionary format returned from
        # the `forecast_silverkite` function
        self.silverkite: SilverkiteForecast = silverkite
        self.silverkite_diagnostics: SilverkiteDiagnostics = silverkite_diagnostics
        self.model_dict = None
        self.pred_cols = None
        self.feature_cols = None
        self.df = None
        self.coef_ = None

        # This is the ``past_df`` used during prediction.
        # Subclasses can set it for prediction use.
        # This ``past_df`` will be combined with the ``train_df`` from training
        # to generate necessary autoregression and lagged regressor terms.
        # If extra history is needed in prediction,
        # users can set this ``self.past_df`` before calling prediction.
        self.past_df = None

        # Predictor category, lazy initialization as None.
        # Will be updated in property function pred_category when needed.
        self._pred_category = None
        self.extra_pred_cols = None  # all silverkite estimators should support this.

        # set by the predict method
        self.forecast = None
        # set by predict method
        self.forecast_x_mat = None
        # set by the summary method
        self.model_summary = None

    def __set_uncertainty_dict(self, X, time_col, value_col):
        """Checks if ``coverage`` is consistent with the ``uncertainty_dict``
        used to train the ``forecast_silverkite`` model. Sets ``uncertainty_dict``
        to a default value if ``coverage`` is provided, and vice versa.

        Parameters
        ----------
        X: `pandas.DataFrame`
            Input timeseries, with timestamp column,
            value column, and any additional regressors.
            The value column is the response, included in
            ``X`` to allow transformation by `sklearn.pipeline.Pipeline`.
        time_col: `str`
            Time column name in ``X``.
        value_col: `str`
            Value column name in ``X``.

        Notes
        -----
        Intended to be called by `fit`.

        ``X`` is necessary to define default parameters for
        ``uncertainty_dict`` if ``coverage`` is provided but ``uncertainty_dict is None``
         or ``uncertainty_dict=="auto"``.
        (NB: ``X`` would not be necessary and this function could called from __init__
        if ``forecast_silverkite`` provides a default value for ``uncertainty_dict``
        given the target coverage).
        """
        period = min_gap_in_seconds(df=X, time_col=time_col)
        simple_freq = get_simple_time_frequency_from_period(period).name

        # Updates `uncertainty_dict` if None or "auto" or missing quantiles,
        # to match ``coverage``.
        # Raises an exception if both are provided and they don't match.
        self.uncertainty_dict = get_silverkite_uncertainty_dict(
            uncertainty=self.uncertainty_dict,
            simple_freq=simple_freq,
            coverage=self.coverage)

        # Updates coverage if None, to match the widest interval of
        # ``uncertainty_dict``. If coverage is not None, they are
        # already consistent, but we set it anyway.
        if self.uncertainty_dict is not None:
            quantiles = self.uncertainty_dict["params"]["quantiles"]
            self.coverage = quantiles[-1] - quantiles[0]

    def fit(
            self,
            X,
            y=None,
            time_col=cst.TIME_COL,
            value_col=cst.VALUE_COL,
            **fit_params):
        """Pre-processing before fitting ``Silverkite`` forecast model.

        Parameters
        ----------
        X: `pandas.DataFrame`
            Input timeseries, with timestamp column,
            value column, and any additional regressors.
            The value column is the response, included in
            ``X`` to allow transformation by `sklearn.pipeline`.
        y: ignored
            The original timeseries values, ignored.
            (The ``y`` for fitting is included in ``X``).
        time_col: `str`
            Time column name in ``X``.
        value_col: `str`
            Value column name in ``X``.
        fit_params: `dict`
            additional parameters for null model.

        Notes
        -----
        Subclasses are expected to call this at the beginning of their ``fit`` method,
        before calling `~greykite.algo.forecast.silverkite.forecast_silverkite.SilverkiteForecast.forecast`.
        """
        # NB: calls `__set_uncertainty_dict` before `super().fit` to ensure
        # coverage is correct before fitting the null model.
        # (null model does not currently use `coverage`, but may in the future.)
        self.__set_uncertainty_dict(
            X=X,
            time_col=time_col,
            value_col=value_col)
        self.df = X

        super().fit(
            X=X,
            y=y,
            time_col=time_col,
            value_col=value_col,
            **fit_params)

    def finish_fit(self):
        """Makes important values of ``self.model_dict`` conveniently accessible.

        To be called by subclasses at the end of their ``fit`` method. Sets
        {``pred_cols``, ``feature_cols``, and ``coef_``}.
        """
        if self.model_dict is None:
            raise ValueError("Must set `self.model_dict` before calling this function.")

        self.pred_cols = self.model_dict["pred_cols"]
        self.feature_cols = self.model_dict["x_mat"].columns
        # model coefficients
        if hasattr(self.model_dict["ml_model"], "coef_"):
            self.coef_ = pd.DataFrame(
                self.model_dict["ml_model"].coef_,
                index=self.feature_cols)

        self._set_silverkite_diagnostics_params()
        return self

    def _set_silverkite_diagnostics_params(self):
        if self.silverkite_diagnostics is None:
            self.silverkite_diagnostics = SilverkiteDiagnostics()
        self.silverkite_diagnostics.set_params(self.pred_category, self.time_col_, self.value_col_)

    def predict(self, X, y=None):
        """Creates forecast for the dates specified in ``X``.

        Parameters
        ----------
        X: `pandas.DataFrame`
            Input timeseries with timestamp column and any additional regressors.
            Timestamps are the dates for prediction.
            Value column, if provided in ``X``, is ignored.
        y: ignored.

        Returns
        -------
        predictions: `pandas.DataFrame`
            Forecasted values for the dates in ``X``. Columns:

                - ``TIME_COL``: dates
                - ``PREDICTED_COL``: predictions
                - ``PREDICTED_LOWER_COL``: lower bound of predictions, optional
                - ``PREDICTED_UPPER_COL``: upper bound of predictions, optional
                - [other columns], optional

            ``PREDICTED_LOWER_COL`` and ``PREDICTED_UPPER_COL`` are present
            if ``self.coverage`` is not None.
        """
        # Returns the cached result if applicable
        cached_predictions = super().predict(X=X)
        if cached_predictions is not None:
            return cached_predictions

        if self.model_dict is None:
            raise NotFittedError("Call `fit` before calling `predict`.")
        if self.pred_cols is None:
            raise NotFittedError("Subclass must call `finish_fit` inside the `fit` method.")

        pred_res = self.silverkite.predict(
            fut_df=X,
            trained_model=self.model_dict,
            past_df=self.past_df,
            new_external_regressor_df=None)  # regressors are included in X
        pred_df = pred_res["fut_df"]
        x_mat = pred_res["x_mat"]
        assert len(pred_df) == len(X), "The returned prediction data must have same number of rows as the input ``X``"
        assert len(x_mat) == len(X), "The returned design matrix (features matrix) must have same number of rows as the input ``X``"

        # Predicts the uncertainty model if not already fit.
        if self.uncertainty_model is not None:
            # The quantile regression model.
            if self.uncertainty_dict.get("uncertainty_method") == UncertaintyMethodEnum.quantile_regression.name:
                pred_df = self.predict_uncertainty(
                    df=pred_df.rename(
                        # The ``self.value_col_`` in ``pred_df`` is the predictions.
                        columns={self.value_col_: cst.PREDICTED_COL}
                    ),
                    predict_params=dict(
                        x_mat=x_mat
                    )
                )

            # In case prediction fails for uncertainty, uses the original output.
            if pred_df is None:
                pred_df = pred_res["fut_df"]

        self.forecast = pred_df
        self.forecast_x_mat = x_mat

        # renames columns to standardized schema
        output_columns = {
            self.time_col_: cst.TIME_COL}
        if cst.PREDICTED_COL in pred_df.columns:
            output_columns[cst.PREDICTED_COL] = cst.PREDICTED_COL
        elif self.value_col_ in pred_df.columns:
            output_columns[self.value_col_] = cst.PREDICTED_COL

        # Checks if uncertainty by "simple_conditional_residuals" is also returned.
        # If so, extract the upper and lower limits of the tuples in
        # ``uncertainty_col`` to be lower and upper limits of the prediction interval.
        # Note that the tuple might have more than two elements if more than two
        # ``quantiles`` are passed in ``uncertainty_dict``.
        uncertainty_col = cst.QUANTILE_SUMMARY_COL
        if uncertainty_col in list(pred_df.columns):
            pred_df[cst.PREDICTED_LOWER_COL] = pred_df[uncertainty_col].apply(
                lambda x: x[0])
            pred_df[cst.PREDICTED_UPPER_COL] = pred_df[uncertainty_col].apply(
                lambda x: x[-1])
            # The following entries are to include the columns in the output,
            # they are not intended to rename the columns.
            output_columns.update({
                uncertainty_col: uncertainty_col})
            if cst.ERR_STD_COL in pred_df.columns:
                output_columns.update({cst.ERR_STD_COL: cst.ERR_STD_COL})
        # Checks if lower/upper columns are in the output df.
        # If so, includes these columns in the final output.
        if cst.PREDICTED_LOWER_COL in pred_df.columns and cst.PREDICTED_UPPER_COL in pred_df.columns:
            output_columns.update({
                cst.PREDICTED_LOWER_COL: cst.PREDICTED_LOWER_COL,
                cst.PREDICTED_UPPER_COL: cst.PREDICTED_UPPER_COL})

        predictions = (pred_df[output_columns.keys()]
                       .rename(output_columns, axis=1))
        # Caches the predictions
        self.cached_predictions_ = predictions
        return predictions

    def forecast_breakdown(
            self,
            grouping_regex_patterns_dict,
            forecast_x_mat=None,
            time_values=None,
            center_components=False,
            denominator=None,
            plt_title="breakdown of forecasts"):
        """Generates silverkite forecast breakdown for groupings given in
        ``grouping_regex_patterns_dict``. Note that this only works for
        additive regression models and not for models such as random forest.

        Parameters
        ----------
        grouping_regex_patterns_dict : `dict` {`str`: `str`}
            A dictionary with group names as keys and regexes as values.
            This dictinary is used to partition the columns into various groups
        forecast_x_mat : `pd.DataFrame`, default None
            The dataframe of design matrix of regression model.
            If None, this will be extracted from the estimator.
        time_values : `list` or `np.array`, default None
            A collection of values (usually timestamps) to be used in the figure.
            It can also be used to join breakdown data with other data when needed.
            If None, and ``forecast_x_mat`` is not passed, timestamps will be extracted
            from the estimator to match the``forecast_x_mat`` which is also extracted
            from the estimator.
            If None, and``forecast_x_mat`` is passed, the timestamps cannot be inferred.
            Therefore we simply create an integer index with size of ``forecast_x_mat``.
        center_components : `bool`, default False
            It determines if components should be centered at their mean and the mean
            be added to the intercept. More concretely if a componet is "x" then it will
            be mapped to "x - mean(x)"; and "mean(x)" will be added to the intercept so
            that the sum of the components remains the same.
        denominator : `str`, default None
            If not None, it will specify a way to divide the components. There are
            two options implemented:

            - "abs_y_mean" : `float`
                The absolute value of the observed mean of the response
            - "y_std" : `float`
                The standard deviation of the observed response

        plt_title : `str`, default "prediction breakdown"
            The title of generated plot

        Returns
        -------
        result : `dict`
            Dictionary returned by `~greykite.algo.common.ml_models.breakdown_regression_based_prediction`
        """
        # If ``forecast_x_mat`` is not passed, we assume its the
        # ``forecast_x_mat`` from the estimator.
        # In this case, we also have access to the corresponding timestamps.
        if forecast_x_mat is None:
            forecast_x_mat = self.forecast_x_mat
            time_values = self.forecast[self.model_dict["time_col"]]

        # If ``time_values`` is not passed or implicitly grabbed from previous step,
        # we simply create an integer index with size of ``forecast_x_mat``.
        # This will be useful in the figure x axis.
        if time_values is None:
            time_values = range(len(forecast_x_mat))

        return breakdown_regression_based_prediction(
            trained_model=self.model_dict,
            x_mat=forecast_x_mat,
            grouping_regex_patterns_dict=grouping_regex_patterns_dict,
            remainder_group_name="OTHER",
            center_components=center_components,
            denominator=denominator,
            index_values=time_values,
            index_col=self.model_dict["time_col"],
            plt_title=plt_title)

    @property
    def pred_category(self):
        """A dictionary that stores the predictor names in each category.

        This property is not initialized until used. This speeds up the
        fitting process. The categories includes

            - "intercept" : the intercept.
            - "time_features" : the predictors that include
              `~greykite.common.constants.TimeFeaturesEnum`
              but not
              `~greykite.common.constants.SEASONALITY_REGEX`.
            - "event_features" : the predictors that include
              `~greykite.common.constants.EVENT_PREFIX`.
            - "trend_features" : the predictors that include
              `~greykite.common.constants.TREND_REGEX`
              but not
              `~greykite.common.constants.SEASONALITY_REGEX`.
            - "seasonality_features" : the predictors that include
              `~greykite.common.constants.SEASONALITY_REGEX`.
            - "lag_features" : the predictors that include
              `~greykite.common.constants.LAG_REGEX`.
            - "regressor_features" : external regressors and other predictors
              manually passed to ``extra_pred_cols``, but not in the categories above.
            - "interaction_features" : the predictors that include
              interaction terms, i.e., including a colon.

        Note that each predictor falls into at least one category.
        Some "time_features" may also be "trend_features".
        Predictors with an interaction are classified into all categories matched by
        the interaction components. Thus, "interaction_features" are already included
        in the other categories.
        """
        if self.model_dict is None:
            raise NotFittedError("Must fit before getting predictor category.")
        if self._pred_category is None:
            # extra_pred_cols could be None/list
            extra_pred_cols = [] if self.extra_pred_cols is None else self.extra_pred_cols
            # regressor_cols could be non-exist/None/list
            # the if catches non-exist and None
            regressor_cols = [] if getattr(self, "regressor_cols", None) is None else getattr(self, "regressor_cols")
            # lagged regressors
            lagged_regressor_dict = getattr(self, "lagged_regressor_dict", None)
            lagged_regressor_cols = []
            if lagged_regressor_dict is not None:
                lagged_regressor_cols = list(lagged_regressor_dict.keys())
            self._pred_category = create_pred_category(
                pred_cols=self.model_dict["x_mat"].columns,
                # extra regressors are specified via "regressor_cols" in simple_silverkite_estimator
                extra_pred_cols=extra_pred_cols + regressor_cols + lagged_regressor_cols,
                df_cols=list(self.model_dict["df"].columns))
        return self._pred_category

    def get_max_ar_order(self):
        """Gets the maximum autoregression order.

        Returns
        -------
        max_ar_order : `int`
            The maximum autoregression order.
        """
        # The Silverkite Family specifies autoregression terms from ``self.autoreg_dict`` parameter.
        autoreg_dict = getattr(self, "autoreg_dict", None)
        if autoreg_dict is None:
            return 0
        if autoreg_dict == "auto":
            freq = getattr(self, "freq", None)
            forecast_horizon = getattr(self, "forecast_horizon", None)
            if freq is None or forecast_horizon is None:
                raise ValueError("The ``autoreg_dict`` is set to 'auto'. "
                                 "To get the default configuration, "
                                 "you need to set ``freq`` and ``forecast_horizon``. "
                                 "However, at least one of them is None.")
            autoreg_dict = SilverkiteForecast()._SilverkiteForecast__get_default_autoreg_dict(
                freq_in_days=to_offset(freq).delta.total_seconds() / 60 / 60 / 24,
                forecast_horizon=forecast_horizon,
                simulation_based=False
            )["autoreg_dict"]
        max_order = min_max_lag_order(
            lag_dict=autoreg_dict.get("lag_dict"),
            agg_lag_dict=autoreg_dict.get("agg_lag_dict")
        )["max_order"]
        return max_order

    def summary(self, max_colwidth=20):
        if self.silverkite_diagnostics is None:
            self._set_silverkite_diagnostics_params()
        return self.silverkite_diagnostics.summary(self.model_dict, max_colwidth)

    def plot_components(self, names=None, title=None):
        if self.model_dict is None:
            raise NotFittedError("Call `fit` before calling `plot_components`.")
        if self.silverkite_diagnostics is None:
            self._set_silverkite_diagnostics_params()
        return self.silverkite_diagnostics.plot_components(self.model_dict, names, title)

    def plot_trend(self, title=None):
        """Convenience function to plot the data and the trend component.

        Parameters
        ----------
        title: `str`, optional, default `None`
            Plot title.

        Returns
        -------
        fig: `plotly.graph_objects.Figure`
            Figure.
        """
        if title is None:
            title = "Trend plot"
        return self.plot_components(names=["trend"], title=title)

    def plot_seasonalities(self, title=None):
        """Convenience function to plot the data and the seasonality components.

        Parameters
        ----------
        title: `str`, optional, default `None`
            Plot title.

        Returns
        -------
        fig: `plotly.graph_objects.Figure`
            Figure.
        """
        if title is None:
            title = "Seasonality plot"
        seas_names = [
            "DAILY_SEASONALITY",
            "WEEKLY_SEASONALITY",
            "MONTHLY_SEASONALITY",
            "QUARTERLY_SEASONALITY",
            "YEARLY_SEASONALITY"]
        return self.plot_components(names=seas_names, title=title)

    def plot_trend_changepoint_detection(self, params=None):
        """Convenience function to plot the original trend changepoint detection results.

        Parameters
        ----------
        params : `dict` or `None`, default `None`
            The parameters in `~greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector.plot`.
            If set to `None`, all components will be plotted.

            Note: seasonality components plotting is not supported currently. ``plot`` parameter must be False.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            Figure.
        """
        if params is None:
            params = dict(
                observation=True,
                observation_original=True,
                trend_estimate=True,
                trend_change=True,
                yearly_seasonality_estimate=True,
                adaptive_lasso_estimate=True,
                seasonality_change=False,  # currently for trend only
                seasonality_change_by_component=False,
                seasonality_estimate=False,
                plot=False)
        else:
            # currently for trend only
            params["seasonality_change"] = False
            params["seasonality_estimate"] = False
            # need to return the figure object
            params["plot"] = False
        return self.model_dict["changepoint_detector"].plot(**params)
