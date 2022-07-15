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
# original author: Albert Chen, Rachit Kumar, Sayan Patra
"""sklearn estimator for prophet"""
import sys


try:
    import prophet
    from prophet.plot import plot_components
except ModuleNotFoundError:
    pass

from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error

from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.logging import pprint
from greykite.sklearn.estimator.base_forecast_estimator import BaseForecastEstimator


class ProphetEstimator(BaseForecastEstimator):
    """Wrapper for Facebook Prophet model.

    Parameters
    ----------
    score_func : callable
        see BaseForecastEstimator

    coverage : float between [0.0, 1.0]
        see BaseForecastEstimator

    null_model_params : dict with arguments to define DummyRegressor null model, optional, default=None
        see BaseForecastEstimator

    add_regressor_dict: dictionary of extra regressors to be added to the model, optional, default=None
        These should be available for training and entire prediction interval.

        Dictionary format::

            add_regressor_dict={  # we can add as many regressors as we want, in the following format
                "reg_col1": {
                    "prior_scale": 10,
                    "standardize": True,
                    "mode": 'additive'
                },
                "reg_col2": {
                    "prior_scale": 20,
                    "standardize": True,
                    "mode": 'multiplicative'
                }
            }

    add_seasonality_dict: dict of custom seasonality parameters to be added to the model, optional, default=None
        parameter details: https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py - refer to
        add_seasonality() function.
        Key is the seasonality component name e.g. 'monthly'; parameters are specified via dict.

        Dictionary format::

            add_seasonality_dict={
                'monthly': {
                    'period': 30.5,
                    'fourier_order': 5
                },
                'weekly': {
                    'period': 7,
                    'fourier_order': 20,
                    'prior_scale': 0.6,
                    'mode': 'additive',
                    'condition_name': 'condition_col'  # takes a bool column in df with True/False values. This means that
                    # the seasonality will only be applied to dates where the condition_name column is True.
                },
                'yearly': {
                    'period': 365.25,
                    'fourier_order': 10,
                    'prior_scale': 0.2,
                    'mode': 'additive'
                }
            }

        Note: If there is a conflict in built-in and custom seasonality e.g. both have "yearly", then custom seasonality
        will be used and Model will throw a warning such as:
        "INFO:prophet:Found custom seasonality named "yearly", disabling built-in yearly seasonality."

    kwargs : additional parameters
        Other parameters are the same as Prophet model, with one exception:
        ``interval_width`` is specified by ``coverage``.

        See source code ``__init__`` for the parameter names, and refer to
        Prophet documentation for a description:

            * https://facebook.github.io/prophet/docs/quick_start.html
            * https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py

    Attributes
    ----------
    model : ``Prophet`` object
        Prophet model object
    forecast : `pandas.DataFrame`
        Output of predict method of ``Prophet``.
    """

    def __init__(
            self,
            score_func=mean_squared_error,
            coverage=0.80,  # to specify interval_width in Prophet
            null_model_params=None,
            growth="linear",
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality="auto",
            holidays=None,
            seasonality_mode="additive",
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            uncertainty_samples=1000,
            add_regressor_dict=None,
            add_seasonality_dict=None):
        if "prophet" not in sys.modules:
            raise ValueError("The module 'prophet' is not installed. Please install manually.")

        # every subclass of BaseForecastEstimator must call super().__init__
        super().__init__(
            score_func=score_func,
            coverage=coverage,
            null_model_params=null_model_params)

        # necessary to set parameters, to ensure get_params() works (used in grid search)
        self.score_func = score_func
        self.coverage = coverage
        self.null_model_params = null_model_params
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.uncertainty_samples = uncertainty_samples
        # additional regressor names and optimization
        self.add_regressor_dict = add_regressor_dict
        # additional seasonality parameters
        self.add_seasonality_dict = add_seasonality_dict
        # set by the fit method
        self.model = None
        # set by the predict method
        self.forecast = None

    def fit(self, X, y=None, time_col=TIME_COL, value_col=VALUE_COL, **fit_params):
        """Fits prophet model.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Input timeseries, with timestamp column,
            value column, and any additional regressors.
            The value column is the response, included in
            X to allow transformation by `sklearn.pipeline.Pipeline`
        y : ignored
            The original timeseries values, ignored.
            (The y for fitting is included in ``X``.)
        time_col : `str`
            Time column name in ``X``
        value_col : `str`
            Value column name in ``X``
        fit_params : `dict`
            additional parameters for null model

        Returns
        -------
        self : self
            Fitted model is stored in ``self.model``.
        """
        super().fit(X, y=y, time_col=time_col, value_col=value_col, **fit_params)

        if self.add_regressor_dict is None:
            fit_columns = [time_col, value_col]
        else:
            reg_cols = list(self.add_regressor_dict.keys())
            fit_columns = [time_col, value_col] + reg_cols

        fit_df = X.reset_index(drop=True)[fit_columns]
        fit_df.rename(columns={time_col: "ds", value_col: "y"}, inplace=True)
        # Prophet expects these column names. Other estimators can use TIME_COL, etc.
        # uses coverage instead of interval_width to set prediction band width. This ensures a common
        # interface for parameters common to every BaseForecastEstimator, usually also needed for forecast evaluation
        # model must be initialized here, not in __init__, to update parameters in grid search
        self.model = prophet.Prophet(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale,
            mcmc_samples=self.mcmc_samples,
            interval_width=(self.coverage or 0.8),  # Prophet can't take None for this param.
            uncertainty_samples=self.uncertainty_samples
        )
        # if extra regressors are given, we add them to temporal features data
        # This implementation assumes that the regressor(s) are provided in time series df, alongside target column.
        if self.add_regressor_dict is not None:
            for reg_col, reg_params in self.add_regressor_dict.items():
                self.model.add_regressor(name=reg_col, **reg_params)

        # if custom seasonality is provided, we supply it to Prophet model
        if self.add_seasonality_dict is not None:
            for seasonality_type, seasonality_params in self.add_seasonality_dict.items():
                self.model.add_seasonality(name=seasonality_type, **seasonality_params)

        self.model.fit(fit_df)
        return self

    def predict(self, X, y=None):
        """Creates forecast for dates specified in ``X``.

        Parameters
        ----------
        X : `pandas.DataFrame`
            Input timeseries with timestamp column and any additional regressors.
            Timestamps are the dates for prediction.
            Value column, if provided in X, is ignored.
        y : ignored

        Returns
        -------
        predictions : `pandas.DataFrame`
            Forecasted values for the dates in ``X``. Columns:

                * TIME_COL dates
                * PREDICTED_COL predictions
                * PREDICTED_LOWER_COL lower bound of predictions, optional
                * PREDICTED_UPPER_COL upper bound of predictions, optional
                * [other columns], optional

            PREDICTED_LOWER_COL and PREDICTED_UPPER_COL are present iff coverage is not None
        """
        # Returns the cached result if applicable
        cached_predictions = super().predict(X=X)
        if cached_predictions is not None:
            return cached_predictions

        # if regressors are not provided, then use time column to predict future. Else, use regressor from predict df
        if self.add_regressor_dict is None:
            predict_columns = [self.time_col_]
        else:
            reg_cols = list(self.add_regressor_dict.keys())
            predict_columns = [self.time_col_] + reg_cols

        fut_df = X.reset_index(drop=True)[predict_columns]
        # prophet expects time_col name to be "ds"
        fut_df.rename(columns={self.time_col_: "ds"}, inplace=True)

        pred_df = self.model.predict(fut_df)
        self.forecast = pred_df     # This is used by the plot_components

        # renames columns to standardized schema
        output_columns = {
            "ds": TIME_COL,
            "yhat": PREDICTED_COL
        }
        if "yhat_lower" in pred_df.columns:
            output_columns["yhat_lower"] = PREDICTED_LOWER_COL
        if "yhat_upper" in pred_df.columns:
            output_columns["yhat_upper"] = PREDICTED_UPPER_COL

        predictions = (pred_df[output_columns.keys()]
                       .rename(output_columns, axis=1))
        # Caches the predictions
        self.cached_predictions_ = predictions
        return predictions

    def summary(self):
        """Prints input parameters and Prophet model parameters.

        Returns
        -------
        log_message : str
            log message printed to logging.info()
        """
        super().summary()
        if self.model is not None:
            log_message(pprint(self.model.params), LoggingLevelEnum.INFO)

    def plot_components(
            self,
            uncertainty=True,
            plot_cap=True,
            weekly_start=0,
            yearly_start=0,
            figsize=None):
        """Plot the ``Prophet`` forecast components on the dataset passed
        to ``predict``.

        Will plot whichever are available of: trend, holidays, weekly
        seasonality, and yearly seasonality.

        Parameters
        ----------
        uncertainty : `bool`, optional, default True
            Boolean to plot uncertainty intervals.
        plot_cap : `bool`, optional, default True
            Boolean indicating if the capacity should be shown
            in the figure, if available.
        weekly_start : `int`, optional, default 0
            Specifying the start day of the weekly seasonality plot.
            0 (default) starts the week on Sunday.
            1 shifts by 1 day to Jan 2, and so on.
        yearly_start : `int`, optional, default 0
            Specifying the start day of the yearly seasonality plot.
            0 (default) starts the year on Jan 1.
            1 shifts by 1 day to Jan 2, and so on.
        figsize : `tuple` , optional, default None
            Width, height in inches.

        Returns
        -------
        fig: `matplotlib.figure.Figure`
            A matplotlib figure.
        """
        if self.model is None:
            raise NotFittedError("The fit method has not been run yet.")

        if self.forecast is None:
            raise RuntimeError("The predict method has not been run yet.")

        try:
            return plot_components(
                m=self.model,
                fcst=self.forecast,
                uncertainty=uncertainty,
                plot_cap=plot_cap,
                weekly_start=weekly_start,
                yearly_start=yearly_start,
                figsize=figsize)
        except AttributeError as e:
            if "'DatetimeIndex'" in repr(e):
                # 'DatetimeIndex' object has no attribute 'weekday_name'
                raise Exception("Prophet 0.5 component plots are incompatible with pandas 1.*. "
                                "Upgrade to prophet:0.6 or higher.")
            else:
                raise e
