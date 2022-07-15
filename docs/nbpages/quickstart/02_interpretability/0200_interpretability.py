"""
Interpretability
================

Silverkite generates easily interpretable forecasting models when using its default ML algorithms (e.g. Ridge).
This is because after transforming the raw features
to basis functions (transformed features), the model uses an additive structure.
Silverkite can break down each forecast into various summable components e.g. long-term growth,
seasonality, holidays, events, short-term growth (auto-regression), regressors impact etc.

The approach to generate these breakdowns consists of two steps:

#. Group the transformed variables into various meaningful groups.
#. Calculate the sum of the features multiplied by their regression coefficients within each group.

These breakdowns then can be used to answer questions such as:

- Question 1: How is the forecast value is generated?
- Question 2: What is driving the change of the forecast as new data comes in?

Forecast components can also help us analyze model behavior and sensitivity.
This is because while it is not feasible to compare a large set of features across two model
settings, it can be quite practical and informative to compare a few well-defined components.
"""

# required imports
import plotly
import warnings
import pandas as pd
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results
from greykite.common.viz.timeseries_plotting import plot_multivariate
warnings.filterwarnings("ignore")


# %%
# Function to load and prepare data
# ---------------------------------
# This is the code to upload and prepare the daily bike-sharing data in Washington DC.
def prepare_bikesharing_data():
    """Loads bike-sharing data and adds proper regressors."""
    dl = DataLoaderTS()
    agg_func = {"count": "sum", "tmin": "mean", "tmax": "mean", "pn": "mean"}
    df = dl.load_bikesharing(agg_freq="daily", agg_func=agg_func)

    # There are some zero values which cause issue for MAPE
    # This adds a small number to all data to avoid that issue
    value_col = "count"
    df[value_col] += 10
    # We drop last value as data might be incorrect as original data is hourly
    df.drop(df.tail(1).index, inplace=True)
    # We only use data from 2018 for demonstration purposes (run time is shorter)
    df = df.loc[df["ts"] > "2018-01-01"]
    df.reset_index(drop=True, inplace=True)

    print(f"\n df.tail(): \n {df.tail()}")

    # Creates useful regressors from existing raw regressors
    df["bin_pn"] = (df["pn"] > 5).map(float)
    df["bin_heavy_pn"] = (df["pn"] > 20).map(float)
    df.columns = [
        "ts",
        value_col,
        "regressor_tmin",
        "regressor_tmax",
        "regressor_pn",
        "regressor_bin_pn",
        "regressor_bin_heavy_pn"]

    forecast_horizon = 7
    train_df = df.copy()
    test_df = df.tail(forecast_horizon).reset_index(drop=True)
    # When using the pipeline (as done in the ``fit_forecast`` below),
    # fitting and prediction are done in one step
    # Therefore for demonstration purpose we remove the response values of last 7 days.
    # This is needed because we are using regressors,
    # and future regressor data must be augmented to ``df``.
    # We mimic that by removal of the values of the response.
    train_df.at[(len(train_df) - forecast_horizon):len(train_df), value_col] = None

    print(f"train_df shape: \n {train_df.shape}")
    print(f"test_df shape: \n {test_df.shape}")
    print(f"train_df.tail(14): \n {train_df.tail(14)}")
    print(f"test_df: \n {test_df}")

    return {
        "train_df": train_df,
        "test_df": test_df}


# %%
# Function to fit silverkite
# --------------------------
# This is the code for fitting a silverkite model to the data.
def fit_forecast(
        df,
        time_col,
        value_col):
    """Fits a daily model for this use case.
    The daily model is a generic silverkite model with regressors."""

    meta_data_params = MetadataParam(
        time_col=time_col,
        value_col=value_col,
        freq="D",
    )

    # Autoregression to be used in the function
    autoregression = {
        "autoreg_dict": {
            "lag_dict": {"orders": [1, 2, 3]},
            "agg_lag_dict": {
                "orders_list": [[7, 7*2, 7*3]],
                "interval_list": [(1, 7), (8, 7*2)]},
            "series_na_fill_func": lambda s: s.bfill().ffill()},
            "fast_simulation": True
    }

    # Changepoints configuration
    # The config includes changepoints both in trend and seasonality
    changepoints = {
        "changepoints_dict": {
            "method": "auto",
            "yearly_seasonality_order": 15,
            "resample_freq": "2D",
            "actual_changepoint_min_distance": "100D",
            "potential_changepoint_distance": "50D",
            "no_changepoint_distance_from_end": "50D"},
        "seasonality_changepoints_dict": {
            "method": "auto",
            "yearly_seasonality_order": 15,
            "resample_freq": "2D",
            "actual_changepoint_min_distance": "100D",
            "potential_changepoint_distance": "50D",
            "no_changepoint_distance_from_end": "50D"}
        }

    regressor_cols = [
        "regressor_tmin",
        "regressor_bin_pn",
        "regressor_bin_heavy_pn",
    ]

    # Model parameters
    model_components = ModelComponentsParam(
        growth=dict(growth_term="linear"),
        seasonality=dict(
            yearly_seasonality=[15],
            quarterly_seasonality=[False],
            monthly_seasonality=[False],
            weekly_seasonality=[7],
            daily_seasonality=[False]
        ),
        custom=dict(
            fit_algorithm_dict=dict(fit_algorithm="ridge"),
            extra_pred_cols=None,
            normalize_method="statistical"
        ),
        regressors=dict(regressor_cols=regressor_cols),
        autoregression=autoregression,
        uncertainty=dict(uncertainty_dict=None),
        events=dict(holiday_lookup_countries=["US"]),
        changepoints=changepoints
     )

    # Evaluation is done on same ``forecast_horizon`` as desired for output
    evaluation_period_param = EvaluationPeriodParam(
        test_horizon=None,
        cv_horizon=forecast_horizon,
        cv_min_train_periods=365*2,
        cv_expanding_window=True,
        cv_use_most_recent_splits=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=0,
        cv_max_splits=5,
    )

    # Runs the forecast model using "SILVERKITE" template
    forecaster = Forecaster()
    result = forecaster.run_forecast_config(
        df=df,
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            coverage=0.95,
            forecast_horizon=forecast_horizon,
            metadata_param=meta_data_params,
            evaluation_period_param=evaluation_period_param,
            model_components_param=model_components
        )
    )

    # Gets cross-validation results
    grid_search = result.grid_search
    cv_results = summarize_grid_search_results(
        grid_search=grid_search,
        decimals=2,
        cv_report_metrics=None)
    cv_results = cv_results.transpose()
    cv_results = pd.DataFrame(cv_results)
    cv_results.columns = ["err_value"]
    cv_results["err_name"] = cv_results.index
    cv_results = cv_results.reset_index(drop=True)
    cv_results = cv_results[["err_name", "err_value"]]

    print(f"\n cv_results: \n {cv_results}")

    return result

# %%
# Loads and prepares data
# -----------------------
# The data is loaded and some information about the input data is printed.
# We use the number of daily rented bikes in Washington DC over time.
# The data is augmented with weather data (precipitation, min/max daily temperature).
data = prepare_bikesharing_data()

# %%
# Fits model to daily data
# ------------------------
# In this step we fit a silverkite model to the data which uses weather regressors,
# holidays, auto-regression etc.
df = data["train_df"]
time_col = "ts"
value_col = "count"
forecast_horizon = 7

result = fit_forecast(
    df=df,
    time_col=time_col,
    value_col=value_col)
trained_estimator = result.model[-1]
# Checks model coefficients and p-values
print("\n Model Summary:")
print(trained_estimator.summary())


# %%
# Grouping of variables
# ---------------------
# Regex expressions are used to group variables in the breakdown plot.
# Each group is given in one key of this dictionary.
# The grouping is done using variable names and for each group multiple regex are given.
# For each group, variables that satisfy EITHER regex are chosen.
# Note that this grouping assumes that regressor variables start with "regressor_".
# Also note that the order of this grouping matters (Python treats the dictionary as ordered in 3.6+).
# That means the variables chosen using regex in top groups will not be picked up again.
# If some variables do not satisfy any of the groupings, they will be grouped into "OTHER".
# The following breakdown dictionary should work for many use cases.
# However, the users can customize it as needed.

grouping_regex_patterns_dict = {
    "regressors": "regressor_.*",  # regressor effects
    "AR": ".*lag",  # autoregression component
    "events": ".*events_.*",  # events and holidays
    "seasonality": ".*quarter.*|.*month.*|.*C\(dow.*|.*C\(dow_hr.*|sin.*|cos.*|.*doq.*|.*dom.*|.*str_dow.*|.*is_weekend.*|.*tow_weekly.*",  # seasonality
    "trend": "ct1|ct2|ct_sqrt|ct3|ct_root3|.*changepoint.*",  # long term trend (includes changepoints)
}

# %%
# Creates forecast breakdown
# --------------------------
# This is generated for observed data plus the prediction data (available in ``df``).
# Each component is centered around zero and the sum of all components is equal to forecast.

breakdown_result = trained_estimator.forecast_breakdown(
    grouping_regex_patterns_dict=grouping_regex_patterns_dict,
    center_components=True,
    plt_title="forecast breakdowns")
forecast_breakdown_df = breakdown_result["breakdown_df_with_index_col"]
forecast_components_fig = breakdown_result["breakdown_fig"]
plotly.io.show(forecast_components_fig)

# %%
# Standardization of the components
# ---------------------------------
# Next we provide a more "standardized" view of the breakdown.
# This is achieved by dividing all components by observed absolute value of the metric.
# By doing so, intercept should be mapped to 1 and the y-axis changes can be viewed
# relative to the average magnitude of the series.
# The sum of all components at each time point will be equal to "forecast / obs_abs_mean".

column_grouping_result = breakdown_result["column_grouping_result"]
component_cols = list(grouping_regex_patterns_dict.keys())
forecast_breakdown_stdzd_df = forecast_breakdown_df.copy()
obs_abs_mean = abs(df[value_col]).mean()
for col in component_cols + ["Intercept", "OTHER"]:
    if col in forecast_breakdown_stdzd_df.columns:
        forecast_breakdown_stdzd_df[col] /= obs_abs_mean
forecast_breakdown_stdzd_fig = plot_multivariate(
    df=forecast_breakdown_stdzd_df,
    x_col=time_col,
    title="forecast breakdowns divided by mean of abs value of response",
    ylabel="component")
forecast_breakdown_stdzd_fig.update_layout(yaxis_range=[-1.1, 1.1])
plotly.io.show(forecast_breakdown_stdzd_fig)

# %%
# Breaking down the predictions
# -----------------------------
# Next we perform a prediction and generate a breakdown plot for that prediction.
test_df = data["test_df"].reset_index()
test_df[value_col] = None
print(f"\n test_df: \n {test_df}")
pred_df = trained_estimator.predict(test_df)
forecast_x_mat = trained_estimator.forecast_x_mat
# Generate the breakdown plot
breakdown_result = trained_estimator.forecast_breakdown(
    grouping_regex_patterns_dict=grouping_regex_patterns_dict,
    forecast_x_mat=forecast_x_mat,
    time_values=pred_df[time_col])

breakdown_fig = breakdown_result["breakdown_fig"]
plotly.io.show(breakdown_fig)


# %%
# Demonstrating a scenario-based breakdown
# ----------------------------------------
# We artificially inject a "bad weather" day into test data on the second day of prediction.
# This is done to observe if the breakdown plot captures a decrease in the collective regressors' effect.
# The impact of the change in the regressor values can be clearly seen in the updated breakdown.

# Altering the test data.
# We alter the normal weather conditions on the second day to heavy precipitation and low temperature.
test_df["regressor_bin_pn"] = [0, 1, 0, 0, 0, 0, 0]
test_df["regressor_bin_heavy_pn"] = [0, 1, 0, 0, 0, 0, 0]
test_df["regressor_tmin"] = [15, 0, 15, 15,  15, 15, 15]
print(f"altered test_df: \n {test_df}")

# Gets predictions and the design matrix used during predictions.
pred_df = trained_estimator.predict(test_df.reset_index())
forecast_x_mat = trained_estimator.forecast_x_mat

# Generates the breakdown plot.
breakdown_result = trained_estimator.forecast_breakdown(
    grouping_regex_patterns_dict=grouping_regex_patterns_dict,
    forecast_x_mat=forecast_x_mat,
    time_values=pred_df[time_col])
breakdown_fig = breakdown_result["breakdown_fig"]
plotly.io.show(breakdown_fig)
