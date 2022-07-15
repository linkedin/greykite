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
"""Exploratory Data Analysis plots."""

import base64
from io import BytesIO

import matplotlib.pyplot as plt
from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector
from greykite.algo.common.holiday_inferrer import HolidayInferrer
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.enums import SeasonalityEnum
from greykite.common.time_properties import min_gap_in_seconds
from greykite.common.time_properties_forecast import get_simple_time_frequency_from_period
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from greykite.framework.input.univariate_time_series import UnivariateTimeSeries


def get_exploratory_plots(
        df,
        time_col,
        value_col,
        freq=None,
        anomaly_info=None,
        output_path=None):
    """Computes multiple exploratory data analysis (EDA) plots to visualize the
    metric in ``value_col``and aid in modeling. The EDA plots are written in
    an `html` file at ``output_path``.

    For details on how to interpret these EDA plots, check the tutorials.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Input timeseries. A data frame which includes the timestamp column
        as well as the value column.
    time_col : `str`
        The column name in ``df`` representing time for the time series data.
        The time column can be anything that can be parsed by pandas DatetimeIndex.
    value_col: `str`
        The column name which has the value of interest to be forecasted.
    freq : `str` or None, default None
        Timeseries frequency, DateOffset alias, If None automatically inferred.
    anomaly_info : `dict` or `list` [`dict`] or None, default None
        Anomaly adjustment info. Anomalies in ``df``
        are corrected before any plotting is done.
    output_path : `str` or None, default None
        Path where the `html` file is written.
        If None, it is set to "EDA_{value_col}.html".

    Returns
    -------
    eda.html : `html` file
        An html file containing the EDA plots is written at ``output_path``.
    """
    # General styles
    style = """
    <style>
        caption {
            text-align: left;
            margin-top: 5px;
            margin-bottom: 0px;
            font-size: 120%;
            padding: 5px;
            font-weight: bold;
        }
        td, th {
            border: 1px solid black;
            padding: 3px;
            text-align: left;
        }
        div {
            white-space: pre-wrap;
        }
    </style>
    """

    # Generates the HTML
    html_header = f"Exploratory plots for {value_col}"
    html_str = f"<!DOCTYPE html><html><h1>{html_header}</h1><body>"
    html_str += style
    html_str += "<h3>Please see the " \
                "<a href='https://linkedin.github.io/greykite/docs/0.4.0/html/gallery/tutorials/0100_forecast_tutorial.html#sphx-glr-gallery-tutorials-0100-forecast-tutorial-py'>tutorials</a> to learn how to interpret these plots.</h3>"  # noqa: E501
    html_str += "<h4>Most of these plots have multiple overlays and traces. Feel free to toggle these on and off by " \
                "clicking the legends to the right of the plots.</h4>"

    ts = UnivariateTimeSeries()
    ts.load_data(
        df=df,
        time_col=time_col,
        value_col=value_col,
        freq=freq,
        anomaly_info=anomaly_info
    )
    period = min_gap_in_seconds(df=ts.df, time_col=TIME_COL)
    simple_freq = get_simple_time_frequency_from_period(period)
    valid_seas = simple_freq.value.valid_seas

    # Metric plot
    html_str += "We first plot the raw timeseries. If 'anomaly_info' is provided, the anomalous data " \
                "is removed before plotting. Be careful of anomalies and missing values, as these can lead " \
                "to sharp drop(s) in the plots."
    fig = ts.plot(title="Raw metric value")
    html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Changepoints
    html_str += "<h2>Changepoints</h2>"
    html_str += "Let's look at few changepoint plots to identify significant systemic changes in the metric value. Please see " \
                "<a href='https://linkedin.github.io/greykite/docs/0.4.0/html/gallery/quickstart/0200_changepoint_detection.html#sphx-glr-gallery-quickstart-0200-changepoint-detection-py'>Changepoint tutorial</a> to learn more."  # noqa: E501
    model = ChangepointDetector()
    res = model.find_trend_changepoints(
        df=ts.df,
        time_col=TIME_COL,
        value_col=VALUE_COL
    )
    fig = model.plot(plot=False)
    fig.update_layout(title_text="changepoints - default configuration", title_x=0.5)
    html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Custom changepoints with less regularization
    res = model.find_trend_changepoints(
        df=ts.df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        yearly_seasonality_order=15,
        regularization_strength=0.4,
        resample_freq="7D",
        potential_changepoint_n=25,
        no_changepoint_distance_from_end="60D"
    )
    fig = model.plot(plot=False)
    fig.update_layout(title_text="changepoints - less regularization", title_x=0.5)
    html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Holidays
    html_str += "<h2>Holiday Effects</h2>"
    html_str += "Now let's look at how different holidays affect the metric value."
    model = HolidayInferrer()
    res = model.infer_holidays(
        df=ts.df,
        countries=("US",),
        pre_search_days=2,
        post_search_days=2,
        baseline_offsets=(-7, 7),
        plot=True
    )
    fig = res["fig"]
    html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Trend
    html_str += "<h2>Trend</h2>"
    html_str += "If trend exists, expect to see a gentle upward / downward slope in the plot. "
    fig = ts.plot_quantiles_and_overlays(
        groupby_time_feature="year_woy",
        show_mean=True,
        show_quantiles=False,
        show_overlays=False,
        overlay_label_time_feature="year",
        overlay_style={"line": {"width": 1}, "opacity": 0.5},
        center_values=False,
        ylabel=ts.original_value_col
    )
    html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Seasonalities
    if valid_seas:
        html_str += "<h2>Seasonalities</h2>"
        html_str += "To assess seasonal patterns we"
        html_str += "<ol>" \
                    "<li> Start with the longest seasonal cycles to see the big picture, then proceed to shorter " \
                    "cycles e.g. yearly -> quarterly -> monthly -> weekly -> daily." \
                    "<li> First check for seasonal effect over the entire timeseries (main effect)." \
                    "<li> If the quantiles are large in the overlay plots we check for interaction effects." \
                    "</ol>"
        html_str += "Seasonality overlay plots are centered to remove the effect of trend and longer seasonal cycles. " \
                    "This helps in isolating the effect against the selected groupby feature. Please see " \
                    "<a href='https://linkedin.github.io/greykite/docs/0.4.0/html/gallery/quickstart/0300_seasonality.html#sphx-glr-gallery-quickstart-0300-seasonality-py'>Seasonality tutorial</a> to learn more."  # noqa: E501
        html_str += "<h4>Note that partial (incomplete) seasonal periods can throw off the mean and should be ignored.</h4>"

    # Yearly Seasonality
    if SeasonalityEnum.YEARLY_SEASONALITY.name in valid_seas:
        html_str += "<h3>Yearly Seasonality (main effect)</h3>"
        html_str += "To check for overall yearly seasonality, we group by day of year (`doy`). Different years are overlaid " \
                    "to provide a sense of consistency in the seasonal pattern between years."
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="doy",
            show_mean=True,
            show_quantiles=True,
            show_overlays=True,
            overlay_label_time_feature="year",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            center_values=True,
            xlabel="day of year",
            ylabel=ts.original_value_col,
            title="yearly seasonality for each year (centered)",
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

        html_str += "<h3>Yearly Seasonality (interaction effect)</h3>"
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="month_dom",
            show_mean=True,
            show_quantiles=True,
            show_overlays=True,
            overlay_label_time_feature="year",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            center_values=True,
            xlabel="month_day of month",
            ylabel=ts.original_value_col,
            title="yearly and monthly seasonality for each year",
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="woy_dow",  # week of year and day of week
            show_mean=True,
            show_quantiles=True,
            show_overlays=True,
            overlay_label_time_feature="year",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            center_values=True,
            xlabel="week of year_day of week",
            ylabel=ts.original_value_col,
            title="yearly and weekly seasonality for each year",
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Quarterly Seasonality
    if SeasonalityEnum.QUARTERLY_SEASONALITY.name in valid_seas:
        html_str += "<h3>Quarterly Seasonality (main effect)</h3>"
        html_str += "To check for overall quarterly seasonality, we group by day of quarter (`doq`). " \
                    "Quarterly pattern for up to 20 randomly selected quarters are shown. In the legend, each overlay " \
                    "is labeled by the first date in the sliding window."
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="doq",
            show_mean=True,
            show_quantiles=True,
            show_overlays=20,  # randomly selects up to 20 overlays
            overlay_label_time_feature="quarter_start",
            center_values=True,
            xlabel="day of quarter",
            ylabel=ts.original_value_col,
            title="quarterly seasonality with overlays"
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

        html_str += "<h3>Quarterly Seasonality (interaction effect)</h3>"
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="doq",
            show_mean=True,
            show_quantiles=False,
            show_overlays=True,
            center_values=True,
            overlay_label_time_feature="year",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            xlabel="day of quarter",
            ylabel=ts.original_value_col,
            title="quarterly seasonality by year"
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Monthly Seasonality
    if SeasonalityEnum.MONTHLY_SEASONALITY.name in valid_seas:
        html_str += "<h3>Monthly Seasonality (main effect)</h3>"
        html_str += "To check overall monthly seasonality, we group by day of month (`dom`). " \
                    "Monthly pattern for up to 20 randomly selected months are shown. In the legend, each overlay " \
                    "is labeled by the first date in the sliding window."
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="dom",
            show_mean=True,
            show_quantiles=True,
            show_overlays=20,
            overlay_label_time_feature="year_month",
            center_values=True,
            xlabel="day of month",
            ylabel=ts.original_value_col,
            title="monthly seasonality with overlays"
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

        html_str += "<h3>Monthly Seasonality (interaction effect)</h3>"
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="dom",
            show_mean=True,
            show_quantiles=False,
            show_overlays=True,
            center_values=True,
            overlay_label_time_feature="year",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            xlabel="day of month",
            ylabel=ts.original_value_col,
            title="monthly seasonality by year"
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Weekly Seasonality
    if SeasonalityEnum.WEEKLY_SEASONALITY.name in valid_seas:
        html_str += "<h3>Weekly Seasonality (main effect)</h3>"
        html_str += "To check overall weekly seasonality, we group by day of week (`str_dow`). " \
                    "Weekly pattern for up to 20 randomly selected weeks are shown. In the legend, each overlay " \
                    "is labeled by the first date in the sliding window."
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="str_dow",
            show_mean=True,
            show_quantiles=True,
            show_overlays=20,  # randomly selects up to 20 overlays
            overlay_label_time_feature="year_woy",
            center_values=True,
            xlabel="day of week",
            ylabel=ts.original_value_col,
            title="weekly seasonality with overlays"
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

        html_str += "<h3>Weekly Seasonality (interaction effect)</h3>"
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="str_dow",
            show_mean=True,
            show_quantiles=False,
            show_overlays=True,
            center_values=True,
            overlay_label_time_feature="year",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            xlabel="day of week",
            ylabel=ts.original_value_col,
            title="weekly seasonality by year",
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="str_dow",
            show_mean=True,
            show_quantiles=False,
            show_overlays=True,
            center_values=True,
            overlay_label_time_feature="month",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            xlabel="day of week",
            ylabel=ts.original_value_col,
            title="weekly seasonality by month",
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Daily Seasonality
    if SeasonalityEnum.DAILY_SEASONALITY.name in valid_seas:
        html_str += "<h3>Daily Seasonality (main effect)</h3>"
        html_str += "To check overall daily seasonality, we group by time of day (`tod`). " \
                    "Daily pattern for up to 20 randomly selected days are shown."
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="tod",
            show_mean=True,
            show_quantiles=True,
            show_overlays=20,
            overlay_label_time_feature="year_woy_dow",
            center_values=True,
            xlabel="time of day",
            ylabel=ts.original_value_col,
            title="daily seasonality with overlays"
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

        html_str += "<h3>Daily Seasonality (interaction effect)</h3>"
        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="tod",
            show_mean=True,
            show_quantiles=False,
            show_overlays=True,
            center_values=True,
            overlay_label_time_feature="year",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            xlabel="time of day",
            ylabel=ts.original_value_col,
            title="daily seasonality by year",
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="tod",
            show_mean=True,
            show_quantiles=False,
            show_overlays=True,
            center_values=True,
            overlay_label_time_feature="month",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            xlabel="time of day",
            ylabel=ts.original_value_col,
            title="daily seasonality by month",
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

        fig = ts.plot_quantiles_and_overlays(
            groupby_time_feature="tod",
            show_mean=True,
            show_quantiles=False,
            show_overlays=True,
            center_values=True,
            overlay_label_time_feature="week",
            overlay_style={"line": {"width": 1}, "opacity": 0.5},
            xlabel="time of day",
            ylabel=ts.original_value_col,
            title="daily seasonality by week",
        )
        html_str += fig.to_html(full_html=False, include_plotlyjs=True)

    # Auto-correlation
    html_str += "<h2>Auto-correlation</h2>"
    html_str += "Partial auto-correlation plot can be a good guide to choose appropriate auto-regression lag terms. " \
                "Use large spikes to model individual lag terms (`lag_dict`). " \
                "Smaller but significant spikes can be grouped under `agg_lag_dict`."
    ts.df[VALUE_COL].fillna(ts.df[VALUE_COL].median(), inplace=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    plot_pacf(ts.df[VALUE_COL].values, lags=40, ax=ax[0])
    plot_acf(ts.df[VALUE_COL].values, lags=40, ax=ax[1])
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html_str += '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    # Writes the HTML
    if not output_path:
        output_path = f"EDA_{value_col}.html"
    with open(output_path, "w+") as f:
        f.write(html_str)
