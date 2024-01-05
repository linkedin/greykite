"""
Simple Anomaly Detection
========================

You can create and evaluate an anomaly detection model with just a few lines of code.

Provide your timeseries as a pandas dataframe with timestamp and value.
Optionally, you can also provide the anomaly labels as a column in the dataframe.

For example, to detect anomalies in daily sessions data, your dataframe could look like this:

.. code-block:: python

    import pandas as pd
    df = pd.DataFrame({
        "date": ["2020-01-08-00", "2020-01-09-00", "2020-01-10-00"],
        "sessions": [10231.0, 12309.0, 12104.0],
        "is_anomaly": [False, True, False]
    })

The time column can be any format recognized by `pandas.to_datetime`.

In this example, we'll load a dataset representing ``log(daily page views)``
on the Wikipedia page for Peyton Manning.
It contains values from 2007-12-10 to 2016-01-20. More dataset info
`here <https://facebook.github.io/prophet/docs/quick_start.html>`_.
"""

import warnings

import plotly
from greykite.common.data_loader import DataLoader
from greykite.detection.detector.config import ADConfig
from greykite.detection.detector.data import DetectorData
from greykite.detection.detector.greykite import GreykiteDetector
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.model_templates import ModelTemplateEnum

warnings.filterwarnings("ignore")

# Loads dataset into pandas DataFrame
dl = DataLoader()
df = dl.load_peyton_manning()

# specify dataset information
metadata = MetadataParam(
    time_col="ts",  # name of the time column ("date" in example above)
    value_col="y",  # name of the value column ("sessions" in example above)
    freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
    # Any format accepted by `pandas.date_range`
)

# %%
# Create an Anomaly Detection Model
# -------------------------------
# Similar to forecasting, you need to provide a forecast config and an
# anomaly detection config. You can choose any of the available forecast model
# templates (see :doc:`/pages/stepbystep/0100_choose_model`).

# In this example, we choose the "AUTO" model template for the forecast config,
# and the default anomaly detection config.
# The Silverkite "AUTO" model template chooses the parameter configuration
# given the input data frequency, forecast horizon and evaluation configs.

anomaly_detector = GreykiteDetector()  # Creates an instance of the Greykite anomaly detector

forecast_config = ForecastConfig(
    model_template=ModelTemplateEnum.AUTO.name,
    forecast_horizon=7,  # forecasts 7 steps ahead
    coverage=None,       # Confidence Interval will be tuned by the AD model
    metadata_param=metadata)

ad_config = ADConfig()  # Default anomaly detection config

detector = GreykiteDetector(
    forecast_config=forecast_config,
    ad_config=ad_config,
    reward=None)

# %%
# Train the Anomaly Detection Model
# ---------------------------------
# You can train the anomaly detection model by calling the ``fit`` method.
# This method takes a ``DetectorData`` object as input.
# The ``DetectorData`` object consists the time series information as a pandas dataframe.
# Optionally, you can also provide the anomaly labels as a column in the dataframe.
# The anomaly labels can also be provided as a list of boolean values.
# The anomaly labels are used to evaluate the model performance.

train_size = int(2700)
df_train = df[:train_size].reset_index(drop=True)
train_data = DetectorData(df=df_train)
detector.fit(data=train_data)

# %%
# Predict with the Anomaly Detection Model
# ---------------------------------------
# You can predict anomalies by calling the ``predict`` method.

test_data = DetectorData(df=df)
test_data = detector.predict(test_data)

# %%
# Evaluate the Anomaly Detection Model
# ------------------------------------
# The output of the anomaly detection model are stored as attributes
# of the ``GreykiteDetector`` object.
# (The interactive plots are generated by ``plotly``: **click to zoom!**)


# %%
# Training
# ^^^^^^^^
# The ``fitted_df`` attribute contains the result on the training data.
# You can plot the result by calling the ``plot`` method with ``phase="train"``.
print(detector.fitted_df)

fig = detector.plot(
    phase="train",
    title="Greykite Detector Peyton Manning - fit phase")
plotly.io.show(fig)

# %%
# Prediction
# ^^^^^^^^^^
# The ``pred_df`` attribute contains the predicted result.
# You can plot the result by calling the ``plot`` method with ``phase="predict"``.

print(detector.pred_df)

fig = detector.plot(
    phase="predict",
    title="Greykite Detector Peyton Manning - predict phase")
plotly.io.show(fig)

# %%
# Model Summary
# ^^^^^^^^^^^^^^^^^
# Model summary allows inspection of individual model terms.
# Check parameter estimates and their significance for insights
# on how the model works and what can be further improved.
# You can call the ``summary`` method to see the model summary.
summary = detector.summary()
print(summary)

# %%
# What's next?
# ------------
# If you're satisfied with the forecast performance, you're done!
#
# For a complete example of how to tune this forecast, see
# :doc:`/gallery/tutorials/0400_anomaly_detection_tutorial`.
