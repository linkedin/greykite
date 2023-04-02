Greykite: A flexible, intuitive and fast forecasting library

.. image:: https://raw.githubusercontent.com/linkedin/greykite/master/LOGO-C8.png
   :height: 300px
   :width: 450px
   :scale: 80%
   :alt: Greykite
   :align: center

Why Greykite?
-------------

The Greykite library provides flexible, intuitive and fast forecasts through its flagship algorithm, Silverkite.

Silverkite algorithm works well on most time series, and is especially adept for those with changepoints in trend or seasonality,
event/holiday effects, and temporal dependencies.
Its forecasts are interpretable and therefore useful for trusted decision-making and insights.

The Greykite library provides a framework that makes it easy to develop a good forecast model,
with exploratory data analysis, outlier/anomaly preprocessing, feature extraction and engineering, grid search,
evaluation, benchmarking, and plotting.
Other open source algorithms can be supported through Greykite’s interface to take advantage of this framework,
as listed below.

For a demo, please see our `quickstart <https://linkedin.github.io/greykite/get_started>`_.

Distinguishing Features
-----------------------

* Flexible design
    * Provides time series regressors to capture trend, seasonality, holidays,
      changepoints, and autoregression, and lets you add your own.
    * Fits the forecast using a machine learning model of your choice.
* Intuitive interface
    * Provides powerful plotting tools to explore seasonality, interactions, changepoints, etc.
    * Provides model templates (default parameters) that work well based on
      data characteristics and forecast requirements (e.g. daily long-term forecast).
    * Produces interpretable output, with model summary to examine individual regressors,
      and component plots to visually inspect the combined effect of related regressors.
* Fast training and scoring
    * Facilitates interactive prototyping, grid search, and benchmarking.
      Grid search is useful for model selection and semi-automatic forecasting of multiple metrics.
* Extensible framework
    * Exposes multiple forecast algorithms in the same interface,
      making it easy to try algorithms from different libraries and compare results.
    * The same pipeline provides preprocessing, cross-validation,
      backtest, forecast, and evaluation with any algorithm.

Algorithms currently supported within Greykite’s modeling framework:

* Silverkite (Greykite’s flagship algorithm)
* `Facebook Prophet <https://facebook.github.io/prophet/>`_
* `Auto Arima <https://alkaline-ml.com/pmdarima/>`_

Notable Components
------------------

Greykite offers components that could be used within other forecasting
libraries or even outside the forecasting context.

* ModelSummary() - R-like summaries of `scikit-learn` and `statsmodels` regression models.
* ChangepointDetector() - changepoint detection based on adaptive lasso, with visualization.
* SimpleSilverkiteForecast() - Silverkite algorithm with `forecast_simple` and `predict` methods.
* SilverkiteForecast() - low-level interface to Silverkite algorithm with `forecast` and `predict` methods.
* ReconcileAdditiveForecasts() - adjust a set of forecasts to satisfy inter-forecast additivity constraints.

Usage Examples
--------------

You can obtain forecasts with only a few lines of code:

.. code-block:: python

    from greykite.common.data_loader import DataLoader
    from greykite.framework.templates.autogen.forecast_config import ForecastConfig
    from greykite.framework.templates.autogen.forecast_config import MetadataParam
    from greykite.framework.templates.forecaster import Forecaster
    from greykite.framework.templates.model_templates import ModelTemplateEnum

    # Defines inputs
    df = DataLoader().load_bikesharing().tail(24*90)  # Input time series (pandas.DataFrame)
    config = ForecastConfig(
         metadata_param=MetadataParam(time_col="ts", value_col="count"),  # Column names in `df`
         model_template=ModelTemplateEnum.AUTO.name,  # AUTO model configuration
         forecast_horizon=24,   # Forecasts 24 steps ahead
         coverage=0.95,         # 95% prediction intervals
     )

    # Creates forecasts
    forecaster = Forecaster()
    result = forecaster.run_forecast_config(df=df, config=config)

    # Accesses results
    result.forecast     # Forecast with metrics, diagnostics
    result.backtest     # Backtest with metrics, diagnostics
    result.grid_search  # Time series CV result
    result.model        # Trained model
    result.timeseries   # Processed time series with plotting functions

For a demo, please see our `quickstart <https://linkedin.github.io/greykite/get_started>`_.

Setup and Installation
----------------------

Greykite is available on Pypi and can be installed with pip:

.. code-block::

    pip install greykite

For more installation tips, see `installation <http://linkedin.github.io/greykite/installation>`_.

Documentation
-------------

Please find our full documentation `here <http://linkedin.github.io/greykite/docs>`_.

Learn More
----------

* `Website <https://linkedin.github.io/greykite>`_
* `Paper <https://doi.org/10.1145/3534678.3539165>`_ (KDD '22 Best Paper Runner-up, Applied Data Science Track)
* `Blog post <https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library>`_

Citation
--------

Please cite Greykite in your publications if it helps your research:

.. code-block::

    @misc{reza2021greykite-github,
      author = {Reza Hosseini and
                Albert Chen and
                Kaixu Yang and
                Sayan Patra and
                Yi Su and
                Rachit Arora},
      title  = {Greykite: a flexible, intuitive and fast forecasting library},
      url    = {https://github.com/linkedin/greykite},
      year   = {2021}
    }

.. code-block::

    @inproceedings{reza2022greykite-kdd,
      author = {Hosseini, Reza and Chen, Albert and Yang, Kaixu and Patra, Sayan and Su, Yi and Al Orjany, Saad Eddin and Tang, Sishi and Ahammad, Parvez},
      title = {Greykite: Deploying Flexible Forecasting at Scale at LinkedIn},
      year = {2022},
      isbn = {9781450393850},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3534678.3539165},
      doi = {10.1145/3534678.3539165},
      booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
      pages = {3007–3017},
      numpages = {11},
      keywords = {forecasting, scalability, interpretable machine learning, time series},
      location = {Washington DC, USA},
      series = {KDD '22}
    }


License
-------

Copyright (c) LinkedIn Corporation. All rights reserved. Licensed under the
`BSD 2-Clause <https://opensource.org/licenses/BSD-2-Clause>`_ License.