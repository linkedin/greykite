Auto-regression
===============
Auto-regression can be utilized to improve forecasts by capturing the remaining temporal correlation after accounting
for other features such as seasonality and growth.

Silverkite allows for specifying lags. For example if Y(t) is the time series of interest, user can easily activate
desired lags such as Y(t-1), Y(t-2) to be used in the model.

Moreover Silverkite allows for more advanced auto-regression in the model by specifying "aggregated lags" which are
predictors constructed by averaging several specified lags.

This technique makes it possible to build models with longer term auto-correlation with only modest increase of the
number of parameters. See `Hosseini et al. (2011) <https://link.springer.com/article/10.1007/s10651-010-0169-1>`_ for an early example of utilizing aggregated lags as predictors.

Silverkite automatically activates auto-regression for forecast horizons which are less than or equal to 30 days.
By default, Silverkite uses lags which are larger than the given forecast horizon periods to avoid having to do
simulations to forecast the future.

User does not need to explicitly specify auto-regression to activate the "auto"
option. However the explicit code is as follows:

.. code-block:: python

    from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam

    autoreg_dict="auto"

    model_components = ModelComponentsParam(
        growth=growth,
        seasonality=seasonality,
        events=events,
        changepoints=changepoints,
        regressors=regressors,
        autoregression=dict(autoreg_dict=autoreg_dict),  # auto-regression input is specified here
        uncertainty=uncertainty,
        custom=custom,
        hyperparameter_override=hyperparameter_override
    )



It is possible to also deactivate auto-regression by letting:


.. code-block:: python

    autoreg_dict = None



Custom Auto-regression
^^^^^^^^^^^^^^^^^^^^^^


There are three ways to specify an auto-regressive structure. Below we give the example code and discuss what each
component means in what follows.


.. code-block:: python

    # Form (1): Specifies regular lags, e.g. lags 1, 2, 7
    lag_dict = dict(orders=[1, 2, 7])
    # Specifies aggregated lags in two forms
    # Form (2): Each group of lags to be aggregated is given in a list
    orders_list = [[7, 7*2, 7*3], [7*4, 7*5, 7*6]] # Two aggregated lag list are specified
    # Form (3): Each tuple has two elements specifying the beginning and end of aggregation
    interval_list = [(1, 7), (8, 7*2)]
    agg_lag_dict = dict(orders_list=orders_list, interval_list=interval_list)
    autoreg_dict = dict(lag_dict=lag_dict, agg_lag_dict=agg_lag_dict)


The aggregated lag can be specified in two ways (forms 2 and 3) as demonstrated above.

In Form (2), each set of lags to be aggregated is passed as a list.
For example [7, 7*2, 7*3] corresponds to (Y(t-7) + Y(t-14) + Y(t-21))/3.
For daily data one can interpret this predictor as the average value of the series on the same day of week in past three weeks.

In Form (3), user can specify the starting and end point of the lags to be aggregated. For example (1, 7) corresponds
to (Y(t-1) + Y(t-2) + ... + Y(t-7)) / 7. One can interpret this predictor is the average value during the week prior to
time t.
