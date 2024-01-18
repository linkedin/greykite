Holidays and Events
===================

Use ``model_components.events`` to specify the holidays and events.

Events indicate times of expected deviations from the usual growth and seasonality pattern.
These may be repeated (e.g. holidays), or one-time (Olympic for a city).

Repeated events are allowed to fall on a different day each year (e.g. Lunar New Year, Thanksgiving).
The model assumes the event has similar effect each time it occurs.

Holidays
--------

Silverkite provides a standard set of holidays by country (imported from ``pypi:holidays-ext``).

Each holiday is mapped to a list of calendar dates where it occurs. For modeling, you can choose to
extend the holiday window to include days before and after the holiday. A separate effect is modeled for each
offset from the actual holiday.

For example, you can choose to model offsets (-2, -1, 0, 1) to capture effects
two days before, and one day after the holiday, by setting ``holiday_pre_num_days=-2``
and ``holiday_post_num_days=1`` below.

If you are not sure which holidays to use, start with our defaults and create a forecast.
Plot forecasts against actuals, and look for large errors. If these happen on holidays,
include relevant countries in ``holiday_lookup_countries`` list.

Silverkite Holidays
^^^^^^^^^^^^^^^^^^^

Options (defaults shown for ``SILVERKITE`` template):

.. code-block:: none

    events : `dict` [`str`, `any`] or None
        Holiday/events configuration dictionary with the following optional keys:

        holiday_lookup_countries : `list` [`str`] or "auto" or None or a list of such values for grid search, optional, default "auto"
            The countries that contain the holidays you intend to model
            (``holidays_to_model_separately``).

            * If "auto", uses a default list of countries
              that contain the default ``holidays_to_model_separately``.
              See `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO`.
            * If a list, must be a list of country names.
            * If None or an empty list, no holidays are modeled.

        holidays_to_model_separately : `list` [`str`] or "auto" or `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES` or None or a list of such values for grid search, optional, default "auto"
            Which holidays to include in the model.
            The model creates a separate key, value for each item in ``holidays_to_model_separately``.
            The other holidays in the countries are grouped together as a single effect.

            * If "auto", uses a default list of important holidays. See
              `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO`.
            * If `~greykite.algo.forecast.silverkite.constants.silverkite_holiday.SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES`,
              uses all available holidays in ``holiday_lookup_countries``. This can often
              create a model that has too many parameters, and should typically be avoided.
            * If a list, must be a list of holiday names.
            * If None or an empty list, all holidays in ``holiday_lookup_countries`` are grouped together
              as a single effect.

            Use ``holiday_lookup_countries`` to provide a list of countries where these holiday occur.
        holiday_pre_num_days: `int` or a list of such values for grid search, default 2
            model holiday effects for pre_num days before the holiday.
            The unit is days, not periods. It does not depend on input data frequency.
        holiday_post_num_days: `int` or a list of such values for grid search, default 2
            model holiday effects for post_num days after the holiday.
            The unit is days, not periods. It does not depend on input data frequency.
        holiday_pre_post_num_dict : `dict` [`str`, (`int`, `int`)] or None, default None
            Overrides ``pre_num`` and ``post_num`` for each holiday in
            ``holidays_to_model_separately``.
            For example, if ``holidays_to_model_separately`` contains "Thanksgiving" and "Labor Day",
            this parameter can be set to ``{"Thanksgiving": [1, 3], "Labor Day": [1, 2]}``,
            denoting that the "Thanksgiving" ``pre_num`` is 1 and ``post_num`` is 3, and "Labor Day"
            ``pre_num`` is 1 and ``post_num`` is 2.
            Holidays not specified use the default given by ``pre_num`` and ``post_num``.
        daily_event_df_dict : `dict` [`str`, `pandas.DataFrame`] or None, default None
            A dictionary of data frames, each representing events data for the corresponding key.
            Specifies additional events to include besides the holidays specified above. The format
            is the same as in `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`.
            The DataFrame has two columns:

                - The first column contains event dates. Must be in a format
                  recognized by `pandas.to_datetime`. Must be at daily
                  frequency for proper join. It is joined against the time
                  in ``df``, converted to a day:
                  ``pd.to_datetime(pd.DatetimeIndex(df[time_col]).date)``.
                - the second column contains the event label for each date

            The column order is important; column names are ignored.
            The event dates must span their occurrences in both the training
            and future prediction period.

            During modeling, each key in the dictionary is mapped to a categorical variable
            named ``f"{EVENT_PREFIX}_{key}"``, whose value at each timestamp is specified
            by the corresponding DataFrame.

            For example, to manually specify a yearly event on September 1
            during a training/forecast period that spans 2020-2022::

                daily_event_df_dict = {
                    "custom_event": pd.DataFrame({
                        "date": ["2020-09-01", "2021-09-01", "2022-09-01"],
                        "label": ["is_event", "is_event", "is_event"]
                    })
                }

            It's possible to specify multiple events in the same df. Two events,
            ``"sep"`` and ``"oct"`` are specified below for 2020-2021::

                daily_event_df_dict = {
                    "custom_event": pd.DataFrame({
                        "date": ["2020-09-01", "2020-10-01", "2021-09-01", "2021-10-01"],
                        "event_name": ["sep", "oct", "sep", "oct"]
                    })
                }

            Use multiple keys if two events may fall on the same date. These events
            must be in separate DataFrames::

                daily_event_df_dict = {
                    "fixed_event": pd.DataFrame({
                        "date": ["2020-09-01", "2021-09-01", "2022-09-01"],
                        "event_name": ["is_event", "is_event", "is_event"]
                    }),
                    "moving_event": pd.DataFrame({
                        "date": ["2020-09-01", "2021-08-28", "2022-09-03"],
                        "event_name": ["is_event", "is_event", "is_event"]
                    }),
                }

            The multiple event specification can be used even if events never overlap. An
            equivalent specification of the second example::

                daily_event_df_dict = {
                    "sep": pd.DataFrame({
                        "date": ["2020-09-01", "2021-09-01"],
                        "event_name": "is_event"
                    }),
                    "oct": pd.DataFrame({
                        "date": ["2020-10-01", "2021-10-01"],
                        "event_name": "is_event"
                    }),
                }

            Note: All these events are automatically added to the model. There is no need
            to specify them in ``extra_pred_cols`` as you would for
            `~greykite.algo.forecast.silverkite.SilverkiteForecast.forecast`.

            Note: Do not use `~greykite.common.constants.EVENT_DEFAULT`
            in the second column. This is reserved to indicate dates that do not
            correspond to an event.


Examples:

.. code-block:: python

    # silverkite defaults
    events = dict(
        holidays_to_model_separately="auto",
        holiday_lookup_countries="auto",
        holiday_pre_num_days=2,
        holiday_post_num_days=2,
        holiday_pre_post_num_dict=None,
        daily_event_df_dict=None
    )

    # Two letter country code is also accepted.
    # Sets holiday_pre_num_days=holiday_post_num_days=0 to only capture effects on the holiday itself.
    # Uses holiday_pre_post_num_dict to customize pre_num and post_num for New Year's Day
    events = dict(
        holidays_to_model_separately = [
            "New Year's Day",
            "Thanksgiving"],
        holiday_lookup_countries = [  # containing countries to lookup holiday dates
            "US",
            "CA"],
        holiday_pre_num_days=0,
        holiday_post_num_days=0,
        holiday_pre_post_num_dict={"New Year's Day": (3, 1)}
    )

    # See the docstring above for examples of ``daily_event_df_dict``.
    # You can use it to add your own events or specify a
    # fully custom holiday configuration.

    # Grid search is possible
    events = dict(
        holidays_to_model_separately = [
            "auto",
            ["New Year's Day", "Thanksgiving"],
            None],
        holiday_lookup_countries = ["auto"],
        holiday_pre_num_days=2,
        holiday_post_num_days=2
    )

If you are not sure which holidays to use, start with our defaults and create a forecast.
Plot forecasts against actuals, and look for large errors. If these happen on holidays,
include them in the model.

To customize this, you will want to see the available holidays.

* How to check the available ``holiday_lookup_countries``:

.. code-block:: python

    from greykite.common.features.timeseries_features import get_available_holiday_lookup_countries

    # See all available countries
    get_available_holiday_lookup_countries()

    # Filter the full list to your countries of interest
    get_available_holiday_lookup_countries(["US", "IN", "EuropeanCentralBank"])

    # Full list for `holidays-ext` 0.0.7, `holidays` 0.13
    >>> get_available_holiday_lookup_countries()
    ['ABW', 'AE', 'AGO', 'AO', 'AR', 'ARE', 'ARG', 'AT', 'AU', 'AUS',
     'AUT', 'AW', 'AZ', 'AZE', 'Angola', 'Argentina', 'Aruba', 'Australia',
     'Austria', 'Azerbaijan', 'BD', 'BDI', 'BE', 'BEL', 'BG', 'BGD', 'BI',
     'BLG', 'BLR', 'BR', 'BRA', 'BW', 'BWA', 'BY', 'Bangladesh', 'Belarus',
     'Belgium', 'Botswana', 'Brazil', 'Bulgaria', 'Burundi', 'CA', 'CAN',
     'CH', 'CHE', 'CHL', 'CHN', 'CL', 'CN', 'CO', 'COL', 'CUW', 'CW', 'CZ',
     'CZE', 'Canada', 'Chile', 'China', 'Colombia', 'Croatia', 'Curacao',
     'Czechia', 'DE', 'DEU', 'DJ', 'DJI', 'DK', 'DNK', 'DO', 'DOM',
     'Denmark', 'Djibouti', 'DominicanRepublic', 'ECB', 'EE', 'EG', 'EGY',
     'ES', 'ESP', 'EST', 'ET', 'ETH', 'Egypt', 'England', 'Estonia',
     'Ethiopia', 'EuropeanCentralBank', 'FI', 'FIN', 'FR', 'FRA', 'Finland',
     'France', 'GB', 'GBR', 'GE', 'GEO', 'GR', 'GRC', 'Georgia', 'Germany',
     'Greece', 'HK', 'HKG', 'HN', 'HND', 'HR', 'HRV', 'HU', 'HUN',
     'HolidaySum', 'Honduras', 'HongKong', 'Hungary', 'ID', 'IE', 'IL',
     'IM', 'IN', 'IND', 'IRL', 'IS', 'ISL', 'ISR', 'IT', 'ITA', 'Iceland',
     'India', 'Indonesia', 'Ireland', 'IsleOfMan', 'Israel', 'Italy', 'JAM',
     'JM', 'JP', 'JPN', 'Jamaica', 'Japan', 'KAZ', 'KE', 'KEN', 'KOR', 'KR',
     'KZ', 'Kazakhstan', 'Kenya', 'Korea', 'LS', 'LSO', 'LT', 'LTU', 'LU',
     'LUX', 'LV', 'LVA', 'Latvia', 'Lesotho', 'Lithuania', 'Luxembourg',
     'MA', 'MEX', 'MK', 'MKD', 'MOR', 'MOZ', 'MW', 'MWI', 'MX', 'MY', 'MYS',
     'MZ', 'Malawi', 'Malaysia', 'Mexico', 'Morocco', 'Mozambique', 'NA',
     'NAM', 'NG', 'NGA', 'NI', 'NIC', 'NL', 'NLD', 'NO', 'NOR', 'NZ', 'NZL',
     'Namibia', 'Netherlands', 'NewZealand', 'Nicaragua', 'Nigeria',
     'NorthMacedonia', 'NorthernIreland', 'Norway', 'PE', 'PER', 'PH', 'PK',
     'PL', 'POL', 'PRT', 'PRTE', 'PRY', 'PT', 'PTE', 'PY', 'Pakistan',
     'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'PortugalExt',
     'RO', 'ROU', 'RS', 'RU', 'RUS', 'Romania', 'Russia', 'SA', 'SAU', 'SE',
     'SG', 'SGP', 'SI', 'SK', 'SRB', 'SVK', 'SVN', 'SWE', 'SZ', 'SZW',
     'SaudiArabia', 'Scotland', 'Serbia', 'Singapore', 'Slovakia',
     'Slovenia', 'SouthAfrica', 'Spain', 'Swaziland', 'Sweden',
     'Switzerland', 'TAR', 'TH', 'TN', 'TR', 'TUN', 'TUR', 'TW', 'TWN',
     'Taiwan', 'Thailand', 'Tunisia', 'Turkey', 'UA', 'UK', 'UKR', 'URY',
     'US', 'USA', 'UY', 'UZ', 'UZB', 'Ukraine', 'UnitedArabEmirates',
     'UnitedKingdom', 'UnitedStates', 'Uruguay', 'Uzbekistan', 'VE', 'VEN',
     'VN', 'VNM', 'Venezuela', 'Vietnam', 'Wales', 'ZA', 'ZAF', 'ZM', 'ZMB',
     'ZW', 'ZWE', 'Zambia', 'Zimbabwe']

* To check the available ``holidays_to_model_separately`` in those countries,
  run ``get_available_holidays_across_countries``:

.. code-block:: python

    from greykite.common.features.timeseries_features import get_available_holidays_across_countries

    # Select your countries
    holiday_lookup_countries = ["US", "IN", "EuropeanCentralBank"]
    # List the holidays
    get_available_holidays_across_countries(
        countries=holiday_lookup_countries,
        year_start=2017,
        year_end=2025)

.. note::

  While holidays are specified at a daily level, you can use interactions with seasonality to capture
  sub-daily holiday effects. For more information, see :doc:`/pages/model_components/0600_custom`.

Holiday Indicators and Neighboring Effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. When holidays are present in the model, we allow for using holiday indicators:

* "is_event": an indicator column which is 1 when the timestamp is either the exact holiday dates or its
  adjacent days.

* "is_event_exact": an indicator of whether the timestamp is exactly on the holiday date.

* "is_event_adjacent": an indicator of whether the timestamp is adjacent to a holiday if ``holiday_pre_num_days``
  or ``holiday_post_num_days`` is not 0.

You may include the interactions between such indicators and other features in ``extra_pred_cols`` like
``extra_pred_col = ["is_event:y_lag1"]``.
See more at :doc:`/pages/model_components/0600_custom`.

Or you may use it as a conditional column in the uncertainty model ``conditional_cols = ["dow", "is_event"]``.

2. Sometimes you may have a weekly time series or the response is daily rolling sum.
In such cases, the whole week, or the whole rolling window is impacted by a holiday within it.
We allow for modeling such holiday neighboring effect by specifying ``daily_event_neighbor_impact`` in ``events``.
For example, you may use ``daily_event_neighbor_impact = 6`` to model rolling 7-day sum holiday effect
in a daily time series. Or you may use
``daily_event_neighbor_impact = lambda x: [x - timedelta(days=x.isocalendar()[2] - 1) + timedelta(days=i) for i in range(7)]``
to model a holiday effect in weekly time series.

Note that this feature works as adding extra dates with the same event name to the holiday model,
therefore the number of events does not increase.

3. There are also cases where you need additional events that are shifted based on existing events.
For example, if we model the week-over-week changes as response, the week after a holiday has a counter effect.
We support an easy way of adding such events using ``daily_event_shifted_effect`` parameter in ``events``.
For example, if we have an event called "Christmas Day", ``daily_event_shifted_effect=["7D"]`` will add a new
event called "Christmas Day_7D_after" which is 7 days after the Christmas Day.

Auto Holiday and Holiday Grouper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Silverkite models support automatically grouped holiday features.
It utilizes the `~greykite.algo.common.holiday_grouper.HolidayGrouper` method to group
holidays based on their estimated impact inferred from the training data. This method avoids creating too many
parameters for each holiday while making sure that holidays
that are different enough will be modeled separately. For more details, see Holiday Inferer and
Holiday Grouper in :doc:`/gallery/quickstart/01_exploration/0200_auto_configuration_tools`.

It's easy to use auto holiday in model components.
In the ``events`` dictionary, specify ``auto_holiday=True``, and the model will automatically pull
holidays from ``holiday_lookup_countries``. Additional events can be passed in through ``daily_event_df_dict``,
which get combined with holidays from ``holiday_lookup_countries`` and used as sources of holidays for grouping. Their
neighboring days will be pulled based on ``holiday_pre_num_days``, ``holiday_post_num_days`` and
``holiday_pre_post_num_dict`` if provided. Each holiday and their neighboring days (e.g. Christmas, Christmas +1 Day,
Christmas +2 Days) will have their holiday impact estimated. The holiday grouper checks their individual effects from
the training data and generates holiday groups. Each holiday and its neighboring day in ``holidays_to_model_separately``
will be modeled separately regardless of whether ``auto_holiday`` is on. Generally, it is recommended to leave
``holidays_to_model_separately`` as ``None`` here unless there is prior knowledge that some holidays or events have
different behaviors as other ones. The ``auto_holiday_params`` can take in additional parameters used by holiday grouper.

.. code-block:: python

    events = dict(
        auto_holiday=True,  # Turns on auto holiday config.
        holidays_to_model_separately=None,  # No holiday is modeled separately.
        holiday_lookup_countries="auto",  # A default list of countries to search.
        holiday_pre_num_days=2,  # Considers 2 days before each holiday.
        holiday_post_num_days=2,  # Considers 2 days after each holiday.
        holiday_pre_post_num_dict=None,  # Specifies if any holiday needs a different count of neighboring days.
        daily_event_df_dict=None,  # Additional events added to holidays from `holiday_lookup_countries`.
        auto_holiday_params=None  # Additional parameters for holiday groupers.
    )

In the example here, by default there will be 5 feature groups generated based on holiday grouping results:
``"events_holiday_group_0"``, ``"events_holiday_group_1"``, ``"events_holiday_group_2"``, ``"events_holiday_group_3"``,
``"events_holiday_group_4"``.

By contrast, if ``auto_holiday=False``, there will also be  5 feature groups generated:
``"events_Other"``, ``"events_Other-1"``, ``"events_Other-2"``, ``"events_Other+1"``, ``"events_Other+2"``.
All holidays will be modeled as one whole group named ``"events_Other"``. Their neighboring days will be modeled in
groups based on their relations to the holidays.

To extend from default settings, additional parameters for holiday groupers can be passed in through
``auto_holiday_params``. The following code lists out all parameters that can be specified and tuned with their
current defaults. Please refer to Holiday Grouper in
:doc:`/gallery/quickstart/01_exploration/0200_auto_configuration_tools` for more information on how it works.

.. code-block:: python

    # The `auto_holiday_params` to pass in for events.
    auto_holiday_params = dict(
        df=None,  # Time series to infer holiday impact.
        time_col=None,  # Time column in `df`.
        value_col=None,  # Value column in `df`.
        holiday_df=None,  # User specified holidays, will replace holidays from `holiday_lookup_countries` and `daily_event_df_dict`.
        holiday_date_col=None,  # Holiday date column in `holiday_df`.
        holiday_name_col=None,  # Holiday name column in `holiday_df`.
        get_suffix_func="wd_we",  # Extended feature group added as suffix, please see `HolidayGrouper`.
        baseline_offsets=[-7, 7],  # The baseline is the average of -7/+7 observations.
        use_relative_score=True,  # Whether to use relative or absolute score when estimating impact.
        min_n_days=1,  # Minimal number of occurrences of an event to be included in grouping.
        min_abs_avg_score=0.03,  # Minimal average score of an event to be kept in consideration.
        clustering_method="kmeans",  # Clustering methods.
        n_clusters=5,  # Number of clusters in k-means clustering.
        bandwidth=None,  # Only used if "kde" is selected for `clustering_method`.
        bandwidth_multiplier=None,  # Only used if "kde" is selected for `clustering_method` and `bandwidth` is `None`.
    )

Example 1: User-specified Time Series for Learning Holiday Impact

By default, the same time series used for training is used to learn holiday impact and generate holiday
groups. The library also allows users to import external time series to train holiday impact. This can be useful when
the training data is too short to learn holiday impact. The external time series
used for generating holiday groups needs at least one time column and one value column.

For example, if we have a time series ``external_df`` in the format below (values are made up):

.. csv-table:: external_df
    :header: dates,values

    2020-01-01,5.22
    2020-01-02,8.88
    2020-01-03,8.72
    ...,...

The input for ``auto_holiday_params`` can then be specified as below:

.. code-block:: python

    # The `auto_holiday_params` to pass in for events.
    auto_holiday_params = dict(
        df=external_df,
        time_col="dates",
        value_col="values"
    )

Example 2: Customized Holiday List

By default, holidays pulled from ``holiday_lookup_countries`` will be combined with events imported from
``daily_event_df_dict`` and serve as the source of holidays for grouping.
When users want to use a completely customized list of holidays, they can specify that through ``holiday_df``.  In
this case, holidays fetched from ``holiday_lookup_countries`` and ``daily_event_df_dict`` will be ignored. Only
holidays included in ``holiday_df`` will be used for grouping.

The ``holiday_df``
should be a `pandas.DataFrame` with at least two columns: one column for dates and one for holiday names. The holiday
dates should cover both training and forecasting periods. For example, assuming that the input ``holiday_df`` looks
like below:

.. csv-table:: holiday_df
    :header: date, event_name

    2020-12-25,Christmas
    2021-01-01,New Year
    ...,...

The input for ``auto_holiday_params`` can then be specified as:

.. code-block:: python

    # The `auto_holiday_params` to pass in for events.
    auto_holiday_params = dict(
        holiday_df=holiday_df,
        holiday_date_col="date",
        holiday_name_col="event_name"
    )

Prophet
^^^^^^^

Options:

.. code-block:: none

    events : `dict` [`str`, `any`] or None
        Holiday/events configuration dictionary with the following optional keys:

        holiday_lookup_countries: `list` [`str`] or "auto" or None, optional.
            default ("auto") uses a default list of countries with a good coverage of global holidays.
            If None or an empty list, no holidays are modeled.
            Must include all countries, for which you want to model holidays.
            Grid search is not supported.
        holidays_prior_scale: `float` or None or list of such values for grid search, default 10.0
            Modulates the strength of the holiday effect.
        holiday_pre_num_days: `list` [`int`] or None, default 2
            Model holiday effects for holiday_pre_num_days days before the holiday
            Grid search is not supported. Must be a list with one element or None.
        holiday_post_num_days: `list` [`int`] or None, default 2
            Model holiday effects for holiday_post_num_days days after the holiday.
            Grid search is not supported. Must be a list with one element or None.

Examples:

.. code-block:: python

    # Prophet template defaults
    events = dict(
        holiday_lookup_countries="auto",
        holiday_pre_num_days=[2],
        holiday_post_num_days=[2]
    )

    # Set holiday_pre_num_days=holiday_post_num_days=0 to only capture effects on the holiday itself
    events = dict(
        holiday_lookup_countries=[
        # Use two letter country code or full country name to look up holiday dates
            "US",
            "Canada"],
        holiday_pre_num_days=[0],
        holiday_post_num_days=[0]
    )

    # Grid search is possible
    events = dict(
        holiday_lookup_countries="auto",
        holiday_pre_num_days=[2],
        holiday_post_num_days=[2],
        holidays_prior_scale=[5.0, 15.0] # grid search over variety of holiday effect strength
    )

.. note::

  We do not currently allow custom events for Prophet.
  As a workaround, you can specify custom events (one-time or recurring)
  as a binary regressor whose value is 1 on the event, and 0 otherwise.
  See :doc:`/pages/model_components/0700_regressors`.


One-time events
---------------

.. note::

  Handle anomalies by classifying them as outliers. The model will smooth
  the value before fitting and should consider it in the volatility model.
  See :doc:`/pages/model_components/1000_override`. The volatility model
  does not yet do so; if volatility is important to you now, do not mark
  the point as an outlier.

  For data issues, set the value to np.nan. The model will smooth the value before fitting
  and consider any residual in the volatility model.

  For expected one-time events that will not repeat, label the point using a custom event. The
  model will fit to the original value and consider any residual in the volatility model.
