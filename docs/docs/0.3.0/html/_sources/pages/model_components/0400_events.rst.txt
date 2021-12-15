Holidays and Events
===================

Use ``model_components.events`` to specify the holidays and events.

Events indicate times of expected deviations from the usual growth and seasonality pattern.
These may be repeated (e.g. holidays), or one-time (Olympic for a city).

Repeated events are allowed to fall on a different day each year (e.g. Lunar New Year, Thanksgiving).
The model assumes the event has similar effect each time it occurs.

Holidays
--------

Silverkite provides a standard set of holidays by country (imported from ``pypi:holidays`` and ``pypi:fbprophet``).

Each holiday is mapped to a list of calendar dates where it occurs. For modeling, you can choose to
extend the holiday window to include days before and after the holiday. A separate effect is modeled for each
offset from the actual holiday.

For example, you can choose to model offsets (-2, -1, 0, 1) to capture effects
two days before, and one day after the holiday, by setting ``holiday_pre_num_days=-2``
and ``holiday_post_num_days=1`` below.

If you are not sure which holidays to use, start with our defaults and create a forecast.
Plot forecasts against actuals, and look for large errors. If these happen on holidays,
include relevant countries in ``holiday_lookup_countries`` list.

Silverkite
^^^^^^^^^^

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
    get_available_holiday_lookup_countries(["UnitedStates", "NewZealand", "EuropeanCentralBank"])

    # Full list for v3.0.0
    >>> get_available_holiday_lookup_countries()
    ['AR', 'AT', 'AU', 'Argentina', 'Australia', 'Austria', 'BD', 'BE', 'BG', 'BR',
     'BY', 'Bangladesh', 'Belarus', 'Belgium', 'Brazil', 'Bulgaria', 'CA', 'CH', 'CN',
     'CO', 'CZ', 'Canada', 'China', 'Colombia', 'Croatia', 'Czech', 'Czechia', 'DE',
     'DK', 'Denmark', 'ECB', 'EG', 'ES', 'Egypt', 'England', 'EuropeanCentralBank',
     'FI', 'FRA', 'Finland', 'France', 'Germany', 'HND', 'HR', 'HU', 'Honduras',
     'Hungary', 'ID', 'IE', 'IN', 'IND', 'IT', 'India', 'Indonesia', 'Ireland',
     'IsleOfMan', 'Italy', 'JP', 'Japan', 'LT', 'LU', 'Lithuania', 'Luxembourg',
     'MX', 'MY', 'Malaysia', 'Mexico', 'NL', 'NO', 'NZ', 'Netherlands', 'NewZealand',
     'NorthernIreland', 'Norway', 'PH', 'PK', 'PL', 'PT', 'PTE', 'Pakistan', 'Philippines',
     'Polish', 'Portugal', 'PortugalExt', 'RU', 'Russia', 'SE', 'SI', 'SK', 'Scotland',
     'Slovak', 'Slovenia', 'SouthAfrica', 'Spain', 'Sweden', 'Switzerland', 'TAR', 'TH',
     'TU', 'Thailand', 'Turkey', 'UA', 'UK', 'US', 'Ukraine', 'UnitedKingdom',
     'UnitedStates', 'VN', 'Vietnam', 'Wales', 'ZA']


* To check the available ``holidays_to_model_separately`` in those countries,
  run ``get_available_holidays_across_countries``:

.. code-block:: python

    from greykite.common.features.timeseries_features import get_available_holidays_across_countries

    # Select your countries
    holiday_lookup_countries = ["UnitedStates", "NewZealand", "EuropeanCentralBank"]
    # List the holidays
    get_available_holidays_across_countries(
        countries=holiday_lookup_countries,
        year_start=2017,
        year_end=2025)

.. note::

  While holidays are specified at a daily level, you can use interactions with seasonality to capture
  sub-daily holiday effects. For more information, see :doc:`/pages/model_components/0600_custom`.


Prophet
^^^^^^^

Options:

.. code-block:: none

    events : `dict` [`str`, `any`] or None
        Holiday/events configuration dictionary with the following optional keys:

        holiday_lookup_countries: `list` [`str`] or "auto" or None, optional.
            default ("auto") uses default list of countries with large contribution to Internet traffic.
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
