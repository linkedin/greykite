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
# original authors: Sayan Patra, Yi Su
"""Class to load datasets."""
import datetime
import os
import warnings
from pathlib import Path

import pandas as pd

from greykite.common.constants import TIME_COL


class DataLoader:
    """Returns datasets included in the library in `pandas.DataFrame` format.

    Attributes
    ----------
    available_datasets : `list` [`str`]
        The names of the available datasets.
    """
    def __init__(self):
        self.available_datasets = self.get_data_inventory()

    @staticmethod
    def get_data_home(data_dir=None, data_sub_dir=None):
        """Returns the folder path ``data_dir/data_sub_dir``.
        If ``data_dir`` is None returns the internal data directory.
        By default the Greykite data dir is set to a folder named 'data' in the project source code.
        Alternatively, it can be set programmatically by giving an explicit folder path.

        Parameters
        ----------
        data_dir : `str` or None, default None
            The path to the input data directory.
        data_sub_dir : `str` or None, default None
            The name of the input data sub directory.
            Updates path by appending to the ``data_dir`` at the end.
            If None, ``data_dir`` path is unchanged.

        Returns
        -------
        data_home : `str`
            Path to the data folder.
        """
        if data_dir is None:
            data_dir = Path(__file__).parents[1].joinpath("data")

        if data_sub_dir is None:
            data_home = os.path.abspath(data_dir)
        else:
            data_home = os.path.abspath(os.path.join(data_dir, data_sub_dir))

        if not os.path.exists(data_home):
            raise ValueError(f"Requested data directory '{data_home}' does not exist.")

        return data_home

    @staticmethod
    def get_data_names(data_path):
        """Returns the names of the ``.csv`` and ``.csv.xz`` files in ``data_path``.

        Parameters
        ----------
        data_path : `str`
            Path to the data folder.

        Returns
        -------
        file_names : `list` [`str`]
            The names of the ``.csv`` and ``.csv.xz`` files in ``data_path``.
        """
        file_names = os.listdir(data_path)
        file_names = [file_name.split(".csv")[0] for file_name in file_names
                      if file_name.endswith((".csv", ".csv.xz"))]

        return file_names

    @staticmethod
    def get_aggregated_data(df, agg_freq=None, agg_func=None):
        """Returns aggregated data.

        Parameters
        ----------
        df : `pandas.DataFrame`.
            The input data must have TIME_COL ("ts") column and the columns in the keys of ``agg_func``.

        agg_freq : `str` or None, default None
            If None, data will not be aggregated and will include all columns.
            Possible values: "daily", "weekly", or "monthly".

        agg_func : `Dict` [`str`, `str`], default None
            A dictionary of the columns to be aggregated and the corresponding aggregating functions.
            Possible aggregating functions include "sum", "mean", "median", "max", "min", etc.
            An example input can be {"col1":"mean", "col2":"sum"}
            If None, data will not be aggregated and will include all columns.

        Returns
        -------
        df : `pandas.DataFrame`
            The aggregated dataframe.
        """
        if TIME_COL not in df.columns:
            raise ValueError(f"{TIME_COL} must be in the DataFrame.")

        if not agg_freq and not agg_func:
            return df
        elif agg_freq and agg_func:
            df_raw = df[list(agg_func.keys())]
            df_raw.insert(0, TIME_COL, pd.to_datetime(df[TIME_COL]))
            if agg_freq == "daily":
                # Aggregate to daily
                df_tmp = df_raw.resample("D", on=TIME_COL).agg(agg_func)
                df_daily = df_tmp.drop(columns=TIME_COL).reset_index() if TIME_COL in df_tmp.columns else df_tmp.reset_index()
                return df_daily
            elif agg_freq == "weekly":
                # Aggregate to weekly
                df_tmp = df_raw.resample("W-MON", on=TIME_COL).agg(agg_func)
                df_weekly = df_tmp.drop(columns=TIME_COL).reset_index() if TIME_COL in df_tmp.columns else df_tmp.reset_index()
                return df_weekly
            elif agg_freq == "monthly":
                # Aggregate to monthly
                df_tmp = df_raw.resample("MS", on=TIME_COL).agg(agg_func)
                df_monthly = df_tmp.drop(columns=TIME_COL).reset_index() if TIME_COL in df_tmp.columns else df_tmp.reset_index()
                return df_monthly
            else:
                warnings.warn("Invalid \"agg_freq\", must be one of \"daily\", \"weekly\" or \"monthly\". "
                              "Non-aggregated data is returned.")
                return df_raw
        else:
            warnings.warn("Both \"agg_freq\" and \"agg_func\" must be provided. "
                          "Non-aggregated data is returned.")
            return df

    def get_data_inventory(self):
        """Returns the names of the available internal datasets.

        Returns
        -------
        file_names : `list` [`str`]
            The names of the available internal datasets.
        """
        file_names = []
        for freq in ["minute", "hourly", "daily", "monthly"]:
            data_path = self.get_data_home(data_dir=None, data_sub_dir=freq)
            file_names.extend(self.get_data_names(data_path=data_path))

        return file_names

    def get_df(self, data_path, data_name):
        """Returns a ``pandas.DataFrame`` containing the dataset from ``data_path/data_name``.
        The input data must be in ``.csv`` or ``.csv.xz`` format.
        Raises a ValueError if the the specified input file is not found.

        Parameters
        ----------
        data_path : `str`
            Path to the data folder.
        data_name : `str`
            Name of the csv file to be loaded from. For example 'peyton_manning'.

        Returns
        -------
        df : `pandas.DataFrame`
            Input dataset.
        """
        file_path = os.path.join(data_path, f"{data_name}.csv")
        if not os.path.exists(file_path):
            file_names = self.get_data_names(data_path)
            raise ValueError(f"Given file path '{file_path}' is not found. Available datasets in "
                             f"data directory '{data_path}' are {file_names}.")
        df = pd.read_csv(file_path, sep=",")
        return df

    def load_peyton_manning(self):
        """Loads the Daily Peyton Manning dataset.

        This dataset contains log daily page views for the Wikipedia page for Peyton Manning.
        One of the primary datasets used for demonstrations by Facebook ``Prophet`` algorithm.
        Source: https://github.com/facebook/prophet/blob/master/examples/example_wp_log_peyton_manning.csv

        Below is the dataset attribute information:

            ts : date of the page view
            y : log of the number of page views

        Returns
        -------
        df : `pandas.DataFrame` object with Peyton Manning data.
            Has the following columns:

                "ts" : date of the page view.
                "y" : log of the number of page views.
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="daily")
        df = self.get_df(data_path=data_path, data_name="daily_peyton_manning")
        return df

    def load_parking(self, system_code_number=None):
        """Loads the Hourly Parking dataset.
        This dataset contains occupancy rates (8:00 to 16:30) from 2016/10/04 to 2016/12/19
        from car parks in Birmingham that are operated by NCP from Birmingham City Council.
        Source: https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham
        UK Open Government Licence (OGL)

        Below is the dataset attribute information:

            SystemCodeNumber: car park ID
            Capacity: car park capacity
            Occupancy: car park occupancy rate
            LastUpdated: date and time of the measure

        Parameters
        ----------
        system_code_number : `str` or None, default None
            If None, occupancy rate is averaged across all the ``SystemCodeNumber``.
            Else only the occupancy rate of the given ``system_code_number`` is returned.

        Returns
        -------
        df : `pandas.DataFrame` object with Parking data.
            Has the following columns:

                "LastUpdated" : time, rounded to the nearest half hour.
                "Capacity" : car park capacity
                "Occupancy" : car park occupancy rate
                "OccupancyRatio" : ``Occupancy`` divided by ``Capacity``.
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="hourly")
        df = self.get_df(data_path=data_path, data_name="hourly_parking")

        # Rounds time column to nearest half hour point
        df["LastUpdated"] = pd.to_datetime(df["LastUpdated"]).dt.round("30min")
        df["OccupancyRatio"] = df["Occupancy"] / df["Capacity"]
        if system_code_number is None:
            df = df.groupby("LastUpdated", as_index=False).mean()
        else:
            df = df[df["SystemCodeNumber"] == system_code_number].reset_index(drop=True)
        return df

    def load_bikesharing(self, agg_freq=None, agg_func=None):
        """Loads the Hourly Bike Sharing Count dataset with possible aggregations.

        This dataset contains aggregated hourly count of the number of rented bikes.
        The data also includes weather data: Maximum Daily temperature (tmax);
        Minimum Daily Temperature (tmin); Precipitation (pn)
        The raw bike-sharing data is provided by Capital Bikeshare.
        Source: https://www.capitalbikeshare.com/system-data
        The raw weather data (Baltimore-Washington INTL Airport)
        https://www.ncdc.noaa.gov/data-access/land-based-station-data

        Below is the dataset attribute information:

            ts : hour and date
            count : number of shared bikes
            tmin : minimum daily temperature
            tmax : maximum daily temperature
            pn : precipitation

        Parameters
        ----------
        Refer to the input of function `get_aggregated_data`.

        Returns
        -------
        df : `pandas.DataFrame` with bikesharing data.
            If no ``freq`` was specified, the returned data has the following columns:

                "date" : day of year
                "ts" : hourly timestamp
                "count" : number of rented bikes across Washington DC.
                "tmin" : minimum daily temperature
                "tmax" : maximum daily temperature
                "pn" : precipitation
            Otherwise, only ``agg_col`` column is returned.
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="hourly")
        df = self.get_df(data_path=data_path, data_name="hourly_bikesharing")
        return self.get_aggregated_data(df=df, agg_freq=agg_freq, agg_func=agg_func)

    def load_solarpower(self, agg_freq=None, agg_func=None):
        """Loads the Hourly Solar Power dataset.

        This dataset contains the solar power production of an Australian wind farm
        from August 2019 to July 2020, with original frequency 4-second.
        We aggregated it to an hourly series and removed any incomplete hours.
        Source: https://zenodo.org/record/4656027#.YrpHbuzMLGp

        Below is the dataset attribute information:

            ts : hourly timestamp
            y : solar power production in MW (megawatt)

        Parameters
        ----------
        agg_freq : `str` or None, default None
            Possible values: "daily", "weekly", or "monthly".
            If None, data will not be aggregated and will include all columns.

        agg_func : `Dict` [`str`, `str`], default None
            A dictionary of the columns to be aggregated and the corresponding aggregating functions.
            Possible aggregating functions include "sum", "mean", "median", "max", "min", etc.
            An example input can be {"col1":"mean", "col2":"sum"}
            If None, data will not be aggregated and will include all columns.

        Returns
        -------
        df : `pandas.DataFrame` object with Solar Power data.
            Has the following columns:

                "ts" : hourly timestamp
                "y" : solar power production in MW (megawatt)
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="hourly")
        df = self.get_df(data_path=data_path, data_name="hourly_solarpower")
        return self.get_aggregated_data(df=df, agg_freq=agg_freq, agg_func=agg_func)

    def load_windpower(self, agg_freq=None, agg_func=None):
        """Loads the Hourly Wind Power dataset.

        This dataset contains the wind power production of an Australian wind farm
        from August 2019 to July 2020, with original frequency 4-second.
        We aggregated it to an hourly series and removed any incomplete hours.
        Source: https://zenodo.org/record/4656032#.YrpJTezMLGp

        Below is the dataset attribute information:

            ts : hourly timestamp
            y : wind power production in MW (megawatt)

        Parameters
        ----------
        agg_freq : `str` or None, default None
            Possible values: "daily", "weekly", or "monthly".
            If None, data will not be aggregated and will include all columns.

        agg_func : `Dict` [`str`, `str`], default None
            A dictionary of the columns to be aggregated and the corresponding aggregating functions.
            Possible aggregating functions include "sum", "mean", "median", "max", "min", etc.
            An example input can be {"col1":"mean", "col2":"sum"}
            If None, data will not be aggregated and will include all columns.

        Returns
        -------
        df : `pandas.DataFrame` object with Wind Power data.
            Has the following columns:

                "ts" : hourly timestamp
                "y" : wind power production in MW (megawatt)
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="hourly")
        df = self.get_df(data_path=data_path, data_name="hourly_windpower")
        return self.get_aggregated_data(df=df, agg_freq=agg_freq, agg_func=agg_func)

    def load_electricity(self, agg_freq=None, agg_func=None):
        """Loads the Hourly Electricity dataset.

        This dataset contains the hourly consumption (in Kilowatt)
        of 321 clients from 2012 to 2014 published by Monash.
        We aggregated them by taking the average across the 321 clients.
        Source: https://zenodo.org/record/4656140#.YrpKtezMJqs

        Below is the dataset attribute information:

            ts : hourly timestamp
            y : average electricity consumption in Kilowatt

        Parameters
        ----------
        agg_freq : `str` or None, default None
            Possible values: "daily", "weekly", or "monthly".
            If None, data will not be aggregated and will include all columns.

        agg_func : `Dict` [`str`, `str`], default None
            A dictionary of the columns to be aggregated and the corresponding aggregating functions.
            Possible aggregating functions include "sum", "mean", "median", "max", "min", etc.
            An example input can be {"col1":"mean", "col2":"sum"}
            If None, data will not be aggregated and will include all columns.

        Returns
        -------
        df : `pandas.DataFrame` object with Electricity data.
            Has the following columns:

                "ts" : hourly timestamp
                "y" : average electricity consumption in Kilowatt
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="hourly")
        df = self.get_df(data_path=data_path, data_name="hourly_electricity")
        return self.get_aggregated_data(df=df, agg_freq=agg_freq, agg_func=agg_func)

    def load_sf_traffic(self, agg_freq=None, agg_func=None):
        """Loads the Hourly San Francisco Bay Area Traffic dataset.

        This dataset contains the road occupancy rates (between 0 and 1)
        measured by different sensors on San Francisco
        Bay area freeways from 2015 to 2016.
        Source: https://zenodo.org/record/4656132#.YrpMxuzMLGp

        Below is the dataset attribute information:

            ts : hourly timestamp
            y : average occupancy rate

        Parameters
        ----------
        agg_freq : `str` or None, default None
            Possible values: "daily", "weekly", or "monthly".
            If None, data will not be aggregated and will include all columns.

        agg_func : `Dict` [`str`, `str`], default None
            A dictionary of the columns to be aggregated and the corresponding aggregating functions.
            Possible aggregating functions include "sum", "mean", "median", "max", "min", etc.
            An example input can be {"col1":"mean", "col2":"sum"}
            If None, data will not be aggregated and will include all columns.

        Returns
        -------
        df : `pandas.DataFrame` object with San Francisco Bay Area Traffic data.
            Has the following columns:

                "ts" : hourly timestamp
                "y" : average occupancy rate
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="hourly")
        df = self.get_df(data_path=data_path, data_name="hourly_sf_traffic")
        return self.get_aggregated_data(df=df, agg_freq=agg_freq, agg_func=agg_func)

    def load_bitcoin_transactions(self, agg_freq=None, agg_func=None):
        """Loads the Daily Bitcoin Transactions dataset.

        This dataset contains the number of Bitcoin transactions from 2009 to 2021.
        The dataset was curated (with missing values filled) by Monash.
        Source: https://zenodo.org/record/5122101#.YrpNFuzMLGp

        Below is the dataset attribute information:

            ts : date
            y : number of transactions

        Parameters
        ----------
        agg_freq : `str` or None, default None
            Possible values: "daily", "weekly", or "monthly".
            If None, data will not be aggregated and will include all columns.

        agg_func : `Dict` [`str`, `str`], default None
            A dictionary of the columns to be aggregated and the corresponding aggregating functions.
            Possible aggregating functions include "sum", "mean", "median", "max", "min", etc.
            An example input can be {"col1":"mean", "col2":"sum"}
            If None, data will not be aggregated and will include all columns.

        Returns
        -------
        df : `pandas.DataFrame` object with Bitcoin Transactions data.
            Has the following columns:

                "ts" : date
                "y" : number of transactions
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="daily")
        df = self.get_df(data_path=data_path, data_name="daily_bitcoin_transactions")
        return self.get_aggregated_data(df=df, agg_freq=agg_freq, agg_func=agg_func)

    def load_sunspot(self):
        """Loads the Sunspot dataset.

        This dataset contains the number of observed sunspots from 1818 to 2020
        published by Monash.
        The original dataset was a daily series, and we aggregate it to a
        monthly time series more than 200 years long.
        Source: https://zenodo.org/record/4654722#.YrpQ4uzMLGp

        Below is the dataset attribute information:

            ts : month start date
            y : average number of sunspots

        Returns
        -------
        df : `pandas.DataFrame` object with Sunspot data.
            Has the following columns:

                "ts" : date
                "y" : average number of sunspots
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="monthly")
        df = self.get_df(data_path=data_path, data_name="monthly_sunspot_monash")
        return df

    def load_fred_housing(self):
        """Loads the FRED House Supply dataset.

        This dataset contains the monthly house supply in the United States
        from 1963 to 2021 obtained from FRED.
        Source: https://fred.stlouisfed.org/series/MSACSR

        Below is the dataset attribute information:

            ts : month start date
            y : monthly supply of new houses

        Returns
        -------
        df : `pandas.DataFrame` object with FRED House Supply data.
            Has the following columns:

                "ts" : date
                "y" : monthly supply of new houses
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="monthly")
        df = self.get_df(data_path=data_path, data_name="monthly_fred_housing")
        return df

    def load_beijing_pm(self, agg_freq=None, agg_func=None):
        """Loads the Beijing Particulate Matter (PM2.5) dataset.
        https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

        This hourly data set contains the PM2.5 data of US Embassy in Beijing. Meanwhile, meteorological data
        from Beijing Capital International Airport are also included.

        The dataset's time period is between Jan 1st, 2010 to Dec 31st, 2014. Missing data are denoted as NA.

        Below is the dataset attribute information:

            No : row number
            year : year of data in this row
            month : month of data in this row
            day : day of data in this row
            hour : hour of data in this row
            pm2.5: PM2.5 concentration (ug/m^3)
            DEWP : dew point (celsius)
            TEMP : temperature (celsius)
            PRES : pressure (hPa)
            cbwd : combined wind direction
            Iws : cumulated wind speed (m/s)
            Is : cumulated hours of snow
            Ir : cumulated hours of rain

        Parameters
        ----------
        Refer to the input of function `get_aggregated_data`.

        Returns
        -------
        df : `pandas.DataFrame` with Beijing PM2.5 data.
            Has the following columns:

                "ts" : hourly timestamp
                "year" : year of data in this row
                "month" : month of data in this row
                "day" : day of data in this row
                "hour" : hour of data in this row
                "pm" : PM2.5 concentration (ug/m^3)
                "dewp" : dew point (celsius)
                "temp" : temperature (celsius)
                "pres" : pressure (hPa)
                "cbwd" : combined wind direction
                "iws" : cumulated wind speed (m/s)
                "is" : cumulated hours of snow
                "ir" : cumulated hours of rain
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="hourly")
        df = self.get_df(data_path=data_path, data_name="hourly_beijing_pm")

        df.drop("No", axis=1, inplace=True)
        df.columns = map(str.lower, df.columns)
        df.rename(columns={"pm2.5": "pm"}, inplace=True)

        hour_string = df["year"].astype(str) + "-" + df["month"].astype(str) + "-" + df["day"].astype(str) + "-" + df["hour"].astype(str)
        hour_datetime = hour_string.apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d-%H"))
        df.insert(0, TIME_COL, hour_datetime)

        return self.get_aggregated_data(df=df, agg_freq=agg_freq, agg_func=agg_func)

    def load_hierarchical_actuals(self):
        """Loads hierarchical actuals.

        This dataset contains synthetic data that satisfy hierarchical constraints.
        Consider the 3-level tree with the parent-child relationships below.
                00        # level 0
              /    \
           10       11    # level 1
          / | \     /\  # noqa: W605
        20 21 22   23 24  # level 2

        There is one root node (00) with 2 children.
        The first child (10) has 3 children.
        The second child (11) has 2 children.

        Let `x_{ij}` be the value of the `j`th node in level `i` of the tree ({`ij`} is shown in diagram above).
        We require the value of a parent to equal the sum of the values of its children.
        There are 3 constraints in this hierarchy, satisfied at all time points:

            * `x_00 = x_10 + x_11`
            * `x_10 = x_20 + x_21 + x_22`
            * `x_11 = x_23 + x_24`

        Below is the dataset attribute information:

            "ts" : date of the (synthetic) observation
            "00" : value for node 00
            "10" : value for node 10
            "11" : value for node 11
            "20" : value for node 20
            "21" : value for node 21
            "22" : value for node 22
            "23" : value for node 23
            "24" : value for node 24

        Returns
        -------
        df : `pandas.DataFrame` object with synthetic hierarchical data.
            Has the following columns:

                "ts" : date of the (synthetic) observation
                "00" : value for node 00
                "10" : value for node 10
                "11" : value for node 11
                "20" : value for node 20
                "21" : value for node 21
                "22" : value for node 22
                "23" : value for node 23
                "24" : value for node 24

            The values satisfy the hierarchical constraints above.
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="daily")
        df = self.get_df(data_path=data_path, data_name="daily_hierarchical_actuals")
        return df

    def load_hierarchical_forecasts(self):
        """Loads hierarchical forecasts.

        This dataset contains forecasts for the actuals
        given by `~greykite.common.data_loader.DataLoader.load_hierarchical_actuals`.
        The attributes are the same.

        Returns
        -------
        df : `pandas.DataFrame` object with forecasts for synthetic hierarchical data.
            Has the following columns:

                "ts" : date of the forecasted value
                "00" : value for node 00
                "10" : value for node 10
                "11" : value for node 11
                "20" : value for node 20
                "21" : value for node 21
                "22" : value for node 22
                "23" : value for node 23
                "24" : value for node 24

            The forecasts do not satisfy the hierarchical constraints.
            The index and columns are identical to
            `~greykite.common.data_loader.DataLoader.load_hierarchical_actuals`.
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="daily")
        df = self.get_df(data_path=data_path, data_name="daily_hierarchical_forecasts")
        return df

    def load_data(self, data_name, **kwargs):
        """Loads dataset by name from the internal data library.

        Parameters
        ----------
        data_name : `str`
            Dataset to load from the internal data library.

        Returns
        -------
        df : ``UnivariateTimeSeries`` object with ``data_name``.
        """
        if data_name == "daily_peyton_manning":
            df = self.load_peyton_manning()
        elif data_name == "hourly_parking":
            df = self.load_parking(**kwargs)
        elif data_name == "hourly_bikesharing":
            df = self.load_bikesharing(**kwargs)
        elif data_name == "hourly_beijing_pm":
            df = self.load_beijing_pm(**kwargs)
        elif data_name == "daily_hierarchical_actuals":
            df = self.load_hierarchical_actuals()
        elif data_name == "daily_hierarchical_forecasts":
            df = self.load_hierarchical_forecasts()
        else:
            data_inventory = self.get_data_inventory()
            raise ValueError(f"Input data name '{data_name}' is not recognized. "
                             f"Must be one of {data_inventory}.")

        return df
