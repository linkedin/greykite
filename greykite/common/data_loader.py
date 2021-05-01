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
"""Class to load datasets."""
import datetime
import os
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
            If None, it is set to '{project_trunk}/data'.
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
            data_dir = Path(__file__).parents[2].joinpath("data")

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

    def load_bikesharing(self):
        """Loads the Hourly Bike Sharing Count dataset.

        This dataset contains aggregated hourly count of the number of rented bikes.
        The data also includes weather data: Maximum Daily temperature (tmax);
        Minimum Daily Temperature (tmin); Precipitation (pn)
        The raw bike-sharing data is provided by Capital Bikeshares.
        Source: https://www.capitalbikeshare.com/system-data
        The raw weather data (Baltimore-Washington INTL Airport)
        https://www.ncdc.noaa.gov/data-access/land-based-station-data

        Below is the dataset attribute information:

            ts : hour and date
            count : number of shared bikes
            tmin : minimum daily temperature
            tmax : maximum daily temperature
            pn : precipitation

        Returns
        -------
        df : `pandas.DataFrame` with bikesharing data.
            Has the following columns:

                "date" : day of year
                "ts" : hourly timestamp
                "count" : number of rented bikes across Washington DC.
                "tmin" : minimum daily temperature
                "tmax" : maximum daily temperature
                "pn" : precipitation
        """
        data_path = self.get_data_home(data_dir=None, data_sub_dir="hourly")
        df = self.get_df(data_path=data_path, data_name="hourly_bikesharing")
        return df

    def load_beijing_pm(self):
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

        Returns
        -------
        df : `pandas.DataFrame` with Beijing PM2.5 data.
            Has the following columns:

                TIME_COL : hourly timestamp
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
            df = self.load_bikesharing()
        elif data_name == "hourly_beijing_pm":
            df = self.load_beijing_pm()
        else:
            data_inventory = self.get_data_inventory()
            raise ValueError(f"Input data name '{data_name}' is not recognized. "
                             f"Must be one of {data_inventory}.")

        return df
