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
"""Class to load datasets in UnivariateTimeSeries format"""

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries


class DataLoaderTS(DataLoader):
    """Returns datasets included in the library in `pandas.DataFrame` or
    `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries` format.

    Extends `~greykite.common.data_loader.DataLoader`
    """
    def __init__(self):
        super().__init__()

    def load_peyton_manning_ts(self):
        """Loads the Daily Peyton Manning dataset.

        This dataset contains log daily page views for the Wikipedia page for Peyton Manning.
        One of the primary datasets used for demonstrations by Facebook ``Prophet`` algorithm.
        Source: https://github.com/facebook/prophet/blob/master/examples/example_wp_log_peyton_manning.csv

        Below is the dataset attribute information:

            ts : date of the page view
            y : log of the number of page views

        Returns
        -------
        ts : `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries`
            Peyton Manning page views data. Time and value column:

                ``time_col`` : "ts"
                    Date of the page view.
                ``value_col`` : "y"
                    Log of the number of page views.
        """
        df = self.load_peyton_manning()
        ts = UnivariateTimeSeries()
        ts.load_data(
            df=df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            freq="1D"
        )
        return ts

    def load_parking_ts(self, system_code_number=None):
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
        ts : `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries`
            Parking data. Time and value column:

                ``time_col`` : "LastUpdated"
                    Date and Time of the Occupancy Rate, rounded to the nearest half hour.
                ``value_col`` : "OccupancyRatio"
                    ``Occupancy`` divided by ``Capacity``.
        """
        df = self.load_parking(system_code_number=system_code_number)
        ts = UnivariateTimeSeries()
        ts.load_data(
            df=df,
            time_col="LastUpdated",
            value_col="OccupancyRatio",
            freq="30min",
        )
        return ts

    def load_bikesharing_ts(self):
        """Loads the Hourly Bike Sharing Count dataset.

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

        Returns
        -------
        ts : `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries`
            Bike Sharing Count data. Time and value column:

                ``time_col`` : "ts"
                    Hour and Date.
                ``value_col`` : "y"
                     Number of rented bikes across Washington DC.

            Additional regressors:

                "tmin" : minimum daily temperature
                "tmax" : maximum daily temperature
                "pn" : precipitation
        """
        df = self.load_bikesharing()
        ts = UnivariateTimeSeries()
        ts.load_data(
            df=df,
            time_col="ts",
            value_col="count",
            freq="H",
            regressor_cols=["tmin", "tmax", "pn"]
        )
        return ts

    def load_beijing_pm_ts(self):
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
        ts : `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries`
            Beijing PM2.5 data. Time and value column:

                ``time_col`` : TIME_COL
                    hourly timestamp
                ``value_col`` : "pm"
                     PM2.5 concentration (ug/m^3)

            Additional regressors:

                "dewp" : dew point (celsius)
                "temp" : temperature (celsius)
                "pres" : pressure (hPa)
                "cbwd" : combined wind direction
                "iws" : cumulated wind speed (m/s)
                "is" : cumulated hours of snow
                "ir" : cumulated hours of rain
        """
        df = self.load_beijing_pm()
        ts = UnivariateTimeSeries()
        ts.load_data(
            df=df,
            time_col=TIME_COL,
            value_col="pm",
            freq="H",
            regressor_cols=["dewp", "temp", "pres", "cbwd", "iws", "is", "ir"]
        )
        return ts

    def load_data_ts(self, data_name, **kwargs):
        """Loads dataset by name from the internal data library.

        Parameters
        ----------
        data_name : `str`
            Dataset to load from the internal data library.

        Returns
        -------
        ts : `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries`
            Has the requested ``data_name``.
        """
        if data_name == "daily_peyton_manning":
            ts = self.load_peyton_manning_ts()
        elif data_name == "hourly_parking":
            ts = self.load_parking_ts(**kwargs)
        elif data_name == "hourly_bikesharing":
            ts = self.load_bikesharing_ts()
        elif data_name == "hourly_beijing_pm":
            ts = self.load_beijing_pm_ts()
        else:
            data_inventory = self.get_data_inventory()
            raise ValueError(f"Input data name '{data_name}' is not recognized. "
                             f"Must be one of {data_inventory}.")

        return ts
