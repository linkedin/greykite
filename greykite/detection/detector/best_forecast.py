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
# original author: Reza Hosseini


from greykite.detection.detector.ad_utils import add_new_params_to_records
from greykite.detection.detector.forecast_based import ForecastBasedDetector


class BestForecastDetector(ForecastBasedDetector):
    """This class purpose is to find the best forecast from given k forecasts
    to act as baseline for anomaly detection.

    This class inherits its parameters and attributes from
    `~greykite.detection.detector.forecast_based.ForecastBasedDetector`
    and all methods apply here as well.

    The only addition is on top of the specified parameters to optimize over, this
    class also searches for the best forecast to use out of a number of forecasts
    passed. It achieves that simply by extending the parameter space, given in the input
    `param_iterable` to have one extra parameter: ``"forecast_id"``.

    Given that assumption, this class further implements the ``fit`` method fully
    and the user does not need to implement ``fit``.

    Therefore user only needs to implement the following:

        - ``add_features_one_df``: Note that the class uses this method to fully implement
            ``add_features`` because the class can assume that the same function can
            be used for all k forecasts.
        - ``calc_with_param``: the prediction logic, assuming the optimal param
            is determined.

    As an example see how the APE based method is implemented easily by inheriting
    from the current class:

        `~greykite.detection.detector.ape_based.APEDetector`


    Parameters
    ----------
    Solely inherited from
        `~greykite.detection.detector.forecast_based.ForecastBasedDetector`


    Attributes
    ----------
    Solely inherited from
        `~greykite.detection.detector.forecast_based.ForecastBasedDetector`
    """
    def __init__(
            self,
            value_cols=None,
            pred_cols=None,
            is_anomaly_col=None,
            join_cols=None,
            reward=None,
            anomaly_percent_dict=None,
            param_iterable=None):
        super().__init__(
            value_cols=value_cols,
            pred_cols=pred_cols,
            is_anomaly_col=is_anomaly_col,
            join_cols=join_cols,
            reward=reward,
            anomaly_percent_dict=anomaly_percent_dict,
            param_iterable=param_iterable)

    def fit(
            self,
            data):
        """
        Parameters
        ----------
        data : `~greykite.detection.detector.ForecastDetectorData`
            Object including the data.

        Returns
        -------
        result : None
            The fit will update ``self.fit_info`` and ``self.fitted_df``
        """
        # Adds the forecast id possibilities to `param_iterable`
        # so that we can optimize over forecasts as well as
        # existing combinations of parameters given in `param_iterable`
        if data.forecast_dfs is not None:
            # First it creates the list of needed forecast ids
            # based on the length of the input `forecast_dfs`
            forecast_ids = list(range(len(data.forecast_dfs)))
            if self.param_iterable is None:
                # It creates a list of dictionaries of length one
                # each prescribing only the `forecast_id`
                param_iterable = [{"forecast_id": x} for x in forecast_ids]
            else:
                param_iterable = add_new_params_to_records(
                    new_params={"forecast_id": forecast_ids},
                    records=self.param_iterable)
        else:
            param_iterable = self.param_iterable

        # Joins the forecast dfs with df
        # each joined df is a join between df and one element of
        # `forecast_dfs`
        # Prepares data
        self.prep_df_for_predict(data)
        # True labels might be needed for some objectives
        # therefore we extract them if available
        # If labels are not available, then `y_true` is set to be None
        data.y_true = None
        if self.is_anomaly_col is not None and data.y_true is None:
            data.y_true = data.df[self.is_anomaly_col]

        self.data = data

        optim_res = self.optimize_param(
            data=data,
            param_iterable=param_iterable)

        self.fit_info = {
            "param": optim_res["best_param"],
            "param_full": optim_res["best_param_full"],
            "obj_value": optim_res["best_obj_value"],
            "param_obj_list": optim_res["param_obj_list"],
            "best_calc_result": optim_res["best_calc_result"]}

        # Gets fitted values and final objective values
        self.fitted_df = self.predict(data)
