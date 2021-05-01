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
"""Functional wrapper around a forecast function.
Calls the forecast function once for each horizon
with the goal of providing the best accuracy for
each horizon.
"""

from functools import partial

import pandas as pd


def forecast_one_by_one_fcn(
        train_forecast_func,
        **model_params):

    """A function which turns a train-forecast function to a function which
    forecast each horizon by fitting its own corresponding model with the goal
    of providing the best accuracy for each horizon.

    Parameters
    ----------
    train_forecast_func : `callable`
        A train forecast function, which gets inputs data (``df``) and produces
        forecast for given horizon for the whole period: 1 to ``forecast_horizon``.

        The input function ``train_forecast_func`` has the following expected signature
        and expected output::

            train_forecast_func(
                df,
                time_col,
                value_col,
                forecast_horizon,
                **model_params)

                    ->

                {
                    "fut_df": fut_df,
                    "trained_model": trained_model,
                    ...}  # potential extra outputs depending on the model
            where

                - fut_df : `pandas.DataFrame`
                    A dataframe with forecast values
                - trained_model : `dict`
                    A dictionary with information for the trained model

        This function then can be composed with ``train_forecast_func`` from
        left as follows::

            forecast_one_by_one_fcn(train_forecast_func)

        to generate a new function with the same inputs as
        ``train_forecast_func`` and same outputs with an extra dict which has
        all trained models per horizon (see the Return Section for more details).

    model_params : additional arguments
        Extra parameters passed to ``trained_forecast_func`` if desired as
        keyed parameters.

    Returns
    -------
    func : `callable`
        A function to compose with the input function ``train_forecast_func`` from
        left and return another function with the same inputs and outputs as
        ``train_forecast_func``::

            func = forecast_one_by_one_fcn(train_forecast_func)

        Note that ``func`` will utilize ``train_forecast_func`` to produce forecasts
        one by one by training one model per horizon.

        As discussed, ``func`` has the same inputs as ``train_forecast_func``
        and same outputs with an extra dict which has all trained models per horizon:

            - fut_df : `pandas.DataFrame`
                With same columns and structure as ``fut_df`` returned by ``train_forecast_func``
            - trained_model : `dict`
                The trained model on the longest horizon passed.
                This is simply application of ``trained_forecast_func`` on the full horizon
            - trained_models_per_horizon: `dict`
                A dictionary with trained models per horizon
                    - key : ``forecasts_horizon`` : `int`
                    - value : trained_model
                        This is the trained_model for that horizon
    """
    def train_forecast_func_one_by_one(
            df,
            time_col,
            value_col,
            forecast_horizon,
            **model_parameters):

        def forecast_kth_time(k):
            """Returns the kth time period forecast.

            Parameters
            ----------
            k : `int`
                The time period for which we require forecast.

            Returns
            -------
            result : `dict`
                A dictionary with following items:

                - fut_df_one_row : `pd.DataFrame`
                    A dataframe with one row with the same format as ``fut_df``
                    which includes the forecast for time ``k``.
                - forecast : `dict`
                    A dictionary which is the forecast result for ``forecast_horizon=k``.

            """
            forecast = train_forecast_func(
                df=df,
                time_col=time_col,
                value_col=value_col,
                forecast_horizon=k,
                **model_parameters)

            fut_df = forecast["fut_df"]
            fut_df_one_row = fut_df.iloc[[k-1]].reset_index(drop=True)
            return {
                "forecast": forecast,
                "fut_df_one_row": fut_df_one_row}

        # Runs the model and stores the results for all horizons
        forecast_per_horizon = {
                k: forecast_kth_time(k) for k in range(1, forecast_horizon+1)}

        # Extracts a forecast as usual for all times in the horizon
        # This is the same as the function called on the whole horizon
        # Note ``"trained_model"`` and ``"fut_df"`` are expected to be contained in "forecast"
        # as they are expected outputs of ``train_forecast_func``
        forecast = forecast_per_horizon[forecast_horizon]["forecast"]
        forecast["trained_models_per_horizon"] = {
            k: forecast_per_horizon[k]["forecast"]["trained_model"] for k in range(1, forecast_horizon+1)}

        # If the forecast horizon is 1, we are done
        # Otherwise, we need to update the returned ``"fut_df"``
        if forecast_horizon > 1:
            fut_df_list = [
                forecast_per_horizon[k]["fut_df_one_row"] for k in range(1, forecast_horizon+1)]
            fut_df = pd.concat(
                fut_df_list,
                axis=0,
                sort=False)
            # Replaces ``fut_df`` with the updated one
            forecast["fut_df"] = fut_df
        return forecast

    return partial(train_forecast_func_one_by_one, **model_params)
