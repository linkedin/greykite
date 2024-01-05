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

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from greykite.detection.detector.data import DetectorData


@dataclass
class CalcResult:
    """This data class represents the standard return of the method:
    `calc_with_param` of `~greykite.detection.detector.optimizer.Optimizer`

    Attributes
    ----------
    data : `object` or None, default None
        This is part of the calculation which includes data e.g. dataframes etc.
        This data is the only part of calculation which is needed to calculate the
        reward during optimization.
    model :`object` or None, default None
        This is returned by the calculation, and it might be a trained model, or
        a useful object generated during the calculation which can be used later
        for example during prediction phase, when the optimizer is a predictor.

    """
    data: Optional[DetectorData] = None
    model: Optional[object] = None


class Optimizer:
    """A class to enable easy implementation of optimization over
    arbitrary parameter spaces and using arbitrary rewards.
    The optimization problem can be stated in pseudo code:

    Maximize_{param} reward(some_function(param))

    Note that 'reward(some_function(param))' can be considered as the objective
    which is to be maximized. The objective is a two part calculation in this framework

        - calculate a function for param
        - calculate the reward for the above

    Here is a more detailed mathematical explanation in more detailed pseudo code:

    Assume:
        - ``param`` is a (potentially multivariate) parameter in a parameter space ``param_iterable``
        - ``calc_with_param`` is a function which depends on ``param``
        - psudeo code: 'calc_result = calc_with_param(param, ...)'
        - psudeo code: 'obj_value = reward.apply(calc_result.data)'

    Goal:
        - optimize (maximize) 'obj_value' across all possible param

    Note that `reward` does not take ``param`` as an input and only applied to
    the updated data calculated using ``param``.

    The class initializes by passing an arbitrary ``reward``
    for optimization and a potentially multivariate parameter
    (given in ``param_iterable``) to optimize.

    The ``reward`` object is required to implement the ``apply`` method
    which is the case for this class:
    `~greykite.detection.detector.reward.Reward`

    The optimization method (``optimize_param``) is the main method in this class and
    works simply by iterating over ``param_iterable`` and calculating the reward
    to choose the optimal parameter.
    The class assumes that larger is better for the reward function, during optimization.

    The classes inherting this class, need to implement ``calc_with_param`` method to be able to use the optimizer and given that implementation.

    Parameters
    ----------
    reward : `~greykite.detection.detector.reward.Reward` or None, default None
        The reward to be used in the optimization.
    param_iterable : iterable or None, default None
        An iterable with every element being a parameter passed to the method
        ``calc_with_param`` which takes ``param`` as one of its arguments and
        ``data`` as the other.
        Each ``param`` can be a dictionary including values for a set of variables.
        The optimizer method (``optimize_param``) will iterate over all the
        parameters to find the best parameter in terms of the specified reward.

    Attributes
    ----------
    data : `dataclasses.dataclass` or None, default None
        A data class object which includes the data for fitting or
        prediction. Depending on the model, this data class might
        include various fields. A simple example is given in
        `~greykite.detection.detector.data.Data`
    fit_info : `dict`
        A dictionary which includes information about the fitted model.
        It is expected that this includes ``"full_param"`` after the fitting
        so that the `predict` function can use that param during the prediction
        and simply call `calc_with_param`.
        In that case the `predict` function does not need further implementation
        in child classes as it's already implemented in this class.
    """
    def __init__(
            self,
            reward=None,
            param_iterable=None):
        self.reward = reward
        self.param_iterable = param_iterable
        # Initialize attributes
        self.data = None
        self.fit_info = {"param_full": None}

    def optimize_param(
            self,
            param_iterable=None,
            data=None,
            default_param=None,
            **kwargs):
        """The optimizer which picks the best possible parameter from the ones
        specified in ``param_iterable``, using the reward specified in the
        class instance.
        This method assumes larger is better for the reward.

        Parameters
        ----------
        param_iterable : iterable or None, default None
            See class docstring.
            If None in this method call, it will be set to
            ``self.param_iterable``
        data : `dataclasses.dataclass` or None, default None
            See class docstring.
        default_param : `dict` or None, default None
            A fixed parameter which is used as the default and for each
            param in ``param_iterable``, it will be used to construct the full
            parameter. For example it can include fixed parameters which are
            calculated separately before optimization occurs.

        Returns
        -------
        result : `dict`
            A dictionary with following items:

            - ``"best_param"``: `dict`
                The best parameter in terms of the specified reward, where larger
                is considered better.
            - ``"best_param_full"``: `dict`
                The default parameter augmented with the best parameter
                to construct the full parameter.
            - ``"best_obj_value"``: `float`
                The best reward value.
            - ``"param_obj_list"``: `list` [`dict`]
            - "best_calc_result": `greykite.detection.detector.optimizer.CalcResult`
                The calculation result at the best parameter.
        """
        if param_iterable is None:
            param_iterable = self.param_iterable
        if default_param is None:
            default_param = {}
        best_obj_value = float("-inf")
        best_param = None
        best_param_full = None
        param_obj_list = []
        best_calc_result = None

        for param in param_iterable:
            full_param = default_param.copy()
            full_param.update(param)
            calc_result = self.calc_with_param(
                param=full_param,
                data=data,
                **kwargs)
            obj_value = self.reward.apply(calc_result.data)

            if obj_value > best_obj_value:
                best_obj_value = obj_value
                best_param = param.copy()
                best_param_full = full_param.copy()
                # In order to preserve the data from the optimal case,
                # we make a copy of the data using `.replace` when it is None
                # Note that `replace` is the method to copy data for `dataclasses`
                # Note that the otherwise the data could be over-written
                # duing the next iterations for the optimizer's for loop
                if dataclasses.is_dataclass(calc_result.data):
                    calc_result.data = dataclasses.replace(calc_result.data)
                best_calc_result = calc_result

            param = param.copy()
            # Adds the reward value as a new key to each param
            param.update({"obj_value": obj_value})
            param_obj_list.append(param)

        return {
            "best_param": best_param,
            "best_param_full": best_param_full,
            "best_obj_value": best_obj_value,
            "param_obj_list": param_obj_list,
            "best_calc_result": best_calc_result}

    @abstractmethod
    def calc_with_param(
            self,
            param,
            data=None,
            **kwargs):
        """This is a calculation step which uses both ``data`` and ``param``.
        This is typically expected to be implemented by user.
        By default, it simply returns the data without any alteration.
        However, in general the data will be altered in various ways
        depending on the ``param`` passed.

        Parameters
        ----------
        param : `Any`
            One element of `param_iterable`.
            Typically a dictionary which includes the values of a set of parameters
            given in its keys. However it could also be simply a float if `param_iterable`
            is a list of floats.
        data : `dataclasses.dataclass` or None, default None
            The `data` is typically updated in this function after
            we use the given `param` in the calculation here.
            The data is then returned as a part of returned `CalcResult`.

        Returns
        -------
        calc_result : `greykite.detection.detector.optimizer.CalcResult`
            The optimization results.
        """
        return CalcResult(data=data, model=None)
