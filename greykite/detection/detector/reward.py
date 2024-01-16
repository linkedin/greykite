# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

import inspect
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
import numbers

from greykite.detection.detector.constants import PenalizeMethod


class Reward:
    """Reward class which is to support very flexible set of rewards
    used in optimization where an objective is to be optimized.

    The main method for this class is `apply` which
    is used when the reward is to be applied to data.
    No assumption is made on the arguments of `apply` to keep this class very
    generic.

    This class enables two powerful mechanisms:

    - taking a simple `reward_func` and construct a penalized version of that
    - starting from existing objectives building more complex ones by adding /
    multiplying / dividing them or use same operations with numbers.

    Using these two mechanisms can enable use to support multi-objective problems.
    For examples, in the context of anomaly detection if recall is to be optimized
    subject to precision being at least 80 percent, then use can enable that by

    def recall(y_true, y_pred):
        # recall function implementation
        ...

    def precision(y_true, y_pred):
        # precision function implementation
        ...

    reward =
        Reward(reward_func=recall) +
        Reward(
            reward_func=precision,
            min_unpenalized_metric=0.8,
            max_unpenalized_metric=None,
            penalty=-inf)

    where the second part will cause the total sum of the objectives to be -inf
    whenever precision is not in the desired range.
    Also note that the "+" operation is defined in this class using the dunder
    method `__add__`.

    One can also combine objectives to achieve more complex objectives from existing ones. For example F1 can be easily expressed in terms of
    precision and recall:

        rec_obj = Reward(reward_func=recall)
        prec_obj =Reward(reward_func=precision)
        f1_obj = (2 * rec_obj * prec_obj) / (rec_obj + prec_obj)

    The penalize mechanism on its own is useful for example in the context of
    anomaly detection without labels, where we only have an idea about the
    expected anomaly percentage in the data. In such a case an objective can be
    constructed for optimization. See
        `~greykite.detection.detector.detector.Detector`
    init to see a construction of such an objective.


    Parameters
    ----------
    reward_func : callable
        The reward function which will be used as the staring point and augmented
        with extra logic depending on other input.
    min_unpenalized : `float`, default `float("-inf")`
        The minimum value of the reward function (`reward_func`) which will
        remain un-penalized.
    max_unpenalized : `float`, default `float("inf")`
        The maximum value of the reward function (`reward_func`) which will
        remain un-penalized.
    penalize_method : `str`, default `PenalizeMethod.ADDITIVE.value`
        The logic of using the penalty. The possibilities are given in
        `~greykite.detection.detector.constants.PenalizeMethod`
    penalty : `float` or None, default None
        The penalty amount. If None, it will be mapped to 0 for additive and
        1 for multiplicative.

    Attributes
    ----------
    None
    """

    def __init__(
            self,
            reward_func,
            min_unpenalized=float("-inf"),
            max_unpenalized=float("inf"),
            penalize_method=PenalizeMethod.ADDITIVE.value,
            penalty=None):
        self.reward_func = reward_func
        self.min_unpenalized = min_unpenalized
        self.max_unpenalized = max_unpenalized
        self.penalize_method = penalize_method
        if penalty is None:
            if penalize_method == PenalizeMethod.ADDITIVE.value:
                penalty = 0
            else:
                penalty = 1
        self.penalty = penalty

    def apply(
            self,
            *args,
            **kwargs):

        obj_value = self.reward_func(
            *args,
            **kwargs)

        if (
                obj_value > self.max_unpenalized or
                obj_value < self.min_unpenalized):

            if self.penalize_method == PenalizeMethod.ADDITIVE.value:
                obj_value += self.penalty
            elif self.penalize_method == PenalizeMethod.MULTIPLICATIVE.value:
                obj_value *= self.penalty
            elif self.penalize_method == PenalizeMethod.PENALTY_ONLY.value:
                obj_value = self.penalty
            elif self.penalize_method is None:
                obj_value = self.penalty
            else:
                raise ValueError(
                    f"penalize_method {self.penalize_method.value} does not exist")

        return obj_value

    def __add__(self, other):
        """Addition of objects or an object with a number (scalar)."""
        if isinstance(other, numbers.Number):
            def reward_func(*args, **kwargs):
                return (
                    self.apply(*args, **kwargs) +
                    other)
        else:
            def reward_func(*args, **kwargs):
                return (
                    self.apply(*args, **kwargs) +
                    other.apply(*args, **kwargs))

        return Reward(reward_func=reward_func)

    def __mul__(self, other):
        """Multiplication of objects or an object with a number."""
        if isinstance(other, numbers.Number):
            def reward_func(*args, **kwargs):
                return (
                    self.apply(*args, **kwargs) *
                    other)
        else:
            def reward_func(*args, **kwargs):
                return (
                    self.apply(*args, **kwargs) *
                    other.apply(*args, **kwargs))

        return Reward(reward_func=reward_func)

    def __truediv__(self, other):
        """Division of objects or an object with a number."""
        if isinstance(other, numbers.Number):
            def reward_func(*args, **kwargs):
                return (
                    self.apply(*args, **kwargs) /
                    other)
        else:
            def reward_func(*args, **kwargs):
                return (
                    self.apply(*args, **kwargs) /
                    other.apply(*args, **kwargs))

        return Reward(reward_func=reward_func)

    # Below defines the above operators from right
    # Addition and multiplication operators are commutative
    # Division is an exception
    def __radd__(self, other):
        """Right addition."""
        return self.__add__(other)

    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """Right division."""
        if isinstance(other, numbers.Number):
            def reward_func(*args, **kwargs):
                return (
                    other /
                    self.apply(*args, **kwargs))
        else:
            def reward_func(*args, **kwargs):
                return (
                    other.apply(*args, **kwargs) /
                    self.apply(*args, **kwargs))

        return Reward(reward_func=reward_func)

    def __str__(self):
        """Print method."""
        reward_func_content = inspect.getsource(self.reward_func)
        return (
            f"\n reward_func:\n {reward_func_content} \n"
            f"min_unpenalized: {self.min_unpenalized} \n"
            f"max_unpenalized: {self.max_unpenalized} \n"
            f"penalize_method: {self.penalize_method} \n"
            f"penalty: {self.penalty} \n")
