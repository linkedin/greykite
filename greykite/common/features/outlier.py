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
"""Basic functionality to identify one-dimensional outliers."""

import warnings
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DetectionResult:
    """This is a dataclass to denote the result of an outlier detection."""
    scores: Optional[pd.Series] = None
    """A series of floats each denoting a score of how much of an outleir a point is.
    The core could be signed for some methods with negative meaning a value is very small and
    the very large for positive."""
    is_outlier: Optional[pd.Series] = None
    """A series of booleans with `True` meaning a point is an outlier and False meaning
    it is not an outlier."""


@dataclass
class DiffMethod:
    """This dataclass is to denote a `diff_method` if a differencing with respect
    to a baseline is needed."""
    name: Optional[str] = None
    """Name of the method."""
    param: Optional[dict] = field(default_factory=dict)
    """Parameters of the method."""


EXPONENTIAL_SMOOTHING_ALPHA = 0.5
"""Default for exponential smoothing `com`.
See the constant below.
"""
EXPONENTIAL_SMOOTHING = DiffMethod(name="es", param={"alpha": EXPONENTIAL_SMOOTHING_ALPHA})
"""This uses exponential smoothing method to calculate a baseline.
This is utilized in `diff_from_baseline` method of `~greykite.common.features.outlier.BaseOutlierDetector`.
See: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
See method details: https://en.wikipedia.org/wiki/Exponential_smoothing
"""

MOVING_MEDIAN_WINDOW = 5
"""The window size for moving aggregation method.
See below."""
MOVING_MEDIAN = DiffMethod(
    name="moving_med",
    param={
        "window": MOVING_MEDIAN_WINDOW,
        "min_periods": 1,
        "center": True})
"""This calculate a moving median as the baseline.
The parameter default are centered window of size 10 and requires only one available data as minimum.
This is utilized in `diff_from_baseline` method of `~greykite.common.features.outlier.BaseOutlierDetector`.
For a longer list of parameters to pass:
See: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
"""


IMPLEMENTED_DIFF_METHODS = ["es", "moving_med"]
"""List of implemented methods for baseline to be used in differencing."""

Z_SCORE_CUTOFF = 5.0
"""Z-score cutoff point."""
IQR_LOWER = 0.1
"""Lower quantile to calculate IQR (Inter-quartile Range).
Note that our default is different from standard."""
IQR_UPPER = 0.9
"""Upper quantile to calculate IQR (Inter-quartile Range).
Note that our default is different from standard."""
TUKEY_CUTOFF = 1.0
"""IQR coefficient to detect outliers.
Note that our default is different from standard.
"""
TRIM_PERCENT = 5.0
"""Default percent for trimming wich is a number between 0 and 100 (typically less then 5)."""


class BaseOutlierDetector:
    """This is the base class for detecting one-dimensional outliers.
        These classes are expected to return (outlier) scores for each point and a
        boolean to express if each point is an outlier or not.

        Additionally, a lower bound (`lower_bound`) and upper bound (`upper_bound`)
        attribute will be available after `fit` to determine the bounds for
        outlier determination.

        This class already implements a few processing steps which are useful across
        various outlier detection methods:

        - `remove_na`: removing NAs
        - `trim`: trimming the data. This is useful for example when using z-score
        - `diff_from_baseline`: fitting a simple baseline and differencing the input from baseline.

    Children of this base class need to implement two abstract methods:

        - `fit_logic`
        - `detect_logic`

    Parameters
    ----------
    trim_percent : `float`, default None
        Trimming percent before calculating the model quantities.
        This removes `trim_percent` of data in symmetric fashion from
        both ends and then it calculates the quantities needed.
        For example in Z-score based method: `ZScoreOutlierDetector` (a child of this class),
        this will remove extreme values before calculating the variance.
    diff_method : `str` or None, default `~greykite.common.features.outlier.MOVING_MEDIAN`
        A simple method to fit a baseline and calculate residuals, then apply
        the approach on the residuals.
        The implemented methods are listed in:
        `~greykite.common.features.outlier.IMPLEMENTED_DIFF_METHODS`

    Attributes
    ----------
    self.lower_bound : `float` or None, default None
        The lower bound for not being outlier which is decided after `self.fit`
        is called.
    self.upper_bound : `float` or None, default None
        The upper bound for not being outlier which is decided after `self.fit`
        is called.
    self.fitted_param : `dict`, default None
        Fitted (method specific) parameters.
        The dictionary keys depend on the model (child of this class).
        This is updated from empty dict after `self.fit` is called.
        Note that `self.fit` calls the abstract method `self.fit_logic`,
        which is implemented by the child class.
    self.y : `pandas.Series` or None, default None
        Input series which is added after `self.fit` is called with data.
    y_diff : `pandas.Series` or None, default None
        Differenced series if a `diff_method` is passed.
    y_na_removed : `pandas.Series` or None, default None
        The vector `y` after removing NAs.
    y_trimmed : `pandas.Series` or None, default None
        The `y` vector after trimming in symmetric fashion using `trim_percent`.
    y_ready_to_fit` : `pandas.Series` or None, default None
        The final vector used in `fit` which essentially finds
        This vector is constructed with the following three steps:

            - (1) differencing if a `diff_method` is passed;
            - (2) removing NAs;
            - (3) trimming.

    self.y_diffed : `pandas.Series` or None, default None
    self.y_na_removed : `pandas.Series` or None, default None
    self.y_trimmed : `pandas.Series` or None, default None
    self.y_ready_to_fit: `pandas.Series` or None, default None
    self.fitted : `~greykite.common.features.outlier.DetectionResult` or None, default None
        Fitted scores and booleans.
        The default for both fields is None, before fit is called.
    self.y_new : `pandas.Series` or None, default None
        This is a series used at `prediction` time.
        Note that in this case prediction means that we want to apply
        the same logic to new data.
        It iw worth noting in most application `fit` is only needed because
        the sole purpose is to do outlier removal.
    self.y_new_ready_to_predict : `pandas.Series` or None, default None
        This is the transformed input for prediction.
        In this case, only differencing might take place, if a `diff_method` is passed.
        Note that there is no need to trim or remove NAs in this case.
    self.predicted : `~greykite.common.features.outlier.DetectionResult` or None, default None
        Predicted scores and booleans, if `self.predict` is called on new data.
    """
    def __init__(
            self,
            trim_percent=TRIM_PERCENT,
            diff_method=MOVING_MEDIAN):
        self.trim_percent = trim_percent
        self.diff_method = diff_method

        # Attributes (See docstring of the class).
        self.lower_bound = None
        self.upper_bound = None

        # Fitted (method specific) parameters.
        # This attibute is updated by the abstract method `self.fit_logic`.
        # That method is the core of the logic for the outlier detection.
        self.fitted_param = {}

        # Input series.
        self.y = None

        # Transformed input series:
        # `y_diff`: differenced series if a `diff_method` is passed.
        # `y_na_removed`: the vector `y` after removing NAs.
        # `y_trimmed`: `y` vector after trimming in symmetric fashion using `trim_percent`.
        # `y_ready_to_fit`: The final vector used in `fit`,
        # which essentially defines
        # the criteria to assign outlier being `True`.
        # This vector is constricted after
        # (1) differencing if a `diff_method` is passed;
        # (2) removing NAs;
        # (3) trimming.
        self.y_diffed = None
        self.y_na_removed = None
        self.y_trimmed = None
        self.y_ready_to_fit = None

        # Fitted scores and booleans.
        self.fitted = DetectionResult(scores=None, is_outlier=None)

        # Prediction related attributes:
        # Prediction time series.
        self.y_new = None
        # This is the transformed input for prediction.
        # In this case, only differencing might take place,
        # if a `diff_method` is passed.
        # Note that there is no need to trim or remove NAs in this case.
        self.y_new_ready_to_predict = None
        # Predicted scores and booleans.
        self.predicted = DetectionResult(scores=None, is_outlier=None)

    @staticmethod
    def remove_na(y):
        """Removes NAs from the input series.
        Also, importantly, will raise error if there are not enough data left (at least 2).

        Parameters
        ----------
        y : `pandas.series`
            A series of floats.

        Returns
        -------
        y_na_removed : `pandas.Series`
            The input series after removing NAs.
        """
        y = pd.Series(y)
        # Removes NAs.
        y_na_removed = y.copy().dropna()
        if len(y_na_removed) < 2:
            raise ValueError(
                f"Length of y after removing NAs is less than 2.\n"
                f"y: {y}\n",
                f"y_na_removed: {y_na_removed}.\n")
        return y_na_removed

    @staticmethod
    def trim(
            y,
            trim_percent=TRIM_PERCENT):
        """This methods performs symmetric trimming from a given series `y`
        in the inputs.
        This means a percent of data from both sides of the
        distribution are cut by calculating a high and low quantile.

        Parameters
        ----------
        y : `pandas.series`
            A series of floats.
        trim_percent : `float`, default `TRIM_PERCENT`.
            Trimming percent for calculating the variance.
            The function first removes this amount of data in symmetric fashion from
            both ends and then it calculates the mean and the variance.

        Returns
        -------
        y_trimmed : `pandas.series`
            A series of floats.
        """
        y_trimmed = y.copy()

        if trim_percent is None or trim_percent == 0:
            return y_trimmed

        if trim_percent < 100 and trim_percent > 0:
            # Calculates half of trimming percent,
            # in order to calculate upper / lower quantiles.
            alpha = 0.5 * (trim_percent / 100.0)
            lower_limit = np.quantile(a=y, q=alpha)
            upper_limit = np.quantile(a=y, q=1 - alpha)
            y_trimmed = [x for x in y if (x <= upper_limit and x >= lower_limit)]
            # If length of trimmed values is less than 2,
            # we revert to the original values.
            if len(y_trimmed) < 2:
                warnings.warn(
                    "After trimming there were less than two values: "
                    "Therefore trimming was disabled."
                    f"\n original y: {y}"
                    f"\n y_trimmed: {y_trimmed}",
                    UserWarning)
                y_trimmed = y
        else:
            raise ValueError(
                f"Trim percent: {trim_percent} needs to be"
                " a value in the interval [0.0, 100.0).")

        return y_trimmed

    @staticmethod
    def diff_from_baseline(y, diff_method):
        """Calculates a baseline for the input series `y` and then removes
        that baseline from `y`, thus creating a diff series.

        Parameters
        ----------
        y : `pandas.series`
            The input series (of floats).
        diff_method : `str`
            A simple method to fit a baseline and calculate residuals, then apply
            the approach on the residuals.
            The implemented methods are listed in:
            `~greykite.common.features.outlier.IMPLEMENTED_DIFF_METHODS`.

        Returns
        -------
        residuals : `pandas.series`
            The series of residuals after applying `diff_method`.
            If `diff_method` is None, then input is not altered.
        """
        y = pd.Series(y).copy()
        name = diff_method.name
        param = diff_method.param

        # Initial values for returned quantities.
        baseline_y = None
        residuals = y.copy()

        if name not in IMPLEMENTED_DIFF_METHODS:
            raise NotImplementedError(f"{name} is not implemented")
        else:
            if name == EXPONENTIAL_SMOOTHING.name:
                if param is None:
                    # `alpha` specifies the coef. in exponential smoothing.
                    # See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html.
                    param = dict(alpha=EXPONENTIAL_SMOOTHING_ALPHA)
                baseline_y = y.ewm(**param).mean()
            elif name == MOVING_MEDIAN.name:
                baseline_y = y.rolling(**param).median()

            # Below we assert that the baseline needs to be of the same size as input.
            assert len(y) == len(baseline_y), "baseline needs to be of the same size as input."
            residuals = y - baseline_y

        return {
            "residuals": residuals,
            "baseline_y": baseline_y}

    @abstractmethod
    def fit_logic(self, y_ready_to_fit):
        """This is an abstract method to be implemented by children of this base class.
        This logic is to be applied to `y_ready_to_fit` and it will update
        `self.fitted_param`.
        All transormations needed to map from `y` to `y_ready_to_fit` are handled
        by this class in the method `fit` below.

        Parameters
        ----------
        y_ready_to_fit : `pandas.series`
            A float series of the (transformed) inputs.

        Returns
        -------
        None. This method updates: `self.fitted_param`.
        """

        # Implement the fit logic here!
        self.fitted_param = None

    @abstractmethod
    def detect_logic(self, y_new):
        """This is an abstract method to be implemented by children of this base class.

        Parameters
        ----------
        y_new : `pandas.series`
            A series of floats. This is the input series to be labeled.

        Returns
        -------
        result :  `~greykite.common.features.outlier.DetectionResult`
            Includes the outlier detection results (see the data class attributes).
        """
        # Implement detect logic here!
        return DetectionResult(scores=None, is_outlier=None)

    def fit(self, y):
        """This method prepares and fits the input data.
        This means that for the input series `y`, it fits the parameters
        which are then used to decide if a point should be an outlier and
        return an outlier score for each point.

        This method first prepares the input series `y` by

            1- differencing from baseline if a `diff_method` is passed.
            2- removing NAs
            3- trimming.

        and then calls the `fit_logic` which is an abstract method to be
        implemented by the users of this class.

        Parameters
        ----------
        y : `pandas.series`
            A series of floats. This is the input series to be labeled.


        Returns
        -------
        None.
        """
        # Initialize all series to be the same as input.
        # We then modify these according to the parameters,
        # For example differencing will only take place if `diff_method` is not None.
        self.y = y.copy()
        self.y_diffed = y.copy()
        self.y_na_removed = y.copy()
        self.y_trimmed = y.copy()
        self.y_ready_to_fit = y.copy()

        # Applies `diff_method` and calculates diffs (residuals).
        if self.diff_method is not None:
            self.y_diffed = self.diff_from_baseline(
                y=self.y,
                diff_method=self.diff_method)["residuals"]

        # Removes NAs.
        self.y_na_removed = self.remove_na(self.y_diffed)

        # Assigns the final vector used for fitting.
        self.y_ready_to_fit = self.y_na_removed.copy()

        if self.trim_percent is not None:
            self.y_ready_to_fit = self.trim(
                y=self.y_ready_to_fit,
                trim_percent=self.trim_percent)
        # Calls the implemented fit logic: `fit_logic`,
        # which is an abstract method to be implemented in child classes.
        # This will update `self.fitted_param`.
        self.fit_logic(y_ready_to_fit=self.y_ready_to_fit)

        # The fitted values must always be obtained by a call to `detect` using original `y`.
        # Note that we pass the original `y` to detect,
        # as we like to get a series of the same size.
        # In other words: there is no need to remove NAs or trim.
        # Also note that the differncing will be handled by the `detect` method.
        self.fitted = self.detect(self.y)

    def detect(self, y_new):
        """This method uses the fitted parameters to decide if a point should
            be labeled as outlier and also provides a score.

            This method performs two steps

                - first does a simple preparation by a call to `diff_from_baseline`;
                - then calls the abstract method: `detect_logic`.

        Parameters
        ----------
        y_new : `pandas.series`
            A series of floats. This is the input series to be labeled.

        Returns
        -------
        result :  `~greykite.common.features.outlier.DetectionResult`
            Includes the outlier detection results (see the data class attributes).

        """
        self.y_new = y_new.copy()
        self.y_new_diffed = y_new.copy()

        if self.diff_method is not None:
            self.y_new_diffed = self.diff_from_baseline(
                y=self.y_new_diffed,
                diff_method=self.diff_method)["residuals"]

        self.predicted = self.detect_logic(self.y_new_diffed)
        return self.predicted


class ZScoreOutlierDetector(BaseOutlierDetector):
    """This is a class for detecting one-dimensional outliers using z-score (based on the normal distribution).
    See https://en.wikipedia.org/wiki/Standard_score as a reference.

    This is a child of `~greykite.common.features.outlier.ZScoreOutlierDetector`,
    which already implements a few processing steps which are useful across
    various outlier detection methods:

        - `remove_na`: removing NAs
        - `trim`: trimming the data. This is useful for example when using z-score
        - `diff_from_baseline`: fitting a simple baseline and differencing the input from baseline.

    For this method:

    - `DetectionResult.scores` are defined as:
        The difference of the value with trimmed mean divided by the trimmed standard deviation
    - `DetectionResult.is_outlier` are defined as: scores which are off by more than `z_score_cutoff`.

    Parameters
    ----------
    z_score_cutoff : `float`, default `Z_SCORE_CUTOFF`
        The normal distribution cut-off used to decide outliers.

    Attribute
    ---------
    fitted_param: `dict` or None, default None
        This is updated after `self.fit` is run on data.
        This dictionary stores:

            - "trimmed_mean": `float`
                Trimmed mean of the input fit data (after trimming).
            - "trimmed_sd": `float`
                Trimmed standard deviation of the input fit data (after trimming).


    Other Parameters and Attributes:
        See `~greykite.common.features.outlier.BaseOutlierDetector` docstring.
    """
    def __init__(
            self,
            z_score_cutoff=Z_SCORE_CUTOFF,
            trim_percent=TRIM_PERCENT,
            diff_method=MOVING_MEDIAN):
        """See class attributes for details on parameters / attributes."""
        self.z_score_cutoff = z_score_cutoff

        super().__init__(
            trim_percent=trim_percent,
            diff_method=diff_method)

    def fit_logic(self, y_ready_to_fit):
        """The logic of the fit.
        This is an abstract method of the base class:
        `~greykite.common.features.outlier.BaseOutlierDetector`
        and it is implemented for z-score here.

        Parameters
        ----------
        y_ready_to_fit: `pandas.series`
            The series which is used for fitting.

        Returns
        -------
        None.

        Updates
        self.fitted_param: `dict`
            Parameters of the z-score model.

                - "trimmed_mean"
                - "trimmed_sd"

        self.lower_bound: `float`
            The lower bound to decide if a point is an outlier.
        self.lower_bound: `float`
            The upper bound to decide is a point is an outlier.

        Indirectly updates (via `self.fit` method of the parent class):

        self.y_diffed: `pandas.Series` or None, default None
        self.y_na_removed: `pandas.Series` or None, default None
        self.y_trimmed: `pandas.Series` or None, default None
        self.y_ready_to_fit: `pandas.Series` or None, default None
        """
        # Calculates z-scores and identifies points with:"
        # Pseudo code: abs(z-score) > `Z_SCORE_CUTOFF` as outliers.
        trimmed_mean = np.mean(y_ready_to_fit)
        trimmed_sd = np.std(y_ready_to_fit)
        self.lower_bound = trimmed_mean - trimmed_sd * self.z_score_cutoff
        self.upper_bound = trimmed_mean + trimmed_sd * self.z_score_cutoff

        self.fitted_param = {
            "trimmed_mean": trimmed_mean,
            "trimmed_sd": trimmed_sd}

    def detect_logic(self, y_new):
        """The logic of outlier detection, after `fit` is done.
        This method uses the fit information to decide which points are
        outliers and what should be their score.

        This is an abstract method of the base class:
        `~greykite.common.features.outlier.BaseOutlierDetector`
        and it is implemented for z-score here.

        For this method: scores are defined as the mean difference divided by the standard deviation
        (as prescribed z-score).

        Parameters
        ----------
        y_new: `pandas.series`
            A series of floats. This is the input series to be labeled.

        Returns
        -------
        result:  `~greykite.common.features.outlier.DetectionResult`
            Includes the outlier detection results (see the data class attributes).
        """
        scores = pd.Series([0] * len(y_new), dtype=float)
        is_outlier = pd.Series([False] * len(y_new))

        # If trimmed sd is zero or not defined, we will not detect any outlier and return.
        if not (self.fitted_param["trimmed_sd"] > 0):
            return DetectionResult(scores=scores, is_outlier=is_outlier)

        scores = (y_new - self.fitted_param["trimmed_mean"]) / self.fitted_param["trimmed_sd"]
        # Boolean series to denote outliers.
        is_outlier = np.abs(scores) > self.z_score_cutoff

        return DetectionResult(scores=scores, is_outlier=is_outlier)


class TukeyOutlierDetector(BaseOutlierDetector):
    """This is a class for detecting one-dimensional outliers.
    This uses the celebrated outlier definition of John Tukey (and named here as such in his recognition):
    Reference: Tukey, J.W., Exploratory data analysis. Addison-Wesley, Reading, 1977

    Note: In Tukey's work:

        - `iqr_lower = 0.25` which is the first quartile
        - `iqr_upper = 0.75` which is the third quartile

    Here we let user specify them and defaults are different.
    Therefore the naming IQR (inter-quartile range)
    is an imperfect naming. However, we think this naming is very widely used and
    it is not worth to come up with new naming.

    For this method:

    - `DetectionResult.scores` are defined as:

        - score is zero if the value is within the IQR
        - score is postive if the value is above the IQR.
            It is the difference from the IQR upper bound devided by IQR length.
        - score is negative if the value is below the IQR.
            It is the difference from the IQR lower bound devided by IQR length.

        Exception: when IQR = 0, we need to handle it.

            - If there is no diff with the corresponding quantile, we set: `score = 0`
            - If there is a diff with corresponding quantile, we set: `score = 2 * tukey_cutoff`
                multiplied by the sign of the difference.

    - `DetectionResult.is_outlier` are defined as: scores which are off by more than `tukey_cutoff`.

    This is a child of `~greykite.common.features.outlier.BaseOutlierDetector`,
    which already implements a few processing steps which are useful across
    various outlier detection methods:

        - `remove_na`: removing NAs
        - `trim`: trimming the data. This is useful for example when using z-score
        - `diff_from_baseline`: fitting a simple baseline and differencing the input from baseline.

    Parameters:
    ----------
    tukey_cutoff : `float`, default `TUKEY_CUTOFF`
        The Tukey cutoff for deciding what is an anomaly.
    iqr_lower : float`, default `IQR_LOWER`
        The smaller quantile used in IQR calculation.
    iqr_upper : float, default `IQR_UPPER`
        The larger quantile used in IQR calculation.

    Attribute
    ---------
    fitted_param: `dict` or None, default None
        This is updated after `self.fit` is run on data.
        This dictionary stores:

            - "quantile_value_lower": `float` or None, default None
                The lower bound of Tukey IQR.
            - "quantile_value_upper": `float` or None, default None
                The upper bound of Tukey IQR.
            - "iqr": `float` or None, default None
                The Tukey IQR (inter-quartile range).

    Other Parameters and Attributes:
        See `~greykite.common.features.outlier.BaseOutlierDetector` docstring.
    """
    def __init__(
            self,
            tukey_cutoff=TUKEY_CUTOFF,
            iqr_lower=IQR_LOWER,
            iqr_upper=IQR_UPPER,
            trim_percent=None,
            diff_method=MOVING_MEDIAN):
        """See class docstring for details on parameters / attributes.

        Note that in this case, the default of `trim_percent` is None, rather than
        `TRIM_PERCENT` used in the base class.
        This is because in this method, trimming is not necessary as the logic
        uses quantiles (which are robust to outliers).
        """
        self.tukey_cutoff = tukey_cutoff
        self.iqr_lower = iqr_lower
        self.iqr_upper = iqr_upper

        super().__init__(
            trim_percent=trim_percent,
            diff_method=diff_method)

    def fit_logic(self, y_ready_to_fit):
        """The logic of the fit.
        This is an abstract method of the base class:
        `~greykite.common.features.outlier.BaseOutlierDetector`
        and it is implemented for Tukey method here.

        Parameters
        ----------
        y_ready_to_fit : `pandas.series`
            A series of floats. This is the input series to be labeled.

        Returns
        -------
        None.

        Updates
        self.fitted_param: `dict` or None, default None
            This is updated after `self.fit` is run on data.
            This dictionary stores:

                - "quantile_value_lower": `float`
                    The lower bound of Tukey IQR.
                - "quantile_value_upper": `float`
                    The upper bound of Tukey IQR.
                - "iqr": `float`
                    The Tukey IQR (inter-quartile range).

        self.lower_bound : `float`
            The lower bound to decide if a point is an outlier.
        self.lower_bound : `float`
            The upper bound to decide is a point is an outlier.

        Indirectly updates (via `self.fit` method of the parent class):

        self.y_diffed : `pandas.Series` or None, default None
        self.y_na_removed : `pandas.Series` or None, default None
        self.y_trimmed : `pandas.Series` or None, default None
        self.y_ready_to_fit : `pandas.Series` or None, default None
        """
        # The value of the distribution at the given quantiles.
        quantile_value_lower = np.quantile(a=y_ready_to_fit, q=self.iqr_lower)
        quantile_value_upper = np.quantile(a=y_ready_to_fit, q=self.iqr_upper)
        # IQR.
        iqr = (quantile_value_upper - quantile_value_lower)
        # Upper and lower bounds after considering the Tukey coef.
        self.lower_bound = quantile_value_lower - (self.tukey_cutoff * iqr)
        self.upper_bound = quantile_value_upper + (self.tukey_cutoff * iqr)

        self.fitted_param = {
            "quantile_value_lower": quantile_value_lower,
            "quantile_value_upper": quantile_value_upper,
            "iqr": iqr
        }

    def detect_logic(self, y_new):
        """The logic of outlier detection, after `fit` is done.
        This method uses the fit information to decide which points are
        outliers and what should be their score.

        This is an abstract method of the base class:
        `~greykite.common.features.outlier.BaseOutlierDetector`
        and it is implemented for Tukey method here.

        For this method:
            - `DetectionResult.scores` are defined as:

                - score is zero if the value is within the IQR
                - score is postive if the value is above the IQR.
                    It is the difference from the IQR upper bound devided by IQR length.
                - score is negative id the value is below the IQR.
                    It is the difference from the IQR lower bound devided by IQR length.

            - `DetectionResult.is_outlier` are defined as: scores which are off by more than `tukey_cutoff`.

        Parameters
        ----------
        y_new : `pandas.series`
            A series of floats. This is the input series to be labeled.

        Returns
        -------
        result : `~greykite.common.features.outlier.DetectionResult`
            Includes the outlier detection results (see the data class attributes).
        """
        scores = pd.Series([0.0] * len(y_new), dtype=float)
        is_outlier = pd.Series([False] * len(y_new))

        # if IQR is zero, we will not detect any outlier and return.
        if self.fitted_param["iqr"] == 0:
            return DetectionResult(scores=scores, is_outlier=is_outlier)

        # First, we find the anchor point for calculating the score.
        # If the point is too large then the upper quantile will be the anchor point.
        # If the point is too small then the lower quantole will be the anchor point.
        for i in range(len(y_new)):
            if y_new[i] < self.fitted_param["quantile_value_lower"]:
                anchor = self.fitted_param["quantile_value_lower"]
                score = (y_new[i] - anchor)
            elif y_new[i] > self.fitted_param["quantile_value_upper"]:
                anchor = self.fitted_param["quantile_value_upper"]
                score = (y_new[i] - anchor)
            else:
                anchor = None
                score = 0

            scores[i] = float(score) / float(self.fitted_param["iqr"])

        is_outlier = np.abs(scores) > self.tukey_cutoff
        return DetectionResult(scores=scores, is_outlier=is_outlier)
