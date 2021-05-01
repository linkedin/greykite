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
"""Pandas dataframe utils for uncertainty interval calculation."""


def offset_tuple_col(df, offset_col, tuple_col):
    """Takes a dataframe (df) with at least two columns
    (`offset_col` and `tuple_col`)
    and offsets all values given in tuple_col using an offset

    Parameters
    ----------
    df: `pd.DataFrame`
        input dataframe which includes desired values
    offset_col: `str`
        The column with the offset value
    tuple_col: `str`
        The column with tuples including real (float/double/int) numbers

    Returns
    -------
    df_offset : `pd.Dataframe`
        A dataframe which is same as ``df`` and only the
        ``tuple_col`` is altered by adding an offset to
        all values in each tuple.
    """
    return df.apply(
        lambda row: tuple(
            e + row[offset_col] for e in row[tuple_col]),
        axis=1)


def limit_tuple_col(df, tuple_col, lower=None, upper=None):
    """This function applies limits (lower bound and upper bound)
    to values (could be any value type for which min and max are defined)
    given in tuples appearing in `tuple_col` column of dataframe `df`.

    Parameters
    ----------
    df: `pd.DataFrame`
        Input DataFrame
    tuple_col: `str`
        The column of interest consisting of tuples
    lower: `int` or `float` or `double` or `str`
        Any type for which order "<" is defined.
        This input has the same type of values in the tuples,
        and denotes the desired lower bound
    upper: `int` or `float` or `double` or `str`
        Any type for which order "<" is defined.
        This input has the same type of values in the tuples,
        and denotes the desired upper bound

    Returns
    -------
    df : `pd.Dataframe`
        Dataframe of same shape as `df` with only values in `tuple_col` altered
        to satisfy the bounds: `lower`, `upper`
    """
    if lower is not None:
        df[tuple_col] = df.apply(
            lambda row: tuple(
                max(e, lower) for e in row[tuple_col]),
            axis=1)

    if upper is not None:
        df[tuple_col] = df.apply(
            lambda row: tuple(
                min(e, upper) for e in row[tuple_col]),
            axis=1)
    return df
