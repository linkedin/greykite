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
"""Color palette for plotting."""

from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.colors import n_colors
from plotly.colors import validate_colors


def get_color_palette(num, colors=DEFAULT_PLOTLY_COLORS):
    """Returns ``num`` of distinct RGB colors.
    If ``num`` is less than or equal to the length of ``colors``, first ``num``
    elements of ``colors`` are returned.
    Else ``num`` elements of colors are interpolated between the first and the last
    colors of ``colors``.

    Parameters
    ----------
    num : `int`
        Number of colors required.
    colors : [`str`, `list` [`str`]], default ``DEFAULT_PLOTLY_COLORS``
        Which colors to use to build the color palette.
        This can be a list of RGB colors or a `str` from ``PLOTLY_SCALES``.

    Returns
    -------
    color_palette: List
        A list consisting ``num`` of RGB colors.
    """
    validate_colors(colors, colortype="rgb")
    if len(colors) == 1:
        return colors * num
    elif len(colors) >= num:
        color_palette = colors[0:num]
    else:
        color_palette = n_colors(
            colors[0],
            colors[-1],
            num,
            colortype="rgb")
    return color_palette
