from plotly.colors import DEFAULT_PLOTLY_COLORS

from greykite.common.testing_utils import assert_equal
from greykite.common.viz.colors_utils import get_color_palette


def test_get_color_palette():
    # color palette length is 1
    color_palette = get_color_palette(5, colors=["rgb(99, 114, 218)"])
    assert_equal(color_palette, ["rgb(99, 114, 218)"] * 5)

    # color palette length less than DEFAULT_PLOTLY_COLORS length
    color_palette = get_color_palette(5, colors=DEFAULT_PLOTLY_COLORS)
    assert_equal(color_palette, DEFAULT_PLOTLY_COLORS[0:5])

    # color palette length greater than DEFAULT_PLOTLY_COLORS length
    color_palette = get_color_palette(15, colors=DEFAULT_PLOTLY_COLORS)
    assert len(color_palette) == 15
    for color in color_palette:
        assert color[0:3] == "rgb"

    # custom colors
    colors = ["rgb(99, 114, 218)", "rgb(0, 145, 202)", "rgb(255, 255, 255)"]
    color_palette = get_color_palette(3, colors=colors)
    assert_equal(color_palette, colors)
