import pytest
from plotly.colors import DEFAULT_PLOTLY_COLORS

from greykite.common.testing_utils import assert_equal
from greykite.common.viz.colors_utils import get_color_palette
from greykite.common.viz.colors_utils import get_distinct_colors


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


def test_get_distinct_colors():
    """Tests the function to get most distinguishable colors."""
    # Under 10 colors, using tab10.
    assert get_distinct_colors(num_colors=1) == get_distinct_colors(num_colors=2)[:1]

    assert get_distinct_colors(num_colors=10) == [
        "rgba(31, 119, 180, 0.95)",
        "rgba(255, 127, 14, 0.95)",
        "rgba(44, 160, 44, 0.95)",
        "rgba(214, 39, 40, 0.95)",
        "rgba(148, 103, 189, 0.95)",
        "rgba(140, 86, 75, 0.95)",
        "rgba(227, 119, 194, 0.95)",
        "rgba(127, 127, 127, 0.95)",
        "rgba(188, 189, 34, 0.95)",
        "rgba(23, 190, 207, 0.95)"
    ]
    # Under 20 colors, using tab20.
    assert get_distinct_colors(num_colors=15) == get_distinct_colors(num_colors=18)[:15]

    assert get_distinct_colors(num_colors=20, opacity=0.9) == [
        "rgba(31, 119, 180, 0.9)",
        "rgba(174, 199, 232, 0.9)",
        "rgba(255, 127, 14, 0.9)",
        "rgba(255, 187, 120, 0.9)",
        "rgba(44, 160, 44, 0.9)",
        "rgba(152, 223, 138, 0.9)",
        "rgba(214, 39, 40, 0.9)",
        "rgba(255, 152, 150, 0.9)",
        "rgba(148, 103, 189, 0.9)",
        "rgba(197, 176, 213, 0.9)",
        "rgba(140, 86, 75, 0.9)",
        "rgba(196, 156, 148, 0.9)",
        "rgba(227, 119, 194, 0.9)",
        "rgba(247, 182, 210, 0.9)",
        "rgba(127, 127, 127, 0.9)",
        "rgba(199, 199, 199, 0.9)",
        "rgba(188, 189, 34, 0.9)",
        "rgba(219, 219, 141, 0.9)",
        "rgba(23, 190, 207, 0.9)",
        "rgba(158, 218, 229, 0.9)"
    ]

    # Under 256 colors, using Viridis.
    assert get_distinct_colors(num_colors=30, opacity=0.85) == [
        "rgba(68, 1, 84, 0.85)",
        "rgba(70, 12, 95, 0.85)",
        "rgba(72, 25, 107, 0.85)",
        "rgba(71, 37, 117, 0.85)",
        "rgba(70, 48, 125, 0.85)",
        "rgba(67, 59, 131, 0.85)",
        "rgba(64, 68, 135, 0.85)",
        "rgba(60, 78, 138, 0.85)",
        "rgba(55, 88, 140, 0.85)",
        "rgba(51, 97, 141, 0.85)",
        "rgba(47, 106, 141, 0.85)",
        "rgba(43, 115, 142, 0.85)",
        "rgba(40, 122, 142, 0.85)",
        "rgba(37, 131, 141, 0.85)",
        "rgba(34, 139, 141, 0.85)",
        "rgba(31, 148, 139, 0.85)",
        "rgba(30, 156, 137, 0.85)",
        "rgba(32, 165, 133, 0.85)",
        "rgba(38, 172, 129, 0.85)",
        "rgba(48, 180, 122, 0.85)",
        "rgba(62, 188, 115, 0.85)",
        "rgba(79, 195, 105, 0.85)",
        "rgba(98, 202, 95, 0.85)",
        "rgba(119, 208, 82, 0.85)",
        "rgba(139, 213, 70, 0.85)",
        "rgba(162, 218, 55, 0.85)",
        "rgba(186, 222, 39, 0.85)",
        "rgba(210, 225, 27, 0.85)",
        "rgba(233, 228, 25, 0.85)",
        "rgba(253, 231, 36, 0.85)"
    ]
    # Can't get more than 256 colors.
    with pytest.raises(
            ValueError,
            match="The maximum number of colors is 256."):
        get_distinct_colors(num_colors=257)

    # Opacity must be between 0 and 1
    with pytest.raises(
            ValueError,
            match="Opacity must be between 0 and 1."):
        get_distinct_colors(num_colors=2, opacity=-1)
