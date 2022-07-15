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
# original author: Albert Chen
"""Defines an image scraper for sphinx-gallery
https://sphinx-gallery.github.io/
which can be used to show plotly output in documentation.
Adapted from the functions in plotly v4.6.0 to support:

    1. List of examples_dirs, gallery_dirs
    2. Do not render png thumbnail to avoid dependencies

Compatible with plotly 5.4.0.
"""

import inspect
import os
import shutil
import warnings
from glob import glob
from pathlib import Path

import plotly
from plotly.io import write_html
from plotly.io import write_image
from plotly.io._base_renderers import ExternalRenderer
from plotly.tools import return_figure_from_figure_or_data


class SphinxGalleryRenderer(ExternalRenderer):
    """Original class: `from plotly.io._base_renderers import SphinxGalleryRenderer`
    Modified to add `render_png` parameter.
    """
    def render(self, fig_dict):
        """Called by plotly.io.show"""
        stack = inspect.stack()
        # Name of .py example script from which plot function was called
        try:
            filename = stack[3].filename  # let's hope this is robust...
        except (IndexError, AttributeError):  # python 2
            filename = stack[3][1]
        # NB: this works for multiple plots in the same file
        # It appears they are executed sequentially
        filename_root, _ = os.path.splitext(filename)
        filename_html = filename_root + ".html"
        filename_png = filename_root + ".png"
        _ = write_html(fig_dict, file=filename_html)

        # Whether to render the html image as a png for the thumbnail
        render_png = False
        if render_png:
            # Requires plotly-orca and xfvb
            # https://github.com/plotly/orca
            plotly.io.orca.config.use_xvfb = True
            figure = return_figure_from_figure_or_data(fig_dict, True)
            write_image(figure, filename_png)
        else:
            # The thumbnail isn't important, so we use a default image
            # Assumes the default image is one level above the .py file
            try:
                filename_default_image = Path(filename_root).parents[1].joinpath('default_thumb.png')
                if not os.path.exists(filename_default_image):
                    # the .py file is in a subfolder (sub-gallery)
                    filename_default_image = Path(filename_root).parents[2].joinpath('default_thumb.png')
                shutil.copyfile(filename_default_image, filename_png)
            except FileNotFoundError:
                warnings.warn(f"Thumbnail not found! Expected default_thumb.png to be "
                              f"in the parent folder of filename ({filename}).")


# Original code: https://github.com/plotly/plotly.py/issues/1459
# Register for use by name
sphinx_renderer = SphinxGalleryRenderer()
plotly.io.renderers['sphinx_gallery'] = sphinx_renderer

# Set as default
plotly.io.renderers.default = 'sphinx_gallery'


def plotly_sg_scraper(block, block_vars, gallery_conf, **kwargs):
    """Scrape Plotly figures for galleries of examples using
    sphinx-gallery.
    Examples should use ``plotly.io.show()`` to display the figure with
    the custom sphinx_gallery renderer.
    Since the sphinx_gallery renderer generates both html and static png
    files, we simply crawl these files and give them the appropriate path.

    Original function:
        from plotly.io._sg_scraper import plotly_sg_scraper
    Which is based on:
        https://sphinx-gallery.github.io/stable/advanced.html#example-2-detecting-image-files-on-disk
    Modified to handle the case where examples_dirs is a list of more than one item.
        Original code failed to render plots for multiple directories:
            - Only the rendered the first directory
            - If multiple directories were specified, matplotlib plots in first directory were not rendered

    Parameters
    ----------
    block : tuple
        A tuple containing the (label, content, line_number) of the block.
    block_vars : dict
        Dict of block variables.
    gallery_conf : dict
        Contains the configuration of Sphinx-Gallery
    **kwargs : dict
        Additional keyword arguments to pass to
        :meth:`~matplotlib.figure.Figure.savefig`, e.g. ``format='svg'``.
        The ``format`` kwarg in particular is used to set the file extension
        of the output file (currently only 'png' and 'svg' are supported).
    Returns
    -------
    rst : str
        The ReSTructuredText that will be rendered to HTML containing
        the images.
    Notes
    -----
    Add this function to the image scrapers
    """
    examples_dir = os.path.dirname(block_vars["src_file"])
    pngs = sorted(glob(os.path.join(examples_dir, "*.png")))
    htmls = sorted(glob(os.path.join(examples_dir, "*.html")))
    image_path_iterator = block_vars["image_path_iterator"]
    image_names = list()
    seen = set()
    for html, png in zip(htmls, pngs):
        if png not in seen:
            seen |= set(png)
            # the incrementor simply increments the filename as long as html/png are found
            # _001.png, _002.png, _003.png, etc.
            this_image_path_png = next(image_path_iterator)
            this_image_path_html = os.path.splitext(this_image_path_png)[0] + ".html"
            image_names.append(this_image_path_html)
            shutil.move(png, this_image_path_png)
            shutil.move(html, this_image_path_html)
    # Use the `figure_rst` helper function to generate rST for image files
    return figure_rst(image_names, gallery_conf["src_dir"])


def figure_rst(figure_list, sources_dir):
    """Generate RST for a list of image filenames.
    Depending on whether we have one or more figures, we use a
    single rst call to 'image' or a horizontal list.

    Original functions:
        from sphinx_gallery.gen_rst import figure_rst
        from plotly.io._sg_scraper import figure_rst
    Modified rST directives from image to raw html

    Parameters
    ----------
    figure_list : list
        List of strings of the figures' absolute paths.
    sources_dir : str
        absolute path of Sphinx documentation sources
    Returns
    -------
    images_rst : str
        rst code to embed the images in the document
    """
    # figure_paths = [os.path.relpath(figure_path, sources_dir)
    #                 .replace(os.sep, '/').lstrip('/')
    #                 for figure_path in figure_list]
    figure_paths = figure_list
    images_rst = ""
    if len(figure_paths) == 1:
        figure_name = figure_paths[0]
        images_rst = SINGLE_HTML % figure_name
    elif len(figure_paths) > 1:
        images_rst = HLIST_HEADER
        for figure_name in figure_paths:
            images_rst += HLIST_HTML_TEMPLATE % figure_name

    return images_rst


# The following strings are used when we have several pictures: we use
# an html div tag that our CSS uses to turn the lists into horizontal
# lists.
HLIST_HEADER = """"""

HLIST_HTML_TEMPLATE = """
    *
        .. raw:: html
            :file: %s
"""

SINGLE_HTML = """
.. raw:: html
    :file: %s
"""
