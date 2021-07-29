============
Contributing
============

Thank you for being a Greykite user and taking the time to contribute. The contribution to Greykite simply consists of
two steps

* Forking the repo
* Submitting a pull request with your change.

Please carefully read this documentation before making contributions, thank you!

Contribution Agreement
----------------------

As a contributor, you represent that the code you submit is your original work or that of your employer
(in which case you represent you have the right to bind your employer).
By submitting code, you (and, if applicable, your employer) are licensing the submitted code to LinkedIn
and the open source community subject to the BSD 2-Clause license.

Contribution Guidelines
-----------------------

Code of Conduct
^^^^^^^^^^^^^^^

Please follow our `Code of Conduct <https://github.com/linkedin/greykite/blob/master/CODE_OF_CONDUCT.rst>`_ when making contributions.

Code Style
^^^^^^^^^^

Greykite follows the `flake8 <https://flake8.pycqa.org/en/latest/>`_ code style.
Contributed code must pass flake8 check before it can be merged.

Greykite follows the `numpydoc guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstrings.
For more details, you can also read `guide from lsst <https://developer.lsst.io/python/numpydoc.html>`_.

Greykite uses `sphinx <https://www.sphinx-doc.org/en/master/>`_
and `sphinx-gallery <https://sphinx-gallery.github.io/stable/index.html>`_ to generate documentation pages.
Please make sure the content are well rendered if you contribute to the documentation.

Please make sure the code/docstring styles are the same as the other code in the same file/module before creating a pull request.

Contribution Steps
^^^^^^^^^^^^^^^^^^

Before contributing
"""""""""""""""""""

* Read everything in this contributing guide.
* Open an issue with the Greykite team about the change you would like to contribute. Large features which have never been discussed are unlikely to be accepted.

Start contributing
""""""""""""""""""

* Fork the repo and clone it to local environment. For details, see `instructions on Github <https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_.
* Create a new virtual-environment with your favorite python version (we recommend python3.7 and python3.8)

  .. code-block::

    {your-python} -m venv greykite-env
    source greykite-env/bin/activate

* Install the dev environment.

  .. code-block::

    pip install -r requirements-dev.txt

* Make your changes. You may want to do it on a separate branch to avoid making master messy.

Makes sure your code works
""""""""""""""""""""""""""

* Make sure the flake8 style check passes.

  .. code-block::

    make flake8

* If you fixed a bug, please write a unit test that indicates the fix. Put the tests in the corresponding directories in greykite/tests. After you write the tests, make sure all tests pass by running

  .. code-block::

    make test

  Please keep a screenshot or a text file of the testing results.

* If you developed a new feature, please write unit tests that covers all code in the feature. Put the tests in the corresponding directories in greykite/tests. After you write the tests, make sure all tests pass and show coverage by running

  .. code-block::

    make coverage

  Please keep a screenshot or a text file of the testing results. Please also document the new feature you developed by writing either a quickstart example (indicates the functionalities of the feature) or a tutorial example (an end-to-end example including the feature as core functionality).

* If you contributed to the documentation, please make sure the sphinx-build completes without any error

  .. code-block::

    make docs

  and that the generated documentation pages look good. The generated docs are in docs/build/docs/greykite/html. Please take a full screenshot of the generated page with your browser.

* Currently we do not allow modifying dependencies for stability reasons.

Ready to submit
"""""""""""""""

* After all steps above are finished, you are ready to submit a pull request (PR).
* Please squash or rebase your changes into a single commit.
* Push your code to your cloned branch and submit a PR.
* Please include any screenshots you have from the previous steps, as well as a screenshot indicating the new feature (notebook, plots, etc.).
