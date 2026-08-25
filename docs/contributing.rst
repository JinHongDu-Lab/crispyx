Contributing
============

Contributions are welcome! This page covers how to set up a development
environment and submit changes.

Development setup
-----------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/<your-user>/crispyx.git
      cd crispyx

2. Create a virtual environment and install in editable mode with test
   dependencies:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate
      pip install -e ".[test,docs]"

3. Verify the test suite passes:

   .. code-block:: bash

      pytest

Running tests
-------------

.. code-block:: bash

   # Run the full suite
   pytest

   # Run a specific test file
   pytest tests/test_workflow.py

   # Run tests with verbose output
   pytest -v

Building documentation
----------------------

.. code-block:: bash

   # Install doc dependencies
   pip install -e ".[docs]"

   # Build HTML docs
   sphinx-build docs docs/_build/html

   # Open in browser
   open docs/_build/html/index.html    # macOS
   xdg-open docs/_build/html/index.html  # Linux

Code style
----------

* Follow PEP 8 conventions.
* Use NumPy-style docstrings for all public functions.
* Type annotations are encouraged for function signatures.
* Keep imports sorted (stdlib → third-party → local).

Submitting changes
------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/my-improvement

2. Make your changes and add tests if applicable.
3. Ensure all tests pass and the documentation builds without errors.
4. Open a pull request against the ``main`` branch.

Reporting issues
----------------

Please use the `GitHub issue tracker
<https://github.com/jinhongdu-lab/crispyx/issues>`_ to report bugs or request
features. Include:

* A minimal reproducible example
* Python version and OS
* crispyx version (``cx.__version__``)
* Full traceback for errors
