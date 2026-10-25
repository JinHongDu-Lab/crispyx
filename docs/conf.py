"""Sphinx configuration for Streamlined CRISPR documentation."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

project = "crispyx"
version = runpy.run_path(str(SRC_PATH / "crispyx" / "_version.py"))["__version__"]
release = version
copyright = "2026, crispyx Team"
author = "crispyx Team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "nbsphinx",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}
intersphinx_timeout = 10

# -- nbsphinx settings -------------------------------------------------------
nbsphinx_execute = "never"  # use pre-existing cell outputs

exclude_patterns = ["_build", "tutorial_outputs"]

# -- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
html_context = {
    "display_github": True,
    "github_user": "jinhongdu-lab",
    "github_repo": "crispyx",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
