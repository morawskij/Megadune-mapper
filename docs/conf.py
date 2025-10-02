# conf.py
# Configuration file for the Sphinx documentation builder

import os
import sys

# Add the repo root to sys.path so autodoc can find your modules
sys.path.insert(0, os.path.abspath(".."))  # since docs are in the root, .. is repo root

# -- Project information -----------------------------------------------------
project = "Megadune mapper"
author = "Jakub Morawski"
release = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",      # pull in docstrings
    "sphinx.ext.napoleon",     # support Google / NumPy style docstrings
    "sphinx.ext.autosummary",
    "myst_parser",             # Markdown support
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]