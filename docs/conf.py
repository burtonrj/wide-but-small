"""Sphinx configuration."""
from datetime import datetime


project = "wide-but-small"
author = "Ross Burton"
copyright = f"{datetime.now().year}, {author}"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
autodoc_typehints = "description"
