# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SpacetimeLib'
copyright = '2022-2023, Kurt Mohler'
author = 'Kurt Mohler'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'




# -- Pull in public methods and `__init__` func docstrings for autoclasses. --
# Doesn't pull in the other dunder methods, so they need to be added in doc source
autoclass_content = 'both'


# -- Options for myst_nb -----------------------------------------------------
# Make docs build fail if there's an exception in a notebook
nb_execution_raise_on_error=True
