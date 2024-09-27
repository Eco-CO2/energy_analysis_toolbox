# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import energy_toolbox

project = "energy_toolbox"
copyright = "2024, energy_toolbox developers"
author = "energy_toolbox developers"
version = energy_toolbox.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# define code executed for each file as if included in it
rst_prolog = """
.. |et| replace:: ``energy_toolbox``
.. |None| replace:: ``None``
.. |NaN| replace:: ``NaN``

.. |flow_rate_conservative| replace:: :py:func:`flow_rate_conservative <cenergy_toolbox.timeseries.power.conservative.flow_rate_conservative`
.. |volume_conservative| replace:: :py:func:`volume_conservative <energy_toolbox.timeseries.power.conservative.volume_conservative`

.. |ETK.start_f| replace:: :py:const:`ETK.start_f <energy_toolbox.keywords.start_f>`
.. |ETK.end_f| replace:: :py:const:`ETK.end_f <energy_toolbox.keywords.end_f>`
.. |ETK.time_f| replace:: :py:const:`ETK.time_f <energy_toolbox.keywords.time_f>`

"""


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "gitlab_url": "https://gitlab.ecoco2.com/recherche/fret21_indicators",
    "logo": {"image_dark": "./_static/logo.png"},
}
html_static_path = ["_static"]
html_sidebars = {
    "using/windows": ["windows-sidebar.html"],
}
html_logo = "./_static/logo.png"
html_favicon = "./_static/favicon.ico"
add_function_parentheses = False

html_title = f"{project} v{version} Manual"
html_context = {
    "version": version,  # Pass the version to the HTML context for JS usage
}
# Custom CSS and JS
html_css_files = [
    "css/custom.css",
]
# -- nbsphinx extension configuration ------------------------------------------
nbsphinx_execute = (
    "never"  # never, always, or auto (only run if no output available)
)

# -- Extension configuration -------------------------------------------------
napoleon_use_rtype = False  # move return type inline
