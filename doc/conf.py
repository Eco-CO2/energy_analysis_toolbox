# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import energy_toolbox
project = 'energy_toolbox'
copyright = '2023, R&D @ Eco CO2'
author = 'R&D @ Eco CO2'
version = energy_toolbox.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# define code executed for each file as if included in it
rst_prolog="""
.. |et| replace:: ``energy_toolbox``
.. |None| replace:: ``None``
.. |NaN| replace:: ``NaN``

.. |flow_rate_conservative| replace:: :py:func:`flow_rate_conservative <cenergy_toolbox.timeseries.power.conservative.flow_rate_conservative`
.. |volume_conservative| replace:: :py:func:`volume_conservative <energy_toolbox.timeseries.power.conservative.volume_conservative`

.. |ETK.start_f| replace:: :py:const:`ETK.start_f <energy_toolbox.keywords.start_f>`
.. |ETK.end_f| replace:: :py:const:`ETK.end_f <energy_toolbox.keywords.end_f>`
.. |ETK.time_f| replace:: :py:const:`ETK.time_f <energy_toolbox.keywords.time_f>`

"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "gitlab_url": "https://gitlab.ecoco2.com/recherche/energy_toolbox",
    "announcement": f"<p>Last version is v{version}</p>",
}
html_static_path = ['_static']
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", 'globaltoc.html',]
}
add_function_parentheses = False

html_title = f"{project} v{version} Manual"

# -- nbsphinx extension configuration ------------------------------------------
nbsphinx_execute = 'never'  # never, always, or auto (only run if no output available)

# -- Extension configuration -------------------------------------------------
napoleon_use_rtype = False  # move return type inline
