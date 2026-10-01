# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, resolve it relative to this configuration file.
#
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_ext'))


# -- Project information -----------------------------------------------------

project = 'pyBaram'
copyright = '2022-2026, AADL, Inha University'
author = 'Jin Seok Park'

# The full version, including alpha/beta/rc tags
import pybaram._version as _version
version = _version.__version__
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'autodoc2',
    'autodoc2_context',
    'sphinx.ext.imgmath',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'sphinx_togglebutton',
]

# MyST pages and static API analysis.
source_suffix = {'.md': 'markdown'}
myst_enable_extensions = ['colon_fence', 'dollarmath', 'substitution', 'deflist']
myst_substitutions = {'version': version, 'release': release}

# Analyse source without importing solver dependencies. Keep the curated API
# sections in the guides rather than generating pages for every module.
autodoc2_packages = [{'path': '../../pybaram', 'auto_mode': False}]
autodoc2_render_plugin = 'myst'
# Existing Python docstrings use reStructuredText; new pages use MyST.
autodoc2_docstring_parser_regexes = [(r'.*', 'rst')]
autodoc2_hidden_objects = ['dunder', 'private', 'inherited']
autodoc2_docstrings = 'all'
autodoc2_class_docstring = 'both'
autodoc2_module_summary = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

inheritance_graph_attrs = dict(
    center='true',
    layout='dot',
    rankdir='LR',
    ranksep='0.05',
    splines='ortho',
    ratio='compress')
inheritance_node_attrs = dict(
    fontsize='6.5',
    penwidth='0.3')
inheritance_edge_attrs = dict(
    penwidth='0.3')

# Inheritance diagrams and API base lists reference internal base classes, including
# internal classes that intentionally do not have standalone API entries.
nitpick_ignore_regex = [
    ('py:class', r'pybaram\.(?:integrators|solvers|plugins)\..*'),
    ('py:obj', r'pybaram\.(?:integrators|solvers|plugins)\..*'),
]

graphviz_output_format = 'svg'

# Document classes at their defining modules so imported aliases do not
# overwrite the source viewer's links back to the API reference.
viewcode_follow_imported_members = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_title = 'pyBaram {} (seawhanlee fork)'.format(release)
html_baseurl = 'https://seawhanlee.github.io/pyBaram/'
html_context = {
    'github_user': 'seawhanlee',
    'github_repo': 'pyBaram',
    'github_version': 'main',
    'doc_path': 'doc/src',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_theme_options = {
    'use_edit_page_button': True,
    'github_url': 'https://github.com/seawhanlee/pyBaram',
}
html_css_files = ['css/custom.css']

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
'preamble': r'\setcounter{secnumdepth}{4}',
'makeindex': '',
'printindex': '',
}

latex_domain_indices = False

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'pyBaram.tex', u'pyBaram Documentation',
   u'Inha University', 'manual'),
]

# -- Options for Bibliography output --------------------------------------------

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
