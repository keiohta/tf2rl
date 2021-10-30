from recommonmark.transform import AutoStructify

project = "TF2RL"
author = "Kei Ohta"
copyright = "2020, Kei Ohta"


extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "recommonmark"]
html_theme = "sphinx_rtd_theme"

autoclass_content = "both"
autodoc_member_order = "bysource"


def setup(app):
    app.add_transform(AutoStructify)
