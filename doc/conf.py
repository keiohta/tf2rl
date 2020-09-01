from recommonmark.transform import AutoStructify

project = "TF2RL"
author = "Kei Ohta"
copyright = "2020, Kei Ohta"


extensions = ["sphinx.ext.autodoc", "recommonmark"]
html_theme = "sphinx_rtd_theme"

def setup(app):
    app.add_transform(AutoStructify)
