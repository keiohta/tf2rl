from setuptools import setup, Extension, find_packages

setup(name="tf2rl",
      install_requires=["cpprb", "tensorflow"],
      url="https://github.com/keiohta/tf2rl",
      packages=find_packages("."),
      extras_require={"examples": ["gym", "roboschool"]})
