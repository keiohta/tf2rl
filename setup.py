from setuptools import setup, find_packages

install_requires = [
    "cpprb>=8.1.1",
    "setuptools>=41.0.0",
    "numpy>=1.16.0",
    "joblib",
    "scipy"
]

extras_require = {
    "tf": ["tensorflow"],
    "tf_gpu": ["tensorflow-gpu"],
    "examples": ["gym[atari]", "opencv-python"],
    "test": ["coveralls", "gym[atari]", "matplotlib"]
}

setup(
    name="tf2rl",
    version="0.1.15",
    description="Deep Reinforcement Learning for TensorFlow2",
    url="https://github.com/keiohta/tf2rl",
    author="Kei Ohta",
    author_email="dev.ohtakei@gmail.com",
    license="MIT",
    packages=find_packages("."),
    install_requires=install_requires,
    extras_require=extras_require)
