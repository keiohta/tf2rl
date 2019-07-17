from setuptools import setup, Extension, find_packages

install_requires = [
    "cpprb>=7.13.3,<8.0.0"
    "joblib",
    "scipy"
]

extras_require = {
    "tf": ["tensorflow==2.0.0b0"],
    "tf_gpu": ["tensorflow-gpu==2.0.0b0"],
    "examples": ["gym", "gym[atari]", "roboschool", "opencv-python"]
}

setup(
    name="tf2rl",
    version="0.1.2",
    description="Deep Reinforcement Learning for TensorFlow2.0",
    url="https://github.com/keiohta/tf2rl",
    author="Kei Ohta",
    author_email="dev.ohtakei@gmail.com",
    license="MIT",
    packages=find_packages("."),
    install_requires=install_requires,
    extras_require=extras_require,
    test_suite='tests')
