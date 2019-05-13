from setuptools import setup, Extension, find_packages

install_requires = [
    "cpprb"
]

extras_require = {
    "tf": ["tensorflow==1.12"],
    "tf_gpu": ["tensorflow-gpu==1.12"],
    "examples": ["gym", "gym[atari]", "roboschool", "opencv-python"]
}

setup(
    name="tf2rl",
    description="Deep Reinforcement Learning for TensorFlow2.0",
    url="https://github.com/keiohta/tf2rl",
    author="Kei Ohta",
    author_email="dev.ohtakei@gmail.com",
    license="MIT",
    packages=find_packages("."),
    install_requires=install_requires,
    extras_require=extras_require)
