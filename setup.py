from setuptools import setup, find_packages

tf_version = "2.3"  # Default Version
compatible_tfp = {"2.3": ["tensorflow~=2.3.0",
                          "tensorflow-probability~=0.11.0"],
                  "2.2": ["tensorflow-probability~=0.10.0"],
                  "2.1": ["tensorflow-probability~=0.8.0"],
                  "2.0": ["tensorflow-probability~=0.8.0"]}

try:
    import tensorflow as tf
    tf_version = tf.version.VERSION.rsplit('.', 1)[0]
except ImportError:
    pass


install_requires = [
    "cpprb>=8.1.1",
    "setuptools>=41.0.0",
    "numpy>=1.16.0",
    "joblib",
    "scipy",
    *compatible_tfp[tf_version]
]

extras_require = {
    "tf": ["tensorflow>=2.0.0"],
    "tf_gpu": ["tensorflow-gpu>=2.0.0"],
    "examples": ["gym[atari]", "opencv-python"],
    "test": ["coveralls", "gym[atari]", "matplotlib", "opencv-python"]
}

setup(
    name="tf2rl",
    version="0.2.0",
    description="Deep Reinforcement Learning for TensorFlow2",
    url="https://github.com/keiohta/tf2rl",
    author="Kei Ohta",
    author_email="dev.ohtakei@gmail.com",
    license="MIT",
    packages=find_packages("."),
    install_requires=install_requires,
    extras_require=extras_require)
