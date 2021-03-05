import numpy as np
import tensorflow as tf


class Normalizer:
    """
    Normalize input data online. This is based on following:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """

    def __init__(self, mean_only=False):
        self._mean_only = mean_only
        self._n = tf.Variable(0, dtype=tf.float32)
        self._mean = tf.Variable(0, dtype=tf.float32)
        self._mean_diff = tf.Variable(0, dtype=tf.float32)
        if not self._mean_only:
            self._var = tf.Variable(0, dtype=tf.float32)

    @tf.function
    def observe(self, x):
        """Compute next mean and std

        Args:
            x: float
                Input data.
        """
        self._n.assign_add(1)
        numerator = x - self._mean
        self._mean.assign_add((x - self._mean) / self._n)
        self._mean_diff.assign_add(numerator * (x - self._mean))
        if not self._mean_only:
            self._var = tf.clip_by_value(
                tf.math.divide_no_nan(self._mean_diff, self._n), 1e-2, 1e+2)

    @tf.function
    def normalize(self, x):
        std = tf.math.sqrt(self._var)
        return tf.math.divide_no_nan(x - self._mean, std)


class NormalizerNumpy:
    def __init__(self):
        self._n = 0
        self._mean = 0
        self._mean_diff = 0
        self._var = 0

    def observe(self, x):
        self._n += 1
        numerator = x - self._mean
        self._mean += (x - self._mean) / self._n
        self._mean_diff += numerator * (x - self._mean)
        self._var = self._mean_diff / self._n

    def normalize(self, x, update=False):
        if update:
            self.observe(x)
        return (x - self._mean) / (np.sqrt(self._var) + 1e-8)

    def get_params(self):
        return self._n, self._mean, self._mean_diff, self._var

    def set_params(self, n, mean, mean_diff, var):
        self._n = n
        self._mean = mean
        self._mean_diff = mean_diff
        self._var = var
