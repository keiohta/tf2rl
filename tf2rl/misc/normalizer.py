import numpy as np
import tensorflow as tf


class Normalizer():
    """
    Normalize input data online. This is based on following:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """

    def __init__(self):
        self.n = tf.Variable(0, dtype=tf.float32)
        self.mean = tf.Variable(0, dtype=tf.float32)
        self.mean_diff = tf.Variable(0, dtype=tf.float32)
        self.var = tf.Variable(0, dtype=tf.float32)

    @tf.function
    def observe(self, x):
        """
        Compute next mean and std

        :param x (float): Input data
        """
        self.n.assign_add(1)
        numerator = x - self.mean
        self.mean.assign_add((x - self.mean) / self.n)
        self.mean_diff.assign_add(numerator * (x - self.mean))
        self.var = tf.clip_by_value(
            self.mean_diff / self.n, 1e-2, 1e+2)

    @tf.function
    def normalize(self, x):
        std = tf.math.sqrt(self.var)
        return tf.math.divide_no_nan(x - self.mean, std)


class NormalizerNumpy:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.mean_diff = 0
        self.var = 0

    def observe(self, x):
        self.n += 1
        numerator = x - self.mean
        self.mean += (x - self.mean) / self.n
        self.mean_diff += numerator * (x - self.mean)
        self.var = self.mean_diff / self.n

    def normalize(self, x):
        std = np.sqrt(self.var)
        return (x - self.mean) / std

