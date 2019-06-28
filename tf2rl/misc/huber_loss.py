import numpy as np
import tensorflow as tf


def huber_loss(x, delta=1.):
    """
    Compute the huber loss.
    https://en.wikipedia.org/wiki/Huber_loss

    :param x (np.ndarray or tf.Tensor): Values to compute the huber loss.
    :param delta (float): Positive floating point value. Represents the
                          maximum possible gradient magnitude.
    :return (tf.Tensor): The huber loss.
    """
    delta = tf.ones_like(x) * delta
    less_than_max = 0.5 * tf.square(x)
    greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
    return tf.where(
        tf.abs(x) <= delta,
        x=less_than_max,
        y=greater_than_max)
