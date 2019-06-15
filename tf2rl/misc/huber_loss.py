import numpy as np
import tensorflow as tf


def huber_loss(x, delta=1.):
    """Compute the huber loss.
    https://en.wikipedia.org/wiki/Huber_loss

    Args:
    x: np.array or tf.Tensor
        Values to compute the huber loss.
    delta: float, optional
        Positive floating point value. Represents the maximum possible
        gradient magnitude.

    Returns:
        tf.Tensor
        The huber loss.
    """
    delta = tf.ones_like(x) * delta
    less_than_max = 0.5 * tf.square(x)
    greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
    return tf.where(
        tf.abs(x) <= delta,
        x=less_than_max,
        y=greater_than_max)
