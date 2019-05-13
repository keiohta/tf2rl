import numpy as np
import tensorflow as tf


def huber_loss(y_target=None, y_pred=None, diff=None, max_grad=1., weights=None):
    """Calculate the huber loss.
    Args:
    y_true: np.array, tf.Tensor
        Target value.
    y_pred: np.array, tf.Tensor
        Predicted value.
    max_grad: float, optional
        Positive floating point value. Represents the maximum possible
        gradient magnitude.
    Returns:
        tf.Tensor
        The huber loss.
    """
    if weights is None:
        weights = np.ones_like(y_target)
    if diff is None:
        diff = tf.abs(y_target - y_pred)
    less_than_max = 0.5 * tf.square(diff)
    greater_than_max = max_grad * (diff - 0.5 * max_grad)
    return weights * tf.where(diff <= max_grad, x=less_than_max, y=greater_than_max)
