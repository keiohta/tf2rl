import numpy as np
from scipy.signal import lfilter


def discount_cumsum(x, discount):
    """Forked from rllab for computing discounted cumulative sums of vectors.

    Args:
        x: np.ndarray or tf.Tensor
            Vector of inputs
        discount: float
            Discount factor

    Returns:
        Discounted cumulative summation. If input is [x0, x1, x2], then the output is:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
    """
    assert isinstance(x, np.ndarray) and x.ndim == 1
    return lfilter(
        b=[1],
        a=[1, float(-discount)],
        x=x[::-1],
        axis=0)[::-1]
