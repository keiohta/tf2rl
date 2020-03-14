"""
MIT License

Copyright (c) 2017 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Modified by keiohta

import numpy as np


class EmpiricalNormalizer:
    """Normalize mean and variance of values based on emprical values.
    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
        until (int or None): If this arg is specified, the link learns input
            values until the sum of batch sizes exceeds it.
    """

    def __init__(self, shape, batch_axis=0, eps=1e-2, dtype=np.float32,
                 until=None, clip_threshold=None):
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self._mean = np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)
        self._var = np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)
        self.count = 0

        # cache
        self._cached_std_inverse = None

    @property
    def mean(self):
        return np.squeeze(self._mean, self.batch_axis).copy()

    @property
    def std(self):
        return np.sqrt(np.squeeze(self._var, self.batch_axis))

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    def experience(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        self.count += count_x
        rate = x.dtype.type(count_x / self.count)

        mean_x = np.mean(x, axis=self.batch_axis, keepdims=True)
        var_x = np.var(x, axis=self.batch_axis, keepdims=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (
            var_x - self._var
            + delta_mean * (mean_x - self._mean)
        )

        # clear cache
        self._cached_std_inverse = None

    def __call__(self, x, update=True):
        """Normalize mean and variance of values based on emprical values.
        Args:
            x (ndarray or Variable): Input values
            update (bool): Flag to learn the input values
        Returns:
            ndarray or Variable: Normalized output values
        """
        if self.count == 0:
            return x

        mean = np.broadcast_to(self._mean, x.shape)
        std_inv = np.broadcast_to(self._std_inverse, x.shape)

        if update:
            self.experience(x)

        normalized = (x - mean) * std_inv
        if self.clip_threshold is not None:
            normalized = np.clip(
                normalized, -self.clip_threshold, self.clip_threshold)
        return normalized

    def inverse(self, y):
        mean = np.broadcast_to(self._mean, y.shape)
        std = np.broadcast_to(np.sqrt(self._var + self.eps), y.shape)
        return y * std + mean
