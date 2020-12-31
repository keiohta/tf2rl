import unittest

import numpy as np

from tf2rl.misc.discount_cumsum import discount_cumsum


class TestDiscountCumSum(unittest.TestCase):
    def test_discount_cumsum(self):
        rewards = np.ones(shape=(3,), dtype=np.float32)
        discount = 0.99
        expected = np.array([
            1 + discount + discount * discount,
            1 + discount,
            1
        ], dtype=np.float32)
        results = discount_cumsum(rewards, discount)
        np.testing.assert_allclose(expected, results)


if __name__ == '__main__':
    unittest.main()
