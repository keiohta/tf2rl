import unittest
import numpy as np

from tf2rl.misc.discount_cumsum import discount_cumsum


class TestUtils(unittest.TestCase):
    def test_discount_cumsum(self):
        x = np.array([1., 1., 1.])
        discount = 0.99
        expected = [
            1. + 1.*discount**1 + 1.*discount**2,
            1. + 1.*discount**1,
            1.]
        results = discount_cumsum(x, discount)
        np.testing.assert_array_equal(results, expected)


if __name__ == '__main__':
    unittest.main()
