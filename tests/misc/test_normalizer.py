import unittest

import numpy as np

from tf2rl.misc.normalizer import Normalizer


class TestNormalizer(unittest.TestCase):
    def test_normalize(self):
        normalizer = Normalizer()
        data = np.array([1, 2, 3, 4, 5])
        for datum in data:
            normalizer.observe(datum)
            data = normalizer.normalize(datum)


if __name__ == "__main__":
    unittest.main()
