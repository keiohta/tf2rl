import unittest

import numpy as np

from tf2rl.misc.normalizer import NormalizerNumpy


class TestNormalizer(unittest.TestCase):
    def test_normalize(self):
        normalizer = NormalizerNumpy()
        data = np.array([1, 2, 3, 4, 5])
        for datum in data:
            normalizer.observe(datum)
            data = normalizer.normalize(datum)

    def test_observe(self):
        normalizer = NormalizerNumpy()
        for _ in range(10000):
            normalizer.observe(np.random.uniform(-1, 1, 10))
        self.assertLessEqual(np.mean(normalizer._mean), 0.10)


if __name__ == "__main__":
    unittest.main()
