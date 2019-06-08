import unittest

from tf2rl.networks.noisy_dense import NoisyDense
from tests.networks.utils import layer_test


class TestNoisyDense(unittest.TestCase):
    def test_sn_dense(self):
        layer_test(
            NoisyDense, kwargs={'units': 3}, input_shape=(3, 2),
            custom_objects={'NoisyDense': NoisyDense})


if __name__ == '__main__':
    unittest.main()
