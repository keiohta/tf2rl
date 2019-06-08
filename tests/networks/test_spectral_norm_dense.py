import unittest

from tf2rl.networks.spectral_norm_dense import SNDense
from tests.networks.utils import layer_test


class TestSNDense(unittest.TestCase):
    def test_sn_dense(self):
        layer_test(
            SNDense, kwargs={'units': 3}, input_shape=(3, 2),
            custom_objects={'SNDense': SNDense})


if __name__ == '__main__':
    unittest.main()
