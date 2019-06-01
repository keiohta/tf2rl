import unittest

import numpy as np

from tf2rl.networks.spectral_norm_dense import SNDense
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import testing_utils


class TestUtils(unittest.TestCase):
    def test_sn_dense(self):
        pass
        from utils import layer_test
        layer_test(
            SNDense, kwargs={'units': 3}, input_shape=(3, 2),
            custom_objects={'SNDense': SNDense})


if __name__ == '__main__':
    unittest.main()
