import unittest

import numpy as np

from tf2rl.networks.atari_model import AtariQFunc, AtariCategoricalActor


class TestNoisyDense(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = np.zeros(shape=(32, 84, 84, 4), dtype=np.uint8)
        cls.output_dim = 18

    def test_atariqfunc(self):
        qfunc = AtariQFunc(state_shape=self.inputs.shape[1:],
                           action_dim=self.output_dim)
        qfunc(self.inputs)

    def test_ataricatgoricalactor(self):
        actor = AtariCategoricalActor(
            state_shape=self.inputs.shape[1:],
            action_dim=self.output_dim)
        actor(self.inputs)


if __name__ == '__main__':
    unittest.main()
