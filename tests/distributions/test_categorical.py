import unittest
import numpy as np

from tf2rl.distributions.categorical import Categorical
from tests.distributions.common import CommonDist


class TestCategorical(CommonDist):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dist = Categorical(dim=cls.dim)
        prob = np.zeros(shape=(cls.dim), dtype=np.float32)
        prob[0] = 1.
        cls.param = {
            "prob": np.tile(prob, (1, 1))}  # [1, dim]
        cls.params = {
            "prob": np.tile(prob, (cls.batch_size, 1))}  # [batch_size, dim]

    def test_kl(self):
        # KL of same distribution should be zero
        np.testing.assert_array_equal(
            self.dist.kl(self.param, self.param),
            np.zeros(shape=(1,)))
        np.testing.assert_array_equal(
            self.dist.kl(self.params, self.params),
            np.zeros(shape=(self.batch_size,)))

        # Add tests with not same distribution

    def test_log_likelihood(self):
        pass

    def test_ent(self):
        pass

    def test_sample(self):
        samples = self.dist.sample(self.param)
        self.assertEqual(samples.shape, (1, 1))
        samples = self.dist.sample(self.params)
        self.assertEqual(samples.shape, (self.batch_size, 1))


if __name__ == '__main__':
    unittest.main()
