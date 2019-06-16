import unittest
import numpy as np

from tf2rl.misc.huber_loss import huber_loss


class TestHuberLoss(unittest.TestCase):
    def test_huber_loss(self):
        """Test of huber loss
        huber_loss() allows two types of inputs:
        - `y_target` and `y_pred`
        - `diff`
        """
        # [1, 1] -> [0.5, 0.5]
        loss = huber_loss(np.array([1., 1.]), delta=1.)
        np.testing.assert_array_equal(
            np.array([0.5, 0.5]),
            loss.numpy())

        # [0,0] and [10, 10] -> [9.5, 9.5]
        loss = huber_loss(np.array([10., 10.]), delta=1.)
        np.testing.assert_array_equal(
            np.array([9.5, 9.5]),
            loss.numpy())

        # [0,0] and [-1, -2] -> [0.5, 1.5]
        loss = huber_loss(np.array([-1., -2.]), delta=1.)
        np.testing.assert_array_equal(
            np.array([0.5, 1.5]),
            loss.numpy())


if __name__ == '__main__':
    unittest.main()
