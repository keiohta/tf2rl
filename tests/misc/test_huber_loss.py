import unittest

import os
import numpy as np
import gym

from tf2rl.misc.huber_loss import huber_loss


class TestHuberLoss(unittest.TestCase):
    def test_huber_loss(self):
        # [0, 0] and [1, 1] -> [0.5, 0.5]
        y_target = np.array([0., 0.])
        y_pred = np.array([1., 1.])
        expected = np.array([0.5, 0.5])
        loss = huber_loss(y_target, y_pred)
        print(loss)
        # self.assertEqual(expected, loss.numpy())

        y_target = np.array([0., 0.])
        y_pred = np.array([10., 10.])
        expected = np.array([10., 10.])
        loss = huber_loss(y_target, y_pred)
        print(loss)
        # self.assertEqual(expected, loss.numpy())


if __name__ == '__main__':
    unittest.main()
