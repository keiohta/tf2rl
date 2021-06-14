import unittest

import numpy as np
import gym

from tf2rl.envs.atari_wrapper import wrap_dqn


# @unittest.skipIf((platform.system() == 'Windows') and (sys.version_info.minor >= 8),
#                  "atari-py doesn't work at Windows with Python3.8 and later")
@unittest.skipIf(True, "atari-py doesn't work at all. see https://github.com/openai/atari-py/issues/82")
class TestAtariWrapper(unittest.TestCase):
    def test_wrap_dqn(self):
        env = wrap_dqn(gym.make("SpaceInvadersNoFrameskip-v4"), wrap_ndarray=True)

        obs = env.reset()
        self.assertEqual(type(obs), np.ndarray)
        self.assertEqual(obs.shape, (84, 84, 4))


if __name__ == "__main__":
    unittest.main()
