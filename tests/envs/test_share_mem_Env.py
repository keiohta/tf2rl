import unittest

import gym
import numpy as np
from tf2rl.envs.share_mem_env import ShmemEnv


class MyTestCase(unittest.TestCase):
    def test_multiple(self):
        def make_env():
            return gym.make("Pendulum-v0")

        env = make_env()
        seeds = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        vec_env = ShmemEnv(make_env, 2, 4, env.observation_space, env.action_space, seeds)
        vec_obs = vec_env.observation()

        list_actions = []
        for i in range(8):
            list_actions.append(env.action_space.sample())
        actions = np.array(list_actions, dtype=np.float32)
        next_obs, rewards, dones, reach_limits = vec_env.step(actions)

        for idx_env in range(4):
            env = make_env()
            env.seed(idx_env + 1)
            obs1 = np.array(env.reset(), dtype=np.float32)
            np.testing.assert_array_equal(vec_obs[idx_env], obs1)

            cur_obs, cur_reward, cur_done, _ = env.step(actions[idx_env])
            np.testing.assert_array_equal(next_obs[idx_env], np.array(cur_obs, dtype=np.float32))
            np.testing.assert_array_equal(rewards[idx_env], np.array(cur_reward, dtype=np.float32))
            np.testing.assert_array_equal(dones[idx_env], cur_done)
            np.testing.assert_array_equal(reach_limits[idx_env], False)

        for idx_env in range(4):
            env = make_env()
            env.seed(idx_env + 5)
            obs2 = np.array(env.reset(), dtype=np.float32)
            np.testing.assert_array_equal(vec_obs[idx_env + 4], obs2)

            cur_obs, cur_reward, cur_done, _ = env.step(actions[idx_env + 4])
            np.testing.assert_array_equal(next_obs[idx_env + 4], np.array(cur_obs, dtype=np.float32))
            np.testing.assert_array_equal(rewards[idx_env + 4], np.array(cur_reward, dtype=np.float32))
            np.testing.assert_array_equal(dones[idx_env + 4], cur_done)
            np.testing.assert_array_equal(reach_limits[idx_env + 4], False)

    def test_long_steps(self):
        def make_env():
            return gym.make("CartPole-v1")

        env = make_env()
        seeds = np.array([[1, 2], [3, 4]])
        vec_env = ShmemEnv(make_env, 2, 2, env.observation_space, env.action_space, seeds)

        list_envs = []
        list_steps = [0] * 4

        for i in range(4):
            env = make_env()
            env.seed(i + 1)
            env.reset()
            list_envs.append(env)

        for i in range(1000):
            list_actions = []
            for i in range(4):
                list_actions.append(env.action_space.sample())
            actions = np.array(list_actions, dtype=env.action_space.dtype)
            next_obs, rewards, dones, reach_limits = vec_env.step(actions)
            reset_obs = vec_env.observation()

            for idx_env in range(4):
                cur_obs, cur_reward, cur_done, _ = list_envs[idx_env].step(actions[idx_env])
                cur_next_obs = cur_obs
                list_steps[idx_env] += 1
                reach_limit = False

                if cur_done:
                    if env._max_episode_steps == list_steps[idx_env]:
                        reach_limit = True

                    cur_next_obs = list_envs[idx_env].reset()
                    list_steps[idx_env] = 0

                np.testing.assert_array_equal(next_obs[idx_env], np.array(cur_obs, dtype=np.float32))
                np.testing.assert_array_equal(rewards[idx_env], np.array(cur_reward, dtype=np.float32))
                np.testing.assert_array_equal(dones[idx_env], cur_done)
                np.testing.assert_array_equal(reach_limits[idx_env], reach_limit)
                np.testing.assert_array_equal(reset_obs[idx_env], np.array(cur_next_obs, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
