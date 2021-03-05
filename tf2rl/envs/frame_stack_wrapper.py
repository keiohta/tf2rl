from collections import deque

import gym
import numpy as np


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, obs_shape, channel_first=False):
        gym.Wrapper.__init__(self, env)

        assert isinstance(obs_shape, tuple) and len(obs_shape) == 3

        self._k = k
        self._channel_first = channel_first

        self._frames = deque([], maxlen=k)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=obs_shape,
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0 if self._channel_first else -1)
