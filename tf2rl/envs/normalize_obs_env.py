import gym

from tf2rl.misc.normalizer import NormalizerNumpy


class NormalizeObsEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.normalizer = NormalizerNumpy()
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return self.normalizer.normalize(obs, update=True)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self.normalizer.normalize(obs, update=True)
        return obs, rew, done, info
