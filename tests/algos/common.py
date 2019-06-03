import unittest
import gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.dqn import DQN


class CommonAlgos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.discrete_env = gym.make("CartPole-v0")
        cls.continuous_env = gym.make("Pendulum-v0")
        cls.batch_size = 32
