import numpy as np
import gym
from gym.spaces import Discrete, Box


def is_discrete(space):
    if isinstance(space, Discrete):
        return True
    elif isinstance(space, Box):
        return False
    else:
        raise NotImplementedError


def get_act_dim(action_space):
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        return action_space.low.size
    else:
        raise NotImplementedError


def get_shape(space):
    if isinstance(space, Discrete):
        return (space.n,)
    elif isinstance(space, Box):
        return space.shape
    else:
        raise ValueError("{} is not currently supported".format(type(space)))


def to_one_hot(input, size):
    if isinstance(input, list):
        input = np.array(input)
    elif not isinstance(input, np.ndarray):
        input = np.array([input])
    if input.ndim == 2:
        input = np.squeeze(input)
    temp = np.zeros((input.shape[0], size))
    temp[np.arange(input.shape[0]), input.astype(np.int32)] = 1
    return temp


def is_mujoco_env(env):
    from gym.envs import mujoco
    if not hasattr(env, "env"):
        return False
    return gym.envs.mujoco.mujoco_env.MujocoEnv in env.env.__class__.__bases__


def is_atari_env(env):
    from gym.envs import atari
    if not hasattr(env, "env"):
        return False
    return gym.envs.atari.atari_env.AtariEnv == env.env.__class__
