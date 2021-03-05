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


def is_mujoco_env(env):
    if not hasattr(env, "env"):
        return False
    return gym.envs.mujoco.mujoco_env.MujocoEnv in env.env.__class__.__bases__


def is_atari_env(env):
    if not hasattr(env, "env"):
        return False
    return gym.envs.atari.atari_env.AtariEnv == env.env.__class__
