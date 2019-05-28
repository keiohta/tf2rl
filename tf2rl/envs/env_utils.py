from gym.spaces import Discrete, Box


def get_act_dim(env):
    if isinstance(env.action_space, Discrete):
        return 1  # env.action_space.n
    elif isinstance(env.action_space, Box):
        return env.action_space.low.size
    else:
        raise NotImplementedError
