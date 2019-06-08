import numpy as np

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.dict import Dict

# from cpprb import NstepReplayBuffer
# from cpprb import NstepPrioritizedReplayBuffer
from cpprb.experimental import ReplayBuffer
from cpprb.experimental import PrioritizedReplayBuffer

from tf2rl.algos.policy_base import OffPolicyAgent


def get_space_size(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [1,]  # space.n
    else:
        raise NotImplementedError("Assuming to use Box or Discrete")


def get_default_rb_dict(size, env):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {
                "shape": get_space_size(env.observation_space)},
            "next_obs": {
                "shape": get_space_size(env.observation_space)},
            "act": {
                "shape": get_space_size(env.action_space)},
            "rew": {},
            "done": {}}}


def get_replay_buffer(policy, env, use_prioritized_rb=False,
                      use_nstep_rb=False, n_step=1, size=None):
    if policy is None or env is None:
        return None

    obs_shape = get_space_size(env.observation_space)
    kwargs = get_default_rb_dict(policy.update_interval, env)

    # on-policy policy
    if not issubclass(type(policy), OffPolicyAgent):
        kwargs["env_dict"]["log_pi"] = {}
        return ReplayBuffer(**kwargs)

    # off-policy policy
    kwargs["size"] = policy.memory_capacity \
        if size is None else size

    # N-step prioritized
    if use_prioritized_rb and use_nstep_rb:
        kwargs["n_step"] = n_step
        kwargs["discount"] = policy.discount
        raise NotImplementedError
        # return NstepPrioritizedReplayBuffer(**kwargs)

    if len(obs_shape) == 3:
        # kwargs["next_of"] = "obs"
        kwargs["env_dict"]["obs"]["dtype"] = np.ubyte
        kwargs["env_dict"]["next_obs"]["dtype"] = np.ubyte
        # kwargs["env_dict"].pop("next_obs")

    # prioritized
    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwargs)

    # N-step
    if use_nstep_rb:
        kwargs["n_step"] = n_step
        kwargs["discount"] = policy.discount
        raise NotImplementedError
        # return NstepReplayBuffer(**kwargs)

    # if isinstance(kwargs["act_dim"], tuple):
    #     kwargs["act_dim"] = kwargs["act_dim"][0]

    return ReplayBuffer(**kwargs)
