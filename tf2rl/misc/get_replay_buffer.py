import numpy as np

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.dict import Dict

# from cpprb import NstepReplayBuffer, NstepPrioritizedReplayBuffer
from cpprb.experimental import ReplayBuffer, PrioritizedReplayBuffer

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.envs.utils import is_discrete


def get_space_size(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [1, ]  # space.n
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
    kwargs = get_default_rb_dict(policy.memory_capacity, env)

    if size is not None:
        kwargs["size"] = size

    # on-policy policy
    if not issubclass(type(policy), OffPolicyAgent):
        kwargs["size"] = policy.horizon
        kwargs["env_dict"].pop("next_obs")
        kwargs["env_dict"].pop("rew")
        # TODO: Remove done. Currently cannot remove because of cpprb implementation
        # kwargs["env_dict"].pop("done")
        kwargs["env_dict"]["logp"] = {}
        kwargs["env_dict"]["ret"] = {}
        kwargs["env_dict"]["adv"] = {}
        if is_discrete(env.action_space):
            kwargs["env_dict"]["act"]["dtype"] = np.int32
        return ReplayBuffer(**kwargs)

    # N-step prioritized
    if use_prioritized_rb and use_nstep_rb:
        kwargs["n_step"] = n_step
        kwargs["discount"] = policy.discount
        raise NotImplementedError
        # return NstepPrioritizedReplayBuffer(**kwargs)

    if len(obs_shape) == 3:
        kwargs["env_dict"]["obs"]["dtype"] = np.ubyte
        kwargs["env_dict"]["next_obs"]["dtype"] = np.ubyte

    # prioritized
    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwargs)

    # N-step
    if use_nstep_rb:
        kwargs["n_step"] = n_step
        kwargs["discount"] = policy.discount
        raise NotImplementedError
        # return NstepReplayBuffer(**kwargs)

    return ReplayBuffer(**kwargs)
