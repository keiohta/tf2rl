import numpy as np

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.dict import Dict

from cpprb import ReplayBuffer, PrioritizedReplayBuffer
from cpprb import NstepReplayBuffer
from cpprb import NstepPrioritizedReplayBuffer
from cpprb.experimental import ReplayBuffer as ImgReplayBuffer
from cpprb.experimental import PrioritizedReplayBuffer as ImgPrioritizedReplayBuffer

from tf2rl.algos.policy_base import OffPolicyAgent


def get_space_size(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return 1  # space.n
    else:
        raise NotImplementedError("Assuming to use Box or Discrete")


def get_replay_buffer(policy, env, use_prioritized_rb, use_nstep_rb, n_step):
    if policy is None or env is None:
        return None

    kwargs = {
        "obs_shape": get_space_size(env.observation_space),
        "act_dim": get_space_size(env.action_space),
        "size": policy.update_interval
    }

    # on-policy policy
    if not issubclass(type(policy), OffPolicyAgent):
        return ReplayBuffer(**kwargs)

    # off-policy policy
    kwargs["size"] = policy.memory_capacity

    # N-step prioritized
    if use_prioritized_rb and use_nstep_rb:
        kwargs["n_step"] = n_step
        kwargs["discount"] = policy.discount
        return NstepPrioritizedReplayBuffer(**kwargs)

    # prioritized
    if use_prioritized_rb:
        if len(kwargs["obs_shape"]) == 3:
            return ImgPrioritizedReplayBuffer(
                kwargs["size"],
                {"obs": {"shape": kwargs["obs_shape"],
                         "dtype": np.ubyte},
                 "act": {"shape": kwargs["act_dim"]},
                 "rew": {},
                 "done": {}},
                 next_of="obs")
        else:
            return PrioritizedReplayBuffer(**kwargs)

    # N-step
    if use_nstep_rb:
        kwargs["n_step"] = n_step
        kwargs["discount"] = policy.discount
        return NstepReplayBuffer(**kwargs)

    if isinstance(kwargs["act_dim"], tuple):
        kwargs["act_dim"] = kwargs["act_dim"][0]

    if len(kwargs["obs_shape"]) == 3:
        return ImgReplayBuffer(
            kwargs["size"],
            {"obs": {"shape": kwargs["obs_shape"],
                     "dtype": np.ubyte},
             "act": {"shape": kwargs["act_dim"]},
             "rew": {},
             "done": {}},
            next_of="obs")
    else:
        return ReplayBuffer(**kwargs)
