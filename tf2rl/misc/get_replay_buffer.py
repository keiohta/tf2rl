from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.dict import Dict

from cpprb import ReplayBuffer, PrioritizedReplayBuffer
from cpprb import NstepReplayBuffer
from cpprb import NstepPrioritizedReplayBuffer

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
        return PrioritizedReplayBuffer(**kwargs)

    # N-step
    if use_nstep_rb:
        kwargs["n_step"] = n_step
        kwargs["discount"] = policy.discount
        return NstepReplayBuffer(**kwargs)

    if isinstance(kwargs["act_dim"], tuple):
        kwargs["act_dim"] = kwargs["act_dim"][0]
    return ReplayBuffer(**kwargs)


if __name__ == '__main__':
    from cpprb import ReplayBuffer
    import numpy as np

    rb = ReplayBuffer(obs_dim=3, act_dim=3, size=10)
    for i in range(10):
        obs_act = np.array([i for _ in range(3)], dtype=np.float64)
        print(obs_act)
        rb.add(obs=obs_act, act=obs_act, next_obs=obs_act, rew=float(i), done=False)
    print(rb.sample(10))
