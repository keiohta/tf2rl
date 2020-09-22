# Hard to test each function, so just execute shortly

import unittest
import gym

from tf2rl.algos.apex import apex_argument, run
from tf2rl.misc.target_update_ops import update_target_variables


class TestApeX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = apex_argument()
        cls.parser.set_defaults(n_training=10)
        cls.parser.set_defaults(param_update_freq=1)
        cls.parser.set_defaults(test_freq=10)
        cls.parser.set_defaults(n_explorer=2)
        cls.parser.set_defaults(n_env=4)
        cls.parser.set_defaults(local_buffer_size=64)

    def test_run_discrete(self):
        from tf2rl.algos.dqn import DQN
        parser = DQN.get_argument(self.parser)
        parser.set_defaults(n_warmup=1)
        args, _ = parser.parse_known_args()

        run(args, env_fn_discrete, policy_fn_discrete,
            get_weights_fn_discrete, set_weights_fn_discrete)

    def test_run_continuous(self):
        from tf2rl.algos.ddpg import DDPG
        parser = DDPG.get_argument(self.parser)
        parser.set_defaults(n_warmup=1)
        args, _ = parser.parse_known_args()

        run(args, env_fn_continuous, policy_fn_continuous,
            get_weights_fn_continuous, set_weights_fn_continuous)


def env_fn_discrete():
    return gym.make("CartPole-v0")


def policy_fn_discrete(env, name, memory_capacity=int(1e6), gpu=-1, *args, **kwargs):
    from tf2rl.algos.dqn import DQN
    return DQN(
        name=name,
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        n_warmup=500,
        target_replace_interval=300,
        batch_size=32,
        memory_capacity=memory_capacity,
        discount=0.99,
        gpu=-1)


def get_weights_fn_discrete(policy):
    return [policy.q_func.weights,
            policy.q_func_target.weights]


def set_weights_fn_discrete(policy, weights):
    q_func_weights, qfunc_target_weights = weights
    update_target_variables(
        policy.q_func.weights, q_func_weights, tau=1.)
    update_target_variables(
        policy.q_func_target.weights, qfunc_target_weights, tau=1.)


def env_fn_continuous():
    return gym.make('Pendulum-v0')


def policy_fn_continuous(env, name, memory_capacity=int(1e6), gpu=-1, *args, **kwargs):
    from tf2rl.algos.ddpg import DDPG
    return DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        n_warmup=500,
        gpu=-1)


def get_weights_fn_continuous(policy):
    return [policy.actor.weights,
            policy.critic.weights,
            policy.critic_target.weights]


def set_weights_fn_continuous(policy, weights):
    actor_weights, critic_weights, critic_target_weights = weights
    update_target_variables(
        policy.actor.weights, actor_weights, tau=1.)
    update_target_variables(
        policy.critic.weights, critic_weights, tau=1.)
    update_target_variables(
        policy.critic_target.weights, critic_target_weights, tau=1.)


if __name__ == '__main__':
    unittest.main()
