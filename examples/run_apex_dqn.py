import gym
import tensorflow as tf

from tf2rl.algos.apex import apex_argument, run
from tf2rl.algos.dqn import DQN
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.networks.atari_model import AtariQFunc


# Prepare env and policy function
class env_fn:
    def __init__(self, env_name):
        self.env_name = env_name

    def __call__(self):
        return gym.make(self.env_name)


class policy_fn:
    def __init__(self, args, n_warmup, target_replace_interval, batch_size,
                 optimizer, epsilon_decay_rate, QFunc):
        self.args = args
        self.n_warmup = n_warmup
        self.target_replace_interval = target_replace_interval
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.epsilon_decay_rate = epsilon_decay_rate
        self.QFunc = QFunc

    def __call__(self, env, name, memory_capacity=int(1e6),
                 gpu=-1, noise_level=0.3):
        return DQN(
            name=name,
            enable_double_dqn=self.args.enable_double_dqn,
            enable_dueling_dqn=self.args.enable_dueling_dqn,
            enable_noisy_dqn=self.args.enable_noisy_dqn,
            enable_categorical_dqn=self.args.enable_categorical_dqn,
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.n,
            n_warmup=self.n_warmup,
            target_replace_interval=self.target_replace_interval,
            batch_size=self.batch_size,
            memory_capacity=memory_capacity,
            discount=0.99,
            epsilon=1.,
            epsilon_min=0.1,
            epsilon_decay_step=self.epsilon_decay_rate,
            optimizer=self.optimizer,
            update_interval=4,
            q_func=self.QFunc,
            gpu=gpu)


def get_weights_fn(policy):
    return [policy.q_func.weights,
            policy.q_func_target.weights]


def set_weights_fn(policy, weights):
    q_func_weights, qfunc_target_weights = weights
    update_target_variables(
        policy.q_func.weights, q_func_weights, tau=1.)
    update_target_variables(
        policy.q_func_target.weights, qfunc_target_weights, tau=1.)


if __name__ == '__main__':
    parser = apex_argument()
    parser = DQN.get_argument(parser)
    parser.add_argument('--atari', action='store_true')
    parser.add_argument('--env-name', type=str,
                        default="SpaceInvadersNoFrameskip-v4")
    args = parser.parse_args()

    if args.atari:
        env_name = args.env_name
        n_warmup = 50000
        target_replace_interval = 10000
        batch_size = 32
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0000625, epsilon=1.5e-4)
        epsilon_decay_rate = int(1e6)
        QFunc = AtariQFunc
    else:
        env_name = "CartPole-v0"
        n_warmup = 500
        target_replace_interval = 300
        batch_size = 32
        optimizer = None
        epsilon_decay_rate = int(1e3)
        QFunc = None

    run(args, env_fn(env_name),
        policy_fn(args, n_warmup, target_replace_interval, batch_size, optimizer,
                  epsilon_decay_rate, QFunc),
        get_weights_fn, set_weights_fn)
