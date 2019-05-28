import gym

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from tf2rl.algos.dqn import DQN
from tf2rl.algos.categorical_dqn import CategoricalDQN
from tf2rl.networks.noisy_dense import NoisyDense
from tf2rl.envs.atari_wrapper import wrap_dqn
from tf2rl.experiments.trainer import Trainer


class QFunc(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=None,
                 name="QFunc", enable_dueling_dqn=False,
                 enable_noisy_dqn=False, enable_categorical_dqn=False,
                 n_atoms=51):
        super().__init__(name=name)
        self._enable_dueling_dqn = enable_dueling_dqn
        self._enable_noisy_dqn = enable_noisy_dqn
        self._enable_categorical_dqn = enable_categorical_dqn
        if enable_categorical_dqn:
            self._action_dim = action_dim
            self._n_atoms = n_atoms
            action_dim = (action_dim + int(enable_dueling_dqn)) * n_atoms

        DenseLayer = NoisyDense if enable_noisy_dqn else Dense

        self.conv1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                            padding='valid', activation='relu')
        self.conv2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2),
                            padding='valid', activation='relu')
        self.conv3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                            padding='valid', activation='relu')
        self.flat = Flatten()
        self.fc1 = DenseLayer(512, activation='relu')
        self.fc2 = DenseLayer(action_dim, activation='linear')

        if self._enable_dueling_dqn:
            self.fc3 = DenseLayer(1, activation='linear')

        input_shape = (1,) + state_shape
        with tf.device("/cpu:0"):
            self(inputs=tf.constant(np.zeros(shape=input_shape,
                                             dtype=np.float32)))

    def call(self, inputs):
        # TODO: This type conversion seems to be bottle neck
        features = tf.divide(tf.cast(inputs, tf.float32),
                             tf.constant(255.))
        features = self.conv1(features)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.flat(features)
        features = self.fc1(features)
        if self._enable_categorical_dqn:
            features = self.fc2(features)
            if self._enable_dueling_dqn:
                features = tf.reshape(
                    features, (-1, self._action_dim+1, self._n_atoms))  # [batch_size, action_dim, n_atoms]
                v_values = tf.reshape(
                    features[:, 0], (-1, 1, self._n_atoms))
                advantages = tf.reshape(
                    features[:, 1:], [-1, self._action_dim, self._n_atoms])
                features = v_values + (advantages - tf.expand_dims(
                    tf.reduce_mean(advantages, axis=1), axis=1))
            else:
                features = tf.reshape(
                    features, (-1, self._action_dim, self._n_atoms))  # [batch_size, action_dim, n_atoms]
            q_dist = tf.keras.activations.softmax(features, axis=2)  # [batch_size, action_dim, n_atoms]
            return tf.clip_by_value(q_dist, 1e-8, 1.0-1e-8)
        else:
            if self._enable_dueling_dqn:
                advantages = self.fc2(features)
                v_values = self.fc3(features)
                q_values = v_values + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
            else:
                q_values = self.fc2(features)
            return q_values


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = DQN.get_argument(parser)
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6))
    parser.add_argument('--env-name', type=str, default="SpaceInvadersNoFrameskip-v4")
    parser.set_defaults(episode_max_steps=108000)
    parser.set_defaults(test_interval=10000)
    parser.set_defaults(max_steps=int(1e9))
    parser.set_defaults(save_model_interval=500000)
    parser.set_defaults(gpu=0)
    args = parser.parse_args()

    env = wrap_dqn(gym.make(args.env_name))
    test_env = wrap_dqn(gym.make(args.env_name), reward_clipping=False)
    # Following parameters are equivalent to DeepMind DQN paper
    # https://www.nature.com/articles/nature14236
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=0.00025, momentum=0.95, epsilon=0.01)
    DQN_class = CategoricalDQN if args.enable_categorical_dqn else DQN
    policy = DQN_class(
        enable_double_dqn=args.enable_double_dqn,
        enable_dueling_dqn=args.enable_dueling_dqn,
        enable_noisy_dqn=args.enable_noisy_dqn,
        enable_categorical_dqn=args.enable_categorical_dqn,
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        n_warmup=50000,
        target_replace_interval=10000,
        batch_size=32,
        memory_capacity=args.replay_buffer_size,
        discount=0.99,
        epsilon=1.,
        epsilon_min=0.1,
        epsilon_decay_step=int(1e6),
        optimizer=optimizer,
        update_interval=4,
        q_func=QFunc,
        gpu=args.gpu)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
