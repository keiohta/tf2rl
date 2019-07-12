import gym

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from tf2rl.algos.ppo import PPO
from tf2rl.networks.noisy_dense import NoisyDense
from tf2rl.envs.atari_wrapper import wrap_dqn
from tf2rl.experiments.trainer import Trainer
from tf2rl.networks.atari_model import AtariCategoricalPolicy


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = PPO.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="SpaceInvadersNoFrameskip-v4")
    parser.set_defaults(episode_max_steps=108000)
    parser.set_defaults(test_interval=10000)
    parser.set_defaults(max_steps=int(1e9))
    parser.set_defaults(save_model_interval=500000)
    parser.set_defaults(gpu=0)
    parser.set_defaults(show_test_images=True)
    args = parser.parse_args()

    env = wrap_dqn(gym.make(args.env_name))
    test_env = wrap_dqn(gym.make(args.env_name), reward_clipping=False)

    actor = AtariCategoricalPolicy
    actor_arg = {
        "state_shape": env.observation_space.shape,
        "action_dim": env.action_space.n}

    # Following parameters are equivalent to DeepMind DQN paper
    # https://www.nature.com/articles/nature14236
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0000625, epsilon=1.5e-4)  # This value is from Rainbow
    policy = PPO(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        batch_size=64,
        discount=0.99,
        epsilon_min=0.1,
        epsilon_decay_step=int(1e6),
        optimizer=optimizer,
        actor=actor,
        actor_arg=actor_arg,
        gpu=args.gpu)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
