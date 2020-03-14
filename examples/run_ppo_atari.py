import gym

import numpy as np
import tensorflow as tf

from tf2rl.algos.ppo import PPO
from tf2rl.envs.atari_wrapper import wrap_dqn
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.networks.atari_model import AtariCategoricalActorCritic


if __name__ == '__main__':
    parser = OnPolicyTrainer.get_argument()
    parser = PPO.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="PongNoFrameskip-v4")
    parser.set_defaults(episode_max_steps=108000)
    parser.set_defaults(horizon=1024)
    parser.set_defaults(test_interval=204800)
    parser.set_defaults(max_steps=int(1e9))
    parser.set_defaults(save_model_interval=2048000)
    parser.set_defaults(gpu=0)
    parser.set_defaults(show_test_images=True)
    args = parser.parse_args()

    env = wrap_dqn(gym.make(args.env_name))
    test_env = wrap_dqn(gym.make(args.env_name), reward_clipping=False)

    state_shape = env.observation_space.shape
    action_dim = env.action_space.n

    actor_critic = AtariCategoricalActorCritic(
        state_shape=state_shape, action_dim=action_dim)

    policy = PPO(
        state_shape=state_shape,
        action_dim=action_dim,
        is_discrete=True,
        actor_critic=actor_critic,
        batch_size=args.batch_size,
        n_epoch=3,
        lr_actor=2.5e-4,
        lr_critic=2.5e-4,
        discount=0.99,
        lam=0.95,
        horizon=args.horizon,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)
    trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
    trainer()
