import gym
import numpy as np
from tf2rl.experiments.mpc_trainer import MPCTrainer, RandomPolicy


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def reward_fn_pendulum(obses, acts):
    is_single_input = obses.ndim == acts.ndim and acts.ndim == 1
    if is_single_input:
        thetas = np.arctan2(obses[1], obses[0])
        theta_dots = obses[2]
    else:
        assert obses.ndim == acts.ndim == 2
        assert obses.shape[0] == acts.shape[0]
        acts = np.squeeze(acts)
        thetas = np.arctan2(obses[:, 1], obses[:, 0])
        theta_dots = obses[:, 2]
        assert thetas.shape == theta_dots.shape == acts.shape

    acts = np.clip(acts, -2, 2)
    costs = angle_normalize(thetas) ** 2 + .1 * theta_dots ** 2 + .001 * (acts ** 2)

    return -costs


if __name__ == "__main__":
    parser = MPCTrainer.get_argument()
    parser.set_defaults(episode_max_steps=200)
    args = parser.parse_args()

    env = gym.make("Pendulum-v0")
    test_env = gym.make("Pendulum-v0")

    policy = RandomPolicy(
        max_action=env.action_space.high[0],
        act_dim=env.action_space.high.size)

    trainer = MPCTrainer(policy, env, args, reward_fn=reward_fn_pendulum, test_env=test_env)
    trainer()
