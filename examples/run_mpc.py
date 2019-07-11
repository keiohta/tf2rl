import numpy as np
import gym

from tf2rl.experiments.mpc_trainer import MPCTrainer


if __name__ == "__main__":
    parser = MPCTrainer.get_argument()
    parser.set_defaults(episode_max_steps=200)
    args = parser.parse_args()

    env = gym.make("Pendulum-v0")

    def reward_fn(obses, next_obses, acts):
        assert obses.ndim == next_obses.ndim == acts.ndim == 2
        assert obses.shape[0] == next_obses.shape[0] == acts.shape[0]
        acts = np.squeeze(acts)
        thetas = np.arctan2(obses[:, 1], obses[:, 0])
        theta_dots = obses[:, 2]

        def angle_normalize(x):
            return (((x+np.pi) % (2*np.pi)) - np.pi)

        acts = np.clip(acts, -2, 2)
        assert thetas.shape == theta_dots.shape == acts.shape
        costs = angle_normalize(thetas)**2 + .1*theta_dots**2 + .001*(acts**2)
        return -costs

    trainer = MPCTrainer(env, args, reward_fn)
    trainer()
