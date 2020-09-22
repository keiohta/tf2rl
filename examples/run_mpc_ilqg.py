import os

from examples.run_ilqg import PendulumILQGEnv, ReacherILQGEnv
from tf2rl.algos.ilqg_mpc import MPCILQG
from tf2rl.experiments.utils import frames_to_gif
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.misc.prepare_output_dir import prepare_output_dir


def main():
    parser = MPCILQG.get_argument()
    parser.add_argument("--save-movie", action="store_true")
    parser.add_argument("--visualize-interval", type=int, default=5)
    parser.add_argument("--env-name", choices=["Pendulum", "Reacher"], default="Reacher")
    args = parser.parse_args()

    if args.env_name == "Pendulum":
        make_env = PendulumILQGEnv
        args.horizon = 100
    elif args.env_name == "Reacher":
        make_env = ReacherILQGEnv
        args.horizon = 30

    logger = initialize_logger(save_log=False)
    output_dir = prepare_output_dir(
        args=args, user_specified_dir="results", suffix="ilqg_mpc")

    ilqg = MPCILQG(make_env, horizon=args.horizon)
    ilqg.initialize()

    real_env = make_env()
    for epi_idx in range(100):
        X, U, cost = [], [], 0.

        # Reset state
        real_env.reset()
        frames = [real_env.render(mode="rgb_array")]
        ilqg.initialize(initial_state=real_env.get_state_vector())
        X.append(real_env.get_state_vector())

        for i in range(ilqg.horizon):
            u = ilqg.get_next_action(state=real_env.get_state_vector())
            cost += real_env.cost_state() + real_env.cost_control(u)
            real_env.step(u)
            frames.append(real_env.render(mode="rgb_array"))

        movie_prefix = "epi_{:03d}_cost_{:.3f}".format(epi_idx, cost)
        logger.info("Epi {}, cost = {:.4f}. Movie is stored to {}".format(
            epi_idx, cost, os.path.join(output_dir, movie_prefix)))
        frames_to_gif(frames, prefix=movie_prefix, save_dir=output_dir)


if __name__ == '__main__':
    main()
