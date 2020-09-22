from tf2rl.algos.ilqg import ILQG
from tf2rl.misc.ilqg_utils import visualize_rollout
from tf2rl.misc.initialize_logger import initialize_logger

from examples.run_ilqg import PendulumILQGEnv, ReacherILQGEnv


def main():
    logger = initialize_logger(save_log=False)

    parser = ILQG.get_argument()
    parser.add_argument("--save-movie", action="store_true")
    parser.add_argument("--visualize-interval", type=int, default=5)
    parser.add_argument("--env-name", choices=["Pendulum", "Reacher"], default="Reacher")
    args = parser.parse_args()

    if args.env_name == "Pendulum":
        make_env = PendulumILQGEnv
        args.horizon = 100
    elif args.env_name == "Reacher":
        make_env = ReacherILQGEnv
        args.horizon = 50

    ilqg = ILQG(make_env)
    ilqg.initialize()

    logger.info("Initial trajectory: T = {} cost = {}".format(len(ilqg.U), ilqg.cost))
    viewer_env = make_env()
    visualize_rollout(viewer_env=viewer_env, initial_state=ilqg.X[0], U=ilqg.U, save_movie=args.save_movie,
                      prefix="{}_{}".format(args.env_name, 0))

    real_env = make_env()

    for _ in range(100):
        # Resulted trajectories
        X, U = [], []
        # Reset state
        real_env.reset()
        ilqg.initialize(initial_state=real_env.get_state_vector())
        X.append(real_env.get_state_vector())

        for i in range(args.horizon):
            ilqg.optimize(max_iter=args.max_iter_each_step)
            logger.info("Step {} trajectory: cost = {}".format(i + 1, ilqg.cost))

        visualize_rollout(viewer_env=viewer_env, initial_state=ilqg.X[0], U=ilqg.U, save_movie=args.save_movie,
                          prefix="{}_{}".format(args.env_name, i + 1))


if __name__ == '__main__':
    main()
