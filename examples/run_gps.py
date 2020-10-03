from examples.run_ilqg import PendulumILQGEnv, ReacherILQGEnv
from tf2rl.experiments.gps_trainer import GPSTrainer


def main():
    parser = GPSTrainer.get_argument()
    parser.add_argument("--env-name", choices=["Pendulum", "Reacher"], default="Reacher")
    args = parser.parse_args()

    if args.env_name == "Pendulum":
        make_env = PendulumILQGEnv
        args.horizon = 100
    elif args.env_name == "Reacher":
        make_env = ReacherILQGEnv
        args.horizon = 30

    trainer = GPSTrainer(make_env, args)
    trainer()


if __name__ == '__main__':
    main()
