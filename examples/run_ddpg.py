import roboschool, gym

from tf2rl.algos.ddpg import DDPG
from tf2rl.trainer.trainer import Trainer


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser.add_argument('--env-name', type=str, default="RoboschoolAnt-v1")
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        batch_size=100)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
