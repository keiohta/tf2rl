import gym

from tf2rl.algos.ddpg import DDPG
from tf2rl.trainer.trainer import Trainer


if __name__ == '__main__':
    args = Trainer.get_argument().parse_args()
    env = gym.make("HalfCheetah-v2")
    test_env = gym.make("HalfCheetah-v2")
    policy = DDPG(
        state_dim=env.observation_space.high.size,
        action_dim=env.action_space.high.size)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
