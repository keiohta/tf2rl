[![Build Status](https://travis-ci.org/keiohta/tf2rl.svg?branch=master)](https://travis-ci.org/keiohta/tf2rl)
[![Coverage Status](https://coveralls.io/repos/github/keiohta/tf2rl/badge.svg?branch=master)](https://coveralls.io/github/keiohta/tf2rl?branch=master)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![GitHub issues open](https://img.shields.io/github/issues/keiohta/tf2rl.svg)]()
[![PyPI version](https://badge.fury.io/py/tf2rl.svg)](https://badge.fury.io/py/tf2rl)

# TF2RL
TF2RL is a deep reinforcement learning library that implements various deep reinforcement learning algorithms using TensorFlow 2.0.

## Algorithms
Following algorithms are supported:

|                          Algorithm                           | Dicrete action | Continuous action |                  Support                   | Category                 |
| :----------------------------------------------------------: | :------------: | :---------------: | :----------------------------------------: | ------------------------ |
| [VPG](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), [PPO](<https://arxiv.org/abs/1707.06347>) |       ✓        |         ✓         |  [GAE](https://arxiv.org/abs/1506.02438)   | Model-free On-policy RL  |
| [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (including [DDQN](https://arxiv.org/abs/1509.06461), [Prior. DQN](https://arxiv.org/abs/1511.05952), [Duel. DQN](https://arxiv.org/abs/1511.06581), [Distrib. DQN](<https://arxiv.org/abs/1707.06887>), [Noisy DQN](<https://arxiv.org/abs/1706.10295>)) |       ✓        |         -         | [ApeX](<https://arxiv.org/abs/1803.00933>) | Model-free Off-policy RL |
| [DDPG](https://arxiv.org/abs/1509.02971) (including [TD3](<https://arxiv.org/abs/1802.09477>), [BiResDDPG](<https://arxiv.org/abs/1905.01072>)) |       -        |         ✓         | [ApeX](<https://arxiv.org/abs/1803.00933>) | Model-free Off-policy RL |
|          [SAC](<https://arxiv.org/abs/1801.01290>)           |       ✓        |         ✓         | [ApeX](<https://arxiv.org/abs/1803.00933>) | Model-free Off-policy RL |
| [GAIL](<https://arxiv.org/abs/1606.03476>), [VAIL](<https://arxiv.org/abs/1810.00821>) (including [Spectral Normalization](<https://arxiv.org/abs/1802.05957>)) |       ✓        |         ✓         |                     -                      | Imitation Learning       |

Following papers have been implementd in tf2rl:

- Model-free On-policy RL
  - [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/vpg.py>)
  - [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/misc/discount_cumsum.py>)
  - [Proximal Policy Optimization Algorithms](<https://arxiv.org/abs/1707.06347>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/ppo.py>)
- Model-free Off-policy RL
  - [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/dqn.py>)
  - [Human-level control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/dqn.py>)
  - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/dqn.py>)
  - [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/dqn.py>)
  - [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/dqn.py>)
  - [A Distributional Perspective on Reinforcement Learning](<https://arxiv.org/abs/1707.06887>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/dqn.py>)
  - [Noisy Networks for Exploration](<https://arxiv.org/abs/1706.10295>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/networks/noisy_dense.py>)
  - [Distributed Prioritized Experience Replay](<https://arxiv.org/abs/1803.00933>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/apex.py>)
  - [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/ddpg.py>)
  - [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](<https://arxiv.org/abs/1801.01290>), [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/sac.py>)
  - [Addressing Function Approximation Error in Actor-Critic Methods](<https://arxiv.org/abs/1802.09477>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/td3.py>)
  - [Deep Residual Reinforcement Learning](<https://arxiv.org/abs/1905.01072>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/bi_res_ddpg.py>)
  - [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207v1), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/sac_discrete.py>)
- Imitation Learning
  - [Generative Adversarial Imitation Learning](<https://arxiv.org/abs/1606.03476>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/gail.py>)
  - [Spectral Normalization for Generative Adversarial Networks](<https://arxiv.org/abs/1802.05957>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/networks/spectral_norm_dense.py>)
  - [Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](<https://arxiv.org/abs/1810.00821>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/vail.py>)

## Installation
You can install `tf2rl` from PyPI:

```bash
$ pip install tf2rl
```

or, you can also install from source:

```bash
$ git clone https://github.com/keiohta/tf2rl.git tf2rl
$ cd tf2rl
$ pip install .
```

## Getting started
Here is a quick example of how to train DDPG agent on a Pendulum environment:

```python
import gym
from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer


parser = Trainer.get_argument()
parser = DDPG.get_argument(parser)
args = parser.parse_args()

env = gym.make("Pendulum-v0")
test_env = gym.make("Pendulum-v0")
policy = DDPG(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=-1,  # Run on CPU. If you want to run on GPU, specify GPU number
    memory_capacity=10000,
    max_action=env.action_space.high[0],
    batch_size=32,
    n_warmup=500)
trainer = Trainer(policy, env, args, test_env=test_env)
trainer()
```

You can check implemented algorithms in [examples](https://github.com/keiohta/tf2rl/tree/master/examples).
For example if you want to train DDPG agent:

```bash
# You must change directory to avoid importing local files
$ cd examples
# For options, please specify --help or read code for options
$ python run_ddpg.py [options]
```

You can see the training progress/results from TensorBoard as follows:

```bash
# When executing `run_**.py`, its logs are automatically generated under `./results`
$ tensorboard --logdir results
```

