[![Build Status](https://travis-ci.org/keiohta/tf2rl.svg?branch=master)](https://travis-ci.org/keiohta/tf2rl)
[![Coverage Status](https://coveralls.io/repos/github/keiohta/tf2rl/badge.svg?branch=master)](https://coveralls.io/github/keiohta/tf2rl?branch=master)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![GitHub issues open](https://img.shields.io/github/issues/keiohta/tf2rl.svg)]() 

# TF2RL
TF2RL is a deep reinforcement learning library that implements various deep reinforcement learning algorithms using TensorFlow 2.0.

## Algorithms
Following algorithms are supported:

|         Class         | Dicrete action | Continuous action |                          Algorithm                           | Support |
| :-------------------: | :------------: | :---------------: | :----------------------------------------------------------: | :-----: |
| Model-free On-policy  |       ✓        |         ✓         |                             VPG                              |   GAE   |
| Model-free Off-policy |       ✓        |         X         | DQN (including DDQN, Prior. DQN, Duel. DQN, Distrib. DQN, Noisy DQN) |  ApeX   |
| Model-free Off-policy |       X        |         ✓         |               DDPG (including BiResDDPG, TD3)                |  ApeX   |
| Model-free Off-policy |       X        |         ✓         |                             SAC                              |  ApeX   |
|  Imitation Learning   |       ✓        |         ✓         |           GAIL (including Spectral Normalization)            |         |

Following papers have been implementd in tf2rl:

- Model-free On-policy
  - [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
  - [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Model-free Off-policy
  - [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
  - [Human-level control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
  - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
  - [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
  - [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
  - [A Distributional Perspective on Reinforcement Learning](<https://arxiv.org/abs/1707.06887>)
  - [Noisy Networks for Exploration](<https://arxiv.org/abs/1706.10295>)
  - [Distributed Prioritized Experience Replay](<https://arxiv.org/abs/1803.00933>)
  - [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
  - [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](<https://arxiv.org/abs/1801.01290>)
  - [Addressing Function Approximation Error in Actor-Critic Methods](<https://arxiv.org/abs/1802.09477>)
  - [Deep Residual Reinforcement Learning](<https://arxiv.org/abs/1905.01072>)
- Imitation Learning
  - [Generative Adversarial Imitation Learning](<https://arxiv.org/abs/1606.03476>)
  - [Spectral Normalization for Generative Adversarial Networks](<https://arxiv.org/abs/1802.05957>)

## Installation
```bash
$ git clone https://github.com/keiohta/tf2rl.git tf2rl
$ cd tf2rl
$ pip install -U numpy pip
$ pip install .
```

TF2RL is built on Google's TensorFlow and requires either `tensorflow` or `tensorflow-gpu`.
To include the TensorFlow with the installation of TF2RL, add the flag `tf` for the normal CPU version, or `tf_gpu` for the GPU version.

```bash
# Install TF2RL with TensorFlow CPU version
$ pip install -e .[tf]
```

Also, if you want to run example codes, add the flag `examples` that install additional dependencies.
If you are developer, set `-e` option, then local modification affects your installation.

```bash
# Install developer mode TF2RL plus TensorFlow GPU version and additional dependencies to run examples
$ pip install -e .[tf-gpu, examples]
```

## Example
- Train DDPG agent
  - If you want to train with only CPU, set `--gpu -1`

```bash
$ cd examples # You must change directory to avoid importing local files.
$ python run_ddpg.py
```

## Usage
- You can see options defined in `Trainer.get_argument`

