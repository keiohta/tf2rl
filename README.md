[![Test](https://github.com/keiohta/tf2rl/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/keiohta/tf2rl/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/keiohta/tf2rl/badge.svg?branch=master)](https://coveralls.io/github/keiohta/tf2rl?branch=master)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![GitHub issues open](https://img.shields.io/github/issues/keiohta/tf2rl.svg)]()
[![PyPI version](https://badge.fury.io/py/tf2rl.svg)](https://badge.fury.io/py/tf2rl)

# TF2RL
TF2RL is a deep reinforcement learning library that implements various deep reinforcement learning algorithms using [TensorFlow 2.x](https://www.tensorflow.org/).

## 1. Algorithms
Following algorithms are supported:

|                          Algorithm                           | Dicrete action | Continuous action |                  Support                   | Category                 |
| :----------------------------------------------------------: | :------------: | :---------------: | :----------------------------------------: | ------------------------ |
| [VPG](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), [PPO](<https://arxiv.org/abs/1707.06347>) |       ✓        |         ✓         |  [GAE](https://arxiv.org/abs/1506.02438)   | Model-free On-policy RL  |
| [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (including [DDQN](https://arxiv.org/abs/1509.06461), [Prior. DQN](https://arxiv.org/abs/1511.05952), [Duel. DQN](https://arxiv.org/abs/1511.06581), [Distrib. DQN](<https://arxiv.org/abs/1707.06887>), [Noisy DQN](<https://arxiv.org/abs/1706.10295>)) |       ✓        |         -         | [ApeX](<https://arxiv.org/abs/1803.00933>) | Model-free Off-policy RL |
| [DDPG](https://arxiv.org/abs/1509.02971) (including [TD3](<https://arxiv.org/abs/1802.09477>), [BiResDDPG](<https://arxiv.org/abs/1905.01072>)) |       -        |         ✓         | [ApeX](<https://arxiv.org/abs/1803.00933>) | Model-free Off-policy RL |
|          [SAC](<https://arxiv.org/abs/1801.01290>)           |       ✓        |         ✓         | [ApeX](<https://arxiv.org/abs/1803.00933>) | Model-free Off-policy RL |
| [CURL](https://arxiv.org/abs/2004.04136), [SAC-AE](https://arxiv.org/abs/1910.01741) |       -        |         ✓         |                     -                      | Model-free Off-policy RL |
| [MPC](https://arxiv.org/abs/1708.02596), [ME-TRPO](https://arxiv.org/abs/1802.10592) |       ✓        |         ✓         |                     -                      | Model-base RL            |
| [GAIL](<https://arxiv.org/abs/1606.03476>), [GAIfO](<https://arxiv.org/abs/1807.06158>), [VAIL](<https://arxiv.org/abs/1810.00821>) (including [Spectral Normalization](<https://arxiv.org/abs/1802.05957>)) |       ✓        |         ✓         |                     -                      | Imitation Learning       |

Following papers have been implemented in tf2rl:

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
  - [Improving Sample Efficiency in Model-Free Reinforcement Learning from Images](https://arxiv.org/abs/1910.01741), [code](https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/sac_ae.py)
  - [CURL: Contrastive Unsupervised Representations for Reinforcement Learning](https://arxiv.org/abs/2004.04136), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/curl_sac.py>)
- Model-base RL
  - [Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596), [code](https://github.com/keiohta/tf2rl/blob/master/tf2rl/experiments/mpc_trainer.py)
  - [Model-Ensemble Trust-Region Policy Optimization](https://arxiv.org/abs/1802.10592), [code](https://github.com/keiohta/tf2rl/blob/master/tf2rl/experiments/me_trpo_trainer.py)
- Imitation Learning
  - [Generative Adversarial Imitation Learning](<https://arxiv.org/abs/1606.03476>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/gail.py>)
  - [Spectral Normalization for Generative Adversarial Networks](<https://arxiv.org/abs/1802.05957>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/networks/spectral_norm_dense.py>)
  - [Generative Adversarial Imitation from Observation](<https://arxiv.org/abs/1807.06158>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/gail.py>)
  - [Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](<https://arxiv.org/abs/1810.00821>), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/vail.py>)

Also, some useful techniques are implemented:

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), [code](https://github.com/keiohta/tf2rl/blob/master/tf2rl/tools/vae.py)
- [D2RL](https://arxiv.org/abs/2010.09163), [code](<https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/d2rl_sac.py>)

## 2. Installation

There are several ways to install tf2rl.
The recommended way is "2.1 Install from PyPI".

If TensorFlow is already installed, we try to identify the best
version of [TensorFlow Probability](https://www.tensorflow.org/probability).

### 2.1 Install from PyPI

You can install `tf2rl` from [PyPI](https://pypi.org/project/tf2rl/):

```bash
$ pip install tf2rl
```

### 2.2 Install from Source Code
You can also install from source:

```bash
$ git clone https://github.com/keiohta/tf2rl.git tf2rl
$ cd tf2rl
$ pip install .
```

### 2.3 Preinstalled Docker Container
Instead of installing tf2rl on your (virtual) system, you can use
preinstalled Docker containers.

Only the first execution requires time to download the container image.

At the following commands, you need to replace `<version>` with the
version tag which you want to use.


#### 2.3.1 CPU Only

The following simple command starts preinstalled container.

```bash
$ docker run -it ghcr.io/keiohta/tf2rl/cpu:<version> bash
```

If you also want to mount your local directory `/local/dir/path` at
container `/mount/point`

```bash
$ docker run -it -v /local/dir/path:/mount/point ghcr.io/keiohta/tf2rl/cpu:<version> bash
```

#### 2.3.2 GPU Support (Linux Only, Experimental)

WARNING: We encountered unsolved errors when running ApeX multiprocess learning.

Requirements
- Linux
- NVIDIA GPU
  - TF2.2 compatible driver
- Docker 19.03 or later


The following simple command starts preinstalled container.

```bash
$ docker run --gpus all -it ghcr.io/keiohta/tf2rl/nvidia:<version> bash
```

If you also want to mount your local directory `/local/dir/path` at
container `/mount/point`


```bash
$ docker run --gpus all -it -v /local/dir/path:/mount/point ghcr.io/keiohta/tf2rl/nvidia:<version> bash
```


If your container can see GPU correctly, you can check inside
container by the following comand;

```bash
$ nvidia-smi
```


## 3. Getting started
Here is a quick example of how to train DDPG agent on a Pendulum environment:

```python
import gym
from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer


parser = Trainer.get_argument()
parser = DDPG.get_argument(parser)
args = parser.parse_args()

env = gym.make("Pendulum-v1")
test_env = gym.make("Pendulum-v1")
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

## 4. Usage
In basic usage, what you need is initializing one of the policy
classes and `Trainer` class.

As a option, tf2rl supports command line program style, so that you
can also pass configuration parameters from command line arguments.


### 4.1 Command Line Program Style

`Trainer` class and policy classes have class method `get_argument`,
which creates or updates
[ArgParser](https://docs.python.org/3/library/argparse.html) object.

You can parse the command line arguments with the
`ArgParser.parse_args` method, which returns `Namespace` object.

Policy's constructor option can be extracted from the `Namespace`
object explicitly. `Trainer` constructor accepts the `Namespace`
object.

```python
from tf2rl.algos.dqn import DQN
from tf2rl.experiments.trainer import Trainer

env = ... # Create gym.env like environment.

parser = DQN.get_argument(Trainer.get_argument())
args = parser.parse_args()

policy = DQN(enable_double_dqn = args.enable_double_dqn,
             enable_dueling_dqn = args.enable_dueling_dqn,
			 enable_noisy_dqn = args.enable_noisy_dqn)
trainer = Trainer(policy, env, args)
trainer()
```


### 4.2 Non Command Line Program Style (e.g. on Jupyter Notebook)

`ArgParser` doesn't fit the usage on Jupyter Notebook like
envrionment. `Trainer` constructor can accept `dict` as `args`
argument instead of `Namespace` object.

```python
from tf2rl.algos.dqn import DQN
from tf2rl.experiments.trainer import Trainer

env = ... # Create gym.env like environment.

policy = DQN( ... )
trainer = Trainer(policy, env, {"max_steps": int(1e+6), ... })
trainer()
```

### 4.3 Results
The `Trainer` class saves logs and models under
`<logdir>/%Y%m%dT%H%M%S.%f`. The default `logdir` is `"results"`, and
it can be changed by `--logdir` command argument or `"logdir"` key in
constructor `args`.

## 5. Citation
```
@misc{ota2020tf2rl,
  author = {Kei Ota},
  title = {TF2RL},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/keiohta/tf2rl/}}
}
```
