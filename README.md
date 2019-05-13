# TF2RL
TF2RL is a deep reinforcement learning library that implements various deep reinforcement learning algorithms using TensorFlow 2.0.

## Algorithms
Following algorithms are supported:
- DQN variants
    - Double DQN, Dueling DQN
- DDPG
- TD3
- SAC
- ApeX (DDPG or TD3 or SAC)

## Installation
```bash
$ git clone https://github.com/keiohta/tf2rl.git tf2rl
$ cd tf2rl
$ pip install -U numpy pip
$ pip install .
```

TF2RL is built on Google's TensorFlow and requires that either `tensorflow` or `tensorflow-gpu` is installed.
To include the TensorFlow with the installation of TF2RL, add the flag `tf` for the normal CPU version or `tf_gpu` for the GPU version.
Note that we DO'NT actually use TF2.0 but TF1.12 because it has not been officially released yet.
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
$ python run_ddpg.py --gpu -1
```
