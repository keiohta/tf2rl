# TF2RL: A Reinforcement Library for TensorFlow 2.0


# Installation
```bash
$ git clone https://github.com/keiohta/tf2rl.git tf2tl
$ cd tf2rl
$ pip install -U numpy pip
$ pip install .
```

If you want to run example codes, you need `pip install .[examples]`
which install additional dependencies.

If you are developer, set `-e` option, then local modification affects
your installation.

# Example
- Train DDPG agent
  - If you want to train with only CPU, set `--gpu -1`

```bash
$ cd examples # You must change directory to avoid importing local files.
$ python run_ddpg.py --gpu -1
```
