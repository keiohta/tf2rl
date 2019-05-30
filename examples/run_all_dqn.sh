#!/bin/bash
# DQN
bash run.sh srun -X -c 3 --gres gpu:1 python examples/run_dqn_atari.py --dir-suffix=dqn
# Prioritized DQN
bash run.sh srun -X -c 3 --gres gpu:1 python examples/run_dqn_atari.py --use-prioritized-rb --dir-suffix=per_dqn
# Dueling DDQN
bash run.sh srun -X -c 3 --gres gpu:1 python examples/run_dqn_atari.py --enable-double-dqn --enable-dueling-dqn --dir-suffix=dueling-ddqn
# Distributional DQN
bash run.sh srun -X -c 3 --gres gpu:1 python examples/run_dqn_atari.py --enable-categorical-dqn --dir-suffix=distributional_dqn
# Noisy Nets
bash run.sh srun -X -c 3 --gres gpu:1 python examples/run_dqn_atari.py --enable-noisy-dqn --dir-suffix=noisy
# Dueling DDQN + Distributional + Noisy
bash run.sh srun -X -c 3 --gres gpu:1 python examples/run_dqn_atari.py --enable-categorical-dqn --dir-suffix=distributional_dqn --enable-categorical-dqn --enable-noisy-dqn --use-prioritized-rb --dir-suffix=dueling_ddqn_distr_noisy_per