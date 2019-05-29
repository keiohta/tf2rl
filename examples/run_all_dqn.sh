rm -rf results
python examples/run_dqn.py --dir-suffix=dqn
# DDQN
python examples/run_dqn.py --enable-double-dqn --dir-suffix=ddqn
# Dueling DDQN
python examples/run_dqn.py --enable-double-dqn --enable-dueling-dqn --dir-suffix=dueling_ddqn
# Noisy Nets
python examples/run_dqn.py --enable-noisy-dqn --dir-suffix=noisy
# Categorical DQN
python examples/run_dqn.py --enable-categorical-dqn --dir-suffix=categorical_dqn
# Dueling DDQN Noisy Nets
python examples/run_dqn.py --enable-double-dqn --enable-dueling-dqn --enable-noisy-dqn --dir-suffix=dueling_ddqn_noisy
# Dueling DDQN Categorical Noisy Nets
python examples/run_dqn.py --enable-double-dqn --enable-dueling-dqn --enable-noisy-dqn --enable-categorical-dqn --dir-suffix=dueling_ddqn_categorical_noisy