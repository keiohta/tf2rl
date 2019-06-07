#!/bin/bash
submit_command="sbatch -c 3 --gres gpu:1 examples/run.sh python -u" 
${submit_command} examples/run_dqn_atari.py --dir-suffix=dqn
${submit_command} examples/run_dqn_atari.py --use-prioritized-rb --dir-suffix=per_dqn
${submit_command} examples/run_dqn_atari.py --enable-double-dqn --enable-dueling-dqn --dir-suffix=dueling-ddqn
${submit_command} examples/run_dqn_atari.py --enable-categorical-dqn --dir-suffix=distributional_dqn
${submit_command} examples/run_dqn_atari.py --enable-noisy-dqn --dir-suffix=noisy
${submit_command} examples/run_dqn_atari.py --enable-categorical-dqn --dir-suffix=distributional_dqn --enable-categorical-dqn --enable-noisy-dqn --use-prioritized-rb --dir-suffix=dueling_ddqn_distr_noisy_per
