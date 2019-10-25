#!/bin/bash -eu

prefix="python -u"
common_arg="--gpu -1 --logging-level WARNING --max-steps 256 --batch-size 32 --dir-suffix TEST"
off_pol_arg="--n-warmup 64 --memory-capacity 256"

# DDPG variants
${prefix} examples/run_ddpg.py ${common_arg} ${off_pol_arg}
${prefix} examples/run_bi_res_ddpg.py ${common_arg} ${off_pol_arg}
${prefix} examples/run_td3.py ${common_arg} ${off_pol_arg}
${prefix} examples/run_sac.py ${common_arg} ${off_pol_arg}
${prefix} examples/run_sac_discrete.py ${common_arg} ${off_pol_arg}

# DQN variants
${prefix} examples/run_dqn.py ${common_arg} ${off_pol_arg}
${prefix} examples/run_dqn.py ${common_arg} ${off_pol_arg} --enable-double-dqn --enable-dueling-dqn
${prefix} examples/run_dqn.py ${common_arg} ${off_pol_arg} --enable-noisy-dqn
${prefix} examples/run_dqn.py ${common_arg} ${off_pol_arg} --enable-categorical-dqn
${prefix} examples/run_dqn.py ${common_arg} ${off_pol_arg} --enable-categorical-dqn --enable-dueling-dqn
# DQN variants for Atari
${prefix} examples/run_dqn_atari.py ${common_arg} ${off_pol_arg}
${prefix} examples/run_dqn_atari.py ${common_arg} ${off_pol_arg} --enable-double-dqn --enable-dueling-dqn
${prefix} examples/run_dqn_atari.py ${common_arg} ${off_pol_arg} --enable-noisy-dqn
${prefix} examples/run_dqn_atari.py ${common_arg} ${off_pol_arg} --enable-categorical-dqn
${prefix} examples/run_dqn_atari.py ${common_arg} ${off_pol_arg} --enable-categorical-dqn --enable-dueling-dqn

# ApeX
apex_arg="--gpu-explorer -1 --gpu-learner -1 --gpu-evaluator -1 --logging-level WARNING --n-training 4 --batch-size 32 --param-update-freq 1 --local-buffer-size 64 --test-freq 1"
${prefix} examples/run_apex_ddpg.py ${apex_arg} --n-env 1
${prefix} examples/run_apex_ddpg.py ${apex_arg} --n-env 64
${prefix} examples/run_apex_dqn.py ${apex_arg} --n-env 1
${prefix} examples/run_apex_dqn.py ${apex_arg} --n-env 64

# GAIL
# TODO: test run_gail_ddpg

# On-policy agents
on_pol_arg="--horizon 64"
${prefix} examples/run_vpg.py ${common_arg} ${on_pol_arg}
${prefix} examples/run_ppo.py ${common_arg} ${on_pol_arg}
${prefix} examples/run_ppo_atari.py ${common_arg} ${on_pol_arg}

# Clean generated files
rm -rf results/*TEST*
